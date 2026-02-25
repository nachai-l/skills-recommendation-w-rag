# functions/core/llm_generate.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

import yaml
from pydantic import BaseModel

from functions.llm.client import build_gemini_client
from functions.llm.prompts import build_common_variables
from functions.llm.runner import run_prompt_yaml_json
from functions.online.pipeline_4a_hybrid_context import run_pipeline_4a_hybrid_context
from functions.utils.paths import resolve_path


# -----------------------------
# Config access helpers (core-safe)
# -----------------------------
def _get(cfg: Any, key: str) -> Any:
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def _get_path(cfg: Any, path: Sequence[str]) -> Any:
    cur = cfg
    for k in path:
        cur = _get(cur, k)
    return cur


# -----------------------------
# File helpers
# -----------------------------
def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be mapping/dict: {p}")
    return obj


# -----------------------------
# Schema loader (Pydantic model)
# -----------------------------
def load_schema_model(schema_py_path: str | Path, *, model_name: str = "Output") -> Type[BaseModel]:
    """
    Load schema/llm_schema.py and return the Pydantic model class.

    Default export expected: Output
    """
    schema_py_path = Path(schema_py_path)
    if not schema_py_path.exists():
        raise FileNotFoundError(f"Missing schema py: {schema_py_path}")

    spec = spec_from_file_location(schema_py_path.stem, str(schema_py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import schema module: {schema_py_path}")

    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    Model = getattr(mod, model_name, None)
    if Model is None:
        candidates = [k for k in dir(mod) if k and k[0].isupper()]
        raise AttributeError(
            f"Schema module has no model named {model_name!r}. Available candidates: {candidates}"
        )

    if not issubclass(Model, BaseModel):
        raise TypeError(f"Schema export {model_name!r} is not a Pydantic BaseModel")

    # Ensure the model is fully defined (Pydantic v2)
    # Provide the module namespace so typing symbols (List/Literal/etc) resolve.
    try:
        Model.model_rebuild(_types_namespace=mod.__dict__)
    except TypeError:
        # Older pydantic versions may not accept _types_namespace
        Model.model_rebuild()
    return Model


# -----------------------------
# Cache id (deterministic)
# -----------------------------
def build_cache_id_p4b(
    *,
    query: str,
    context: str,
    llm_schema_txt: str,
    prompt_path: str,
    model_name: str,
    temperature: float,
    top_k: int,
) -> str:
    """
    Deterministic cache_id for runner (filename-safe).
    """
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\n--prompt--\n")
    h.update(prompt_path.encode("utf-8"))
    h.update(b"\n--model--\n")
    h.update(model_name.encode("utf-8"))
    h.update(b"\n--temp--\n")
    h.update(str(temperature).encode("utf-8"))
    h.update(b"\n--topk--\n")
    h.update(str(top_k).encode("utf-8"))
    h.update(b"\n--schema--\n")
    h.update(llm_schema_txt.encode("utf-8"))
    h.update(b"\n--context--\n")
    h.update(context.encode("utf-8"))
    return f"p4b__{h.hexdigest()}"


# -----------------------------
# Result container (optional)
# -----------------------------
@dataclass
class Pipeline4BResult:
    query: str
    top_k: int
    alpha: float
    cache_id: str
    llm_validated: Dict[str, Any]
    retrieval_results: Optional[list[dict]] = None
    context: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


# -----------------------------
# Core runner
# -----------------------------
def run_pipeline_4b_generate_core(
    *,
    params: Any,
    repo_root: str | Path,
    query: str,
    top_k: Optional[int],
    debug: bool,
    parameters_path: str,
    credentials_path: str,
    schema_model_name: str,
    include_retrieval_results: bool,
    top_k_vector: Optional[int],
    top_k_bm25: Optional[int],
) -> Dict[str, Any]:
    """
    Core implementation for Pipeline 4b.
    Online wrapper should pass already-loaded params + repo_root.

    Returns: JSON-serializable dict payload (API-friendly).
    """
    repo_root = Path(repo_root)

    # --- Resolve top_k ---
    top_k_default = int(_get_path(params, ["rag", "vector_search", "top_k_default"]))
    k_out = int(top_k) if top_k is not None else top_k_default

    # --- Paths from parameters.yaml ---
    prompt_path_raw = str(_get_path(params, ["prompts", "generation", "path"]))
    schema_txt_raw = str(_get_path(params, ["llm_schema", "txt_path"]))
    schema_py_raw = str(_get_path(params, ["llm_schema", "py_path"]))

    prompt_path = resolve_path(prompt_path_raw, base_dir=repo_root)
    schema_txt_path = resolve_path(schema_txt_raw, base_dir=repo_root)
    schema_py_path = resolve_path(schema_py_raw, base_dir=repo_root)

    # --- LLM config ---
    llm_cfg = _get(params, "llm")
    model_name_override = str(_get(llm_cfg, "model_name"))
    temperature = float(_get(llm_cfg, "temperature"))
    max_retries = int(_get(llm_cfg, "max_retries"))

    # --- Cache config ---
    cache_cfg = _get(params, "cache")
    cache_dir_raw = str(_get(cache_cfg, "dir"))
    cache_enabled = bool(_get(cache_cfg, "enabled"))
    cache_force = bool(_get(cache_cfg, "force"))
    dump_failures = bool(_get(cache_cfg, "dump_failures"))
    cache_dir = resolve_path(cache_dir_raw, base_dir=repo_root)

    # 1) Build context via 4a (force meta so criteria always available)
    p4a = run_pipeline_4a_hybrid_context(
        query=query,
        top_k=k_out,
        debug=debug,
        parameters_path=parameters_path,
        include_meta=True,
        include_internal_idx=True,
        top_k_vector=top_k_vector,
        top_k_bm25=top_k_bm25,
    )
    context_text = str(p4a.get("context") or "")
    retrieval_results = p4a.get("results") or []

    # 2) Load schema text for prompt injection
    llm_schema_txt = _read_text(schema_txt_path)

    # 3) Prompt variables (safe brace renderer is inside prompts.py + runner)
    variables = build_common_variables(
        context=context_text,
        llm_schema=llm_schema_txt,
        extra={"query": (query or "").strip()},
    )

    # 4) Load schema model for validation
    schema_model = load_schema_model(schema_py_path, model_name=schema_model_name)

    # 5) Gemini client ctx from credentials.yaml + model override
    credentials = _load_yaml(resolve_path(credentials_path, base_dir=repo_root))
    client_ctx = build_gemini_client(credentials, model_name_override=model_name_override)

    # 6) Deterministic cache_id
    cache_id = build_cache_id_p4b(
        query=(query or "").strip(),
        context=context_text,
        llm_schema_txt=llm_schema_txt,
        prompt_path=str(prompt_path),
        model_name=client_ctx["model_name"],
        temperature=temperature,
        top_k=k_out,
    )

    # 7) Run runner (handles prompt render, JSON extraction, retries, validation, cache, failure dumps)
    parsed_model = run_prompt_yaml_json(
        prompt_path=str(prompt_path),
        variables=variables,
        schema_model=schema_model,
        client_ctx=client_ctx,
        temperature=temperature,
        max_retries=max_retries,
        cache_dir=str(cache_dir),
        cache_id=cache_id,
        force=cache_force,
        write_cache=cache_enabled,
        dump_failures=dump_failures,
    )

    validated_json = parsed_model.model_dump()

    payload: Dict[str, Any] = {
        "query": (query or "").strip(),
        "top_k": k_out,
        "alpha": float(p4a.get("alpha") or 0.0),
        "cache_id": cache_id,
        "llm_validated": validated_json,
    }

    if include_retrieval_results:
        payload["retrieval_results"] = retrieval_results
        payload["context"] = context_text

    if debug:
        payload["debug"] = {
            "prompt_path": str(prompt_path),
            "schema_txt_path": str(schema_txt_path),
            "schema_py_path": str(schema_py_path),
            "model_name": client_ctx["model_name"],
            "temperature": temperature,
            "max_retries": max_retries,
            "cache_dir": str(cache_dir),
            "cache_enabled": cache_enabled,
            "cache_force": cache_force,
            "dump_failures": dump_failures,
            "context_chars": len(context_text),
            "num_retrieval_results": len(retrieval_results),
        }

    return payload


__all__ = [
    "Pipeline4BResult",
    "build_cache_id_p4b",
    "load_schema_model",
    "run_pipeline_4b_generate_core",
]