from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

import yaml
from pydantic import BaseModel

from functions.llm.client import build_gemini_client
from functions.llm.prompts import build_common_variables, json_dumps_stable
from functions.llm.runner import run_prompt_yaml_json
from functions.utils.paths import resolve_path


def _get(cfg: Any, key: str) -> Any:
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def _get_path(cfg: Any, path: Sequence[str]) -> Any:
    cur = cfg
    for k in path:
        cur = _get(cur, k)
    return cur


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


def load_schema_model(schema_py_path: str | Path, *, model_name: str) -> Type[BaseModel]:
    """
    Load schema/llm_schema.py and return the requested Pydantic model.

    IMPORTANT: call model_rebuild() with module namespace to avoid
    'not fully defined' errors in Pydantic v2.
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
        raise AttributeError(f"Schema module has no model named {model_name!r}. Available: {candidates}")

    if not issubclass(Model, BaseModel):
        raise TypeError(f"Schema export {model_name!r} is not a Pydantic BaseModel")

    # Pydantic v2: ensure typing symbols / forward refs resolve
    try:
        Model.model_rebuild(_types_namespace=mod.__dict__)
    except TypeError:
        Model.model_rebuild()

    return Model


def build_cache_id_p4c(
    *,
    query: str,
    context: str,
    output_json: str,
    prompt_path: str,
    model_name: str,
    temperature: float,
) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\n--prompt--\n")
    h.update(prompt_path.encode("utf-8"))
    h.update(b"\n--model--\n")
    h.update(model_name.encode("utf-8"))
    h.update(b"\n--temp--\n")
    h.update(str(temperature).encode("utf-8"))
    h.update(b"\n--output_json--\n")
    h.update(output_json.encode("utf-8"))
    h.update(b"\n--context--\n")
    h.update(context.encode("utf-8"))
    return f"p4c__{h.hexdigest()}"


def run_pipeline_4c_judge_core(
    *,
    params: Any,
    repo_root: str | Path,
    query: str,
    llm_validated: Dict[str, Any],
    context: str,
    parameters_path: str,
    credentials_path: str,
    judge_model_name: str,
    debug: bool,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)

    # Paths
    judge_prompt_path_raw = str(_get_path(params, ["prompts", "judge", "path"]))
    schema_py_raw = str(_get_path(params, ["llm_schema", "py_path"]))
    schema_txt_raw = str(_get_path(params, ["llm_schema", "txt_path"]))

    judge_prompt_path = resolve_path(judge_prompt_path_raw, base_dir=repo_root)
    schema_py_path = resolve_path(schema_py_raw, base_dir=repo_root)
    schema_txt_path = resolve_path(schema_txt_raw, base_dir=repo_root)

    # LLM config (reuse same model for now; later you can add llm_judge section)
    llm_cfg = _get(params, "llm")
    model_name_override = str(_get(llm_cfg, "model_name"))
    temperature = float(_get(llm_cfg, "temperature"))
    max_retries = int(_get(llm_cfg, "max_retries"))

    # Cache
    cache_cfg = _get(params, "cache")
    cache_dir_raw = str(_get(cache_cfg, "dir"))
    cache_enabled = bool(_get(cache_cfg, "enabled"))
    cache_force = bool(_get(cache_cfg, "force"))
    dump_failures = bool(_get(cache_cfg, "dump_failures"))
    cache_dir = resolve_path(cache_dir_raw, base_dir=repo_root)

    # Stable JSON for injection
    output_json = json_dumps_stable(llm_validated)

    # Optional: schema txt for judge prompt injection (depends on judge.yaml)
    llm_schema_txt = _read_text(schema_txt_path)

    # Variables (MOST COMMON placeholders used by judge prompts)
    # - {output_json}
    # - {context}
    # - (optional) {llm_schema}
    # - {query}
    variables = build_common_variables(
        context=str(context or ""),
        llm_schema=llm_schema_txt,
        output_json=output_json,
        extra={"query": (query or "").strip()},
    )

    # Schema model for judge output
    judge_schema_model = load_schema_model(schema_py_path, model_name=judge_model_name)

    # Gemini client ctx
    credentials = _load_yaml(resolve_path(credentials_path, base_dir=repo_root))
    client_ctx = build_gemini_client(credentials, model_name_override=model_name_override)

    # cache_id
    cache_id = build_cache_id_p4c(
        query=(query or "").strip(),
        context=str(context or ""),
        output_json=output_json,
        prompt_path=str(judge_prompt_path),
        model_name=client_ctx["model_name"],
        temperature=temperature,
    )

    parsed = run_prompt_yaml_json(
        prompt_path=str(judge_prompt_path),
        variables=variables,
        schema_model=judge_schema_model,
        client_ctx=client_ctx,
        temperature=temperature,
        max_retries=max_retries,
        cache_dir=str(cache_dir),
        cache_id=cache_id,
        force=cache_force,
        write_cache=cache_enabled,
        dump_failures=dump_failures,
    )

    payload: Dict[str, Any] = {
        "query": (query or "").strip(),
        "cache_id": cache_id,
        "judge_validated": parsed.model_dump(),
    }

    if debug:
        payload["debug"] = {
            "judge_prompt_path": str(judge_prompt_path),
            "judge_model_name": judge_model_name,
            "model_name": client_ctx["model_name"],
            "temperature": temperature,
            "max_retries": max_retries,
            "cache_dir": str(cache_dir),
            "cache_enabled": cache_enabled,
            "cache_force": cache_force,
            "dump_failures": dump_failures,
        }

    return payload


__all__ = ["run_pipeline_4c_judge_core", "build_cache_id_p4c", "load_schema_model"]