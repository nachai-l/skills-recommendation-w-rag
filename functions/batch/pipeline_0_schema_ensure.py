"""
Pipeline 0 — Ensure schema/llm_schema.py exists.

Behavior
- If schema file exists and force_regenerate=false: no-op.
- If schema missing and auto_generate=true: generate via LLM prompt file.
- If force_regenerate=true: archive old schema and regenerate.

Forward compatibility
- If params.prompts.schema_auto_py_generation.path exists, use it
- Else fallback to "prompts/schema_auto_py_generation.yaml"

Schema generation context requirement (your rule)
- {context} is required by prompts/schema_auto_py_generation.yaml
- Use prompts/generation.yaml (rendered) as {context}
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional, Sequence

from functions.core.schema_postprocess import postprocess_schema_py, validate_schema_ast
from functions.utils.config import ensure_dirs, load_credentials, load_parameters
from functions.utils.logging import get_logger

DEFAULT_SCHEMA_AUTO_PROMPT_PATH = "prompts/schema_auto_py_generation.yaml"
DEFAULT_GENERATION_PROMPT_PATH = "prompts/generation.yaml"


def _get(obj: Any, dotted_path: str, default: Any = None) -> Any:
    """
    Safe nested getter supporting Pydantic models and dicts.
    Example: _get(params, "prompts.schema_auto_py_generation.path", None)
    """
    cur: Any = obj
    for part in dotted_path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        return default
    return cur if cur is not None else default


def _extract_python_code(text: str) -> str:
    """
    Extract python code from an LLM response.
    Accepts:
    - raw python
    - ```python fenced blocks
    - ``` fenced blocks (assumed python)
    If multiple blocks exist, uses the first python-ish block.
    """
    if not text:
        return ""

    m = re.search(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return text.strip()


def _archive_existing(schema_path: Path, archive_dir: Path) -> Optional[Path]:
    """
    Archive current schema file with timestamp.
    Returns archive path if archived.
    """
    if not schema_path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_path = archive_dir / f"{schema_path.stem}_{ts}{schema_path.suffix}"
    shutil.copy2(schema_path, archived_path)
    return archived_path


def _validate_importable(schema_path: Path) -> None:
    """
    Validate that schema module is importable.
    Also checks at least one Pydantic BaseModel is present (best-effort).
    """
    spec = importlib.util.spec_from_file_location("llm_schema_generated", str(schema_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {schema_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(f"Generated schema is not importable: {exc}") from exc

    found_model = False
    for _, v in vars(module).items():
        try:
            mro = getattr(v, "__mro__", ())
            if any(getattr(c, "__name__", "") == "BaseModel" for c in mro):
                found_model = True
                break
        except Exception:
            continue

    if not found_model:
        raise RuntimeError(
            "Generated schema imported successfully but no Pydantic BaseModel subclass was found. "
            "Ensure your schema prompt generates a BaseModel output schema."
        )


def _stable_id(parts: Sequence[str]) -> str:
    blob = "\n".join([p if p is not None else "" for p in parts]).encode("utf-8", errors="replace")
    return sha1(blob).hexdigest()


def _build_client_ctx(params: Any, creds: Any) -> dict:
    """
    Build the client_ctx required by functions.llm.runner.* APIs:
      client_ctx = {"client": <google.genai.Client>, "model_name": "..."}
    """
    try:
        from google import genai  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "google-genai is required for real pipeline execution. "
            "Install it (e.g. `pip install google-genai`) and try again."
        ) from e

    api_key_env = _get(creds, "gemini.api_key_env", "GEMINI_API_KEY")
    model_name = getattr(params.llm, "model_name", None) or _get(creds, "gemini.model_name", None)
    if not model_name:
        raise RuntimeError("Missing model_name. Set params.llm.model_name or credentials.gemini.model_name")

    api_key_val = os.getenv(str(api_key_env))
    if not api_key_val:
        raise RuntimeError(
            f"Missing Gemini API key in env var: {api_key_env}. "
            "Export it before running, e.g. `export GEMINI_API_KEY=...`"
        )

    client = genai.Client(api_key=api_key_val)
    return {"client": client, "model_name": model_name}


def _find_placeholders(text: str) -> set[str]:
    """
    Extract {placeholders} used by str.format() in a template string.
    Very small + safe: only captures {name} (no format specs parsing).
    """
    if not isinstance(text, str) or not text:
        return set()
    return set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)}", text))


def _render_generation_prompt_as_context(params: Any) -> str:
    """
    Build {context} for schema auto-generation by rendering prompts/generation.yaml.

    Important:
    - generation.yaml may contain placeholders beyond {context}/{llm_schema}/{output_json}
      e.g. {schema}, {role}, etc.
    - For Pipeline 0 we render it with safe empty values to avoid KeyError.
    """
    from functions.llm.prompts import load_prompt_file, render_prompt_blocks

    gen_prompt_path = _get(params, "prompts.generation.path", None) or DEFAULT_GENERATION_PROMPT_PATH
    p = Path(str(gen_prompt_path))
    if not p.exists():
        raise FileNotFoundError(f"Generation prompt not found for schema context: {p}")

    prompt_dict = load_prompt_file(p)

    # Collect placeholders across all blocks
    needed: set[str] = set()
    for k in ("system", "user", "assistant_prefix"):
        needed |= _find_placeholders(prompt_dict.get(k, "") or "")

    # Provide safe defaults so rendering cannot KeyError
    safe_vars = {name: "" for name in needed}

    # Always include these common ones (even if not detected)
    safe_vars.setdefault("context", "")
    safe_vars.setdefault("llm_schema", "")
    safe_vars.setdefault("output_json", "")
    safe_vars.setdefault("schema", "")

    rendered = render_prompt_blocks(prompt_dict, safe_vars).strip()

    return (
        "INPUT SPEC SOURCE: prompts/generation.yaml\n"
        "Below is the rendered generation prompt template (placeholders blank).\n"
        "Derive the expected JSON output shape from this.\n\n"
        f"{rendered}\n"
    )


def _call_llm_generate_schema(prompt_path: str | Path, params: Any, creds: Any) -> str:
    """
    REAL schema generation via canonical runner:
      functions.llm.runner.run_prompt_yaml_text(...)

    Required:
    - schema_auto_py_generation.yaml requires {context}
    - we provide {context} from rendered prompts/generation.yaml
    """
    from functions.llm.prompts import build_common_variables
    from functions.llm.runner import run_prompt_yaml_text

    client_ctx = _build_client_ctx(params=params, creds=creds)

    temperature = float(getattr(params.llm, "temperature", 0.3))
    max_retries = int(getattr(params.llm, "max_retries", 3))

    # cache:
    #   enabled: true
    #   dir: artifacts/cache
    #   force: false
    #   dump_failures: true
    cache_enabled = bool(getattr(params.cache, "enabled", True))
    cache_dir = str(getattr(params.cache, "dir", "artifacts/cache"))
    dump_failures = bool(getattr(params.cache, "dump_failures", True))

    schema_ctx = _render_generation_prompt_as_context(params)

    # Deterministic cache id for schema generation (include context source)
    cache_id = "pipeline0_schema_py__" + _stable_id(
        [
            str(prompt_path),
            str(_get(params, "prompts.generation.path", None) or DEFAULT_GENERATION_PROMPT_PATH),
            str(client_ctx.get("model_name", "")),
            str(temperature),
            str(max_retries),
        ]
    )

    variables = build_common_variables(context=schema_ctx, llm_schema="")

    text = run_prompt_yaml_text(
        prompt_path=str(prompt_path),
        variables=variables,
        client_ctx=client_ctx,
        temperature=temperature,
        max_retries=max_retries,
        cache_dir=cache_dir,
        cache_id=cache_id,
        force=bool(getattr(params.cache, "force", False)),
        write_cache=cache_enabled,
        dump_failures=dump_failures,
        strip_code_fences=True,  # still strip fences if model disobeys
        min_chars=20,
        must_contain=["BaseModel"],
    )

    # Extra safety: if prompt returns extra junk, extract a python block if present
    return _extract_python_code(text)


def main() -> int:
    logger = get_logger(__name__)

    params = load_parameters()
    creds = load_credentials()
    ensure_dirs(params)

    schema_path = Path(params.llm_schema.py_path)
    archive_dir = Path(params.llm_schema.archive_dir)

    auto_generate = bool(getattr(params.llm_schema, "auto_generate", True))
    force_regen = bool(getattr(params.llm_schema, "force_regenerate", False))

    if schema_path.exists() and not force_regen:
        logger.info("Pipeline 0: schema exists; no-op. path=%s", schema_path)
        return 0

    if not auto_generate and not schema_path.exists():
        raise RuntimeError(f"Schema missing but llm_schema.auto_generate=false. Missing path: {schema_path}")

    if schema_path.exists() and force_regen:
        archived = _archive_existing(schema_path, archive_dir)
        logger.info("Archived existing schema to: %s", archived)

    prompt_path_str = _get(params, "prompts.schema_auto_py_generation.path", None)
    prompt_path = Path(prompt_path_str) if prompt_path_str else Path(DEFAULT_SCHEMA_AUTO_PROMPT_PATH)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Schema auto-generation prompt not found: {prompt_path}")

    logger.info("Generating schema via prompt: %s", prompt_path)

    raw = _call_llm_generate_schema(prompt_path=prompt_path, params=params, creds=creds)

    # post-LLM deterministic hardening (fences + model_config + __all__)
    py_code = postprocess_schema_py(raw, required_exports=("LLMOutput", "JudgeResult"))

    if not py_code.strip():
        raise RuntimeError("LLM returned empty schema code.")

    try:
        validate_schema_ast(py_code)
    except ValueError as exc:
        raise RuntimeError(f"Generated schema failed static safety check: {exc}") from exc

    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(py_code.strip() + "\n", encoding="utf-8")

    logger.info("Wrote schema to: %s (bytes=%s)", schema_path, schema_path.stat().st_size)

    _validate_importable(schema_path)
    logger.info("Schema validated: importable + BaseModel found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
