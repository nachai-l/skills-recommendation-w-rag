# functions/core/schema_text.py
"""
Schema Text Helpers (Pipeline 1)

Goal
- Produce prompt-ready schema text for `{llm_schema}`.
- Deterministic output (no LLM), JSON-only (preferred).

Approach (preferred)
- Import schema/llm_schema.py as a module
- Extract required exports: LLMOutput, JudgeResult
- Convert to JSON Schema using Pydantic v2 `model_json_schema()`
- Dump as stable JSON (sorted keys, fixed indent)

Important
- Avoid sys.modules collisions by using a unique module name per import.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable

from functions.core.schema_postprocess import validate_schema_ast

DEFAULT_EXPORTS = ("LLMOutput", "JudgeResult")


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _unique_module_name(py_path: Path) -> str:
    # unique per file path + content hash (stable-ish, avoids collisions)
    blob = (str(py_path.resolve()) + "|" + str(py_path.stat().st_mtime_ns)).encode("utf-8", errors="replace")
    h = sha1(blob).hexdigest()[:12]
    return f"llm_schema_runtime_{h}"


def _load_module_from_path(py_path: Path) -> Any:
    """
    Import a module from a file path safely, avoiding module-name collisions.
    """
    if not py_path.exists():
        raise FileNotFoundError(f"Schema .py not found: {py_path}")

    module_name = _unique_module_name(py_path)

    # Make sure we don't reuse a stale module object
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Validate schema code with AST safety check before execution
    source_code = py_path.read_text(encoding="utf-8")
    validate_schema_ast(source_code)

    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {py_path}")

    module = importlib.util.module_from_spec(spec)

    # Register in sys.modules so relative imports (if any) behave predictably
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as e:
        # Clean up on failure to avoid poisoning future imports
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Failed to load schema module {py_path}: {e}") from e

    return module


def build_prompt_schema_json_from_module(
    module: Any,
    required_exports: Iterable[str] = DEFAULT_EXPORTS,
) -> Dict[str, Any]:
    """
    Build a single JSON schema blob containing each required export's JSON schema.

    Output shape:
    {
      "title": "LLM Schema",
      "type": "object",
      "models": {
        "LLMOutput": { ...pydantic json schema... },
        "JudgeResult": { ...pydantic json schema... }
      }
    }
    """
    models: Dict[str, Any] = {}

    for name in required_exports:
        if not hasattr(module, name):
            raise RuntimeError(f"Schema module missing required export: {name}")

        model = getattr(module, name)

        if not hasattr(model, "model_json_schema"):
            raise RuntimeError(
                f"Export {name} is not a Pydantic v2 model (missing model_json_schema)."
            )

        models[name] = model.model_json_schema()

    return {
        "title": "LLM Schema",
        "type": "object",
        "models": models,
    }


def extract_python_code_from_text(text: str) -> str:
    """
    Fallback utility: extract python code if the input is fenced.
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


def extract_public_schema_text_from_py(py_code: str) -> str:
    """
    Conservative fallback: return cleaned python text (NOT prompt-ideal).
    """
    if not py_code:
        return ""

    py_code = extract_python_code_from_text(py_code)
    text = py_code.strip()

    # remove main footer (noise)
    text = re.sub(r"\nif __name__ == ['\"]__main__['\"]:\s*.*\Z", "", text, flags=re.DOTALL)

    return text.strip() + "\n" if text.strip() else ""


def build_llm_schema_txt_from_py_file(
    py_path: str | Path,
    required_exports: Iterable[str] = DEFAULT_EXPORTS,
    *,
    allow_fallback: bool = False,
) -> str:
    """
    Produce prompt-ready schema text for `{llm_schema}`.

    Default behavior:
    - MUST return JSON schema text (raises if import/json schema extraction fails)

    Optional:
    - allow_fallback=True returns cleaned python source if import fails
      (not recommended for production prompts)
    """
    p = Path(py_path)
    if not p.exists():
        raise FileNotFoundError(f"Schema .py not found: {p}")

    try:
        module = _load_module_from_path(p)
        schema_obj = build_prompt_schema_json_from_module(module, required_exports=required_exports)
        txt = _stable_json_dumps(schema_obj)
        if not txt.strip():
            raise RuntimeError("Derived schema JSON is empty.")
        return txt
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Failed to derive JSON schema from {p}: {exc}") from exc

        # fallback (deterministic but not prompt-ideal)
        py_code = p.read_text(encoding="utf-8")
        txt = extract_public_schema_text_from_py(py_code)
        if not txt.strip():
            raise RuntimeError("Derived fallback schema text is empty.")
        return txt
