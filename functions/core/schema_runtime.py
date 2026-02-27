# functions/core/schema_runtime.py
"""
Schema runtime loader

Intent
- Dynamically load schema/llm_schema.py at runtime and resolve required Pydantic models.
- Centralize import + model resolution so pipelines stay thin.

Contract
- Required model:
  - LLMOutput (pydantic.BaseModel subclass)
- Optional model:
  - JudgeResult (pydantic.BaseModel subclass)

Notes
- This module only loads and resolves models. It does NOT:
  - generate schema files
  - postprocess schema text
  - perform AST safety validation
Those responsibilities live in schema generation/postprocess modules.
"""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
from typing import Optional, Tuple, Type

from pydantic import BaseModel


def load_schema_module(schema_py_path: str | Path):
    """
    Dynamically import a schema module from a .py path and return the loaded module.

    Raises:
      FileNotFoundError: if schema_py_path does not exist
      RuntimeError: if import spec cannot be created
    """
    p = Path(schema_py_path)
    if not p.exists():
        raise FileNotFoundError(f"Schema py not found: {p}")

    spec = importlib_util.spec_from_file_location("llm_schema_runtime", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {p}")

    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def resolve_schema_models(schema_py_path: str | Path) -> Tuple[Type[BaseModel], Optional[Type[BaseModel]]]:
    """
    Resolve schema models from a schema .py file.

    Returns:
      (LLMOutput_model, JudgeResult_model_or_None)

    Notes:
    - Calls model_rebuild() to resolve deferred annotations (Literal/ForwardRef/etc).
    - Passes the module namespace so Pydantic can resolve typing symbols imported
      inside the dynamically-loaded module.
    """
    mod = load_schema_module(schema_py_path)

    gen_model = getattr(mod, "LLMOutput", None)
    if gen_model is None or not isinstance(gen_model, type) or not issubclass(gen_model, BaseModel):
        raise RuntimeError("Schema must define a Pydantic BaseModel named 'LLMOutput'")

    judge_model = getattr(mod, "JudgeResult", None)
    if judge_model is not None:
        if not isinstance(judge_model, type) or not issubclass(judge_model, BaseModel):
            raise RuntimeError("'JudgeResult' exists but is not a Pydantic BaseModel")

    # Rebuild models to resolve deferred annotations (e.g. Literal, ForwardRef).
    # Pass the module namespace so Pydantic can find types like Literal that
    # were imported in the schema file but aren't in sys.modules (dynamic load).
    ns = vars(mod)
    gen_model.model_rebuild(_types_namespace=ns)
    if judge_model is not None:
        judge_model.model_rebuild(_types_namespace=ns)

    return gen_model, judge_model


__all__ = ["load_schema_module", "resolve_schema_models"]
