# functions/core/schema_runtime.py
"""
Schema Runtime Loader

Intent
- Load schema/llm_schema.py dynamically at runtime and resolve required Pydantic models.
- Centralize the import/validation logic so Pipeline 4 (and later Pipeline 5/6) stay thin.

Contract
- Required model:
    - LLMOutput (pydantic.BaseModel subclass)
- Optional model:
    - JudgeResult (pydantic.BaseModel subclass)
"""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
from typing import Optional, Tuple, Type

from pydantic import BaseModel


def load_schema_module(schema_py_path: str | Path):
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
