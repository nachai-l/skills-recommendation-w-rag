# tests/test_schema_runtime.py

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from functions.core.schema_runtime import load_schema_module, resolve_schema_models


def test_load_schema_module_success(tmp_path: Path):
    schema_py = tmp_path / "llm_schema.py"
    schema_py.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel, ConfigDict

            class LLMOutput(BaseModel):
                model_config = ConfigDict(extra="forbid")
                foo: str
            """
        ).lstrip(),
        encoding="utf-8",
    )

    mod = load_schema_module(schema_py)
    assert hasattr(mod, "LLMOutput")


def test_load_schema_module_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_schema_module(tmp_path / "missing.py")


def test_resolve_schema_models_requires_llmoutput(tmp_path: Path):
    schema_py = tmp_path / "llm_schema.py"
    schema_py.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel

            class SomethingElse(BaseModel):
                x: str
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="LLMOutput"):
        resolve_schema_models(schema_py)


def test_resolve_schema_models_returns_optional_judgeresult(tmp_path: Path):
    schema_py = tmp_path / "llm_schema.py"
    schema_py.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel, ConfigDict

            class LLMOutput(BaseModel):
                model_config = ConfigDict(extra="forbid")
                foo: str

            class JudgeResult(BaseModel):
                model_config = ConfigDict(extra="forbid")
                passed: bool
                feedback: str = ""
            """
        ).lstrip(),
        encoding="utf-8",
    )

    gen_model, judge_model = resolve_schema_models(schema_py)
    assert gen_model.__name__ == "LLMOutput"
    assert judge_model is not None
    assert judge_model.__name__ == "JudgeResult"


def test_resolve_schema_models_rejects_invalid_judgeresult_type(tmp_path: Path):
    schema_py = tmp_path / "llm_schema.py"
    schema_py.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel, ConfigDict

            class LLMOutput(BaseModel):
                model_config = ConfigDict(extra="forbid")
                foo: str

            # Not a BaseModel
            class JudgeResult:
                pass
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="JudgeResult"):
        resolve_schema_models(schema_py)


def test_resolve_schema_models_rebuilds_deferred_literal(tmp_path: Path):
    """
    If a schema uses Literal but has deferred annotations (e.g. from __future__),
    model_rebuild() in resolve_schema_models should make the model usable.
    """
    schema_py = tmp_path / "llm_schema.py"
    schema_py.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from typing import Literal
            from pydantic import BaseModel, ConfigDict

            class LLMOutput(BaseModel):
                model_config = ConfigDict(extra="forbid")
                foo: str

            class JudgeResult(BaseModel):
                model_config = ConfigDict(extra="forbid")
                verdict: Literal["PASS", "FAIL"]
                feedback: str = ""
            """
        ).lstrip(),
        encoding="utf-8",
    )

    gen_model, judge_model = resolve_schema_models(schema_py)
    assert judge_model is not None
    # This would raise "not fully defined" without model_rebuild()
    instance = judge_model(verdict="PASS", feedback="ok")
    assert instance.verdict == "PASS"
