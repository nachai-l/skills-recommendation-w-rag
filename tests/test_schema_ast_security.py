# tests/test_schema_ast_security.py
"""
Integration tests for schema AST validation security.

Verifies that:
- validate_schema_ast() rejects malicious code (os, sys, subprocess, eval, exec, etc.)
- validate_schema_ast() accepts valid Pydantic schema code
- Schema loading paths in llm_generate, llm_judge, and schema_text all enforce
  AST validation before executing the schema module
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from functions.core.schema_postprocess import validate_schema_ast


# ---------------------------------------------------------------------------
# Valid schemas (should PASS)
# ---------------------------------------------------------------------------

VALID_SCHEMA = """\
from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict

class SkillItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_name: str = Field(...)
    relevance_score: float = Field(..., ge=0.0, le=1.0)

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_summary: str
    recommended_skills: List[SkillItem]

class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["PASS", "FAIL"]
    score: int = Field(..., ge=0, le=100)
    reasons: List[str] = Field(default_factory=list)

__all__ = ["LLMOutput", "JudgeResult", "SkillItem"]
"""


def test_valid_schema_passes_ast_validation():
    validate_schema_ast(VALID_SCHEMA)


def test_valid_schema_with_module_docstring():
    code = '"""Schema module."""\n' + VALID_SCHEMA
    validate_schema_ast(code)


# ---------------------------------------------------------------------------
# Malicious schemas (should FAIL)
# ---------------------------------------------------------------------------

MALICIOUS_IMPORT_OS = """\
import os
from pydantic import BaseModel

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_IMPORT_SUBPROCESS = """\
import subprocess
from pydantic import BaseModel

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_FROM_OS = """\
from os import system
from pydantic import BaseModel

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_EVAL = """\
from pydantic import BaseModel

class LLMOutput(BaseModel):
    x: str = eval("'malicious'")
"""

MALICIOUS_EXEC = """\
from pydantic import BaseModel

exec("import os")

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_FUNCTION_DEF = """\
from pydantic import BaseModel

def backdoor():
    pass

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_OPEN = """\
from pydantic import BaseModel

data = open("/etc/passwd").read()

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_DUNDER_IMPORT = """\
from pydantic import BaseModel

mod = __import__("os")

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_IMPORTLIB = """\
import importlib
from pydantic import BaseModel

class LLMOutput(BaseModel):
    x: str
"""

MALICIOUS_BARE_CALL = """\
from pydantic import BaseModel

print("hello")

class LLMOutput(BaseModel):
    x: str
"""


@pytest.mark.parametrize(
    "code,desc",
    [
        (MALICIOUS_IMPORT_OS, "import os"),
        (MALICIOUS_IMPORT_SUBPROCESS, "import subprocess"),
        (MALICIOUS_FROM_OS, "from os import system"),
        (MALICIOUS_EVAL, "eval() usage"),
        (MALICIOUS_EXEC, "exec() usage"),
        (MALICIOUS_FUNCTION_DEF, "function definition at top level"),
        (MALICIOUS_OPEN, "open() usage"),
        (MALICIOUS_DUNDER_IMPORT, "__import__() usage"),
        (MALICIOUS_IMPORTLIB, "import importlib"),
        (MALICIOUS_BARE_CALL, "bare function call at top level"),
    ],
)
def test_malicious_code_rejected(code: str, desc: str):
    with pytest.raises(ValueError):
        validate_schema_ast(code)


def test_syntax_error_rejected():
    with pytest.raises(ValueError, match="syntax error"):
        validate_schema_ast("class Foo(:\n    pass")


# ---------------------------------------------------------------------------
# Integration: schema loading paths enforce AST validation
# ---------------------------------------------------------------------------

def _write_malicious_schema(tmp_path: Path) -> Path:
    p = tmp_path / "malicious_schema.py"
    p.write_text(MALICIOUS_IMPORT_OS, encoding="utf-8")
    return p


def test_llm_generate_load_schema_rejects_malicious(tmp_path: Path):
    from functions.core.llm_generate import load_schema_model

    schema_path = _write_malicious_schema(tmp_path)
    with pytest.raises(ValueError, match="Disallowed import"):
        load_schema_model(schema_path, model_name="LLMOutput")


def test_llm_judge_load_schema_rejects_malicious(tmp_path: Path):
    from functions.core.llm_judge import load_schema_model

    schema_path = _write_malicious_schema(tmp_path)
    with pytest.raises(ValueError, match="Disallowed import"):
        load_schema_model(schema_path, model_name="LLMOutput")


def test_schema_text_load_module_rejects_malicious(tmp_path: Path):
    from functions.core.schema_text import _load_module_from_path

    schema_path = _write_malicious_schema(tmp_path)
    with pytest.raises(ValueError, match="Disallowed import"):
        _load_module_from_path(schema_path)


# ---------------------------------------------------------------------------
# Positive integration: valid schema loads successfully
# ---------------------------------------------------------------------------

def _write_valid_schema(tmp_path: Path) -> Path:
    p = tmp_path / "valid_schema.py"
    p.write_text(VALID_SCHEMA, encoding="utf-8")
    return p


def test_llm_generate_load_schema_accepts_valid(tmp_path: Path):
    from functions.core.llm_generate import load_schema_model

    schema_path = _write_valid_schema(tmp_path)
    Model = load_schema_model(schema_path, model_name="LLMOutput")
    inst = Model.model_validate(
        {"analysis_summary": "ok", "recommended_skills": []}
    )
    assert inst.analysis_summary == "ok"


def test_llm_judge_load_schema_accepts_valid(tmp_path: Path):
    from functions.core.llm_judge import load_schema_model

    schema_path = _write_valid_schema(tmp_path)
    Model = load_schema_model(schema_path, model_name="JudgeResult")
    inst = Model.model_validate({"verdict": "PASS", "score": 90, "reasons": ["ok"]})
    assert inst.verdict == "PASS"
