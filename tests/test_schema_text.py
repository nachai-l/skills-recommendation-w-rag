# tests/test_schema_text.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from functions.core.schema_text import (
    build_llm_schema_txt_from_py_file,
    extract_public_schema_text_from_py,
)


MIN_SCHEMA_PY = """
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_name: str = Field(...)

class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["PASS","FAIL"] = Field(...)
    reasons: List[str] = Field(default_factory=list)

__all__ = ["LLMOutput", "JudgeResult"]
"""


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_build_llm_schema_txt_from_py_file_returns_json_schema_text(tmp_path: Path):
    py = _write(tmp_path, "schema/llm_schema.py", MIN_SCHEMA_PY)

    txt = build_llm_schema_txt_from_py_file(py)
    assert txt.strip().startswith("{")

    obj = json.loads(txt)
    assert obj["title"] == "LLM Schema"
    assert "models" in obj
    assert "LLMOutput" in obj["models"]
    assert "JudgeResult" in obj["models"]

    llm = obj["models"]["LLMOutput"]
    assert "properties" in llm
    assert "question_name" in llm["properties"]


def test_build_llm_schema_txt_from_py_file_raises_if_missing(tmp_path: Path):
    missing = tmp_path / "nope.py"
    with pytest.raises(FileNotFoundError):
        build_llm_schema_txt_from_py_file(missing)


def test_extract_public_schema_text_from_py_fallback_strips_fences():
    raw = "x\n```python\nfrom pydantic import BaseModel\n\nclass A(BaseModel):\n  x: int\n```\ny\n"
    out = extract_public_schema_text_from_py(raw)
    assert "```" not in out
    assert "class A" in out
