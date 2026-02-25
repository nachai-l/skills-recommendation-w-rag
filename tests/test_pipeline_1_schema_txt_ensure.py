# tests/test_pipeline_1_schema_txt_ensure.py

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import functions.batch.pipeline_1_schema_txt_ensure as p1


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _mk_params(
    *,
    py_path: str = "schema/llm_schema.py",
    txt_path: str = "schema/llm_schema.txt",
    archive_dir: str = "archived/",
    force_regenerate: bool = False,
):
    return SimpleNamespace(
        llm_schema=SimpleNamespace(
            py_path=py_path,
            txt_path=txt_path,
            archive_dir=archive_dir,
            force_regenerate=force_regenerate,
        ),
        # ensure_dirs reads these but we monkeypatch ensure_dirs to no-op
        cache=SimpleNamespace(dir="artifacts/cache"),
        artifacts=SimpleNamespace(
            outputs_dir="artifacts/outputs",
            reports_dir="artifacts/reports",
            logs_dir="artifacts/logs",
        ),
        outputs=SimpleNamespace(
            psv_path="artifacts/outputs/output.psv",
            jsonl_path="artifacts/outputs/output.jsonl",
        ),
        report=SimpleNamespace(
            md_path="artifacts/reports/report.md",
            html_path="artifacts/reports/report.html",
        ),
    )


MIN_SCHEMA_PY = """
from pydantic import BaseModel, ConfigDict

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: str

class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    y: str
"""


def test_pipeline_1_generates_txt_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "schema/llm_schema.py", MIN_SCHEMA_PY)

    params = _mk_params(force_regenerate=False)

    monkeypatch.setattr(p1, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p1, "ensure_dirs", lambda *a, **k: None)

    assert p1.main() == 0

    out_txt = tmp_path / "schema/llm_schema.txt"
    assert out_txt.exists()

    t = out_txt.read_text(encoding="utf-8")
    obj = json.loads(t)

    assert obj["title"] == "LLM Schema"
    assert "models" in obj
    assert "LLMOutput" in obj["models"]
    assert "JudgeResult" in obj["models"]


def test_pipeline_1_noop_when_txt_exists_and_not_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "schema/llm_schema.py", MIN_SCHEMA_PY)
    _write(tmp_path, "schema/llm_schema.txt", "EXISTING\n")

    params = _mk_params(force_regenerate=False)

    monkeypatch.setattr(p1, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p1, "ensure_dirs", lambda *a, **k: None)

    assert p1.main() == 0
    assert (tmp_path / "schema/llm_schema.txt").read_text(encoding="utf-8") == "EXISTING\n"
