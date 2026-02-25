from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import BaseModel, ConfigDict

from functions.core.llm_judge import (
    build_cache_id_p4c,
    load_schema_model,
    run_pipeline_4c_judge_core,
)


def _write_schema_py(tmp_path: Path) -> Path:
    """
    Write a schema module with JudgeResult + LLMOutput (to match your patterns),
    and include a typing import so we can validate model_rebuild behavior.
    """
    p = tmp_path / "llm_schema.py"
    p.write_text(
        """
from typing import List, Literal
from pydantic import BaseModel, ConfigDict

class RecommendedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_name: str
    relevance_score: float
    reasoning: str
    evidence: List[str]

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_summary: str
    recommended_skills: List[RecommendedSkill]

class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["PASS", "FAIL"]
    score: int
    reasons: List[str]
""".strip(),
        encoding="utf-8",
    )
    return p


def _write_schema_txt(tmp_path: Path) -> Path:
    p = tmp_path / "llm_schema.txt"
    p.write_text("SCHEMA_TEXT", encoding="utf-8")
    return p


def _write_judge_prompt_yaml(tmp_path: Path) -> Path:
    """
    Minimal judge prompt yaml: runner requires 'user'. Uses {llm_schema},{context},{output_json},{query}.
    """
    p = tmp_path / "judge.yaml"
    p.write_text(
        """
name: judge_test
version: 1
system: |
  system
user: |
  {llm_schema}
  {query}
  {context}
  {output_json}
""".strip(),
        encoding="utf-8",
    )
    return p


def _write_credentials_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "credentials.yaml"
    p.write_text("gemini:\n  api_key_env: GEMINI_API_KEY\n", encoding="utf-8")
    return p


def test_build_cache_id_p4c_deterministic():
    cid1 = build_cache_id_p4c(
        query="q",
        context="ctx",
        output_json='{"a":1}',
        prompt_path="prompts/judge.yaml",
        model_name="m",
        temperature=1.0,
    )
    cid2 = build_cache_id_p4c(
        query="q",
        context="ctx",
        output_json='{"a":1}',
        prompt_path="prompts/judge.yaml",
        model_name="m",
        temperature=1.0,
    )
    assert cid1 == cid2
    assert cid1.startswith("p4c__")


def test_load_schema_model_rebuilds_ok(tmp_path: Path):
    schema_py = _write_schema_py(tmp_path)
    Model = load_schema_model(schema_py, model_name="JudgeResult")
    assert issubclass(Model, BaseModel)
    # Validate it can actually validate (implies rebuild ok)
    inst = Model.model_validate({"verdict": "PASS", "score": 90, "reasons": ["ok"]})
    assert inst.verdict == "PASS"


def test_run_pipeline_4c_judge_core_happy_path(monkeypatch, tmp_path: Path):
    repo_root = tmp_path

    schema_py = _write_schema_py(repo_root)
    schema_txt = _write_schema_txt(repo_root)
    judge_prompt = _write_judge_prompt_yaml(repo_root)
    cred_path = _write_credentials_yaml(repo_root)

    # Minimal params needed by core
    params: Dict[str, Any] = {
        "prompts": {"judge": {"path": str(judge_prompt)}},
        "llm_schema": {"py_path": str(schema_py), "txt_path": str(schema_txt)},
        "llm": {"model_name": "gemini-3-flash-preview", "temperature": 1.0, "max_retries": 5},
        "cache": {"dir": "artifacts/cache", "enabled": True, "force": False, "dump_failures": True},
    }

    # fake build_gemini_client
    monkeypatch.setattr(
        "functions.core.llm_judge.build_gemini_client",
        lambda credentials, model_name_override=None: {"client": object(), "model_name": model_name_override or "m"},
    )

    # fake runner: validate variables injection keys
    class DummyJudge(BaseModel):
        model_config = ConfigDict(extra="forbid")
        verdict: str
        score: int
        reasons: list[str]

    def fake_run_prompt_yaml_json(
        prompt_path: str,
        variables: Dict[str, Any],
        schema_model,
        client_ctx: Dict[str, Any],
        **kwargs,
    ):
        assert Path(prompt_path).name == "judge.yaml"
        assert variables["llm_schema"] == "SCHEMA_TEXT"
        assert variables["query"] == "data scientist"
        assert variables["context"] == "CTX_BLOCK"
        assert "output_json" in variables and variables["output_json"].startswith("{")
        # Should be JudgeResult model imported from schema
        assert hasattr(schema_model, "model_validate")
        return DummyJudge(verdict="PASS", score=95, reasons=["ok"])

    monkeypatch.setattr("functions.core.llm_judge.run_prompt_yaml_json", fake_run_prompt_yaml_json)

    out = run_pipeline_4c_judge_core(
        params=params,
        repo_root=repo_root,
        query="data scientist",
        llm_validated={"analysis_summary": "x", "recommended_skills": []},
        context="CTX_BLOCK",
        parameters_path="configs/parameters.yaml",
        credentials_path=str(cred_path),
        judge_model_name="JudgeResult",
        debug=True,
    )

    assert out["query"] == "data scientist"
    assert out["cache_id"].startswith("p4c__")
    assert out["judge_validated"]["verdict"] == "PASS"
    assert out["judge_validated"]["score"] == 95
    assert isinstance(out["judge_validated"]["reasons"], list)
    assert "debug" in out