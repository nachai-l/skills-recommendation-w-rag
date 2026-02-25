from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import BaseModel, ConfigDict

from functions.core.llm_generate import (
    build_cache_id_p4b,
    load_schema_model,
    run_pipeline_4b_generate_core,
)


def _write_schema_py(tmp_path: Path, *, model_name: str = "Output") -> Path:
    """
    Write a minimal Pydantic v2 schema module exporting `Output`.
    """
    p = tmp_path / "llm_schema.py"
    p.write_text(
        f"""
from pydantic import BaseModel, ConfigDict

class {model_name}(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_summary: str
    recommended_skills: list[dict]
""".strip(),
        encoding="utf-8",
    )
    return p


def _write_schema_txt(tmp_path: Path) -> Path:
    p = tmp_path / "llm_schema.txt"
    p.write_text("SCHEMA_TEXT", encoding="utf-8")
    return p


def _write_prompt_yaml(tmp_path: Path) -> Path:
    """
    Minimal prompt YAML matching your runner requirement: must have `user`.
    Uses {llm_schema}, {query}, {context}.
    """
    p = tmp_path / "generation.yaml"
    p.write_text(
        """
name: test_prompt
version: 1
system: |
  System block
user: |
  {llm_schema}
  {query}
  {context}
""".strip(),
        encoding="utf-8",
    )
    return p


def test_load_schema_model_ok(tmp_path: Path):
    schema_py = _write_schema_py(tmp_path)
    Model = load_schema_model(schema_py, model_name="Output")
    assert issubclass(Model, BaseModel)

    inst = Model.model_validate({"analysis_summary": "ok", "recommended_skills": []})
    assert inst.analysis_summary == "ok"


def test_load_schema_model_missing_export_raises(tmp_path: Path):
    schema_py = _write_schema_py(tmp_path, model_name="NotOutput")
    with pytest.raises(AttributeError):
        load_schema_model(schema_py, model_name="Output")


def test_build_cache_id_p4b_is_deterministic():
    cid1 = build_cache_id_p4b(
        query="q",
        context="ctx",
        llm_schema_txt="schema",
        prompt_path="prompts/generation.yaml",
        model_name="m",
        temperature=0.3,
        top_k=10,
    )
    cid2 = build_cache_id_p4b(
        query="q",
        context="ctx",
        llm_schema_txt="schema",
        prompt_path="prompts/generation.yaml",
        model_name="m",
        temperature=0.3,
        top_k=10,
    )
    assert cid1 == cid2
    assert cid1.startswith("p4b__")


def test_run_pipeline_4b_generate_core_happy_path(monkeypatch, tmp_path: Path):
    """
    End-to-end core behavior with all external dependencies mocked.
    Validates injection variable keys and runner call parameters.
    """
    # ---- Arrange repo_root with files ----
    repo_root = tmp_path

    prompt_path = _write_prompt_yaml(repo_root)
    schema_txt_path = _write_schema_txt(repo_root)
    schema_py_path = _write_schema_py(repo_root)

    # Credentials yaml required by build_gemini_client (mocked anyway)
    cred_path = repo_root / "credentials.yaml"
    cred_path.write_text("gemini:\n  api_key_env: GEMINI_API_KEY\n", encoding="utf-8")

    # Minimal params dict matching your core logic
    params: Dict[str, Any] = {
        "rag": {"vector_search": {"top_k_default": 5}},
        "prompts": {"generation": {"path": str(prompt_path)}},
        "llm_schema": {"txt_path": str(schema_txt_path), "py_path": str(schema_py_path)},
        "llm": {"model_name": "gemini-3-flash-preview", "temperature": 1.0, "max_retries": 5},
        "cache": {"dir": "artifacts/cache", "enabled": True, "force": False, "dump_failures": True},
    }

    # ---- Mock Pipeline 4a ----
    def fake_p4a(**kwargs):
        return {
            "query": kwargs["query"],
            "top_k": kwargs["top_k"],
            "alpha": 0.6,
            "results": [{"skill_id": "1", "meta": {"skill_text": "x"}}],
            "context": "CTX_BLOCK",
        }

    monkeypatch.setattr(
        "functions.core.llm_generate.run_pipeline_4a_hybrid_context",
        fake_p4a,
    )

    # ---- Mock Gemini client creation ----
    def fake_build_client(credentials, model_name_override=None):
        return {"client": object(), "model_name": model_name_override or "model"}

    monkeypatch.setattr(
        "functions.core.llm_generate.build_gemini_client",
        fake_build_client,
    )

    # ---- Mock runner ----
    # We validate variable injection keys match prompt placeholders.
    class DummyOut(BaseModel):
        model_config = ConfigDict(extra="forbid")
        analysis_summary: str
        recommended_skills: list[dict]

    def fake_run_prompt_yaml_json(
        prompt_path: str,
        variables: Dict[str, Any],
        schema_model,
        client_ctx: Dict[str, Any],
        **kwargs,
    ):
        assert Path(prompt_path).name == "generation.yaml"
        assert "llm_schema" in variables
        assert "context" in variables
        assert "query" in variables
        assert variables["llm_schema"] == "SCHEMA_TEXT"
        assert variables["context"] == "CTX_BLOCK"
        assert variables["query"] == "data scientist"

        # schema_model should be the imported Output model by default
        assert hasattr(schema_model, "model_validate")
        assert "model_name" in client_ctx

        # Return a model instance as runner does
        return DummyOut(analysis_summary="ok", recommended_skills=[])

    monkeypatch.setattr(
        "functions.core.llm_generate.run_prompt_yaml_json",
        fake_run_prompt_yaml_json,
    )

    # ---- Act ----
    out = run_pipeline_4b_generate_core(
        params=params,
        repo_root=repo_root,
        query="data scientist",
        top_k=10,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path=str(cred_path),
        schema_model_name="Output",
        include_retrieval_results=False,
        top_k_vector=20,
        top_k_bm25=20,
    )

    # ---- Assert ----
    assert out["query"] == "data scientist"
    assert out["top_k"] == 10
    assert out["alpha"] == 0.6
    assert out["cache_id"].startswith("p4b__")
    assert out["llm_validated"]["analysis_summary"] == "ok"
    assert out["llm_validated"]["recommended_skills"] == []
    assert "debug" in out


def test_run_pipeline_4b_generate_core_includes_retrieval_when_enabled(monkeypatch, tmp_path: Path):
    repo_root = tmp_path

    prompt_path = _write_prompt_yaml(repo_root)
    schema_txt_path = _write_schema_txt(repo_root)
    schema_py_path = _write_schema_py(repo_root)
    cred_path = repo_root / "credentials.yaml"
    cred_path.write_text("gemini:\n  api_key_env: GEMINI_API_KEY\n", encoding="utf-8")

    params: Dict[str, Any] = {
        "rag": {"vector_search": {"top_k_default": 5}},
        "prompts": {"generation": {"path": str(prompt_path)}},
        "llm_schema": {"txt_path": str(schema_txt_path), "py_path": str(schema_py_path)},
        "llm": {"model_name": "gemini-3-flash-preview", "temperature": 1.0, "max_retries": 5},
        "cache": {"dir": "artifacts/cache", "enabled": True, "force": False, "dump_failures": True},
    }

    monkeypatch.setattr(
        "functions.core.llm_generate.run_pipeline_4a_hybrid_context",
        lambda **kwargs: {
            "alpha": 0.6,
            "results": [{"skill_id": "A"}],
            "context": "CTX",
        },
    )
    monkeypatch.setattr(
        "functions.core.llm_generate.build_gemini_client",
        lambda credentials, model_name_override=None: {"client": object(), "model_name": model_name_override or "m"},
    )

    class DummyOut(BaseModel):
        model_config = ConfigDict(extra="forbid")
        analysis_summary: str
        recommended_skills: list[dict]

    monkeypatch.setattr(
        "functions.core.llm_generate.run_prompt_yaml_json",
        lambda **kwargs: DummyOut(analysis_summary="ok", recommended_skills=[]),
    )

    out = run_pipeline_4b_generate_core(
        params=params,
        repo_root=repo_root,
        query="q",
        top_k=None,
        debug=False,
        parameters_path="configs/parameters.yaml",
        credentials_path=str(cred_path),
        schema_model_name="Output",
        include_retrieval_results=True,
        top_k_vector=None,
        top_k_bm25=None,
    )

    assert "retrieval_results" in out
    assert "context" in out
    assert out["retrieval_results"] == [{"skill_id": "A"}]
    assert out["context"] == "CTX"