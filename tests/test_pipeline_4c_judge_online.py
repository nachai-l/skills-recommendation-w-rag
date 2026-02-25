from __future__ import annotations

from typing import Any, Dict

from functions.online.pipeline_4c_judge import run_pipeline_4c_judge


def test_pipeline_4c_online_calls_4b_and_core(monkeypatch):
    # fake load_parameters + repo_root
    monkeypatch.setattr("functions.online.pipeline_4c_judge.load_parameters", lambda _: {"dummy": True})
    monkeypatch.setattr("functions.online.pipeline_4c_judge.repo_root_from_parameters_path", lambda _: "/repo_root")

    # fake pipeline 4b output
    def fake_p4b(**kwargs):
        return {
            "cache_id": "p4b__abc",
            "llm_validated": {"analysis_summary": "ok", "recommended_skills": []},
            "context": "CTX",
        }

    monkeypatch.setattr("functions.online.pipeline_4c_judge.run_pipeline_4b_generate", fake_p4b)

    # capture args passed to core
    captured: Dict[str, Any] = {}

    def fake_core(**kwargs):
        captured.update(kwargs)
        return {
            "query": kwargs["query"],
            "cache_id": "p4c__xyz",
            "judge_validated": {"verdict": "PASS", "score": 90, "reasons": ["ok"]},
        }

    monkeypatch.setattr("functions.online.pipeline_4c_judge.run_pipeline_4c_judge_core", fake_core)

    out = run_pipeline_4c_judge(
        query="q",
        top_k=20,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        schema_model_name_generation="LLMOutput",
        judge_model_name="JudgeResult",
        include_retrieval_results=True,
        top_k_vector=20,
        top_k_bm25=20,
    )

    assert out["query"] == "q"
    assert out["cache_id"] == "p4c__xyz"
    assert out["judge_validated"]["verdict"] == "PASS"
    assert out["generation_cache_id"] == "p4b__abc"

    # ensure wiring correctness
    assert captured["repo_root"] == "/repo_root"
    assert captured["params"] == {"dummy": True}
    assert captured["llm_validated"]["analysis_summary"] == "ok"
    assert captured["context"] == "CTX"
    assert captured["judge_model_name"] == "JudgeResult"


def test_pipeline_4c_online_reuses_provided_p4b_payload(monkeypatch):
    # fake load_parameters + repo_root
    monkeypatch.setattr("functions.online.pipeline_4c_judge.load_parameters", lambda _: {"dummy": True})
    monkeypatch.setattr("functions.online.pipeline_4c_judge.repo_root_from_parameters_path", lambda _: "/repo_root")

    # ensure 4b is NOT called when p4b_payload is provided
    def fail_if_called(**kwargs):
        raise AssertionError("run_pipeline_4b_generate should not be called when p4b_payload is provided")

    monkeypatch.setattr("functions.online.pipeline_4c_judge.run_pipeline_4b_generate", fail_if_called)

    captured: Dict[str, Any] = {}

    def fake_core(**kwargs):
        captured.update(kwargs)
        return {
            "query": kwargs["query"],
            "cache_id": "p4c__xyz",
            "judge_validated": {"verdict": "PASS", "score": 90, "reasons": ["ok"]},
            "debug": {},
        }

    monkeypatch.setattr("functions.online.pipeline_4c_judge.run_pipeline_4c_judge_core", fake_core)

    provided_p4b = {
        "cache_id": "p4b__provided",
        "llm_validated": {"analysis_summary": "ok", "recommended_skills": []},
        "context": "CTX",
    }

    out = run_pipeline_4c_judge(
        query="q",
        top_k=20,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        schema_model_name_generation="LLMOutput",
        judge_model_name="JudgeResult",
        include_retrieval_results=True,
        top_k_vector=20,
        top_k_bm25=20,
        p4b_payload=provided_p4b,
    )

    assert out["query"] == "q"
    assert out["cache_id"] == "p4c__xyz"
    assert out["judge_validated"]["verdict"] == "PASS"
    assert out["generation_cache_id"] == "p4b__provided"

    # ensure core received provided p4b content
    assert captured["llm_validated"]["analysis_summary"] == "ok"
    assert captured["context"] == "CTX"
    assert out.get("debug", {}).get("used_provided_p4b_payload") is True