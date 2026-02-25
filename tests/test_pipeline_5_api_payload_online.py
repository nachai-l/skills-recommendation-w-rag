from __future__ import annotations

from typing import Any, Dict

import pytest

from functions.online.pipeline_5_api_payload import run_pipeline_5_api_payload


def test_pipeline_5_builds_payload_and_ignores_judge_details(monkeypatch):
    # fake 4b
    def fake_4b(**kwargs):
        return {
            "query": kwargs["query"],
            "top_k": 10,
            "cache_id": "p4b__x",
            "llm_validated": {
                "analysis_summary": "sum",
                "recommended_skills": [
                    {"skill_name": "data science", "relevance_score": 0.9, "reasoning": "r", "evidence": ["e"]}
                ],
            },
            "retrieval_results": [
                {
                    "skill_id": "745_cluster",
                    "skill_name": "data science",
                    "source": "cluster",
                    "meta": {"skill_text": "text", "Foundational_Criteria": "F"},
                }
            ],
            "context": "CTX",
        }

    # fake 4c: PASS
    def fake_4c(**kwargs):
        return {"judge_validated": {"verdict": "PASS", "score": 90, "reasons": ["ok"]}, "cache_id": "p4c__y"}

    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4b_generate", fake_4b)
    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4c_judge", fake_4c)

    out = run_pipeline_5_api_payload(
        query="q",
        top_k=10,
        debug=True,
        require_judge_pass=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
    )

    assert "payload" in out
    assert out["payload"]["query"] == "q"
    assert len(out["payload"]["recommended_skills"]) == 1
    assert out["payload"]["recommended_skills"][0]["Foundational_Criteria"] == "F"
    assert out["meta"]["generation_cache_id"] == "p4b__x"


def test_pipeline_5_blocks_when_judge_fails(monkeypatch):
    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4b_generate",
        lambda **kwargs: {
            "query": kwargs["query"],
            "top_k": 10,
            "cache_id": "p4b__x",
            "llm_validated": {"analysis_summary": "sum", "recommended_skills": []},
            "retrieval_results": [],
            "context": "CTX",
        },
    )
    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4c_judge",
        lambda **kwargs: {"judge_validated": {"verdict": "FAIL", "score": 10, "reasons": ["bad"]}},
    )

    with pytest.raises(RuntimeError):
        run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=True)