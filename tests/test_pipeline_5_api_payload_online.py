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

    with pytest.raises(RuntimeError, match="attempt"):
        run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=True)


# ---------------------------------------------------------------------------
# Judge retry tests
# ---------------------------------------------------------------------------

def _make_p4b_payload(query: str) -> Dict[str, Any]:
    return {
        "query": query,
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


def test_judge_retry_succeeds_after_initial_fail(monkeypatch):
    """Judge fails on first attempt, passes on second — payload is returned normally."""
    p4b_calls: list[bool] = []

    def fake_4b(**kwargs):
        p4b_calls.append(kwargs.get("force_regenerate", False))
        return _make_p4b_payload(kwargs["query"])

    verdicts = iter(["FAIL", "PASS"])

    def fake_4c(**kwargs):
        return {"judge_validated": {"verdict": next(verdicts), "score": 80, "reasons": []}}

    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4b_generate", fake_4b)
    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4c_judge", fake_4c)

    out = run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=True)

    assert "payload" in out
    # Initial P4b call (no force) + one retry (force=True)
    assert len(p4b_calls) == 2
    assert p4b_calls[0] is False
    assert p4b_calls[1] is True


def test_judge_retry_force_regenerate_on_every_retry(monkeypatch):
    """Every retry after the first must call P4b with force_regenerate=True."""
    p4b_calls: list[bool] = []

    def fake_4b(**kwargs):
        p4b_calls.append(kwargs.get("force_regenerate", False))
        return _make_p4b_payload(kwargs["query"])

    # Fail twice, pass on third attempt
    verdicts = iter(["FAIL", "FAIL", "PASS"])

    def fake_4c(**kwargs):
        return {"judge_validated": {"verdict": next(verdicts), "score": 70, "reasons": []}}

    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4b_generate", fake_4b)
    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4c_judge", fake_4c)

    out = run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=True)

    assert "payload" in out
    assert len(p4b_calls) == 3
    assert p4b_calls[0] is False          # initial call
    assert p4b_calls[1] is True           # retry 1
    assert p4b_calls[2] is True           # retry 2


def test_judge_retry_exhausted_raises_runtime_error(monkeypatch):
    """When judge always returns FAIL, RuntimeError is raised after all retries."""
    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4b_generate",
        lambda **kwargs: _make_p4b_payload(kwargs["query"]),
    )
    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4c_judge",
        lambda **kwargs: {"judge_validated": {"verdict": "FAIL", "score": 5, "reasons": ["bad"]}},
    )

    with pytest.raises(RuntimeError, match="attempt"):
        run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=True)


def test_judge_not_called_when_require_judge_pass_false(monkeypatch):
    """When require_judge_pass=False, P4c is never called regardless of its output."""
    judge_called = []

    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4b_generate",
        lambda **kwargs: _make_p4b_payload(kwargs["query"]),
    )

    def fake_4c(**kwargs):
        judge_called.append(True)
        return {"judge_validated": {"verdict": "FAIL", "score": 0, "reasons": []}}

    monkeypatch.setattr("functions.online.pipeline_5_api_payload.run_pipeline_4c_judge", fake_4c)

    out = run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=False)

    assert "payload" in out
    assert judge_called == []


def test_judge_attempts_in_debug_output(monkeypatch):
    """debug=True includes judge_attempts count in the debug block."""
    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4b_generate",
        lambda **kwargs: _make_p4b_payload(kwargs["query"]),
    )
    # Fail once, then pass
    verdicts = iter(["FAIL", "PASS"])
    monkeypatch.setattr(
        "functions.online.pipeline_5_api_payload.run_pipeline_4c_judge",
        lambda **kwargs: {"judge_validated": {"verdict": next(verdicts), "score": 80, "reasons": []}},
    )

    out = run_pipeline_5_api_payload(query="q", top_k=10, require_judge_pass=True, debug=True)

    assert out["debug"]["judge_attempts"] == 2