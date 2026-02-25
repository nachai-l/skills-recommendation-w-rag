from __future__ import annotations

import pytest

from functions.core.api_payload import build_api_payload_from_4b


def test_build_api_payload_joins_meta_by_skill_name():
    llm_validated = {
        "analysis_summary": "x",
        "recommended_skills": [
            {"skill_name": "Data Science", "relevance_score": 0.9, "reasoning": "r", "evidence": ["e"]},
            {"skill_name": "DS 5", "relevance_score": 0.8, "reasoning": "r2", "evidence": ["e2"]},
        ],
    }
    retrieval_results = [
        {
            "skill_id": "745_cluster",
            "skill_name": "data science",
            "source": "cluster",
            "score_hybrid": 0.6,
            "meta": {
                "skill_text": "text1",
                "Foundational_Criteria": "F",
                "Intermediate_Criteria": "I",
                "Advanced_Criteria": "A",
            },
        },
        {
            "skill_id": "KSTZ...",
            "skill_name": "ds 5",
            "source": "lightcast",
            "score_hybrid": 0.5,
            "meta": {"skill_text": "text2"},
        },
    ]

    payload, dbg = build_api_payload_from_4b(
        query="q",
        llm_validated=llm_validated,
        retrieval_results=retrieval_results,
        top_k=10,
    )

    assert payload["query"] == "q"
    assert len(payload["recommended_skills"]) == 2
    assert dbg.num_missing_meta == 0

    a = payload["recommended_skills"][0]
    assert a["skill_name"] == "Data Science"
    assert a["skill_id"] == "745_cluster"
    assert a["skill_text"] == "text1"
    assert a["Foundational_Criteria"] == "F"


def test_build_api_payload_missing_meta_allowed_by_default():
    llm_validated = {
        "analysis_summary": "x",
        "recommended_skills": [{"skill_name": "Unknown Skill", "relevance_score": 0.5, "reasoning": "r", "evidence": []}],
    }
    payload, dbg = build_api_payload_from_4b(
        query="q",
        llm_validated=llm_validated,
        retrieval_results=[],
        top_k=10,
        require_all_meta=False,
    )
    assert len(payload["recommended_skills"]) == 1
    assert dbg.num_missing_meta == 1


def test_build_api_payload_missing_meta_can_fail_when_required():
    llm_validated = {
        "analysis_summary": "x",
        "recommended_skills": [{"skill_name": "Unknown Skill", "relevance_score": 0.5, "reasoning": "r", "evidence": []}],
    }
    with pytest.raises(ValueError):
        build_api_payload_from_4b(
            query="q",
            llm_validated=llm_validated,
            retrieval_results=[],
            top_k=10,
            require_all_meta=True,
        )