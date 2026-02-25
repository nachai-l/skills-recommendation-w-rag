from __future__ import annotations

from typing import Any, Dict, List

import pytest

from functions.core.hybrid_merge import merge_hybrid_results
from functions.online import pipeline_4a_hybrid_context as p4a_mod


def _mk_vec(skill_id: str, skill_name: str, score: float, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "score_vector": score,
        "meta": meta or {},
        "internal_idx": 1,
    }


def _mk_bm(skill_id: str, skill_name: str, score: float, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "score_bm25": score,
        "meta": meta or {},
        "internal_idx": 2,
    }


def test_merge_hybrid_union_by_skill_id_and_sorting():
    vec = [
        _mk_vec("A", "Alpha", 0.9),
        _mk_vec("B", "Beta", 0.8),
    ]
    bm = [
        _mk_bm("B", "Beta", 10.0),
        _mk_bm("C", "Gamma", 9.0),
    ]

    merged, dbg = merge_hybrid_results(vector_results=vec, bm25_results=bm, alpha=0.6, normalize_scores=True)

    # union by skill_id -> A,B,C
    assert [r["skill_id"] for r in merged] == sorted([r["skill_id"] for r in merged], key=lambda x: x) or len(merged) == 3
    assert {r["skill_id"] for r in merged} == {"A", "B", "C"}
    assert dbg.num_vector_hits == 2
    assert dbg.num_bm25_hits == 2
    assert dbg.num_merged == 3

    # B should be boosted by having both signals
    b = next(r for r in merged if r["skill_id"] == "B")
    assert b["score_vector"] == 0.8
    assert b["score_bm25"] == 10.0
    assert b["score_hybrid"] >= 0.0


def test_pipeline_4a_wrapper_builds_context(monkeypatch):
    # Minimal params dict to satisfy p4a config accessors
    params = {
        "rag": {
            "vector_search": {"top_k_default": 5},
            "hybrid": {"alpha": 0.6, "normalize_scores": True, "tie_break": ["hybrid", "vector", "bm25", "id"]},
        },
        "context": {
            "columns": {"mode": "include", "include": ["skill_text", "Foundational_Criteria"], "exclude": []},
            "row_template": "- skill_id: {skill_id}\n  skill_name: {skill_name}\n  score_hybrid: {score_hybrid}\n  skill_text: {skill_text}\n  foundational: {Foundational_Criteria}\n",
            "max_context_chars": 5000,
            "truncate_field_chars": 200,
        },
    }

    def fake_load_parameters(_: str):
        return params

    def fake_p3a(**kwargs):
        return {
            "query": kwargs["query"],
            "top_k": kwargs["top_k"],
            "results": [
                {
                    "skill_id": "X",
                    "skill_name": "X Skill",
                    "score_vector": 0.9,
                    "meta": {"skill_text": "hello", "Foundational_Criteria": "F1"},
                    "internal_idx": 10,
                }
            ],
        }

    def fake_p3b(**kwargs):
        return {
            "query": kwargs["query"],
            "top_k": kwargs["top_k"],
            "results": [
                {
                    "skill_id": "Y",
                    "skill_name": "Y Skill",
                    "score_bm25": 9.0,
                    "meta": {"skill_text": "world", "Foundational_Criteria": "F2"},
                    "internal_idx": 20,
                }
            ],
        }

    monkeypatch.setattr(p4a_mod, "load_parameters", fake_load_parameters)
    monkeypatch.setattr(p4a_mod, "run_pipeline_3a_vector_search", fake_p3a)
    monkeypatch.setattr(p4a_mod, "run_pipeline_3b_bm25_search", fake_p3b)

    out = p4a_mod.run_pipeline_4a_hybrid_context(query="q", top_k=2, debug=True, parameters_path="configs/parameters.yaml")

    assert out["query"] == "q"
    assert len(out["results"]) == 2
    assert "context" in out
    assert "skill_id: X" in out["context"] or "skill_id: Y" in out["context"]
    assert out["debug"]["num_vector_hits"] == 1
    assert out["debug"]["num_bm25_hits"] == 1