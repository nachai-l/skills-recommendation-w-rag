from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from functions.core.hybrid_merge import (
    HybridMergeDebug,
    merge_hybrid_results,
    normalize_minmax,
)


def _mk_vec(
    skill_id: str,
    skill_name: str = "",
    score_vector: float = 0.0,
    *,
    meta: Optional[Dict[str, Any]] = None,
    internal_idx: Optional[int] = 1,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "score_vector": score_vector,
        "internal_idx": internal_idx,
    }
    if meta is not None:
        r["meta"] = meta
    if source is not None:
        r["source"] = source
    return r


def _mk_bm(
    skill_id: str,
    skill_name: str = "",
    score_bm25: float = 0.0,
    *,
    meta: Optional[Dict[str, Any]] = None,
    internal_idx: Optional[int] = 2,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "score_bm25": score_bm25,
        "internal_idx": internal_idx,
    }
    if meta is not None:
        r["meta"] = meta
    if source is not None:
        r["source"] = source
    return r


# -------------------------
# normalize_minmax
# -------------------------
def test_normalize_minmax_empty():
    assert normalize_minmax([]) == []


def test_normalize_minmax_all_equal_returns_ones():
    out = normalize_minmax([5.0, 5.0, 5.0])
    assert out == [1.0, 1.0, 1.0]


def test_normalize_minmax_basic_range():
    out = normalize_minmax([2.0, 4.0, 6.0])
    assert out[0] == 0.0
    assert out[1] == 0.5
    assert out[2] == 1.0


# -------------------------
# merge_hybrid_results
# -------------------------
def test_merge_union_by_skill_id_and_defaults():
    vec = [_mk_vec("A", "Alpha", 0.9)]
    bm = [_mk_bm("B", "Beta", 10.0)]

    merged, dbg = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=0.6,
        normalize_scores=False,
    )

    assert {r["skill_id"] for r in merged} == {"A", "B"}

    a = next(r for r in merged if r["skill_id"] == "A")
    b = next(r for r in merged if r["skill_id"] == "B")

    assert a["score_vector"] == 0.9
    assert a["score_bm25"] == 0.0
    assert b["score_vector"] == 0.0
    assert b["score_bm25"] == 10.0

    assert isinstance(dbg, HybridMergeDebug)
    assert dbg.num_vector_hits == 1
    assert dbg.num_bm25_hits == 1
    assert dbg.num_merged == 2


def test_merge_prefers_vector_meta_when_both_present():
    vec_meta = {"skill_text": "vec_text", "Foundational_Criteria": "F_vec"}
    bm_meta = {"skill_text": "bm_text", "Foundational_Criteria": "F_bm"}

    vec = [_mk_vec("X", "X Skill", 0.9, meta=vec_meta)]
    bm = [_mk_bm("X", "X Skill", 10.0, meta=bm_meta)]

    merged, _ = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=0.6,
        normalize_scores=True,
    )

    x = merged[0]
    assert x["skill_id"] == "X"
    assert x["meta"]["skill_text"] == "vec_text"
    assert x["meta"]["Foundational_Criteria"] == "F_vec"


def test_merge_uses_bm25_meta_if_vector_meta_empty():
    vec = [_mk_vec("X", "X Skill", 0.9, meta={})]
    bm_meta = {"skill_text": "bm_text", "Foundational_Criteria": "F_bm"}
    bm = [_mk_bm("X", "X Skill", 10.0, meta=bm_meta)]

    merged, _ = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=0.6,
        normalize_scores=True,
    )
    x = merged[0]
    assert x["meta"]["skill_text"] == "bm_text"
    assert x["meta"]["Foundational_Criteria"] == "F_bm"


def test_merge_fills_missing_skill_name_and_source_from_bm25():
    vec = [_mk_vec("X", "", 0.9, meta={"skill_text": "vec"}, source="")]
    bm = [_mk_bm("X", "NameFromBM25", 10.0, meta={"skill_text": "bm"}, source="src_bm")]

    merged, _ = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=0.6,
        normalize_scores=True,
    )
    x = merged[0]
    assert x["skill_name"] == "NameFromBM25"
    assert x["source"] == "src_bm"


def test_merge_internal_idx_fields_captured():
    vec = [{"skill_id": "X", "skill_name": "X", "score_vector": 0.8, "internal_idx": 11, "meta": {}}]
    bm = [{"skill_id": "X", "skill_name": "X", "score_bm25": 9.0, "internal_idx": 22, "meta": {}}]

    merged, _ = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=0.6,
        normalize_scores=False,
    )
    x = merged[0]
    assert x["internal_idx_vector"] == 11
    assert x["internal_idx_bm25"] == 22


def test_merge_hybrid_score_formula_with_normalization_all_equal_lists():
    # all-equal -> norm becomes 1.0 for every item
    vec = [_mk_vec("A", "A", 0.5)]
    bm = [_mk_bm("A", "A", 2.0)]
    alpha = 0.6

    merged, _ = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=alpha,
        normalize_scores=True,
    )
    a = merged[0]
    assert a["score_vector_norm"] == 1.0
    assert a["score_bm25_norm"] == 1.0
    assert a["score_hybrid"] == pytest.approx(alpha * 1.0 + (1.0 - alpha) * 1.0)


def test_merge_stable_sort_tie_breaks_by_skill_id_when_scores_equal():
    # Make all scores equal so tie-break is skill_id asc
    vec = [
        _mk_vec("B", "B", 1.0),
        _mk_vec("A", "A", 1.0),
    ]
    bm: List[Dict[str, Any]] = []

    merged, _ = merge_hybrid_results(
        vector_results=vec,
        bm25_results=bm,
        alpha=0.6,
        normalize_scores=False,
    )

    assert [r["skill_id"] for r in merged] == ["A", "B"]