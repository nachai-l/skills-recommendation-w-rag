# functions/core/hybrid_merge.py
"""
Hybrid merge (Pipeline 4a) — deterministic union of vector + BM25 results.

Intent
- Merge FAISS (vector) and BM25 (lexical) retrieval outputs into a single ranked list.
- Produce a stable merged row shape used downstream for:
  - context rendering (4a)
  - LLM generation (4b)
  - API payload joins (5)

Core behaviors
- Union merge by `skill_id` (preferred key).
- Optional min-max normalization per modality (vector and BM25) to make scores comparable.
- Hybrid score: alpha * vector_norm + (1 - alpha) * bm25_norm
- Deterministic sorting / tie-break:
  score_hybrid desc -> score_vector desc -> score_bm25 desc -> skill_id asc

Meta policy
- Prefer FAISS meta when both exist (usually richer / canonical for full text fields).
- If FAISS meta is empty, allow BM25 meta to backfill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_minmax(scores: List[float]) -> List[float]:
    """
    Deterministic min-max normalization.

    Rules:
    - empty -> []
    - all-equal (min==max) -> all 1.0 (keeps signal rather than zeroing it)
    """
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    denom = mx - mn
    return [(s - mn) / denom for s in scores]


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best-effort float conversion (None/invalid -> default)."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _get_meta(result_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract meta dict from a result row robustly.
    Your core to_api_payload(...) likely uses 'meta', but we support a few fallbacks.
    """
    m = result_row.get("meta")
    if isinstance(m, dict):
        return m
    m = result_row.get("metadata")
    if isinstance(m, dict):
        return m
    return {}


def _get_skill_id(result_row: Dict[str, Any]) -> str:
    """Extract canonical skill_id (fallback to meta.skill_id)."""
    sid = result_row.get("skill_id")
    if sid:
        return str(sid)
    meta = _get_meta(result_row)
    if meta.get("skill_id"):
        return str(meta["skill_id"])
    # last resort (should not happen)
    return ""


def _get_skill_name(result_row: Dict[str, Any]) -> str:
    """Extract skill_name (fallback to meta.skill_name and legacy BM25 'title')."""
    name = result_row.get("skill_name")
    if name:
        return str(name)
    meta = _get_meta(result_row)
    if meta.get("skill_name"):
        return str(meta["skill_name"])
    # BM25 rows use 'title' in corpus; to_api_payload normally maps to skill_name already.
    if result_row.get("title"):
        return str(result_row["title"])
    return ""


def _get_source(result_row: Dict[str, Any]) -> str:
    """Extract source (fallback to meta.source)."""
    src = result_row.get("source")
    if src:
        return str(src)
    meta = _get_meta(result_row)
    if meta.get("source"):
        return str(meta["source"])
    return ""


def _get_internal_idx(result_row: Dict[str, Any]) -> Optional[int]:
    """Extract internal_idx if present and int-castable; else None."""
    v = result_row.get("internal_idx")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


@dataclass
class HybridMergeDebug:
    """Debug counters for merge coverage and raw score ranges."""
    num_vector_hits: int
    num_bm25_hits: int
    num_merged: int
    normalize_scores: bool
    alpha: float
    vector_raw_min: float
    vector_raw_max: float
    bm25_raw_min: float
    bm25_raw_max: float


def merge_hybrid_results(
    *,
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    alpha: float,
    normalize_scores: bool,
) -> Tuple[List[Dict[str, Any]], HybridMergeDebug]:
    """
    Union merge by skill_id with deterministic scoring + stable tie-break:
      score_hybrid desc -> score_vector desc -> score_bm25 desc -> skill_id asc
    """
    # -------------
    # Collect raw scores (for optional normalization and debug ranges)
    # -------------
    vec_scores_raw: List[float] = []
    bm_scores_raw: List[float] = []

    for r in vector_results:
        vec_scores_raw.append(_safe_float(r.get("score_vector"), 0.0))
    for r in bm25_results:
        bm_scores_raw.append(_safe_float(r.get("score_bm25"), 0.0))

    vec_norm = normalize_minmax(vec_scores_raw) if normalize_scores else vec_scores_raw
    bm_norm = normalize_minmax(bm_scores_raw) if normalize_scores else bm_scores_raw

    # -------------
    # Index by skill_id
    # -------------
    merged: Dict[str, Dict[str, Any]] = {}

    # Vector first (prefer FAISS meta if conflict)
    for i, r in enumerate(vector_results):
        sid = _get_skill_id(r)
        if not sid:
            continue

        meta = _get_meta(r)
        merged[sid] = {
            "skill_id": sid,
            "skill_name": _get_skill_name(r),
            "source": _get_source(r),
            "score_vector": _safe_float(r.get("score_vector"), 0.0),
            "score_bm25": 0.0,
            "score_vector_norm": _safe_float(vec_norm[i], 0.0),
            "score_bm25_norm": 0.0,
            "internal_idx_vector": _get_internal_idx(r),
            "internal_idx_bm25": None,
            # carry full meta (preferred)
            "meta": meta if isinstance(meta, dict) else {},
        }

    # BM25 next (fill gaps, but do not overwrite existing FAISS meta by default)
    for j, r in enumerate(bm25_results):
        sid = _get_skill_id(r)
        if not sid:
            continue

        meta = _get_meta(r)
        if sid in merged:
            merged[sid]["score_bm25"] = _safe_float(r.get("score_bm25"), 0.0)
            merged[sid]["score_bm25_norm"] = _safe_float(bm_norm[j], 0.0)
            merged[sid]["internal_idx_bm25"] = _get_internal_idx(r)

            # If FAISS meta is empty, allow BM25 meta to populate context
            if not merged[sid].get("meta") and isinstance(meta, dict):
                merged[sid]["meta"] = meta

            # If skill_name/source missing, fill from BM25
            if not merged[sid].get("skill_name"):
                merged[sid]["skill_name"] = _get_skill_name(r)
            if not merged[sid].get("source"):
                merged[sid]["source"] = _get_source(r)
        else:
            merged[sid] = {
                "skill_id": sid,
                "skill_name": _get_skill_name(r),
                "source": _get_source(r),
                "score_vector": 0.0,
                "score_bm25": _safe_float(r.get("score_bm25"), 0.0),
                "score_vector_norm": 0.0,
                "score_bm25_norm": _safe_float(bm_norm[j], 0.0),
                "internal_idx_vector": None,
                "internal_idx_bm25": _get_internal_idx(r),
                "meta": meta if isinstance(meta, dict) else {},
            }

    # -------------
    # Compute hybrid score
    # -------------
    for sid, row in merged.items():
        sv = _safe_float(row.get("score_vector_norm"), 0.0)
        sb = _safe_float(row.get("score_bm25_norm"), 0.0)
        row["score_hybrid"] = alpha * sv + (1.0 - alpha) * sb

    # -------------
    # Stable sorting
    # -------------
    merged_rows = list(merged.values())
    merged_rows.sort(
        key=lambda r: (
            -_safe_float(r.get("score_hybrid"), 0.0),
            -_safe_float(r.get("score_vector"), 0.0),
            -_safe_float(r.get("score_bm25"), 0.0),
            str(r.get("skill_id") or ""),
        )
    )

    dbg = HybridMergeDebug(
        num_vector_hits=len(vector_results),
        num_bm25_hits=len(bm25_results),
        num_merged=len(merged_rows),
        normalize_scores=bool(normalize_scores),
        alpha=float(alpha),
        vector_raw_min=min(vec_scores_raw) if vec_scores_raw else 0.0,
        vector_raw_max=max(vec_scores_raw) if vec_scores_raw else 0.0,
        bm25_raw_min=min(bm_scores_raw) if bm_scores_raw else 0.0,
        bm25_raw_max=max(bm_scores_raw) if bm_scores_raw else 0.0,
    )
    return merged_rows, dbg


__all__ = ["normalize_minmax", "merge_hybrid_results", "HybridMergeDebug"]