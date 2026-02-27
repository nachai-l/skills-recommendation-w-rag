# functions/core/api_payload.py
"""
API payload builder (Pipeline 5) — join LLM output with retrieval metadata.

Intent
- Produce the final API `payload` returned by `/v1/recommend-skills`.
- Join LLM-validated recommendations (pipeline 4b output) with retrieval rows/meta
  (pipelines 3a/3b/4a) to enrich each skill with canonical IDs and full text fields.

Join strategy
- Key: normalized `skill_name` (lowercased, trimmed, collapsed whitespace).
- If multiple retrieval rows share the same normalized name, prefer the “best” row by:
  score_hybrid (desc) -> score_vector (desc) -> score_bm25 (desc).

Join policy
- If `require_all_meta=True`, raise when any recommended skill cannot be joined to meta.
- Otherwise, log a warning and allow empty text fields + None skill_id/source.

Outputs
- payload: dict with {query, analysis_summary, recommended_skills[]}
- debug: BuildApiPayloadDebug summary counters for observability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _norm_name(s: str) -> str:
    """Normalize a skill name for join keys (case-insensitive, whitespace-stable)."""
    # Collapse whitespace to reduce join misses caused by formatting differences.
    return " ".join((s or "").strip().lower().split())


def _get_meta_from_retrieval_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Return row['meta'] if it is a dict; otherwise return an empty dict."""
    m = r.get("meta")
    return m if isinstance(m, dict) else {}


@dataclass
class BuildApiPayloadDebug:
    """Lightweight debug counters for payload build / join quality."""
    num_llm_recs: int
    num_retrieval_rows: int
    num_joined: int
    num_missing_meta: int


def build_api_payload_from_4b(
    *,
    query: str,
    llm_validated: Dict[str, Any],
    retrieval_results: List[Dict[str, Any]],
    top_k: int,
    require_all_meta: bool = False,
) -> Tuple[Dict[str, Any], BuildApiPayloadDebug]:
    """
    Build final API payload.

    Joins LLM recommendations with retrieval meta by normalized skill_name.

    If require_all_meta=True:
      - raise if any recommended skill cannot be joined to retrieval meta.
    """
    recs = llm_validated.get("recommended_skills") or []
    if not isinstance(recs, list):
        recs = []

    # Build lookup from retrieval results: normalized skill_name -> best row
    # If duplicates exist, prefer highest score_hybrid, then score_vector, then score_bm25
    def _score_tuple(rr: Dict[str, Any]) -> tuple:
        return (
            float(rr.get("score_hybrid") or 0.0),
            float(rr.get("score_vector") or 0.0),
            float(rr.get("score_bm25") or 0.0),
        )

    lookup: Dict[str, Dict[str, Any]] = {}
    for rr in retrieval_results or []:
        name = rr.get("skill_name") or ""
        key = _norm_name(str(name))
        if not key:
            continue
        if key not in lookup or _score_tuple(rr) > _score_tuple(lookup[key]):
            lookup[key] = rr

    out_recs: List[Dict[str, Any]] = []
    missing = 0

    for item in recs[: max(int(top_k), 0) or len(recs)]:
        if not isinstance(item, dict):
            continue

        # Use LLM-provided name as the display name; retrieval row provides canonical IDs/meta.
        skill_name = str(item.get("skill_name") or "").strip()
        key = _norm_name(skill_name)

        rr = lookup.get(key)
        meta = _get_meta_from_retrieval_row(rr) if rr else {}

        # If retrieval row exists, use its skill_id/source as canonical
        skill_id = (rr or {}).get("skill_id") if rr else None
        source = (rr or {}).get("source") if rr else None

        # If not found, try to backfill from meta itself (best-effort).
        if not skill_id and meta.get("skill_id"):
            skill_id = meta.get("skill_id")
        if not source and meta.get("source"):
            source = meta.get("source")

        # Join policy: either strict (raise) or permissive (warn + keep empty meta fields).
        if not meta:
            missing += 1
            if require_all_meta:
                raise ValueError(f"Missing retrieval meta for recommended skill_name={skill_name!r}")
            else:
                logger.warning(
                    "No retrieval meta for recommended skill_name=%r "
                    "(skill_id and source will be None)",
                    skill_name,
                )

        out_recs.append(
            {
                "skill_id": skill_id,
                "skill_name": skill_name,
                "source": source,
                "relevance_score": item.get("relevance_score"),
                "reasoning": item.get("reasoning"),
                "evidence": item.get("evidence") or [],
                # Full text fields (may be empty if not joined)
                "skill_text": meta.get("skill_text", ""),
                "Foundational_Criteria": meta.get("Foundational_Criteria", ""),
                "Intermediate_Criteria": meta.get("Intermediate_Criteria", ""),
                "Advanced_Criteria": meta.get("Advanced_Criteria", ""),
            }
        )

    payload = {
        "query": (query or "").strip(),
        "analysis_summary": llm_validated.get("analysis_summary", ""),
        "recommended_skills": out_recs,
    }

    dbg = BuildApiPayloadDebug(
        num_llm_recs=len(recs),
        num_retrieval_rows=len(retrieval_results or []),
        num_joined=len(out_recs) - missing,
        num_missing_meta=missing,
    )
    return payload, dbg


__all__ = ["build_api_payload_from_4b", "BuildApiPayloadDebug"]