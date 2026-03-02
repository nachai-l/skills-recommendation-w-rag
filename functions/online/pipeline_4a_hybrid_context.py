# functions/online/pipeline_4a_hybrid_context.py
"""
Pipeline 4a — hybrid merge + context construction

Thin orchestration layer.

Responsibilities
- Load parameters.yaml
- Call Pipeline 3a (FAISS vector search) and Pipeline 3b (BM25 lexical search)
- Union merge by skill_id
- Optional score normalization + hybrid score computation
- Deterministic tie-break ordering
- Render `{context}` using context.row_template with:
  - column selection over meta
  - per-field truncation
  - overall max_context_chars cap
- Return an API-friendly payload for downstream 4b/5

Core logic lives in:
- functions/core/hybrid_merge.py
- functions/core/context_render.py

ONLINE safety notes
- No LLM generation here.
- Retrieval + merge + context rendering only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from functions.core.context_render import render_context_rows
from functions.core.hybrid_merge import merge_hybrid_results
from functions.online.pipeline_3a_vector_search import run_pipeline_3a_vector_search
from functions.online.pipeline_3b_bm25_search import run_pipeline_3b_bm25_search
from functions.utils.config import load_parameters
from functions.utils.config_access import cfg_get as _get, cfg_get_path as _get_path
from functions.utils.logging import get_logger

logger = get_logger(__name__)


def run_pipeline_4a_hybrid_context(
    *,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    parameters_path: str = "configs/parameters.yaml",
    include_meta: bool = True,
    include_internal_idx: bool = True,
    # Optional: allow different retrieval depths than output
    top_k_vector: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pipeline 4a — hybrid merge + context construction

    Thin orchestration layer.

    Responsibilities
    - Load parameters.yaml
    - Call Pipeline 3a (FAISS vector search) and Pipeline 3b (BM25 lexical search)
    - Union merge by skill_id
    - Optional score normalization + hybrid score computation
    - Deterministic tie-break ordering
    - Render `{context}` using context.row_template with:
      - column selection over meta
      - per-field truncation
      - overall max_context_chars cap
    - Return an API-friendly payload for downstream 4b/5

    Core logic lives in:
    - functions/core/hybrid_merge.py
    - functions/core/context_render.py

    ONLINE safety notes
    - No LLM generation here.
    - Retrieval + merge + context rendering only.
    """
    params = load_parameters(parameters_path)

    top_k_default = int(_get_path(params, ["rag", "vector_search", "top_k_default"]))
    k_out = int(top_k) if top_k is not None else top_k_default

    # Hybrid config
    alpha = float(_get_path(params, ["rag", "hybrid", "alpha"]))
    normalize_scores = bool(_get_path(params, ["rag", "hybrid", "normalize_scores"]))

    # Context config
    ctx_cols_mode = str(_get_path(params, ["context", "columns", "mode"]) or "all")
    ctx_cols_include = list(_get_path(params, ["context", "columns", "include"]) or [])
    ctx_cols_exclude = list(_get_path(params, ["context", "columns", "exclude"]) or [])
    row_template = str(_get_path(params, ["context", "row_template"]) or "")
    max_context_chars = int(_get_path(params, ["context", "max_context_chars"]) or 30000)
    truncate_field_chars = int(_get_path(params, ["context", "truncate_field_chars"]) or 2000)

    # Retrieval depth can differ from output depth (e.g., retrieve deeper, then cut to k_out).
    k_vec = int(top_k_vector) if top_k_vector is not None else k_out
    k_bm = int(top_k_bm25) if top_k_bm25 is not None else k_out

    if debug:
        logger.info(
            "P4a hybrid start",
            extra={
                "query_len": len((query or "").strip()),
                "top_k_out": k_out,
                "top_k_vector": k_vec,
                "top_k_bm25": k_bm,
                "alpha": alpha,
                "normalize_scores": normalize_scores,
                "max_context_chars": max_context_chars,
            },
        )

    # Retrieve (always include_meta=True internally for context robustness)
    p3a = run_pipeline_3a_vector_search(
        query=query,
        top_k=k_vec,
        debug=debug,
        parameters_path=parameters_path,
        include_meta=True,
        include_internal_idx=True,
    )
    p3b = run_pipeline_3b_bm25_search(
        query=query,
        top_k=k_bm,
        debug=debug,
        parameters_path=parameters_path,
        include_meta=True,
        include_internal_idx=True,
    )

    vec_results = list(p3a.get("results") or [])
    bm_results = list(p3b.get("results") or [])

    merged_rows, dbg_merge = merge_hybrid_results(
        vector_results=vec_results,
        bm25_results=bm_results,
        alpha=alpha,
        normalize_scores=normalize_scores,
    )

    # Cut to output k (context and output results are aligned to this truncated list).
    merged_rows = merged_rows[:k_out]

    # Render {context} for 4b prompt injection.
    context_text, dbg_ctx = render_context_rows(
        rows=merged_rows,
        row_template=row_template,
        max_context_chars=max_context_chars,
        truncate_field_chars=truncate_field_chars,
        columns_mode=ctx_cols_mode,
        columns_include=ctx_cols_include,
        columns_exclude=ctx_cols_exclude,
    )

    # Shape output rows based on include flags
    out_results: list[dict[str, Any]] = []
    for r in merged_rows:
        out: dict[str, Any] = {
            "skill_id": r.get("skill_id"),
            "skill_name": r.get("skill_name"),
            "source": r.get("source"),
            "score_vector": r.get("score_vector", 0.0),
            "score_bm25": r.get("score_bm25", 0.0),
            "score_hybrid": r.get("score_hybrid", 0.0),
        }
        if include_meta:
            out["meta"] = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        if include_internal_idx:
            out["internal_idx_vector"] = r.get("internal_idx_vector")
            out["internal_idx_bm25"] = r.get("internal_idx_bm25")
        out_results.append(out)

    payload: Dict[str, Any] = {
        "query": (query or "").strip(),
        "top_k": k_out,
        "alpha": alpha,
        "results": out_results,
        "context": context_text,
    }

    if debug:
        payload["debug"] = {
            "num_vector_hits": dbg_merge.num_vector_hits,
            "num_bm25_hits": dbg_merge.num_bm25_hits,
            "num_merged": dbg_merge.num_merged,
            "normalize_scores": dbg_merge.normalize_scores,
            "alpha": dbg_merge.alpha,
            "vector_raw_min": dbg_merge.vector_raw_min,
            "vector_raw_max": dbg_merge.vector_raw_max,
            "bm25_raw_min": dbg_merge.bm25_raw_min,
            "bm25_raw_max": dbg_merge.bm25_raw_max,
            "context_chars": dbg_ctx.context_chars,
            "context_truncated": dbg_ctx.context_truncated,
            "rows_rendered": dbg_ctx.rows_rendered,
        }

        logger.info(
            "P4a hybrid done",
            extra={
                "top_k_out": k_out,
                "num_results": len(out_results),
                "context_chars": dbg_ctx.context_chars,
                "context_truncated": dbg_ctx.context_truncated,
            },
        )

    return payload


__all__ = ["run_pipeline_4a_hybrid_context"]