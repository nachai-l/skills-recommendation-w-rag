# functions/core/vector_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Types
# -----------------------------
EmbedFn = Callable[[str], np.ndarray]  # should return shape (dim,) or (1, dim)


@dataclass(frozen=True)
class VectorHit:
    skill_id: str
    skill_name: str
    score_vector: float
    internal_idx: int
    meta: Dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    query: str
    top_k: int
    results: List[VectorHit]
    debug: Optional[Dict[str, Any]] = None


# -----------------------------
# Helpers
# -----------------------------
def _l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize vectors row-wise.
    vecs: (nq, dim)
    """
    if vecs.ndim != 2:
        raise ValueError(f"vecs must be 2D (nq, dim), got shape={vecs.shape}")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vecs / norms


def _ensure_2d_float32_contiguous(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)


def _stable_sort_hits(hits: List[VectorHit]) -> List[VectorHit]:
    """
    Stable deterministic tie-break:
      - score_vector desc
      - skill_id asc
      - internal_idx asc
    """
    return sorted(
        hits,
        key=lambda h: (-float(h.score_vector), str(h.skill_id), int(h.internal_idx)),
    )


# -----------------------------
# Core logic
# -----------------------------
def vector_search_faiss(
    *,
    query: str,
    top_k: int,
    embed_fn: EmbedFn,
    store: Any,
    dim: Optional[int] = None,
    normalize: bool = True,
    debug: bool = False,
    # meta keys (keeps this robust if meta schema evolves)
    meta_skill_id_key: str = "skill_id",
    meta_skill_name_key: str = "skill_name",
) -> VectorSearchResult:
    """
    Online vector search (FAISS) — core logic.

    Contract:
    - No LLM calls here
    - Embedding is injected via embed_fn
    - store is a loaded FAISS store (e.g., functions.core.index_store.FaissIndexStore)

    Required store interface:
      - store.search(query_embeddings: np.ndarray, top_k: int) -> (scores, indices)
      - store.get_meta(internal_idx: int) -> dict | None
      - store.dim -> int (optional but supported by your FaissIndexStore)
      - store.meta -> list (optional; not required here)

    Args:
      query: user query string (must be non-empty after strip)
      top_k: number of hits to return (>0)
      embed_fn: callable that embeds query -> np.ndarray (dim,) or (1, dim)
      store: FAISS store instance
      dim: expected embedding dim (if None, infer from store.dim)
      normalize: whether to L2-normalize query embedding before search
      debug: include debug block in output
      meta_skill_id_key/meta_skill_name_key: keys in meta dict

    Returns:
      VectorSearchResult with list[VectorHit]
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")

    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be > 0")

    # Determine expected dim
    store_dim = int(getattr(store, "dim", 0) or 0)
    expected_dim = int(dim) if dim is not None else store_dim
    if expected_dim <= 0:
        raise ValueError("Unable to determine embedding dimension (dim). Provide dim or ensure store.dim is set.")

    # Embed query
    emb = embed_fn(q)
    emb2d = _ensure_2d_float32_contiguous(emb)

    if emb2d.shape[0] != 1:
        raise ValueError(f"embed_fn must return a single query embedding; got shape={emb2d.shape}")

    if emb2d.shape[1] != expected_dim:
        raise ValueError(f"Embedding dim mismatch: got {emb2d.shape[1]}, expected {expected_dim}")

    if normalize:
        emb2d = _l2_normalize(emb2d)

    # FAISS search
    scores, indices = store.search(emb2d, k)

    scores = _ensure_2d_float32_contiguous(scores)
    indices = _ensure_2d_float32_contiguous(indices).astype(np.int64)

    # Build hits
    hits: List[VectorHit] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        internal_idx = int(idx)
        if internal_idx < 0:
            # keep safe if FAISS returns -1 when not enough docs
            continue

        meta = store.get_meta(internal_idx)
        if meta is None or not isinstance(meta, dict):
            # Hard fail: alignment invariant broken
            raise ValueError(f"Missing or invalid meta for internal_idx={internal_idx}")

        skill_id = meta.get(meta_skill_id_key)
        skill_name = meta.get(meta_skill_name_key)

        if not skill_id or not isinstance(skill_id, str):
            raise ValueError(f"meta[{meta_skill_id_key}] missing/invalid for internal_idx={internal_idx}")
        if not skill_name or not isinstance(skill_name, str):
            # If your meta sometimes stores name under a different key, pass meta_skill_name_key accordingly.
            raise ValueError(f"meta[{meta_skill_name_key}] missing/invalid for internal_idx={internal_idx}")

        hits.append(
            VectorHit(
                skill_id=skill_id,
                skill_name=skill_name,
                score_vector=float(score),
                internal_idx=internal_idx,
                meta=meta,
            )
        )

    hits = _stable_sort_hits(hits)

    dbg: Optional[Dict[str, Any]] = None
    if debug:
        dbg = {
            "expected_dim": expected_dim,
            "store_dim": store_dim,
            "returned_k": k,
            "num_hits": len(hits),
            "score_vector_min": float(min((h.score_vector for h in hits), default=0.0)),
            "score_vector_max": float(max((h.score_vector for h in hits), default=0.0)),
            "normalize": bool(normalize),
        }

    return VectorSearchResult(query=q, top_k=k, results=hits, debug=dbg)


def to_api_payload(result: VectorSearchResult, *, include_meta: bool = True, include_internal_idx: bool = True) -> Dict[str, Any]:
    """
    Convert VectorSearchResult to JSON-serializable dict for online pipeline / API.

    Controls:
    - include_meta: include full meta dict per hit
    - include_internal_idx: include internal_idx per hit
    """
    out_results: List[Dict[str, Any]] = []
    for h in result.results:
        row: Dict[str, Any] = {
            "skill_id": h.skill_id,
            "skill_name": h.skill_name,
            "score_vector": h.score_vector,
        }
        if include_internal_idx:
            row["internal_idx"] = h.internal_idx
        if include_meta:
            row["meta"] = h.meta
        out_results.append(row)

    payload: Dict[str, Any] = {
        "query": result.query,
        "top_k": result.top_k,
        "results": out_results,
    }
    if result.debug is not None:
        payload["debug"] = result.debug
    return payload