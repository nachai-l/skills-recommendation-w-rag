# functions/core/index_store.py
"""
functions.core.index_store

Purpose
-------
Provide a minimal ONLINE-safe access layer for persisted vector retrieval artifacts.

In this Skills Recommendation repo, we use:
- FAISS for vector search (Pipeline 3a)
- BM25 for lexical search (Pipeline 3b) — stored separately (not in this module)

ONLINE safety contract
----------------------
- MUST NOT call LLMs.
- MUST NOT embed text.
- MUST only load persisted artifacts and perform retrieval.

FAISS backend (MVP)
-------------------
- IndexFlatIP (exact search)
- Assumes vectors are L2-normalized, so inner product == cosine similarity

Artifact alignment invariant
----------------------------
- faiss_meta.jsonl row order MUST align with FAISS internal ids:
    internal_idx i <-> meta[i]

Planned production backend
--------------------------
- Vertex AI Vector Search (managed ANN)
- We keep the `search()` contract small so online pipeline logic stays stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss


# -------------------------------------------------------------------------------------------------
# ONLINE: FaissIndexStore (NO LLM, NO embedding)
# -------------------------------------------------------------------------------------------------
@dataclass
class FaissIndexStore:
    """
    Vector index store for ONLINE retrieval (FAISS implementation).

    Required artifacts:
      - index_path: FAISS index file (e.g., artifacts/index_store/faiss.index)
      - meta_path:  JSONL metadata aligned to FAISS internal ids
    """

    index_path: str
    meta_path: str
    _faiss_index: faiss.Index
    _meta: List[Dict[str, Any]]

    @classmethod
    def load(cls, *, index_path: str, meta_path: str) -> "FaissIndexStore":
        """
        Load persisted FAISS index + JSONL metadata.

        Raises:
          FileNotFoundError:
            If required artifacts are missing.
          ValueError:
            If metadata JSONL is invalid or mismatched with index size.
        """
        ip = str(index_path)
        mp = str(meta_path)

        if not Path(ip).exists():
            raise FileNotFoundError(f"FAISS index not found: {ip}")
        if not Path(mp).exists():
            raise FileNotFoundError(f"FAISS meta not found: {mp}")

        index = faiss.read_index(ip)

        meta: List[Dict[str, Any]] = []
        with open(mp, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Invalid JSONL at line {line_no}: expected object/dict")
                meta.append(obj)

        # Alignment guard: meta length must match faiss.ntotal (when ntotal available)
        try:
            ntotal = int(getattr(index, "ntotal", 0))
        except Exception:
            ntotal = 0

        if ntotal > 0 and len(meta) != ntotal:
            raise ValueError(
                f"Meta length mismatch: len(meta)={len(meta)} vs faiss.ntotal={ntotal}. "
                "Meta JSONL row order must align with FAISS internal ids."
            )

        return cls(index_path=ip, meta_path=mp, _faiss_index=index, _meta=meta)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors via FAISS.

        Args:
          query_embeddings:
            2D array shape (nq, dim). Will be converted to float32 contiguous.
          top_k:
            Number of neighbors per query (k > 0).

        Returns:
          scores:
            (nq, top_k) float32 similarity scores.
          indices:
            (nq, top_k) int64 internal ids (row indices into meta JSONL).

        Notes:
        - For IndexFlatIP with L2-normalized vectors, score == cosine similarity.
        """
        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.asarray(query_embeddings)

        if query_embeddings.ndim != 2:
            raise ValueError("query_embeddings must be a 2D array of shape (nq, dim)")

        k = int(top_k)
        if k <= 0:
            raise ValueError("top_k must be > 0")

        q = np.asarray(query_embeddings, dtype=np.float32)
        q = np.ascontiguousarray(q)

        scores, indices = self._faiss_index.search(q, k)
        return np.asarray(scores, dtype=np.float32), np.asarray(indices, dtype=np.int64)

    def get_meta(self, internal_idx: int) -> Optional[Dict[str, Any]]:
        """Return metadata dict for an internal FAISS row id."""
        i = int(internal_idx)
        if i < 0 or i >= len(self._meta):
            return None
        return self._meta[i]

    @property
    def meta(self) -> List[Dict[str, Any]]:
        """Metadata list aligned to FAISS internal ids."""
        return self._meta

    @property
    def dim(self) -> int:
        """FAISS vector dimension."""
        d = getattr(self._faiss_index, "d", None)
        return int(d) if d is not None else 0


# -------------------------------------------------------------------------------------------------
# Process-local cache (ONLINE)
# -------------------------------------------------------------------------------------------------
_FAISS_STORE_CACHE: Dict[str, FaissIndexStore] = {}


def get_faiss_store_cached(*, index_path: str, meta_path: str) -> FaissIndexStore:
    """
    Load and cache FaissIndexStore per resolved (index_path, meta_path).

    Notes:
    - Cache is process-local. In multi-worker serving, each worker has its own cache.
    - On Cloud Run scale-to-zero, first request will be cold.
    """
    key = f"{Path(index_path).resolve()}||{Path(meta_path).resolve()}"
    cached = _FAISS_STORE_CACHE.get(key)
    if cached is not None:
        return cached

    store = FaissIndexStore.load(index_path=index_path, meta_path=meta_path)
    _FAISS_STORE_CACHE[key] = store
    return store


__all__ = [
    "FaissIndexStore",
    "get_faiss_store_cached",
]