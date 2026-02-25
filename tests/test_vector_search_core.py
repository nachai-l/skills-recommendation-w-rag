# tests/test_vector_search_core.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from functions.core.vector_search import to_api_payload, vector_search_faiss


class _StubStore:
    def __init__(self, *, dim: int, meta_by_idx: Dict[int, Dict[str, Any]], search_ret: Tuple[np.ndarray, np.ndarray]):
        self._dim = dim
        self._meta_by_idx = meta_by_idx
        self._search_ret = search_ret

    @property
    def dim(self) -> int:
        return self._dim

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert query_embeddings.ndim == 2
        assert query_embeddings.shape[0] == 1
        assert int(top_k) > 0
        return self._search_ret

    def get_meta(self, internal_idx: int) -> Optional[Dict[str, Any]]:
        return self._meta_by_idx.get(int(internal_idx))


def test_core_vector_search_happy_path() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([3.0, 0.0, 0.0], dtype=np.float32)  # will be normalized

    meta = {
        10: {"skill_id": "S001", "skill_name": "Python", "source": "taxonomy"},
        20: {"skill_id": "S002", "skill_name": "SQL", "source": "taxonomy"},
    }
    scores = np.array([[0.9, 0.8]], dtype=np.float32)
    indices = np.array([[10, 20]], dtype=np.int64)

    store = _StubStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))

    result = vector_search_faiss(
        query="  data engineer  ",
        top_k=2,
        embed_fn=embed_fn,
        store=store,
        dim=3,
        normalize=True,
        debug=True,
    )

    assert result.query == "data engineer"
    assert result.top_k == 2
    assert result.debug is not None
    assert len(result.results) == 2
    assert result.results[0].skill_id == "S001"
    assert pytest.approx(result.results[0].score_vector, rel=1e-6) == 0.9


def test_core_vector_search_empty_query_raises() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    store = _StubStore(dim=3, meta_by_idx={}, search_ret=(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.int64)))

    with pytest.raises(ValueError, match="query must be non-empty"):
        vector_search_faiss(query="   ", top_k=1, embed_fn=embed_fn, store=store, dim=3)


def test_core_vector_search_top_k_invalid_raises() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    store = _StubStore(dim=3, meta_by_idx={}, search_ret=(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.int64)))

    with pytest.raises(ValueError, match="top_k must be > 0"):
        vector_search_faiss(query="x", top_k=0, embed_fn=embed_fn, store=store, dim=3)


def test_core_vector_search_dim_mismatch_raises() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # dim=3

    meta = {0: {"skill_id": "S001", "skill_name": "Python"}}
    scores = np.array([[0.9]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)
    store = _StubStore(dim=4, meta_by_idx=meta, search_ret=(scores, indices))

    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        vector_search_faiss(query="x", top_k=1, embed_fn=embed_fn, store=store, dim=4)


def test_core_vector_search_filters_negative_indices() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    meta = {
        5: {"skill_id": "S010", "skill_name": "A"},
        7: {"skill_id": "S020", "skill_name": "B"},
    }
    scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    indices = np.array([[5, -1, 7]], dtype=np.int64)

    store = _StubStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))

    result = vector_search_faiss(query="x", top_k=3, embed_fn=embed_fn, store=store, dim=3)

    assert [h.internal_idx for h in result.results] == [5, 7]


def test_core_vector_search_missing_meta_raises() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    meta = {
        1: {"skill_id": "S001", "skill_name": "Python"},
        # missing meta for 2
    }
    scores = np.array([[0.9, 0.8]], dtype=np.float32)
    indices = np.array([[1, 2]], dtype=np.int64)

    store = _StubStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))

    with pytest.raises(ValueError, match="Missing or invalid meta"):
        vector_search_faiss(query="x", top_k=2, embed_fn=embed_fn, store=store, dim=3)


def test_core_vector_search_stable_tie_break() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    meta = {
        2: {"skill_id": "S200", "skill_name": "Z"},
        1: {"skill_id": "S100", "skill_name": "A"},
        3: {"skill_id": "S100", "skill_name": "A-dup"},
    }
    scores = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    indices = np.array([[2, 1, 3]], dtype=np.int64)

    store = _StubStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))

    result = vector_search_faiss(query="tie", top_k=3, embed_fn=embed_fn, store=store, dim=3)
    got = [(h.skill_id, h.internal_idx) for h in result.results]
    assert got == [("S100", 1), ("S100", 3), ("S200", 2)]


def test_to_api_payload_flags() -> None:
    def embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    meta = {0: {"skill_id": "S001", "skill_name": "Python", "source": "taxonomy"}}
    scores = np.array([[0.9]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)
    store = _StubStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))

    result = vector_search_faiss(query="x", top_k=1, embed_fn=embed_fn, store=store, dim=3)

    payload = to_api_payload(result, include_meta=False, include_internal_idx=False)
    assert payload["results"][0]["skill_id"] == "S001"
    assert "meta" not in payload["results"][0]
    assert "internal_idx" not in payload["results"][0]