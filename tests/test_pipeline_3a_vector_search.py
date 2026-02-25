# tests/test_pipeline_3a_vector_search.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

import functions.online.pipeline_3a_vector_search as p3a


class _StubFaissStore:
    def __init__(self, *, dim: int, meta_by_idx: Dict[int, Dict[str, Any]], search_ret: Tuple[np.ndarray, np.ndarray]):
        self._dim = dim
        self._meta_by_idx = meta_by_idx
        self._search_ret = search_ret

    @property
    def dim(self) -> int:
        return self._dim

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Basic sanity checks like FAISS store would enforce
        assert isinstance(query_embeddings, np.ndarray)
        assert query_embeddings.ndim == 2
        assert int(top_k) > 0
        return self._search_ret

    def get_meta(self, internal_idx: int) -> Optional[Dict[str, Any]]:
        return self._meta_by_idx.get(int(internal_idx))


def _base_params(*, dim: int = 3, top_k_default: int = 3) -> Dict[str, Any]:
    return {
        "index_store": {
            "faiss": {
                "index_path": "artifacts/index_store/faiss.index",
                "meta_path": "artifacts/index_store/faiss_meta.jsonl",
            }
        },
        "rag": {
            "vector_search": {"top_k_default": top_k_default},
            "corpus": {
                "id_col": "skill_id",
                "title_col": "skill_name",
            },
        },
        "embeddings": {
            "model_name": "stub-embedder",
            "task_type": "SEMANTIC_SIMILARITY",
            "dim": dim,
            "normalize": True,
        },
    }


def _monkeypatch_pipeline_deps(monkeypatch: pytest.MonkeyPatch, *, params: Dict[str, Any], store: _StubFaissStore) -> None:
    # Avoid reading configs/parameters.yaml
    monkeypatch.setattr(p3a, "load_parameters", lambda _path: params)

    # Resolve paths as deterministic Path objects (no filesystem dependency)
    monkeypatch.setattr(p3a, "resolve_path", lambda p, *, base_dir: Path(p))

    # Avoid FAISS I/O
    monkeypatch.setattr(p3a, "get_faiss_store_cached", lambda *, index_path, meta_path: store)

    # Avoid network embedding; return fixed vector
    def _embed_fn(q: str) -> np.ndarray:
        # Return shape (dim,) so core reshapes to (1, dim)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(p3a, "_build_embed_fn_from_repo", lambda _params: _embed_fn)


def test_pipeline_3a_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params(dim=3, top_k_default=3)

    meta = {
        10: {"skill_id": "S001", "skill_name": "Python", "source": "taxonomy"},
        20: {"skill_id": "S002", "skill_name": "SQL", "source": "taxonomy"},
        30: {"skill_id": "S003", "skill_name": "Data Analysis", "source": "taxonomy"},
    }
    scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    indices = np.array([[10, 20, 30]], dtype=np.int64)

    store = _StubFaissStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))
    _monkeypatch_pipeline_deps(monkeypatch, params=params, store=store)

    out = p3a.run_pipeline_3a_vector_search(query="recommend skills for data scientist", top_k=None, debug=True)

    assert out["query"] == "recommend skills for data scientist"
    assert out["top_k"] == 3
    assert "debug" in out
    assert len(out["results"]) == 3

    r0 = out["results"][0]
    assert r0["skill_id"] == "S001"
    assert r0["skill_name"] == "Python"
    assert pytest.approx(r0["score_vector"], rel=1e-6) == 0.9
    assert r0["internal_idx"] == 10
    assert r0["meta"]["source"] == "taxonomy"


def test_pipeline_3a_empty_query_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params(dim=3, top_k_default=3)

    meta = {0: {"skill_id": "S001", "skill_name": "Python"}}
    scores = np.array([[0.9]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)
    store = _StubFaissStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))

    _monkeypatch_pipeline_deps(monkeypatch, params=params, store=store)

    with pytest.raises(ValueError, match="query must be non-empty"):
        p3a.run_pipeline_3a_vector_search(query="   ")


def test_pipeline_3a_dim_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params(dim=4, top_k_default=3)

    meta = {0: {"skill_id": "S001", "skill_name": "Python"}}
    scores = np.array([[0.9]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)
    store = _StubFaissStore(dim=4, meta_by_idx=meta, search_ret=(scores, indices))

    # Patch deps except embed_fn (we override to return wrong dim)
    monkeypatch.setattr(p3a, "load_parameters", lambda _path: params)
    monkeypatch.setattr(p3a, "resolve_path", lambda p, *, base_dir: Path(p))
    monkeypatch.setattr(p3a, "get_faiss_store_cached", lambda *, index_path, meta_path: store)

    def _bad_embed_fn(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # dim=3 but expected=4

    monkeypatch.setattr(p3a, "_build_embed_fn_from_repo", lambda _params: _bad_embed_fn)

    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        p3a.run_pipeline_3a_vector_search(query="hello", top_k=1)


def test_pipeline_3a_filters_negative_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params(dim=3, top_k_default=3)

    meta = {
        5: {"skill_id": "S010", "skill_name": "A"},
        7: {"skill_id": "S020", "skill_name": "B"},
    }
    scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    indices = np.array([[5, -1, 7]], dtype=np.int64)  # includes invalid -1
    store = _StubFaissStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))
    _monkeypatch_pipeline_deps(monkeypatch, params=params, store=store)

    out = p3a.run_pipeline_3a_vector_search(query="x", top_k=3)

    assert len(out["results"]) == 2
    assert [r["internal_idx"] for r in out["results"]] == [5, 7]


def test_pipeline_3a_stable_tie_break(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params(dim=3, top_k_default=3)

    # Same score for both → tie-break should be by skill_id asc, then internal_idx asc
    meta = {
        2: {"skill_id": "S200", "skill_name": "Z"},
        1: {"skill_id": "S100", "skill_name": "A"},
        3: {"skill_id": "S100", "skill_name": "A-dup"},  # same skill_id, different internal
    }
    scores = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    indices = np.array([[2, 1, 3]], dtype=np.int64)
    store = _StubFaissStore(dim=3, meta_by_idx=meta, search_ret=(scores, indices))
    _monkeypatch_pipeline_deps(monkeypatch, params=params, store=store)

    out = p3a.run_pipeline_3a_vector_search(query="tie", top_k=3)

    # Expected order:
    # - S100 (internal 1) then S100 (internal 3) then S200 (internal 2)
    got = [(r["skill_id"], r["internal_idx"]) for r in out["results"]]
    assert got == [("S100", 1), ("S100", 3), ("S200", 2)]