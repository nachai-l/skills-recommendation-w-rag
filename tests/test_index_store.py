from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import faiss

from functions.core.index_store import (
    FaissIndexStore,
    get_faiss_store_cached,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_faiss_index(path: Path, vectors: np.ndarray) -> None:
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError("vectors must be 2D")
    # Ensure contig + normalized (IndexFlatIP expects inner-product; normalized => cosine)
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    d = int(vecs.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    faiss.write_index(index, str(path))


def test_faiss_index_store_load_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FaissIndexStore.load(index_path=str(tmp_path / "nope.index"), meta_path=str(tmp_path / "nope.jsonl"))


def test_faiss_index_store_load_invalid_jsonl(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "faiss_meta.jsonl"

    vecs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _build_faiss_index(index_path, vecs)

    meta_path.write_text('{"ok": 1}\nNOT_JSON\n', encoding="utf-8")

    with pytest.raises(ValueError) as e:
        FaissIndexStore.load(index_path=str(index_path), meta_path=str(meta_path))
    assert "invalid jsonl" in str(e.value).lower()


def test_faiss_index_store_load_meta_length_mismatch(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "faiss_meta.jsonl"

    vecs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _build_faiss_index(index_path, vecs)

    # Only 1 meta row for 2 vectors => mismatch
    _write_jsonl(meta_path, [{"skill_id": "s1"}])

    with pytest.raises(ValueError) as e:
        FaissIndexStore.load(index_path=str(index_path), meta_path=str(meta_path))
    assert "meta length mismatch" in str(e.value).lower()


def test_faiss_index_store_search_shapes_and_types(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "faiss_meta.jsonl"

    # 3 vectors in 2D (already unit-ish for simplicity)
    vecs = np.asarray(
        [
            [1.0, 0.0],  # idx 0
            [0.0, 1.0],  # idx 1
            [0.7071068, 0.7071068],  # idx 2
        ],
        dtype=np.float32,
    )
    _build_faiss_index(index_path, vecs)

    meta = [
        {"skill_id": "s0", "skill_name": "A"},
        {"skill_id": "s1", "skill_name": "B"},
        {"skill_id": "s2", "skill_name": "C"},
    ]
    _write_jsonl(meta_path, meta)

    store = FaissIndexStore.load(index_path=str(index_path), meta_path=str(meta_path))
    assert store.dim == 2
    assert len(store.meta) == 3

    # Query close to [1,0]
    q = np.asarray([[1.0, 0.0]], dtype=np.float32)
    scores, idx = store.search(q, top_k=2)

    assert scores.shape == (1, 2)
    assert idx.shape == (1, 2)
    assert scores.dtype == np.float32
    assert idx.dtype == np.int64

    # Best match should be internal idx 0
    assert int(idx[0, 0]) == 0
    assert float(scores[0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_faiss_index_store_search_validates_inputs(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "faiss_meta.jsonl"

    vecs = np.asarray([[1.0, 0.0]], dtype=np.float32)
    _build_faiss_index(index_path, vecs)
    _write_jsonl(meta_path, [{"skill_id": "s0"}])

    store = FaissIndexStore.load(index_path=str(index_path), meta_path=str(meta_path))

    with pytest.raises(ValueError):
        store.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)  # not 2D

    with pytest.raises(ValueError):
        store.search(np.asarray([[1.0, 0.0]], dtype=np.float32), top_k=0)


def test_get_meta_returns_row_or_none(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "faiss_meta.jsonl"

    vecs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _build_faiss_index(index_path, vecs)
    _write_jsonl(meta_path, [{"skill_id": "s0"}, {"skill_id": "s1"}])

    store = FaissIndexStore.load(index_path=str(index_path), meta_path=str(meta_path))
    assert store.get_meta(0) == {"skill_id": "s0"}
    assert store.get_meta(1) == {"skill_id": "s1"}
    assert store.get_meta(-1) is None
    assert store.get_meta(999) is None


def test_get_faiss_store_cached_returns_same_instance(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "faiss_meta.jsonl"

    vecs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _build_faiss_index(index_path, vecs)
    _write_jsonl(meta_path, [{"skill_id": "s0"}, {"skill_id": "s1"}])

    s1 = get_faiss_store_cached(index_path=str(index_path), meta_path=str(meta_path))
    s2 = get_faiss_store_cached(index_path=str(index_path), meta_path=str(meta_path))

    assert s1 is s2  # cached per resolved paths
    assert len(s1.meta) == 2