# tests/test_ingestion.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from functions.core.index_store import FaissIndexStore
from functions.core.ingestion import (
    build_skill_document_text_for_embedding,
    build_and_persist_indexes,
)


class _StubEmbedder:
    """
    Deterministic stub embedder:
    - returns L2-normalized float32 vectors
    - stable for unit tests (no external calls)
    """

    def __init__(self, dim: int = 4):
        self.dim = int(dim)

    def embed_documents(self, texts: List[str], *, task_type: str) -> np.ndarray:
        n = len(texts)
        mat = np.zeros((n, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = (t or "").encode("utf-8", errors="ignore")
            mat[i, 0] = float(len(s))
            mat[i, 1] = float(sum(s) % 997)
            mat[i, 2] = float((sum(s[::2]) if s else 0) % 997)
            mat[i, 3] = float((sum(s[1::2]) if len(s) > 1 else 0) % 997)

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        mat = mat / norms
        return np.ascontiguousarray(mat, dtype=np.float32)


def test_build_skill_document_text_for_embedding_contains_key_sections() -> None:
    row = {
        "skill_id": "s1",
        "skill_name": "active listening",
        "skill_text": "Pay attention and respond.",
        "source": "cluster",
        "Foundational_Criteria": "F1",
        "Intermediate_Criteria": "I1",
        "Advanced_Criteria": "A1",
    }
    doc = build_skill_document_text_for_embedding(
        row,
        id_col="skill_id",
        title_col="skill_name",
        text_col="skill_text",
        source_col="source",
    )
    assert "skill_id: s1" in doc
    assert "skill_name: active listening" in doc
    assert "source: cluster" in doc
    assert "skill_text:" in doc
    assert "criteria:" in doc
    assert "Foundational_Criteria:" in doc
    assert "Intermediate_Criteria:" in doc
    assert "Advanced_Criteria:" in doc


def test_build_and_persist_indexes_writes_all_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "skill_id": "s0",
                "skill_name": "A",
                "skill_text": "Alpha text",
                "source": "cluster",
                "Foundational_Criteria": "F0",
                "Intermediate_Criteria": "I0",
                "Advanced_Criteria": "A0",
            },
            {
                "skill_id": "s1",
                "skill_name": "B",
                "skill_text": "Beta text",
                "source": "cluster",
                "Foundational_Criteria": "F1",
                "Intermediate_Criteria": "I1",
                "Advanced_Criteria": "A1",
            },
            {
                "skill_id": "s2",
                "skill_name": "C",
                "skill_text": "Gamma text",
                "source": "cluster",
                "Foundational_Criteria": "F2",
                "Intermediate_Criteria": "I2",
                "Advanced_Criteria": "A2",
            },
        ]
    )

    cfg = {
        "embeddings": {
            "model_name": "stub",
            "dim": 4,
            "task_type": "RETRIEVAL_DOCUMENT",
            # keep defaults for bm25 shaping (doesn't affect this test)
        },
        "rag": {
            "corpus": {
                "id_col": "skill_id",
                "title_col": "skill_name",
                "text_col": "skill_text",
                "source_col": "source",
            },
            # Optional: prove bm25 shaping knobs exist (not required)
            "bm25": {"doc_mode": "title_plus_text", "max_text_chars": 800},
        },
        "index_store": {
            "dir": str(tmp_path / "index_store"),
            "faiss": {
                "index_path": str(tmp_path / "index_store" / "faiss.index"),
                "meta_path": str(tmp_path / "index_store" / "faiss_meta.jsonl"),
            },
            "bm25": {
                "corpus_path": str(tmp_path / "index_store" / "bm25_corpus.jsonl"),
                "stats_path": str(tmp_path / "index_store" / "bm25_stats.json"),
            },
        },
    }

    res = build_and_persist_indexes(df, cfg, embedder=_StubEmbedder(dim=4))

    # Files exist
    assert Path(res.paths["faiss_index_path"]).exists()
    assert Path(res.paths["faiss_meta_path"]).exists()
    assert Path(res.paths["bm25_corpus_path"]).exists()
    assert Path(res.paths["bm25_stats_path"]).exists()
    assert Path(res.paths["manifest_path"]).exists()

    # Manifest sanity
    manifest = json.loads(Path(res.paths["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["num_docs"] == 3
    assert manifest["dim"] == 4
    assert manifest["backend"]["vector"] == "faiss"
    assert manifest["backend"]["bm25"] == "local"

    # FAISS + meta alignment sanity by loading store
    store = FaissIndexStore.load(
        index_path=res.paths["faiss_index_path"],
        meta_path=res.paths["faiss_meta_path"],
    )
    assert store.dim == 4
    assert len(store.meta) == 3

    # Build an embedding-query using the same embedding-doc builder, so nearest neighbor should be itself.
    row0 = df.iloc[0].to_dict()
    doc0 = build_skill_document_text_for_embedding(
        row0,
        id_col="skill_id",
        title_col="skill_name",
        text_col="skill_text",
        source_col="source",
    )
    q = _StubEmbedder(dim=4).embed_documents([doc0], task_type="RETRIEVAL_QUERY")

    scores, idx = store.search(q, top_k=2)
    assert scores.shape == (1, 2)
    assert idx.shape == (1, 2)
    assert idx.dtype == np.int64
    assert scores.dtype == np.float32

    # Since we embedded doc0, the top neighbor should be internal idx 0
    assert int(idx[0, 0]) == 0

    # BM25 corpus JSONL count
    bm25_lines = Path(res.paths["bm25_corpus_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(bm25_lines) == 3

    # BM25 stats JSON
    stats = json.loads(Path(res.paths["bm25_stats_path"]).read_text(encoding="utf-8"))
    assert stats["num_docs"] == 3
    assert stats["id_col"] == "skill_id"