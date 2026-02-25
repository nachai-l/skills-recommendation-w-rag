# tests/test_pipeline_2_ingest_index.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from functions.batch.pipeline_2_ingest_index import main as pipeline2_main


class _FakeEmbedder:
    def embed_documents(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        # deterministic embedding: length-based 3D vectors
        mat = []
        for t in texts:
            n = float(len(t))
            v = np.array([n, n + 1.0, n + 2.0], dtype=np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            mat.append(v)
        return np.stack(mat, axis=0).astype(np.float32)


def test_pipeline2_builds_artifacts(tmp_path: Path, monkeypatch):
    # Arrange: create tiny corpus csv
    csv_path = tmp_path / "skill_taxonomy.csv"
    csv_path.write_text(
        "skill_id,skill_name,skill_text,source,Foundational_Criteria,Intermediate_Criteria,Advanced_Criteria\n"
        "1,SkillA,TextA,cluster,F1,I1,A1\n"
        "2,SkillB,TextB,cluster,F2,I2,A2\n",
        encoding="utf-8",
    )

    # parameters.yaml
    params_path = tmp_path / "parameters.yaml"
    params_path.write_text(
        f"""
run:
  name: skills_recommendation_via_rag
  timezone: Asia/Bangkok

input:
  path: {csv_path.as_posix()}
  format: csv
  encoding: utf-8
  required_columns:
    - skill_id
    - skill_name
    - skill_text
    - source
    - Foundational_Criteria
    - Intermediate_Criteria
    - Advanced_Criteria

rag:
  corpus:
    id_col: skill_id
    title_col: skill_name
    text_col: skill_text
    source_col: source

embeddings:
  model_name: gemini-embedding-001
  dim: null

index_store:
  dir: {tmp_path.as_posix()}/index_store
  faiss:
    index_path: {tmp_path.as_posix()}/index_store/faiss.index
    meta_path: {tmp_path.as_posix()}/index_store/faiss_meta.jsonl
  bm25:
    corpus_path: {tmp_path.as_posix()}/index_store/bm25_corpus.jsonl
    stats_path: {tmp_path.as_posix()}/index_store/bm25_stats.json

prompts:
  generation:
    path: prompts/generation.yaml
""",
        encoding="utf-8",
    )

    # Monkeypatch embedder factory
    from functions import utils as _  # noqa: F401

    import functions.utils.text_embeddings as te

    monkeypatch.setattr(te, "build_embedding_model", lambda **kwargs: _FakeEmbedder())

    # Act
    rc = pipeline2_main(parameters_path=str(params_path))
    assert rc == 0

    # Assert artifacts exist
    faiss_index = tmp_path / "index_store" / "faiss.index"
    faiss_meta = tmp_path / "index_store" / "faiss_meta.jsonl"
    bm25_corpus = tmp_path / "index_store" / "bm25_corpus.jsonl"
    bm25_stats = tmp_path / "index_store" / "bm25_stats.json"

    assert faiss_index.exists()
    assert faiss_meta.exists()
    assert bm25_corpus.exists()
    assert bm25_stats.exists()

    # meta rows align
    meta_lines = faiss_meta.read_text(encoding="utf-8").strip().splitlines()
    assert len(meta_lines) == 2
    m0 = json.loads(meta_lines[0])
    assert m0["skill_id"] == "1"
    assert m0["skill_name"] == "SkillA"

    # stats
    stats = json.loads(bm25_stats.read_text(encoding="utf-8"))
    assert stats["num_docs"] == 2
    assert stats["dim"] == 3