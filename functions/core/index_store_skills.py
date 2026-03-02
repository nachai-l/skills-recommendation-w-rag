# functions/core/index_store_skills.py
"""
Index-store helpers (skills corpus) — build/persist FAISS + BM25 artifacts.

Intent
- Provide small, deterministic utilities used by the batch indexing pipeline (Pipeline 2)
  to create and persist retrieval artifacts for the skills corpus:
  - FAISS IndexFlatIP for vector search (cosine similarity with L2-normalized vectors)
  - BM25 corpus + stats files (lexical retrieval; stats computed elsewhere)

Notes
- This module is used in the batch/indexing path (NOT the online serving path).
- Determinism comes from:
  - stable input ordering
  - stable serialization (JSON/JSONL) and fixed FAISS index type (IndexFlatIP)

Artifact alignment invariant
- FAISS internal ids must align with meta JSONL row order:
    internal_idx i <-> meta_rows[i]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import faiss


@dataclass(frozen=True)
class SkillIndexPaths:
    """Canonical artifact paths for skill retrieval indexes (FAISS + BM25)."""
    faiss_index_path: str
    faiss_meta_path: str
    bm25_corpus_path: str
    bm25_stats_path: str


def build_faiss_index_ip(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS IndexFlatIP for cosine similarity (requires L2-normalized vectors).

    Notes:
    - IndexFlatIP performs exact inner-product search.
    - If vectors are L2-normalized upstream, inner product == cosine similarity.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D array (N, D), got shape={embeddings.shape}")

    emb = np.ascontiguousarray(embeddings.astype(np.float32))
    n, d = emb.shape
    if n <= 0 or d <= 0:
        raise ValueError(f"Invalid embeddings shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL (one JSON object per line) in UTF-8 (ensure_ascii=False)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    """Write pretty JSON in UTF-8 (ensure_ascii=False) for human readability."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    """Persist a FAISS index to disk (creates parent dir if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def load_faiss_index(path: str | Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found: {p}")
    return faiss.read_index(str(p))


def validate_alignment(n_rows: int, meta_rows: List[Dict[str, Any]]) -> None:
    """Ensure FAISS ntotal (or corpus size) matches meta JSONL row count."""
    if n_rows != len(meta_rows):
        raise ValueError(f"Meta alignment mismatch: n_rows={n_rows} meta_rows={len(meta_rows)}")


def build_bm25_corpus_rows(
    *,
    skill_ids: List[str],
    docs: List[str],
    titles: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build minimal BM25 corpus rows.

    Output schema (per row):
    - skill_id: str
    - skill_name: Optional[str]
    - source: Optional[str]
    - doc: str

    Notes:
    - Online pipeline tokenizes and computes DF/IDF from this corpus.
    - Caller controls ordering; ordering should be stable for reproducibility.
    """
    if len(skill_ids) != len(docs):
        raise ValueError("skill_ids and docs length mismatch")

    out: List[Dict[str, Any]] = []
    for i in range(len(skill_ids)):
        out.append(
            {
                "skill_id": skill_ids[i],
                "skill_name": titles[i] if titles else None,
                "source": sources[i] if sources else None,
                "doc": docs[i],
            }
        )
    return out