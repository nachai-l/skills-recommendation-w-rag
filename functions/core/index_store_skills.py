# functions/core/index_store_skills.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import faiss


@dataclass(frozen=True)
class SkillIndexPaths:
    faiss_index_path: str
    faiss_meta_path: str
    bm25_corpus_path: str
    bm25_stats_path: str


def build_faiss_index_ip(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS IndexFlatIP for cosine similarity (requires L2-normalized vectors).
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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def load_faiss_index(path: str | Path) -> faiss.Index:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found: {p}")
    return faiss.read_index(str(p))


def validate_alignment(n_rows: int, meta_rows: List[Dict[str, Any]]) -> None:
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
    Minimal BM25 corpus format (online pipeline can tokenize + compute idf).
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