from __future__ import annotations

"""
functions.core.ingestion

Core ingestion + index building logic (used by Pipeline 2).

Responsibilities (core, not orchestrator)
- Build deterministic "document text" per skill row for:
  - FAISS embedding index (vector retrieval)
  - BM25 corpus (lexical retrieval)
- Embed documents (dependency-injected embedder)
- Build FAISS IndexFlatIP (assumes vectors are L2-normalized)
- Persist:
  - faiss.index
  - faiss_meta.jsonl   (row order aligned to FAISS internal ids)
  - bm25_corpus.jsonl
  - bm25_stats.json    (minimal stats)
  - manifest.json      (traceability)

Non-responsibilities
- Reading parameters.yaml (pipeline will do)
- Reading the input table (pipeline will do)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import faiss

from functions.utils.text import normalize_ws, to_context_str
# from functions.utils.text_embeddings import build_embedding_model


# --------------------------------------------------------------------------------------
# Config helpers (dict-like access to params without hard dependency on config models)
# --------------------------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object/dict: {path}")
            rows.append(obj)
    return rows


# --------------------------------------------------------------------------------------
# Document building
# --------------------------------------------------------------------------------------
_DEFAULT_CRITERIA_COLS = [
    "Foundational_Criteria",
    "Intermediate_Criteria",
    "Advanced_Criteria",
]


def build_skill_document_text(
    row: Dict[str, Any],
    *,
    id_col: str,
    title_col: str,
    text_col: str,
    source_col: Optional[str],
    criteria_cols: Optional[List[str]] = None,
) -> str:
    """
    Build a deterministic document string for embeddings/BM25.

    Notes
    - Uses normalize_ws() on derived strings only.
    - Does not mutate the input row.
    """
    criteria_cols = criteria_cols or _DEFAULT_CRITERIA_COLS

    sid = normalize_ws(to_context_str(row.get(id_col)))
    title = normalize_ws(to_context_str(row.get(title_col)))
    text = to_context_str(row.get(text_col))
    src = normalize_ws(to_context_str(row.get(source_col))) if source_col else ""

    parts: List[str] = []
    if sid:
        parts.append(f"skill_id: {sid}")
    if title:
        parts.append(f"skill_name: {title}")
    if src:
        parts.append(f"source: {src}")

    if text:
        parts.append("skill_text:")
        parts.append(normalize_ws(text))

    # Criteria blocks (keep stable order)
    crit_chunks: List[str] = []
    for c in criteria_cols:
        v = to_context_str(row.get(c))
        if v:
            crit_chunks.append(f"{c}: {normalize_ws(v)}")

    if crit_chunks:
        parts.append("criteria:")
        parts.extend(crit_chunks)

    # Final doc
    return "\n".join(parts).strip() or " "


def build_corpus_rows(
    df: pd.DataFrame,
    *,
    id_col: str,
    title_col: str,
    text_col: str,
    source_col: Optional[str],
    criteria_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build:
    - documents: List[str] (for embeddings)
    - faiss_meta_rows: List[dict] aligned to doc order
    - bm25_rows: List[dict] aligned to doc order
    """
    criteria_cols = criteria_cols or _DEFAULT_CRITERIA_COLS

    docs: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    bm25_rows: List[Dict[str, Any]] = []

    # Deterministic: preserve df row order
    for _, r in df.iterrows():
        row = r.to_dict()

        doc = build_skill_document_text(
            row,
            id_col=id_col,
            title_col=title_col,
            text_col=text_col,
            source_col=source_col,
            criteria_cols=criteria_cols,
        )
        docs.append(doc)

        # Meta rows are for online: keep minimal + useful fields
        meta: Dict[str, Any] = {
            "skill_id": to_context_str(row.get(id_col)),
            "skill_name": to_context_str(row.get(title_col)),
            "skill_text": to_context_str(row.get(text_col)),
        }
        if source_col:
            meta["source"] = to_context_str(row.get(source_col))
        for c in criteria_cols:
            if c in row:
                meta[c] = to_context_str(row.get(c))

        meta_rows.append(meta)

        # BM25 corpus rows
        bm25: Dict[str, Any] = {
            "id": to_context_str(row.get(id_col)),
            "title": to_context_str(row.get(title_col)),
            "text": doc,  # BM25 corpus uses the same doc text for now
        }
        if source_col:
            bm25["source"] = to_context_str(row.get(source_col))
        bm25_rows.append(bm25)

    return docs, meta_rows, bm25_rows


# --------------------------------------------------------------------------------------
# Index building + persistence
# --------------------------------------------------------------------------------------
@dataclass
class IngestionResult:
    num_docs: int
    dim: int
    paths: Dict[str, str]
    manifest: Dict[str, Any]


def build_and_persist_indexes(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    embedder: Optional[Any] = None,
) -> IngestionResult:
    """
    Core ingestion entrypoint.

    cfg expects (dict-like, matching parameters.yaml structure):
      - rag.corpus.{id_col,title_col,text_col,source_col}
      - embeddings.{model_name,dim,task_type,normalize}
      - index_store.{faiss.index_path, faiss.meta_path, bm25.corpus_path, bm25.stats_path}
      - index_store.dir (optional, for manifest only)

    embedder:
      - must implement: embed_documents(texts: List[str], *, task_type: str) -> np.ndarray
      - If None, we create one from build_embedding_model() using cfg.embeddings.*
    """
    id_col = _cfg_get(cfg, ["rag", "corpus", "id_col"], "skill_id")
    title_col = _cfg_get(cfg, ["rag", "corpus", "title_col"], "skill_name")
    text_col = _cfg_get(cfg, ["rag", "corpus", "text_col"], "skill_text")
    source_col = _cfg_get(cfg, ["rag", "corpus", "source_col"], None)

    faiss_index_path = _cfg_get(cfg, ["index_store", "faiss", "index_path"])
    faiss_meta_path = _cfg_get(cfg, ["index_store", "faiss", "meta_path"])
    bm25_corpus_path = _cfg_get(cfg, ["index_store", "bm25", "corpus_path"])
    bm25_stats_path = _cfg_get(cfg, ["index_store", "bm25", "stats_path"])
    index_store_dir = _cfg_get(cfg, ["index_store", "dir"], None)

    if not faiss_index_path or not faiss_meta_path or not bm25_corpus_path or not bm25_stats_path:
        raise ValueError("Missing index_store paths in cfg (faiss.index_path/meta_path, bm25.corpus_path/stats_path)")

    docs, meta_rows, bm25_rows = build_corpus_rows(
        df,
        id_col=id_col,
        title_col=title_col,
        text_col=text_col,
        source_col=source_col,
    )

    # ------------------------------------------------------------
    # Test mode: deterministically truncate corpus BEFORE embedding
    # (must truncate docs + meta + bm25 together to preserve alignment)
    # ------------------------------------------------------------
    test_mode = bool(_cfg_get(cfg, ["embeddings", "test_mode"], False))
    if test_mode:
        batch_size = int(_cfg_get(cfg, ["embeddings", "batch_size"], 128) or 128)
        test_max_iter = int(_cfg_get(cfg, ["embeddings", "test_max_iteration"], 30) or 30)
        if batch_size <= 0:
            raise ValueError("embeddings.batch_size must be > 0 in test_mode")
        if test_max_iter <= 0:
            raise ValueError("embeddings.test_max_iteration must be > 0 in test_mode")

        max_docs = min(len(docs), batch_size * test_max_iter)
        docs = docs[:max_docs]
        meta_rows = meta_rows[:max_docs]
        bm25_rows = bm25_rows[:max_docs]

    # Build embedder if not injected
    if embedder is None:
        from functions.utils.text_embeddings import build_embedding_model

        # ✅ define these first (fix "Unresolved reference")
        model_name = _cfg_get(cfg, ["embeddings", "model_name"], "gemini-embedding-001")
        out_dim_raw = _cfg_get(cfg, ["embeddings", "dim"], None)
        out_dim = int(out_dim_raw) if out_dim_raw not in (None, 0, "0", "") else None

        batch_size = int(_cfg_get(cfg, ["embeddings", "batch_size"], 128) or 128)
        max_workers = int(_cfg_get(cfg, ["embeddings", "max_workers"], 1) or 1)
        cache_dir = _cfg_get(cfg, ["embeddings", "cache_dir"], None)
        normalize = bool(_cfg_get(cfg, ["embeddings", "normalize"], True))

        report_every = int(_cfg_get(cfg, ["embeddings", "report_log_per_n_batch"], 10) or 10)
        test_mode = bool(_cfg_get(cfg, ["embeddings", "test_mode"], False))
        test_max_iter = int(_cfg_get(cfg, ["embeddings", "test_max_iteration"], 30) or 30)

        embedder = build_embedding_model(
            model_name=str(model_name),
            output_dim=out_dim,
            batch_size=batch_size,
            max_workers=max_workers,
            cache_dir=cache_dir,
            normalize=normalize,
            report_log_per_n_batch=report_every,
            test_mode=test_mode,
            test_max_iteration=test_max_iter,
        )

    task_type = _cfg_get(cfg, ["embeddings", "task_type"], "RETRIEVAL_DOCUMENT") or "RETRIEVAL_DOCUMENT"

    emb = embedder.embed_documents(docs, task_type=str(task_type))
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"embed_documents must return 2D array, got shape={emb.shape}")

    n, d = int(emb.shape[0]), int(emb.shape[1])

    # Build FAISS index (exact, deterministic)
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(emb, dtype=np.float32))

    # Persist FAISS index + meta
    _ensure_parent_dir(faiss_index_path)
    faiss.write_index(index, str(faiss_index_path))
    _write_jsonl(faiss_meta_path, meta_rows)

    # Persist BM25 corpus + minimal stats
    _write_jsonl(bm25_corpus_path, bm25_rows)

    stats = {
        "num_docs": n,
        "dim": d,
        "id_col": id_col,
        "title_col": title_col,
        "text_col": text_col,
        "source_col": source_col,
        "bm25_row_schema": list(bm25_rows[0].keys()) if bm25_rows else [],
    }
    _ensure_parent_dir(bm25_stats_path)
    with open(bm25_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    manifest = {
        "kind": "skills_index_store",
        "backend": {"vector": "faiss", "bm25": "local"},
        "num_docs": n,
        "dim": d,
        "embeddings": {
            "task_type": str(task_type),
            "model_name": _cfg_get(cfg, ["embeddings", "model_name"], None),
            "output_dim": _cfg_get(cfg, ["embeddings", "dim"], None),
        },
        "columns": {
            "id_col": id_col,
            "title_col": title_col,
            "text_col": text_col,
            "source_col": source_col,
        },
        "paths": {
            "index_store_dir": index_store_dir,
            "faiss_index_path": str(faiss_index_path),
            "faiss_meta_path": str(faiss_meta_path),
            "bm25_corpus_path": str(bm25_corpus_path),
            "bm25_stats_path": str(bm25_stats_path),
        },
    }

    # Persist manifest next to index_store.dir if available, else next to faiss index.
    if index_store_dir:
        manifest_path = str(Path(index_store_dir) / "manifest.json")
    else:
        manifest_path = str(Path(faiss_index_path).parent / "manifest.json")

    _ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    paths = {
        "faiss_index_path": str(faiss_index_path),
        "faiss_meta_path": str(faiss_meta_path),
        "bm25_corpus_path": str(bm25_corpus_path),
        "bm25_stats_path": str(bm25_stats_path),
        "manifest_path": str(manifest_path),
    }

    # Alignment guard (quick sanity)
    re_meta = _read_jsonl(faiss_meta_path)
    if len(re_meta) != n:
        raise RuntimeError(f"faiss_meta.jsonl length mismatch: len(meta)={len(re_meta)} vs n={n}")

    return IngestionResult(num_docs=n, dim=d, paths=paths, manifest=manifest)


__all__ = [
    "IngestionResult",
    "build_skill_document_text",
    "build_corpus_rows",
    "build_and_persist_indexes",
]