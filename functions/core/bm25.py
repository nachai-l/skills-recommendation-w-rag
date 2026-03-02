# functions/core/bm25.py
"""
BM25 (in-process) — minimal deterministic implementation

Intent
- Provide a small, dependency-free BM25 scorer for online lexical retrieval.
- Load a JSONL BM25 corpus (skill_id + doc + optional metadata).
- Tokenize, compute DF/IDF, and score queries using Okapi BM25.

Design goals
- Deterministic ranking with stable tie-breaks.
- Fast enough for ~O(10^4–10^5) docs in-memory (e.g., ~38k skills).
- No external dependencies (pure Python + stdlib).

Notes
- Tokenization is intentionally simple and deterministic.
  Language-specific tokenizers (Thai/Japanese) should be implemented separately.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Tokenization
# -----------------------------
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer controls for BM25 (kept simple for determinism)."""
    tokenizer: str = "simple"  # simple | whitespace
    lower: bool = True
    remove_punct: bool = True
    min_token_len: int = 2


def tokenize(text: str, cfg: TokenizerConfig) -> List[str]:
    """Tokenize text deterministically using the configured options."""
    s = text or ""
    if cfg.lower:
        s = s.lower()
    if cfg.remove_punct:
        s = _PUNCT_RE.sub(" ", s)

    # NOTE: cfg.tokenizer is currently informational (split() is whitespace-based).
    toks = s.split()

    if cfg.min_token_len and cfg.min_token_len > 1:
        toks = [t for t in toks if len(t) >= cfg.min_token_len]

    return toks


# -----------------------------
# BM25 core
# -----------------------------
@dataclass(frozen=True)
class BM25Config:
    """Okapi BM25 parameters and tokenization config."""
    k1: float = 1.5
    b: float = 0.75
    tokenizer: TokenizerConfig = TokenizerConfig()


@dataclass(frozen=True)
class BM25Doc:
    """A single BM25 document row aligned to corpus order."""
    doc_id: int  # internal index aligned to corpus row order
    skill_id: str
    skill_name: Optional[str]
    source: Optional[str]
    doc: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class BM25Hit:
    """A scored BM25 search hit."""
    skill_id: str
    skill_name: str
    score_bm25: float
    internal_idx: int
    meta: Dict[str, Any]


@dataclass
class BM25Index:
    """In-memory BM25 index (DF/IDF + per-doc TF maps)."""
    docs: List[BM25Doc]
    doc_len: List[int]
    avgdl: float
    df: Dict[str, int]
    idf: Dict[str, float]
    tf: List[Dict[str, int]]
    cfg: BM25Config

    @property
    def n_docs(self) -> int:
        return len(self.docs)


def _compute_idf(n_docs: int, df: int) -> float:
    """Compute Okapi BM25 IDF with a stable smoothing term."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def build_bm25_index_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    cfg: BM25Config,
    id_key: str = "skill_id",
    name_key: str = "skill_name",
    source_key: str = "source",
    doc_key: str = "doc",
    # fallback keys (keeps this resilient if corpus format changes)
    id_fallback_keys: Sequence[str] = ("skill_id", "id", "skillId", "skill_code"),
    name_fallback_keys: Sequence[str] = ("skill_name", "name", "title", "label", "skill"),
    doc_fallback_keys: Sequence[str] = ("doc", "skill_text", "text", "content"),
) -> BM25Index:
    """
    Build a BM25Index from corpus rows.

    Notes:
    - Keeps a full `meta` dict per doc for downstream enrichment / API payload joins.
    - Applies best-effort key fallback for robustness across corpus schema variants.
    """
    docs: List[BM25Doc] = []
    tf: List[Dict[str, int]] = []
    df: Dict[str, int] = {}
    doc_len: List[int] = []

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"Invalid corpus row at i={i}: expected dict")

        # ---- id ----
        skill_id = r.get(id_key)
        if not skill_id:
            for k in id_fallback_keys:
                if k == id_key:
                    continue
                skill_id = r.get(k)
                if skill_id:
                    break
        if not skill_id or not isinstance(skill_id, str):
            raise ValueError(
                f"Missing/invalid id at i={i}. "
                f"Expected key={id_key!r} (or fallback {list(id_fallback_keys)}). "
                f"Row keys={sorted(list(r.keys()))[:50]}"
            )

        # ---- name ----
        skill_name = r.get(name_key)
        if skill_name is None or (isinstance(skill_name, str) and not skill_name.strip()):
            skill_name = None
            for k in name_fallback_keys:
                if k == name_key:
                    continue
                v = r.get(k)
                if v is None:
                    continue
                v = str(v).strip()
                if v:
                    skill_name = v
                    break
        elif not isinstance(skill_name, str):
            skill_name = str(skill_name).strip() or None

        # ---- source ----
        source = r.get(source_key)
        if source is not None and not isinstance(source, str):
            source = str(source)

        # ---- doc ----
        doc_text = r.get(doc_key)
        if doc_text is None:
            for k in doc_fallback_keys:
                if k == doc_key:
                    continue
                doc_text = r.get(k)
                if doc_text is not None:
                    break
        if doc_text is None:
            doc_text = ""
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        meta = dict(r)

        # Canonicalize meta fields so downstream access is stable
        meta.setdefault(id_key, skill_id)
        meta.setdefault(doc_key, doc_text)
        if skill_name is not None:
            meta.setdefault(name_key, skill_name)
        if source is not None:
            meta.setdefault(source_key, source)

        tokens = tokenize(doc_text, cfg.tokenizer)
        dl = len(tokens)
        doc_len.append(dl)

        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        tf.append(counts)

        # DF counts unique terms per doc (use counts keys, not raw tokens list).
        for t in counts.keys():
            df[t] = df.get(t, 0) + 1

        docs.append(
            BM25Doc(
                doc_id=i,
                skill_id=skill_id,
                skill_name=skill_name,
                source=source,
                doc=doc_text,
                meta=meta,
            )
        )

    n = len(docs)
    avgdl = (sum(doc_len) / n) if n > 0 else 0.0
    idf: Dict[str, float] = {t: _compute_idf(n, dfi) for t, dfi in df.items()}

    return BM25Index(
        docs=docs,
        doc_len=doc_len,
        avgdl=avgdl,
        df=df,
        idf=idf,
        tf=tf,
        cfg=cfg,
    )


def load_bm25_corpus_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load BM25 corpus rows from JSONL (one dict/object per line)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BM25 corpus not found: {p}")

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
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
            rows.append(obj)
    return rows


def bm25_score_query(index: BM25Index, query: str) -> List[float]:
    """Compute BM25 scores for all docs (aligned to index.docs order)."""
    q = (query or "").strip()
    if not q:
        return [0.0] * index.n_docs

    q_tokens = tokenize(q, index.cfg.tokenizer)
    if not q_tokens:
        return [0.0] * index.n_docs

    q_counts: Dict[str, int] = {}
    for t in q_tokens:
        q_counts[t] = q_counts.get(t, 0) + 1

    k1 = float(index.cfg.k1)
    b = float(index.cfg.b)
    avgdl = float(index.avgdl) if index.avgdl > 0 else 1.0

    scores = [0.0] * index.n_docs

    for doc_id in range(index.n_docs):
        tf_map = index.tf[doc_id]
        dl = float(index.doc_len[doc_id])

        denom_norm = k1 * (1.0 - b + b * (dl / avgdl))

        s = 0.0
        for t, qtf in q_counts.items():
            f = tf_map.get(t, 0)
            if f <= 0:
                continue
            idf = index.idf.get(t)
            if idf is None:
                continue
            s += float(qtf) * idf * (f * (k1 + 1.0)) / (f + denom_norm)

        scores[doc_id] = float(s)

    return scores


def bm25_search(
    *,
    index: BM25Index,
    query: str,
    top_k: int,
    debug: bool = False,
    meta_skill_id_key: str = "skill_id",
    meta_skill_name_key: str = "skill_name",
) -> Tuple[List[BM25Hit], Optional[Dict[str, Any]]]:
    """
    Search BM25 index and return top-k hits (deterministic order).

    Tie-break:
    - score_bm25 desc
    - skill_id asc
    - internal_idx asc
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")

    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be > 0")

    scores = bm25_score_query(index, q)
    candidates: List[Tuple[float, int]] = [(scores[i], i) for i in range(len(scores)) if scores[i] > 0.0]

    hits: List[BM25Hit] = []
    for score, internal_idx in candidates:
        doc = index.docs[internal_idx]
        meta = doc.meta

        skill_id = meta.get(meta_skill_id_key) or doc.skill_id
        skill_name = meta.get(meta_skill_name_key) or doc.skill_name or ""

        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError(f"Invalid skill_id in meta for internal_idx={internal_idx}")
        if not isinstance(skill_name, str) or not skill_name.strip():
            skill_name = (doc.skill_name or "").strip() or skill_id

        hits.append(
            BM25Hit(
                skill_id=skill_id,
                skill_name=skill_name,
                score_bm25=float(score),
                internal_idx=int(internal_idx),
                meta=meta,
            )
        )

    # Deterministic ranking for stable results across runs.
    hits.sort(key=lambda h: (-h.score_bm25, str(h.skill_id), int(h.internal_idx)))
    hits = hits[:k]

    dbg = None
    if debug:
        dbg = {
            "n_docs": index.n_docs,
            "avgdl": index.avgdl,
            "num_candidates_scored_gt0": len(candidates),
            "returned_k": k,
            "num_hits": len(hits),
            "score_bm25_min": float(min((h.score_bm25 for h in hits), default=0.0)),
            "score_bm25_max": float(max((h.score_bm25 for h in hits), default=0.0)),
        }

    return hits, dbg


def to_api_payload(
    *,
    query: str,
    top_k: int,
    hits: List[BM25Hit],
    debug: Optional[Dict[str, Any]] = None,
    include_meta: bool = True,
    include_internal_idx: bool = True,
) -> Dict[str, Any]:
    """Convert BM25 hits into a JSON-serializable payload (for debugging/inspection)."""
    out_hits: List[Dict[str, Any]] = []
    for h in hits:
        row: Dict[str, Any] = {
            "skill_id": h.skill_id,
            "skill_name": h.skill_name,
            "score_bm25": h.score_bm25,
        }
        if include_internal_idx:
            row["internal_idx"] = h.internal_idx
        if include_meta:
            row["meta"] = h.meta
        out_hits.append(row)

    payload: Dict[str, Any] = {"query": query, "top_k": top_k, "results": out_hits}
    if debug is not None:
        payload["debug"] = debug
    return payload


__all__ = [
    "TokenizerConfig",
    "BM25Config",
    "BM25Index",
    "BM25Hit",
    "load_bm25_corpus_jsonl",
    "build_bm25_index_from_rows",
    "bm25_search",
    "to_api_payload",
]