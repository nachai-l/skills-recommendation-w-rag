# functions/online/pipeline_3b_bm25_search.py
from __future__ import annotations

"""
Pipeline 3b — Online BM25 Search (Lexical)

Thin orchestration layer.

Responsibilities:
- Load config (parameters.yaml)
- Resolve BM25 corpus path (bm25_corpus.jsonl)
- Build and cache BM25Index in-process
- Run BM25 search for a query
- Return API-friendly payload

Core logic lives in:
    functions/core/bm25.py
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from functions.core.bm25 import (
    BM25Config,
    TokenizerConfig,
    bm25_search,
    build_bm25_index_from_rows,
    load_bm25_corpus_jsonl,
    to_api_payload,
)
from functions.utils.config import load_parameters
from functions.utils.config_access import cfg_get as _get, cfg_get_path as _get_path
from functions.utils.logging import get_logger
from functions.utils.paths import repo_root_from_parameters_path, resolve_path

logger = get_logger(__name__)


# -----------------------------
# Process-local cache (ONLINE)
# -----------------------------
_BM25_CACHE_MAX_SIZE = 2  # evict oldest entries when exceeded
_BM25_CACHE: Dict[str, Any] = {}  # key -> BM25Index


def _bm25_cache_key(
    *,
    corpus_path: str | Path,
    cfg: BM25Config,
    id_key: str,
    name_key: str,
    source_key: str,
    doc_key: str,
) -> str:
    # Stringify config to avoid subtle cache mismatches across workers / reloads
    tok = cfg.tokenizer
    return (
        f"{Path(corpus_path).resolve()}||"
        f"id={id_key}||name={name_key}||source={source_key}||doc={doc_key}||"
        f"k1={cfg.k1}||b={cfg.b}||"
        f"tok={tok.tokenizer}||lower={tok.lower}||punct={tok.remove_punct}||minlen={tok.min_token_len}"
    )


def get_bm25_index_cached(
    *,
    corpus_path: str | Path,
    cfg: BM25Config,
    id_key: str,
    name_key: str,
    source_key: str,
    doc_key: str,
) -> Any:
    """
    Load corpus JSONL, build BM25Index, cache per (path + cfg + key mapping).

    Notes:
    - Cache is process-local. In multi-worker serving, each worker has its own cache.
    - On Cloud Run cold start, first request builds the index.
    """
    key = _bm25_cache_key(
        corpus_path=corpus_path,
        cfg=cfg,
        id_key=id_key,
        name_key=name_key,
        source_key=source_key,
        doc_key=doc_key,
    )
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        return cached

    rows = load_bm25_corpus_jsonl(corpus_path)

    # Build index with configurable field mapping (robust to corpus schema differences)
    index = build_bm25_index_from_rows(
        rows,
        cfg=cfg,
        id_key=id_key,
        name_key=name_key,
        source_key=source_key,
        doc_key=doc_key,
    )

    # Evict oldest entries if cache is full
    while len(_BM25_CACHE) >= _BM25_CACHE_MAX_SIZE:
        oldest_key = next(iter(_BM25_CACHE))
        del _BM25_CACHE[oldest_key]

    _BM25_CACHE[key] = index
    return index


# -----------------------------
# Public entrypoint
# -----------------------------
def run_pipeline_3b_bm25_search(
    *,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    parameters_path: str = "configs/parameters.yaml",
    include_meta: bool = True,
    include_internal_idx: bool = True,
) -> Dict[str, Any]:
    """
    Run online BM25 lexical search.

    Args:
      query: user query string
      top_k: override; if None use rag.vector_search.top_k_default (same default as 3a)
      debug: include debug fields in output
      parameters_path: path to parameters.yaml
      include_meta: include meta dict for each hit
      include_internal_idx: include internal idx (row id in corpus order)

    Returns:
      JSON-serializable dict payload.
    """
    params = load_parameters(parameters_path)

    corpus_path_raw = _get_path(params, ["index_store", "bm25", "corpus_path"])
    top_k_default = int(_get_path(params, ["rag", "vector_search", "top_k_default"]))
    k = int(top_k) if top_k is not None else top_k_default

    # BM25 hyperparams + tokenizer config
    bm25_cfg_raw = _get_path(params, ["rag", "bm25"])
    cfg = BM25Config(
        k1=float(_get(bm25_cfg_raw, "k1")),
        b=float(_get(bm25_cfg_raw, "b")),
        tokenizer=TokenizerConfig(
            tokenizer=str(_get(bm25_cfg_raw, "tokenizer")),
            lower=bool(_get(bm25_cfg_raw, "lower")),
            remove_punct=bool(_get(bm25_cfg_raw, "remove_punct")),
            min_token_len=int(_get(bm25_cfg_raw, "min_token_len")),
        ),
    )

    # Resolve corpus path relative to repo root (CWD-independent)
    repo_root = repo_root_from_parameters_path(parameters_path)
    corpus_path = resolve_path(str(corpus_path_raw), base_dir=repo_root)

    # BM25 corpus field mapping — read from config (consistent with pipeline 3a),
    # with defaults matching the canonical corpus schema.
    def _cfg_or_default(cfg: Any, key: str, default: str) -> str:
        try:
            return str(_get(cfg, key))
        except (KeyError, AttributeError):
            return default

    corpus_cfg = _get_path(params, ["rag", "corpus"])
    id_key = _cfg_or_default(corpus_cfg, "id_col", "id")
    name_key = _cfg_or_default(corpus_cfg, "title_col", "title")
    source_key = _cfg_or_default(corpus_cfg, "source_col", "source")
    doc_key = _cfg_or_default(corpus_cfg, "text_col", "text")

    if debug:
        logger.info(
            "P3b BM25 search start",
            extra={
                "query_len": len((query or "").strip()),
                "top_k": k,
                "corpus_path": str(corpus_path),
                "k1": cfg.k1,
                "b": cfg.b,
                "id_key": id_key,
                "name_key": name_key,
                "source_key": source_key,
                "doc_key": doc_key,
            },
        )

    index = get_bm25_index_cached(
        corpus_path=corpus_path,
        cfg=cfg,
        id_key=id_key,
        name_key=name_key,
        source_key=source_key,
        doc_key=doc_key,
    )

    hits, dbg = bm25_search(
        index=index,
        query=query,
        top_k=k,
        debug=debug,
        meta_skill_id_key=id_key,
        meta_skill_name_key=name_key,
    )

    payload = to_api_payload(
        query=(query or "").strip(),
        top_k=k,
        hits=hits,
        debug=dbg,
        include_meta=include_meta,
        include_internal_idx=include_internal_idx,
    )

    if debug:
        logger.info(
            "P3b BM25 search done",
            extra={
                "top_k": k,
                "num_results": len(payload.get("results", [])),
                "score_bm25_max": payload.get("debug", {}).get("score_bm25_max") if payload.get("debug") else None,
            },
        )

    return payload


__all__ = ["run_pipeline_3b_bm25_search", "get_bm25_index_cached"]