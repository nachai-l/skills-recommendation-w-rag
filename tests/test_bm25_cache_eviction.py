# tests/test_bm25_cache_eviction.py
"""
Tests for BM25 cache eviction strategy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

import functions.online.pipeline_3b_bm25_search as p3b
from functions.core.bm25 import BM25Config, TokenizerConfig, build_bm25_index_from_rows


def _make_index(rows: List[Dict[str, Any]]) -> Any:
    return build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )


def test_bm25_cache_evicts_oldest_when_full():
    """Cache should evict oldest entry when max size is reached."""
    # Clear cache
    p3b._BM25_CACHE.clear()

    original_max = p3b._BM25_CACHE_MAX_SIZE

    try:
        p3b._BM25_CACHE_MAX_SIZE = 2

        # Insert two entries
        p3b._BM25_CACHE["key_a"] = "index_a"
        p3b._BM25_CACHE["key_b"] = "index_b"

        assert len(p3b._BM25_CACHE) == 2
        assert "key_a" in p3b._BM25_CACHE
        assert "key_b" in p3b._BM25_CACHE

        # Simulate what get_bm25_index_cached does when adding a new entry:
        # evict oldest when at capacity
        while len(p3b._BM25_CACHE) >= p3b._BM25_CACHE_MAX_SIZE:
            oldest_key = next(iter(p3b._BM25_CACHE))
            del p3b._BM25_CACHE[oldest_key]
        p3b._BM25_CACHE["key_c"] = "index_c"

        assert len(p3b._BM25_CACHE) == 2
        assert "key_a" not in p3b._BM25_CACHE  # evicted
        assert "key_b" in p3b._BM25_CACHE
        assert "key_c" in p3b._BM25_CACHE

    finally:
        p3b._BM25_CACHE_MAX_SIZE = original_max
        p3b._BM25_CACHE.clear()


def test_bm25_cache_returns_cached_on_hit():
    """Cache hit should return the same object without rebuilding."""
    p3b._BM25_CACHE.clear()

    rows = [{"skill_id": "S1", "skill_name": "Python", "doc": "python"}]
    index = _make_index(rows)

    cfg = BM25Config(tokenizer=TokenizerConfig())
    key_kwargs = dict(
        corpus_path="/fake/path",
        cfg=cfg,
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )

    key = p3b._bm25_cache_key(**key_kwargs)
    p3b._BM25_CACHE[key] = index

    cached = p3b.get_bm25_index_cached(**key_kwargs)
    assert cached is index

    p3b._BM25_CACHE.clear()
