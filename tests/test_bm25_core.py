# tests/test_bm25_core.py
from __future__ import annotations

import pytest

from functions.core.bm25 import (
    BM25Config,
    TokenizerConfig,
    bm25_search,
    build_bm25_index_from_rows,
    tokenize,
)


def test_tokenize_basic_lower_punct_minlen() -> None:
    cfg = TokenizerConfig(tokenizer="simple", lower=True, remove_punct=True, min_token_len=2)
    toks = tokenize("Hello, WORLD!! a iOS-Dev", cfg)
    # "a" is dropped (minlen=2); punctuation removed; lowercased
    assert toks == ["hello", "world", "ios", "dev"]


def test_build_index_and_search_returns_expected_top_hit() -> None:
    rows = [
        {"skill_id": "S1", "skill_name": "Python", "doc": "python pandas numpy data analysis"},
        {"skill_id": "S2", "skill_name": "SQL", "doc": "sql query join group by database"},
        {"skill_id": "S3", "skill_name": "Docker", "doc": "docker container build image deploy"},
    ]
    cfg = BM25Config(k1=1.5, b=0.75, tokenizer=TokenizerConfig())
    index = build_bm25_index_from_rows(
        rows,
        cfg=cfg,
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )

    hits, dbg = bm25_search(index=index, query="python data", top_k=2, debug=True)
    assert dbg is not None
    assert len(hits) >= 1
    assert hits[0].skill_id == "S1"
    assert hits[0].score_bm25 > 0.0


def test_search_empty_query_raises() -> None:
    rows = [{"skill_id": "S1", "skill_name": "Python", "doc": "python pandas"}]
    index = build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )
    with pytest.raises(ValueError, match="query must be non-empty"):
        bm25_search(index=index, query="   ", top_k=5)


def test_search_top_k_invalid_raises() -> None:
    rows = [{"skill_id": "S1", "skill_name": "Python", "doc": "python pandas"}]
    index = build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )
    with pytest.raises(ValueError, match="top_k must be > 0"):
        bm25_search(index=index, query="python", top_k=0)


def test_search_returns_empty_when_no_tokens_match() -> None:
    rows = [
        {"skill_id": "S1", "skill_name": "Python", "doc": "python pandas"},
        {"skill_id": "S2", "skill_name": "SQL", "doc": "sql join"},
    ]
    index = build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )
    hits, dbg = bm25_search(index=index, query="kubernetes", top_k=5, debug=True)
    assert hits == []
    assert dbg is not None
    assert dbg["num_hits"] == 0


def test_stable_tie_break_score_then_skill_id_then_internal_idx() -> None:
    # Construct identical docs so scores tie
    rows = [
        {"skill_id": "S200", "skill_name": "B", "doc": "same tokens here"},
        {"skill_id": "S100", "skill_name": "A", "doc": "same tokens here"},
        {"skill_id": "S100", "skill_name": "A-dup", "doc": "same tokens here"},
    ]
    index = build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )
    hits, _ = bm25_search(index=index, query="same tokens", top_k=3)

    got = [(h.skill_id, h.internal_idx) for h in hits]
    # Score ties → sort by skill_id asc → internal_idx asc
    assert got == [("S100", 1), ("S100", 2), ("S200", 0)]


def test_build_index_accepts_fallback_id_key() -> None:
    # Uses "id" instead of "skill_id" → should be accepted via fallback logic
    rows = [
        {"id": "S1", "skill_name": "Python", "doc": "python pandas"},
        {"id": "S2", "skill_name": "SQL", "doc": "sql join"},
    ]
    index = build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",  # intentionally "wrong" primary key
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )

    hits, _ = bm25_search(index=index, query="python", top_k=2)
    assert len(hits) >= 1
    assert hits[0].skill_id == "S1"