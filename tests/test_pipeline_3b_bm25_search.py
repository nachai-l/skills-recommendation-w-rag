# tests/test_pipeline_3b_bm25_search.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

import functions.online.pipeline_3b_bm25_search as p3b


def _base_params(*, top_k_default: int = 3) -> Dict[str, Any]:
    return {
        "index_store": {"bm25": {"corpus_path": "artifacts/index_store/bm25_corpus.jsonl"}},
        "rag": {
            "vector_search": {"top_k_default": top_k_default},
            "corpus": {"id_col": "skill_id", "title_col": "skill_name"},
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "tokenizer": "simple",
                "lower": True,
                "remove_punct": True,
                "min_token_len": 2,
            },
        },
    }


def test_pipeline_3b_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params(top_k_default=3)

    # Patch config load
    monkeypatch.setattr(p3b, "load_parameters", lambda _path: params)
    # Patch path resolution
    monkeypatch.setattr(p3b, "repo_root_from_parameters_path", lambda _p: Path("."))
    monkeypatch.setattr(p3b, "resolve_path", lambda p, *, base_dir: Path(p))

    # Build a small in-memory index (no file IO)
    rows: List[Dict[str, Any]] = [
        {"skill_id": "S1", "skill_name": "Python", "doc": "python pandas numpy"},
        {"skill_id": "S2", "skill_name": "SQL", "doc": "sql join database"},
        {"skill_id": "S3", "skill_name": "Docker", "doc": "docker container image"},
    ]
    from functions.core.bm25 import BM25Config, TokenizerConfig, build_bm25_index_from_rows

    index = build_bm25_index_from_rows(
        rows,
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )

    # New signature includes key mapping args; easiest is accept **kwargs
    monkeypatch.setattr(p3b, "get_bm25_index_cached", lambda **kwargs: index)

    out = p3b.run_pipeline_3b_bm25_search(query="python", top_k=None, debug=True)

    assert out["query"] == "python"
    assert out["top_k"] == 3
    assert "debug" in out
    assert len(out["results"]) >= 1
    assert out["results"][0]["skill_id"] == "S1"
    assert out["results"][0]["score_bm25"] > 0.0


def test_pipeline_3b_empty_query_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _base_params()
    monkeypatch.setattr(p3b, "load_parameters", lambda _path: params)
    monkeypatch.setattr(p3b, "repo_root_from_parameters_path", lambda _p: Path("."))
    monkeypatch.setattr(p3b, "resolve_path", lambda p, *, base_dir: Path(p))

    from functions.core.bm25 import BM25Config, TokenizerConfig, build_bm25_index_from_rows

    index = build_bm25_index_from_rows(
        [{"skill_id": "S1", "skill_name": "Python", "doc": "python"}],
        cfg=BM25Config(tokenizer=TokenizerConfig()),
        id_key="skill_id",
        name_key="skill_name",
        source_key="source",
        doc_key="doc",
    )

    monkeypatch.setattr(p3b, "get_bm25_index_cached", lambda **kwargs: index)

    with pytest.raises(ValueError, match="query must be non-empty"):
        p3b.run_pipeline_3b_bm25_search(query="   ")