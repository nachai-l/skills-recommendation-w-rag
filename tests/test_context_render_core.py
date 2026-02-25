from __future__ import annotations

from typing import Any, Dict, List

import pytest

from functions.core.context_render import (
    ContextRenderDebug,
    render_context_rows,
    select_meta_fields,
)


def test_select_meta_fields_all_mode_returns_full_meta():
    meta = {"a": 1, "b": 2}
    out = select_meta_fields(meta=meta, mode="all", include=[], exclude=[])
    assert out == meta


def test_select_meta_fields_include_mode_fills_missing_keys():
    meta = {"a": "x"}
    out = select_meta_fields(meta=meta, mode="include", include=["a", "b"], exclude=[])
    assert out["a"] == "x"
    assert out["b"] == ""  # missing filled deterministically


def test_select_meta_fields_exclude_mode_removes_keys():
    meta = {"a": 1, "b": 2, "c": 3}
    out = select_meta_fields(meta=meta, mode="exclude", include=[], exclude=["b"])
    assert "b" not in out
    assert out["a"] == 1
    assert out["c"] == 3


def _mk_row(
    skill_id: str,
    skill_name: str,
    *,
    meta: Dict[str, Any] | None = None,
    source: str = "",
    score_vector: float = 0.0,
    score_bm25: float = 0.0,
    score_hybrid: float = 0.0,
) -> Dict[str, Any]:
    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "source": source,
        "score_vector": score_vector,
        "score_bm25": score_bm25,
        "score_hybrid": score_hybrid,
        "meta": meta or {},
    }


def test_render_context_rows_basic_and_safe_missing_placeholders():
    rows = [
        _mk_row(
            "A",
            "Alpha",
            meta={"skill_text": "hello"},
            score_hybrid=0.9,
        )
    ]

    # template references unknown placeholder {missing} -> should become ""
    tmpl = "- id={skill_id} name={skill_name} hybrid={score_hybrid} text={skill_text} missing={missing}\n"

    ctx, dbg = render_context_rows(
        rows=rows,
        row_template=tmpl,
        max_context_chars=10000,
        truncate_field_chars=1000,
        columns_mode="include",
        columns_include=["skill_text"],
        columns_exclude=[],
    )

    assert "id=A" in ctx
    assert "name=Alpha" in ctx
    assert "hybrid=0.9" in ctx
    assert "text=hello" in ctx
    assert "missing=" in ctx  # exists but blank
    assert isinstance(dbg, ContextRenderDebug)
    assert dbg.rows_rendered == 1
    assert dbg.context_truncated is False


def test_render_context_rows_truncate_field_chars_applies_to_strings():
    long_text = "x" * 500
    rows = [_mk_row("A", "Alpha", meta={"skill_text": long_text}, score_hybrid=0.1)]

    tmpl = "- {skill_id} {skill_text}\n"
    ctx, _ = render_context_rows(
        rows=rows,
        row_template=tmpl,
        max_context_chars=10000,
        truncate_field_chars=50,
        columns_mode="include",
        columns_include=["skill_text"],
        columns_exclude=[],
    )

    # "x"*50 must appear; longer should be cut
    assert "x" * 50 in ctx
    assert "x" * 51 not in ctx


def test_render_context_rows_enforces_max_context_chars_stops_adding_rows():
    rows = [
        _mk_row("A", "Alpha", meta={"skill_text": "a" * 100}, score_hybrid=0.1),
        _mk_row("B", "Beta", meta={"skill_text": "b" * 100}, score_hybrid=0.2),
        _mk_row("C", "Gamma", meta={"skill_text": "c" * 100}, score_hybrid=0.3),
    ]
    tmpl = "- {skill_id}\n  {skill_text}\n"

    # Force very small max so not all rows can fit
    ctx, dbg = render_context_rows(
        rows=rows,
        row_template=tmpl,
        max_context_chars=140,  # small
        truncate_field_chars=1000,
        columns_mode="include",
        columns_include=["skill_text"],
        columns_exclude=[],
    )

    assert dbg.context_truncated is True
    assert dbg.rows_rendered in (1, 2)  # depends on exact formatting length
    # It should contain at least the first row
    assert "- A" in ctx


def test_render_context_rows_first_row_too_large_hard_truncates():
    rows = [
        _mk_row("A", "Alpha", meta={"skill_text": "x" * 5000}, score_hybrid=0.1),
    ]
    tmpl = "- {skill_id}\n  {skill_text}\n"

    ctx, dbg = render_context_rows(
        rows=rows,
        row_template=tmpl,
        max_context_chars=200,  # smaller than first row would be
        truncate_field_chars=10000,  # don't field-truncate, force max_context truncation
        columns_mode="include",
        columns_include=["skill_text"],
        columns_exclude=[],
    )

    assert dbg.context_truncated is True
    assert dbg.rows_rendered == 1
    assert len(ctx) <= 200


def test_render_context_rows_empty_template_returns_empty():
    rows = [_mk_row("A", "Alpha", meta={"skill_text": "hello"})]
    ctx, dbg = render_context_rows(
        rows=rows,
        row_template="",
        max_context_chars=1000,
        truncate_field_chars=100,
        columns_mode="all",
        columns_include=[],
        columns_exclude=[],
    )
    assert ctx == ""
    assert dbg.rows_rendered == 0
    assert dbg.context_truncated is False