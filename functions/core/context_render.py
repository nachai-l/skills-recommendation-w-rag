# functions/core/context_render.py
"""
Context renderer (Pipeline 4a) — deterministic row templating with size bounds.

Intent
- Render retrieval rows into a single `{context}` string for LLM prompts.
- Apply:
  - column selection over `meta` (all/include/exclude)
  - per-field truncation (string fields only)
  - overall context character budget (`max_context_chars`)
- Keep rendering deterministic and tolerant to missing template keys.

Key behaviors
- Uses `format_map()` with a SafeFormatDict so missing placeholders render as "" (no KeyError).
- Scores always come from the merged retrieval row (not from meta).
- Enforces a hard cap so returned context never exceeds `max_context_chars`.
- Adds a trailing newline when possible (helps prompt readability) without exceeding the cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class _SafeFormatDict(dict):
    """format_map helper: missing keys become empty strings instead of raising KeyError."""
    def __missing__(self, key: str) -> str:
        return ""


def _truncate_str(s: Any, max_chars: int) -> str:
    """Deterministically truncate a value (stringify; None -> '')."""
    if s is None:
        return ""
    text = str(s)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def select_meta_fields(
    *,
    meta: Dict[str, Any],
    mode: str,
    include: List[str],
    exclude: List[str],
) -> Dict[str, Any]:
    """
    Select meta fields according to the configured columns mode.

    mode:
      - all: include all meta keys
      - include: include only keys listed in `include` (missing -> "")
      - exclude: include all except keys listed in `exclude`
    """
    if not isinstance(meta, dict):
        return {}

    mode = (mode or "all").strip().lower()
    if mode == "all":
        out = dict(meta)
    elif mode == "include":
        out = {k: meta.get(k, "") for k in include}
    elif mode == "exclude":
        out = {k: v for k, v in meta.items() if k not in set(exclude)}
    else:
        # fallback to all
        out = dict(meta)

    # Deterministic key stability: ensure included keys exist even if missing in meta.
    if mode == "include":
        for k in include:
            out.setdefault(k, "")
    return out


@dataclass
class ContextRenderDebug:
    """Debug counters for rendered context size and join coverage."""
    context_chars: int
    context_truncated: bool
    rows_rendered: int


def render_context_rows(
    *,
    rows: List[Dict[str, Any]],
    row_template: str,
    max_context_chars: int,
    truncate_field_chars: int,
    columns_mode: str,
    columns_include: List[str],
    columns_exclude: List[str],
) -> tuple[str, ContextRenderDebug]:
    """
    Render context using row_template, enforcing per-field truncation and overall max_context_chars.
    """
    rendered_parts: List[str] = []
    total = 0
    truncated = False
    count = 0

    tmpl = (row_template or "").rstrip()
    if not tmpl:
        return "", ContextRenderDebug(context_chars=0, context_truncated=False, rows_rendered=0)

    for r in rows:
        meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        meta_selected = select_meta_fields(
            meta=meta,
            mode=columns_mode,
            include=columns_include,
            exclude=columns_exclude,
        )

        # Build row_data for template:
        # - scores always come from merged row
        # - meta fills skill_text + criteria etc
        row_data: Dict[str, Any] = {}
        row_data.update(meta_selected)

        # Common top-level fields
        row_data.setdefault("skill_id", r.get("skill_id", ""))
        row_data.setdefault("skill_name", r.get("skill_name", ""))
        row_data.setdefault("source", r.get("source", ""))

        # Scores (not taken from meta)
        row_data["score_vector"] = r.get("score_vector", 0.0)
        row_data["score_bm25"] = r.get("score_bm25", 0.0)
        row_data["score_hybrid"] = r.get("score_hybrid", 0.0)

        # Truncate string fields deterministically (do not touch numeric scores)
        for k, v in list(row_data.items()):
            if isinstance(v, str) or v is None:
                row_data[k] = _truncate_str(v, truncate_field_chars)

        # Format row with missing-key tolerance.
        row_str = tmpl.format_map(_SafeFormatDict(row_data)).rstrip()

        if not row_str:
            continue

        # Ensure newline separation between rows.
        block = row_str + "\n"
        block_len = len(block)

        # Enforce overall context budget.
        if max_context_chars and max_context_chars > 0 and (total + block_len) > max_context_chars:
            if total == 0:
                # First row already too large -> hard truncate the block
                block = block[:max_context_chars].rstrip()
                rendered_parts.append(block)
                total = len(block)
                truncated = True
                count = 1
            else:
                truncated = True
            break

        rendered_parts.append(block)
        total += block_len
        count += 1

    # Assemble final context (no extra whitespace).
    context = "".join(rendered_parts).rstrip()

    # Add trailing newline only if it won't violate max_context_chars
    if rendered_parts:
        if not max_context_chars or max_context_chars <= 0:
            context = context + "\n"
        else:
            if len(context) < max_context_chars:
                context = context + "\n"
            # else: already at limit, do not add newline

    # Final hard cap safety (never exceed max_context_chars)
    if max_context_chars and max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars].rstrip()

    dbg = ContextRenderDebug(
        context_chars=len(context),
        context_truncated=truncated,
        rows_rendered=count,
    )
    return context, dbg


__all__ = ["render_context_rows", "select_meta_fields", "ContextRenderDebug"]