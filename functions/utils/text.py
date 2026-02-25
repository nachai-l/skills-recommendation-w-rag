"""
Text utilities (small, dependency-free)

Intent
- Keep tiny helpers that are shared across IO / processing modules.
- Avoid trimming cell values (traceability); only use for column headers unless explicitly needed.
- Deterministic behavior only (no randomness, no environment-dependent logic).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, List, Tuple


# ---------------------------------------------------------------------
# Basic whitespace helpers
# ---------------------------------------------------------------------


def trim_lr(s: str) -> str:
    """
    Trim leading/trailing whitespace (left+right).
    Keep internal whitespace unchanged.
    """
    return str(s).strip()


def normalize_ws(s: str) -> str:
    """
    Normalize whitespace deterministically:
    - Trim ends
    - Collapse internal whitespace (spaces/newlines/tabs) into single spaces

    Use ONLY for derived strings (context building), not for mutating source cells.
    """
    return " ".join(str(s).split()).strip()


# ---------------------------------------------------------------------
# Safe truncation
# ---------------------------------------------------------------------


def safe_truncate(s: Any, max_chars: int, ellipsis: str = "…") -> Tuple[str, bool]:
    """
    Truncate a value to max_chars with a trailing ellipsis (default: '…').

    Returns (result, applied).
    If max_chars <= 0, returns (stringified, False).

    Note: accepts Any to avoid upstream type surprises; caller still controls
    whether truncation is applied to source vs derived strings.
    """
    s2 = to_context_str(s)
    if not max_chars or max_chars <= 0:
        return s2, False
    if len(s2) <= max_chars:
        return s2, False
    return s2[:max_chars].rstrip() + ellipsis, True


# ---------------------------------------------------------------------
# Safe conversions
# ---------------------------------------------------------------------


def to_context_str(val: Any) -> str:
    """
    Convert arbitrary values into a context-safe string:
    - None -> ""
    - float NaN -> ""
    - everything else -> str(val)

    Note: dependency-free (no pandas). Handles float NaN via math.isnan.
    """
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val)


def json_stringify_if_needed(val: Any) -> Any:
    """
    Deterministically stringify dict/list for flat exports (PSV/CSV).

    - dict/list -> valid JSON string (UTF-8, sorted keys)
    - everything else -> returned unchanged

    Safe for repeated calls (idempotent for strings).
    Does NOT double-escape quotes.
    """
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, sort_keys=True)
    return val


# ---------------------------------------------------------------------
# Tokenization (BM25 helper)
# ---------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def simple_tokenize(
    text: Any,
    *,
    lower: bool = True,
    remove_punct: bool = True,
    min_token_len: int = 2,
) -> List[str]:
    """
    Dependency-free tokenizer for BM25 (deterministic).

    Steps (configurable):
    - stringify + normalize whitespace
    - optional lowercasing
    - optional punctuation removal (unicode-aware)
    - split on whitespace
    - filter short tokens

    Note: This is intentionally simple; if you later need language-specific
    tokenization (Thai/Japanese), add a separate tokenizer mode in core.
    """
    s = normalize_ws(to_context_str(text))
    if not s:
        return []
    if lower:
        s = s.lower()
    if remove_punct:
        s = _PUNCT_RE.sub(" ", s)
        s = normalize_ws(s)
    toks = s.split()
    if min_token_len and min_token_len > 1:
        toks = [t for t in toks if len(t) >= min_token_len]
    return toks


# ---------------------------------------------------------------------
# PSV-safe helpers
# ---------------------------------------------------------------------


def sanitize_psv_value(val: Any) -> Any:
    """
    Ensure a value is safe for single-line PSV output.

    Rules:
    - None -> ""
    - float NaN -> ""
    - int/float/bool -> unchanged (except NaN)
    - Normalize CRLF -> LF
    - Escape:
        newline -> \\n
        tab     -> \\t
        pipe    -> \\|

    Deterministic.
    """
    if val is None:
        return ""

    if isinstance(val, float) and math.isnan(val):
        return ""

    if isinstance(val, (int, float, bool)):
        return val

    s = str(val)

    # Normalize Windows newlines first
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Escape characters that break PSV structure
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    s = s.replace("|", "\\|")

    return s