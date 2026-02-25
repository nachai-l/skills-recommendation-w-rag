# functions/io/writers.py
"""
Writers (Deterministic Artifacts I/O)

Intent
- Provide a single, deterministic way to write pipeline outputs to disk.
- Standardize how artifacts are persisted across pipelines:
  - JSON (readable, deterministic)
  - JSONL for record-level outputs (LLM results, bundles, traceability)
  - CSV for tabular metrics and reports
  - DELIMITED for PSV/TSV exports (Pipeline 5)

External calls
- json.dumps
- pandas.DataFrame.to_csv
- pathlib.Path
- functions.utils.logging.get_logger

Primary functions
- ensure_parent_dir(path) -> None
- write_json(path, obj) -> None
- write_jsonl(path, records) -> None
- write_csv(path, df) -> None
- write_delimited(path, df, sep="|") -> None
- write_psv(path, df) -> None

Key behaviors / guarantees
- **Deterministic output**
  - JSON:
    - UTF-8 encoding
    - sort_keys=True to ensure stable key ordering
    - ensure_ascii=False to preserve Unicode text
    - pretty indent (readability)
  - JSONL:
    - One JSON object per line
    - UTF-8 encoding
    - sort_keys=True to ensure stable key ordering
    - ensure_ascii=False to preserve Unicode text
  - CSV/DELIMITED:
    - UTF-8 encoding
    - index=False
    - Column order is exactly df.columns (caller-controlled)

- **Filesystem safety**
  - ensure_parent_dir() is called automatically before writing.
  - Parent directories are created recursively and idempotently.
  - Safe to call repeatedly across pipeline stages.

- **Observability**
  - All write operations emit INFO-level logs with:
    - file path
    - bytes written (JSON)
    - number of rows written (JSONL)
    - number of rows/columns (CSV/DELIMITED)

Design notes
- Writers are intentionally *thin*:
  - No schema enforcement
  - No column mutation
  - No validation logic
- Schema correctness and ordering are responsibilities of upstream pipeline stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from functions.utils.logging import get_logger


def ensure_parent_dir(path: str | Path) -> None:
    """
    Ensure parent directory exists for the given file path.
    Idempotent and safe.
    """
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """
    Write an object to JSON deterministically.

    Notes:
    - Caller must ensure `obj` is JSON-serializable (dict/list/str/num/bool/None),
      or provide a pre-converted representation (e.g., df.to_dict(orient="records")).
    """
    logger = get_logger(__name__)
    ensure_parent_dir(path)

    p = Path(path)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, indent=indent)
    p.write_text(text + "\n", encoding="utf-8")

    # Use len(text) for determinism even on some filesystems; still log actual size if available.
    try:
        size = p.stat().st_size
    except OSError:
        size = len((text + "\n").encode("utf-8"))

    logger.info("Wrote JSON: %s (bytes=%d)", str(p), int(size))


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """
    Write records to JSONL deterministically:
    - UTF-8
    - One JSON object per line
    - sort_keys=True for stable output
    - ensure_ascii=False to preserve Unicode text
    """
    logger = get_logger(__name__)
    ensure_parent_dir(path)

    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")

    logger.info("Wrote JSONL: %s (rows=%d)", str(p), len(records))


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    """
    Write DataFrame to CSV deterministically:
    - UTF-8
    - index=False
    - stable column order as df.columns
    """
    return write_delimited(path, df, sep=",")


def write_delimited(path: str | Path, df: pd.DataFrame, *, sep: str = "|") -> None:
    """
    Write DataFrame to a delimited text file deterministically (e.g., PSV/TSV):
    - UTF-8
    - index=False
    - stable column order as df.columns
    - Caller controls sep (default: '|')

    Notes:
    - We set lineterminator='\\n' for stable output across platforms.
    - We explicitly control quoting/escaping so JSON-in-cells (e.g., judge_reasons_json)
      stays readable and does NOT get double-quoted to ""..."" by pandas.
    """
    import csv  # local import to keep this function copy/paste-ready

    logger = get_logger(__name__)
    ensure_parent_dir(path)

    p = Path(path)

    # QUOTE_MINIMAL keeps output readable while preserving validity.
    # escapechar + doublequote=False prevents pandas from doubling quotes inside JSON strings.
    df.to_csv(
        p,
        index=False,
        encoding="utf-8",
        sep=sep,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        doublequote=False,
    )

    logger.info(
        "Wrote DELIMITED: %s (sep=%s rows=%d, cols=%d)",
        str(p),
        str(sep),
        int(df.shape[0]),
        int(df.shape[1]),
    )

def write_psv(path: str | Path, df: pd.DataFrame) -> None:
    """
    Convenience wrapper for PSV (pipe-separated values).
    """
    return write_delimited(path, df, sep="|")


__all__ = [
    "ensure_parent_dir",
    "write_json",
    "write_jsonl",
    "write_csv",
    "write_delimited",
    "write_psv",
]
