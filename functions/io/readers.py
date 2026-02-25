# functions/io/readers.py
"""
Reader (Unified CSV/TSV/PSV/XLSX + JSON)

Intent
- Provide a unified reader for multiple input formats, driven by parameters.yaml:
  - csv / tsv / psv
  - xlsx with configurable sheet (default: "sheet1")
- Provide a thin JSON reader for pipeline artifacts (e.g., pipeline2_input.json)

External calls
- pandas.read_csv / pandas.read_excel
- json.loads
- functions.utils.text.trim_lr (used for trimming column headers only)

Primary functions
- read_input_table(path, fmt, sheet_name=None, encoding="utf-8") -> pandas.DataFrame
- validate_required_columns(df, required_columns) -> None (raise ValueError if missing)
- read_json(path, encoding="utf-8") -> Any

Key behaviors / guarantees
- **File existence check**: raises FileNotFoundError if the input file path does not exist.
- **Supported formats**: csv, tsv, psv, xlsx (case-insensitive).
- **Delimiter mapping**:
  - csv -> ","
  - tsv -> "\\t"
  - psv -> "|"
- **No cell-value trimming**:
  - This reader intentionally does not trim cell values to preserve traceability.
  - Only column names are trimmed defensively.

JSON reader
- read_json() is for pipeline artifacts and config-like blobs.
- No schema enforcement; callers validate shape.

Validation
- validate_required_columns(df, required_columns):
  - Computes missing columns by exact name match against df.columns.
  - Raises ValueError with both missing and found columns for debugging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import pandas as pd

from functions.utils.text import trim_lr

InputFormat = Literal["csv", "tsv", "psv", "xlsx"]

_DELIMS = {
    "csv": ",",
    "tsv": "\t",
    "psv": "|",
}


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """
    Raise ValueError if any required column is missing.
    """
    required = list(required_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def _trim_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive: strip whitespace around column names only.
    """
    df = df.copy()
    df.columns = [trim_lr(str(c)) for c in df.columns]
    return df


def _read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {str(p)}")
    return p.read_text(encoding=encoding)


def read_json(path: str | Path, *, encoding: str = "utf-8") -> Any:
    """
    Read a JSON file and return the parsed object.

    Notes:
    - This is a thin helper for pipeline artifacts (e.g., pipeline2_input.json).
    - No schema enforcement; callers should validate expected keys/types.
    """
    raw = _read_text(path, encoding=encoding)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {str(Path(path))} | {e}") from e


def read_input_table(
    path: str | Path,
    fmt: InputFormat,
    sheet_name: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read an input table from csv/tsv/psv/xlsx.

    Notes:
    - This function DOES NOT trim cell values (to preserve traceability).
    - Column names are trimmed defensively.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {str(p)}")

    fmt2 = str(fmt).lower().strip()
    if fmt2 in ("csv", "tsv", "psv"):
        delim = _DELIMS[fmt2]  # type: ignore[index]
        df = pd.read_csv(p, sep=delim, encoding=encoding, dtype=str, keep_default_na=False)
        return _trim_column_names(df)

    if fmt2 == "xlsx":
        sheet = sheet_name or "sheet1"
        df = pd.read_excel(p, sheet_name=sheet, dtype=str, keep_default_na=False)
        return _trim_column_names(df)

    raise ValueError(f"Unsupported input format: {fmt}. Expected one of: csv|tsv|psv|xlsx")


__all__ = [
    "InputFormat",
    "validate_required_columns",
    "read_input_table",
    "read_json",
]
