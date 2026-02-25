import pandas as pd
import pytest
from pathlib import Path

from functions.io.readers import read_input_table, validate_required_columns


def test_validate_required_columns_ok():
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})
    validate_required_columns(df, ["a", "b"])


def test_validate_required_columns_missing():
    df = pd.DataFrame({"a": ["1"]})
    with pytest.raises(ValueError) as e:
        validate_required_columns(df, ["a", "b"])
    msg = str(e.value).lower()
    assert "missing required columns" in msg
    assert "found columns" in msg


def test_read_input_table_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_input_table(tmp_path / "nope.csv", "csv")  # type: ignore[arg-type]


def test_read_input_table_unsupported_format(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(ValueError) as e:
        read_input_table(p, "txt")  # type: ignore[arg-type]
    assert "unsupported input format" in str(e.value).lower()


def test_read_csv_trims_column_names_only(tmp_path: Path):
    # column names include spaces; values include spaces and must be preserved
    p = tmp_path / "in.csv"
    p.write_text("  col_a  , col_b \n  v1  ,  v2  \n", encoding="utf-8")

    df = read_input_table(p, "csv")

    assert list(df.columns) == ["col_a", "col_b"]
    # values preserved exactly (no trimming)
    assert df.loc[0, "col_a"] == "  v1  "
    assert df.loc[0, "col_b"] == "  v2  "


def test_read_tsv_parses_tabs(tmp_path: Path):
    p = tmp_path / "in.tsv"
    p.write_text("a\tb\n1\t2\n", encoding="utf-8")

    df = read_input_table(p, "tsv")

    assert list(df.columns) == ["a", "b"]
    assert df.loc[0, "a"] == "1"
    assert df.loc[0, "b"] == "2"


def test_read_psv_parses_pipes(tmp_path: Path):
    p = tmp_path / "in.psv"
    p.write_text("a|b\n1|2\n", encoding="utf-8")

    df = read_input_table(p, "psv")

    assert list(df.columns) == ["a", "b"]
    assert df.loc[0, "a"] == "1"
    assert df.loc[0, "b"] == "2"


def test_read_csv_case_insensitive_fmt(tmp_path: Path):
    p = tmp_path / "in.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    df = read_input_table(p, "CSV")  # type: ignore[arg-type]
    assert df.loc[0, "a"] == "1"


def test_read_xlsx_default_sheet_sheet1(tmp_path: Path):
    p = tmp_path / "in.xlsx"

    df_in = pd.DataFrame({" a ": ["1"], "b": ["2"]})
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        df_in.to_excel(w, sheet_name="sheet1", index=False)

    df = read_input_table(p, "xlsx")
    assert list(df.columns) == ["a", "b"]
    assert df.loc[0, "a"] == "1"
    assert df.loc[0, "b"] == "2"


def test_read_xlsx_sheet_override(tmp_path: Path):
    p = tmp_path / "in.xlsx"

    df1 = pd.DataFrame({"x": ["1"]})
    df2 = pd.DataFrame({"y": ["2"]})

    with pd.ExcelWriter(p, engine="openpyxl") as w:
        df1.to_excel(w, sheet_name="sheet1", index=False)
        df2.to_excel(w, sheet_name="Sheet2", index=False)  # different sheet

    df = read_input_table(p, "xlsx", sheet_name="Sheet2")
    assert list(df.columns) == ["y"]
    assert df.loc[0, "y"] == "2"


def test_read_xlsx_default_sheet_missing_raises(tmp_path: Path):
    # If default sheet is not "sheet1", read_input_table should fail unless overridden
    p = tmp_path / "in.xlsx"
    df_in = pd.DataFrame({"a": ["1"]})

    with pd.ExcelWriter(p, engine="openpyxl") as w:
        df_in.to_excel(w, sheet_name="Sheet1", index=False)  # NOT "sheet1"

    with pytest.raises(ValueError):
        read_input_table(p, "xlsx")  # default tries "sheet1"
