import json
from pathlib import Path

import pandas as pd

from functions.io.writers import ensure_parent_dir, write_csv, write_jsonl, write_delimited


def test_ensure_parent_dir_creates(tmp_path: Path):
    out = tmp_path / "a" / "b" / "file.jsonl"
    assert not out.parent.exists()
    ensure_parent_dir(out)
    assert out.parent.exists()


def test_write_jsonl_writes_deterministically(tmp_path: Path):
    out = tmp_path / "x" / "out.jsonl"
    records = [
        {"b": 2, "a": 1},
        {"z": "ไทย", "c": 3},
    ]
    write_jsonl(out, records)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    # sort_keys=True => "a" before "b"
    assert lines[0] == json.dumps({"b": 2, "a": 1}, ensure_ascii=False, sort_keys=True)

    # should preserve unicode
    obj = json.loads(lines[1])
    assert obj["z"] == "ไทย"


def test_write_csv_writes_and_creates_parent(tmp_path: Path):
    out = tmp_path / "reports" / "t.csv"
    df = pd.DataFrame({"b": [2, 3], "a": [1, 4]})

    write_csv(out, df)

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "b,a"  # preserves df.columns order
    assert text[1] == "2,1"
    assert text[2] == "3,4"


def test_write_delimited_does_not_doublequote_json_strings(tmp_path):
    """
    Ensure JSON-like strings remain readable and are not turned into ""..."" by pandas.

    We do NOT assert an exact backslash count because pandas/csv may escape backslashes
    differently depending on quoting/escape settings. Instead, we check:
    - no doubled quotes ("")
    - JSON-ish structure is still present
    - the embedded token ("ok") survives the roundtrip text write
    """
    df = pd.DataFrame(
        [
            {
                "judge_reasons_json": '["A", "B", "He said \\"ok\\""]',
                "plain": "x",
            }
        ]
    )

    out_path = tmp_path / "out.psv"
    write_delimited(out_path, df, sep="|")

    text = out_path.read_text(encoding="utf-8")
    lines = text.strip("\n").split("\n")
    assert len(lines) == 2  # header + 1 data line

    row = lines[1]

    # should NOT contain doubled quotes patterns like ""A""
    assert '""' not in row

    # still looks JSON-ish and still contains the payload token
    assert "[" in row and "]" in row
    assert "He said" in row
    assert "ok" in row
