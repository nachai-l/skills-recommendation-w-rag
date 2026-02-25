import pytest
from functions.io.readers import read_json

def test_read_json_ok(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    obj = read_json(p)
    assert obj["a"] == 1

def test_read_json_invalid_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"a": 1', encoding="utf-8")  # missing }
    with pytest.raises(ValueError):
        read_json(p)
