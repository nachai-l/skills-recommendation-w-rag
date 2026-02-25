# tests/test_hashing.py

from __future__ import annotations

from pathlib import Path

from functions.utils.hashing import sha1_file, sha1_text


def test_sha1_text_is_deterministic():
    a = sha1_text("hello")
    b = sha1_text("hello")
    c = sha1_text("hello!")
    assert a == b
    assert a != c
    assert isinstance(a, str)
    assert len(a) == 40  # sha1 hex


def test_sha1_text_handles_unicode():
    a = sha1_text("สวัสดี")
    b = sha1_text("สวัสดี")
    assert a == b
    assert len(a) == 40


def test_sha1_file_is_deterministic(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("abc\n", encoding="utf-8")

    h1 = sha1_file(p)
    h2 = sha1_file(p)
    assert h1 == h2
    assert len(h1) == 40

    # modify content -> hash must change
    p.write_text("abcd\n", encoding="utf-8")
    h3 = sha1_file(p)
    assert h3 != h1
