# tests/test_verbosity.py
from __future__ import annotations

import logging

import pytest

from functions.utils.verbosity import (
    VerbosityLogger,
    VerbositySpec,
    clamp_verbose,
    item_log_every_n,
)


def test_clamp_verbose():
    assert clamp_verbose(None) == 0
    assert clamp_verbose(-10) == 0
    assert clamp_verbose(0) == 0
    assert clamp_verbose(3) == 3
    assert clamp_verbose(10) == 10
    assert clamp_verbose(999) == 10


def test_item_log_every_n():
    # verbose<=0 => never
    assert item_log_every_n(verbose=0) is None
    assert item_log_every_n(verbose=-1) is None

    # verbose 1..10 => use itself
    assert item_log_every_n(verbose=1) == 1
    assert item_log_every_n(verbose=3) == 3
    assert item_log_every_n(verbose=10) == 10

    # clamp >10
    assert item_log_every_n(verbose=99) == 10


def test_verbosity_logger_emits_by_threshold(caplog):
    logger = logging.getLogger("t_verbose")
    v = VerbosityLogger(logger, verbose=3)

    # vmin=4 should not emit
    with caplog.at_level(logging.INFO):
        v.log(4, "info", "nope %s", "x")
    assert "nope x" not in caplog.text

    # vmin=3 should emit
    with caplog.at_level(logging.INFO):
        v.log(3, "info", "ok %s", "y")
    assert "ok y" in caplog.text


def test_verbosity_logger_level_mapping(caplog):
    logger = logging.getLogger("t_levels")
    v = VerbosityLogger(logger, verbose=10)

    with caplog.at_level(logging.DEBUG):
        v.log(1, "warning", "w %s", "1")
        v.log(1, "error", "e %s", "2")
        v.log(1, "debug", "d %s", "3")
        v.log(1, "info", "i %s", "4")

    assert "w 1" in caplog.text
    assert "e 2" in caplog.text
    assert "d 3" in caplog.text
    assert "i 4" in caplog.text


def test_verbosity_spec_prefix_short_ids():
    s = VerbositySpec()
    p = s.item_prefix(item_no=7, n_total=50, work_id="abcdef012345", cache_id="1234567890abcdef")
    assert "[  7/50]" in p
    assert "wid=abcdef01" in p
    assert "cid=12345678" in p
