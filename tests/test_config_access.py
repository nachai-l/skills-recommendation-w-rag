# tests/test_config_access.py
"""
Tests for the shared config access helpers.
"""
from __future__ import annotations

import pytest

from functions.utils.config_access import cfg_get, cfg_get_path


class _AttrCfg:
    """Simple attribute-based config for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_cfg_get_dict():
    assert cfg_get({"a": 1}, "a") == 1


def test_cfg_get_attr():
    obj = _AttrCfg(b=2)
    assert cfg_get(obj, "b") == 2


def test_cfg_get_dict_missing_raises():
    with pytest.raises(KeyError):
        cfg_get({}, "missing")


def test_cfg_get_attr_missing_raises():
    with pytest.raises(AttributeError):
        cfg_get(_AttrCfg(), "missing")


def test_cfg_get_path_nested_dict():
    cfg = {"a": {"b": {"c": 42}}}
    assert cfg_get_path(cfg, ["a", "b", "c"]) == 42


def test_cfg_get_path_nested_attr():
    inner = _AttrCfg(c=99)
    mid = _AttrCfg(b=inner)
    outer = _AttrCfg(a=mid)
    assert cfg_get_path(outer, ["a", "b", "c"]) == 99


def test_cfg_get_path_mixed_dict_attr():
    inner = _AttrCfg(val=7)
    cfg = {"outer": inner}
    assert cfg_get_path(cfg, ["outer", "val"]) == 7


def test_cfg_get_path_empty_path():
    cfg = {"a": 1}
    assert cfg_get_path(cfg, []) == cfg
