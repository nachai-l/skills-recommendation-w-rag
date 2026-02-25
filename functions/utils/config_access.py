# functions/utils/config_access.py
"""
Shared config access helpers.

Provides dict-or-attribute traversal for config objects (supports both
raw dicts from YAML and Pydantic config models).
"""
from __future__ import annotations

from typing import Any, Sequence


def cfg_get(cfg: Any, key: str) -> Any:
    """Get a value from a dict-like or attribute-like config object."""
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def cfg_get_path(cfg: Any, path: Sequence[str]) -> Any:
    """Traverse nested config by a sequence of keys/attrs."""
    cur = cfg
    for k in path:
        cur = cfg_get(cur, k)
    return cur


__all__ = ["cfg_get", "cfg_get_path"]
