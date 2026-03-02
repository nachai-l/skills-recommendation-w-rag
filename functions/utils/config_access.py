# functions/utils/config_access.py
"""
Config access helpers (dict-or-attribute traversal).

Intent
- Provide a tiny compatibility layer so callers can read config values from either:
  - raw dicts (e.g., yaml.safe_load output), or
  - typed config models (e.g., Pydantic BaseModel / dataclasses)

Primary APIs
- cfg_get(cfg, key) -> Any
- cfg_get_path(cfg, path) -> Any

Notes
- This module is intentionally strict:
  - dict access uses `cfg[key]` (raises KeyError if missing)
  - attribute access uses `getattr(cfg, key)` (raises AttributeError if missing)
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
