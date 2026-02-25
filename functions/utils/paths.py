# functions/utils/paths.py
"""
Path helpers

Intent
- Make pipeline behavior independent of CWD.
- Standardize: configs/parameters.yaml => repo root.

This is used heavily in pipelines to resolve relative prompt/schema paths.
"""
from __future__ import annotations

from pathlib import Path


def repo_root_from_parameters_path(parameters_path: str | Path) -> Path:
    """
    Given configs/parameters.yaml, return repo root.
    Works for absolute or relative paths.
    """
    p = Path(parameters_path).resolve()
    # .../repo/configs/parameters.yaml -> .../repo
    return p.parents[1]


def resolve_path(path_like: str | Path, *, base_dir: str | Path) -> Path:
    """
    Resolve a path relative to base_dir unless already absolute.
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (Path(base_dir) / p).resolve()


__all__ = ["repo_root_from_parameters_path", "resolve_path"]
