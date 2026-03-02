# functions/utils/paths.py
"""
Path helpers (repo-root aware)

Intent
- Make pipeline behavior independent of the current working directory (CWD).
- Provide a consistent way to resolve repo-relative paths starting from
  `configs/parameters.yaml` (treated as an anchor to infer repo root).

Used by:
- Pipelines that need to resolve prompt/schema/artifact paths deterministically.
"""

from __future__ import annotations

from pathlib import Path


def repo_root_from_parameters_path(parameters_path: str | Path) -> Path:
    """
    Given a path to `configs/parameters.yaml`, return the repo root.

    Works for absolute or relative paths by resolving first.
    Example:
      .../repo/configs/parameters.yaml -> .../repo
    """
    p = Path(parameters_path).resolve()
    # .../repo/configs/parameters.yaml -> .../repo
    return p.parents[1]


def resolve_path(path_like: str | Path, *, base_dir: str | Path) -> Path:
    """
    Resolve a path relative to base_dir unless it is already absolute.

    Returns a resolved Path (absolute).
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (Path(base_dir) / p).resolve()


__all__ = ["repo_root_from_parameters_path", "resolve_path"]
