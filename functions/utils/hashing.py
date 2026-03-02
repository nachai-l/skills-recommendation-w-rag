# functions/utils/hashing.py
"""
Hashing utilities (deterministic fingerprints)

Intent
- Provide small, reusable hashing helpers for stable cache keys / fingerprints across pipelines.
- Keep hashing logic centralized to avoid drift and inconsistent encodings.

Notes
- Uses SHA1 for short, stable digests (40 hex chars).
- Intended for fingerprinting and cache keys, not cryptographic security.
"""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def sha1_text(s: str) -> str:
    """
    SHA1 hex digest of a text string (UTF-8, replacement on errors).
    """
    return sha1(s.encode("utf-8", errors="replace")).hexdigest()


def sha1_file(path: PathLike) -> str:
    """
    SHA1 hex digest of a file's raw bytes.
    """
    p = Path(path)
    b = p.read_bytes()
    return sha1(b).hexdigest()


__all__ = ["sha1_text", "sha1_file"]
