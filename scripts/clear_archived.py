#!/usr/bin/env python3
"""
Clear archived/ directory.

Safe:
- Resolves repo root from this script location
- Refuses to delete outside repo
- Prints summary
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo = _repo_root()
    archived = repo / "archived"

    if not archived.exists():
        print(f"[SKIP] {archived} (not found)")
        return 0

    if not archived.is_dir():
        print(f"[ERROR] {archived} is not a directory")
        return 1

    count = 0
    for child in archived.iterdir():
        if child.is_file():
            child.unlink()
            count += 1
        elif child.is_dir():
            shutil.rmtree(child)
            count += 1

    print(f"[OK] Cleared {count} entries in: {archived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
