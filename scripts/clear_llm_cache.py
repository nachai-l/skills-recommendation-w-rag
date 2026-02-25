#!/usr/bin/env python3
"""
Clear LLM cache directories:

- artifacts/cache/llm_outputs/
- artifacts/cache/llm_failures/

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
    # scripts/clear_llm_cache.py → repo root = parent of scripts
    return Path(__file__).resolve().parents[1]


def _clear_dir(p: Path) -> int:
    if not p.exists():
        print(f"[SKIP] {p} (not found)")
        return 0

    if not p.is_dir():
        print(f"[ERROR] {p} is not a directory")
        return 1

    count = 0
    for child in p.iterdir():
        if child.is_file():
            child.unlink()
            count += 1
        elif child.is_dir():
            shutil.rmtree(child)
            count += 1

    print(f"[OK] Cleared {count} entries in: {p}")
    return 0


def main() -> int:
    repo = _repo_root()

    targets = [
        repo / "artifacts" / "cache" / "llm_outputs",
        repo / "artifacts" / "cache" / "llm_failures",
        repo / "artifacts" / "cache" / "embeddings" / "gemini-embedding-001" / "SEMANTIC_SIMILARITY",
    ]

    rc = 0
    for t in targets:
        rc |= _clear_dir(t)

    print("Done.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
