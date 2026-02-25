# functions/batch/pipeline_1_schema_txt_ensure.py
"""
Pipeline 1 — Ensure schema/llm_schema.txt exists.

Behavior
- If txt exists and llm_schema.force_regenerate=false: no-op.
- If txt missing: generate from schema/llm_schema.py deterministically.
- If force_regenerate=true: archive old txt and regenerate.

Cleanup
- Clears Pipeline 0 schema-generation cache files (pipeline0_*.txt) on entry.

Preferred output
- JSON Schema (text) derived from Pydantic models via model_json_schema().
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from functions.core.schema_text import build_llm_schema_txt_from_py_file
from functions.utils.config import load_parameters, ensure_dirs
from functions.utils.logging import configure_logging_from_params, get_logger
from functions.utils.paths import repo_root_from_parameters_path, resolve_path

def _utc_ts_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _archive_existing(txt_path: Path, archive_dir: Path) -> Optional[Path]:
    if not txt_path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_ts_compact()
    archived_path = archive_dir / f"{txt_path.stem}_{ts}{txt_path.suffix}"
    shutil.copy2(txt_path, archived_path)
    return archived_path


def _clear_pipeline0_schema_cache(cache_dir: Path, logger) -> int:
    """
    Remove Pipeline 0 schema-generation cache files:
      {cache_dir}/pipeline0_*.txt

    Rationale
    - These files are intermediate caches used during schema generation.
    - Once Pipeline 1 runs, schema/llm_schema.py is already canonical and
      schema/llm_schema.txt is generated deterministically from it.
    """
    if not cache_dir.exists():
        return 0

    n = 0
    for f in cache_dir.glob("pipeline0_*.txt"):
        if not f.is_file():
            continue
        try:
            f.unlink()
            n += 1
        except Exception as e:
            logger.debug("Failed deleting %s: %s", str(f), str(e))
    return n


def main(*, parameters_path: str | Path = "configs/parameters.yaml") -> int:
    params = load_parameters(parameters_path)

    configure_logging_from_params(
        params,
        level=str(getattr(getattr(params, "run", None), "log_level", "INFO")),
        log_file=getattr(getattr(params, "run", None), "log_file", None),
    )
    logger = get_logger(__name__)

    repo_root = repo_root_from_parameters_path(parameters_path)

    py_path = resolve_path(params.llm_schema.py_path, base_dir=repo_root)
    txt_path = resolve_path(params.llm_schema.txt_path, base_dir=repo_root)
    archive_dir = resolve_path(getattr(params.llm_schema, "archive_dir", "archived"), base_dir=repo_root)
    cache_dir = resolve_path(getattr(params.cache, "dir", "artifacts/cache"), base_dir=repo_root)

    # Cleanup Pipeline 0 schema-generation caches (best-effort)
    n_cleaned = _clear_pipeline0_schema_cache(cache_dir, logger)
    if n_cleaned:
        logger.debug("Cleared %d Pipeline 0 schema cache file(s) in %s", n_cleaned, str(cache_dir))

    force_regen = bool(getattr(params.llm_schema, "force_regenerate", False))

    if txt_path.exists() and not force_regen:
        logger.info("Pipeline 1: schema txt exists; no-op. path=%s", str(txt_path))
        return 0

    if not py_path.exists():
        raise RuntimeError(
            f"Pipeline 1 requires schema py to exist, but it is missing: {py_path}. "
            "Run Pipeline 0 first."
        )

    if txt_path.exists() and force_regen:
        archived = _archive_existing(txt_path, archive_dir)
        logger.info("Archived existing schema txt to: %s", str(archived) if archived else "(none)")

    txt = build_llm_schema_txt_from_py_file(py_path)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(txt, encoding="utf-8")

    logger.info("Wrote schema txt to: %s (bytes=%s)", str(txt_path), txt_path.stat().st_size)
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
