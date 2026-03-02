# functions/batch/pipeline_2_ingest_index.py
"""
Pipeline 2 — Ingest + Index (FAISS + BM25)

Thin orchestrator for the batch indexing step.

Responsibilities:
- Load `configs/parameters.yaml`
- Configure logging and ensure output directories exist
- Read the skills corpus input table and validate required columns
- Delegate index building/persistence to core ingestion

Core logic lives in:
- `functions/core/ingestion.py` (FAISS + BM25 build + artifact persistence)

Outputs:
- `artifacts/index_store/*` (FAISS index, BM25 corpus/stats, manifest)
"""


from __future__ import annotations

from typing import Any, Dict

from functions.core.ingestion import build_and_persist_indexes
from functions.io.readers import read_input_table, validate_required_columns
from functions.utils.config import ensure_dirs, load_parameters
from functions.utils.logging import configure_logging_from_params, get_logger

def main(*, parameters_path: str = "configs/parameters.yaml") -> int:
    """Run Pipeline 2 end-to-end. Returns 0 on success (CLI-friendly)."""

    # Load typed parameters (single source of truth for paths, corpus schema, RAG settings).
    params = load_parameters(parameters_path)

    # Configure logging early so downstream steps (read/validate/build) are observable.
    configure_logging_from_params(params, level=params.run.log_level, log_file=params.run.log_file)
    logger = get_logger(__name__)

    # Ensure artifact/cache directories exist before writing indexes/manifests.
    ensure_dirs(params)

    # 1) Read corpus table (skills taxonomy / corpus)
    df = read_input_table(
        params.input.path,
        params.input.format,
        sheet_name=params.input.sheet,
        encoding=params.input.encoding,
    )

    # Enforce a minimal schema contract for index construction.
    if params.input.required_columns:
        validate_required_columns(df, params.input.required_columns)

    # 2) Build + persist FAISS + BM25 artifacts (core implementation)
    cfg: Dict[str, Any] = params.model_dump()
    result = build_and_persist_indexes(df=df, cfg=cfg)

    logger.info(
        "Pipeline 2 completed: num_docs=%s dim=%s manifest=%s",
        result.num_docs,
        result.dim,
        result.paths.get("manifest_path"),
    )
    return 0


__all__ = ["main"]