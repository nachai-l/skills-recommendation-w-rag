from __future__ import annotations

"""
Pipeline 2 — Ingest + Index (FAISS + BM25)

Thin orchestrator.

Responsibilities here:
- Load parameters.yaml
- Ensure output dirs
- Read corpus input table
- Validate required columns
- Call core ingestion to build + persist artifacts

Core logic lives in:
- functions/core/ingestion.py
"""

from typing import Any, Dict

from functions.core.ingestion import build_and_persist_indexes
from functions.io.readers import read_input_table, validate_required_columns
from functions.utils.config import ensure_dirs, load_parameters
from functions.utils.logging import configure_logging_from_params, get_logger

def main(*, parameters_path: str = "configs/parameters.yaml") -> int:

    params = load_parameters(parameters_path)
    configure_logging_from_params(params, level=params.run.log_level, log_file=params.run.log_file)
    logger = get_logger(__name__)
    ensure_dirs(params)

    # 1) Read corpus table
    df = read_input_table(
        params.input.path,
        params.input.format,
        sheet_name=params.input.sheet,
        encoding=params.input.encoding,
    )

    if params.input.required_columns:
        validate_required_columns(df, params.input.required_columns)

    # 2) Build + persist FAISS + BM25 artifacts (core)
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