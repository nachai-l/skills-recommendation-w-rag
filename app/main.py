# app/main.py
"""
FastAPI service entrypoint for Skills Recommendation API.

Exposes a small HTTP surface over the deterministic skill recommendation pipeline:
- `GET /healthz` for liveness checks
- `POST /v1/recommend-skills` to run `run_pipeline_5_api_payload` and return a typed response

Configuration:
- `PARAMETERS_PATH` and `CREDENTIALS_PATH` can be provided via environment variables.
  Defaults point to local repo paths for dev (`configs/parameters.yaml`, `configs/credentials.yaml`).

Error handling:
- All unexpected exceptions are logged server-side (with stack trace) and returned as a
  generic 500 to avoid leaking internal details to clients.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from functions.online.pipeline_5_api_payload import run_pipeline_5_api_payload

APP_NAME = "skills_recommendation_api"
APP_VERSION = "0.1.0"

logger = logging.getLogger(__name__)

# Default config paths (override via env in prod / Cloud Run)
DEFAULT_PARAMETERS_PATH = os.environ.get("PARAMETERS_PATH") or "configs/parameters.yaml"
DEFAULT_CREDENTIALS_PATH = os.environ.get("CREDENTIALS_PATH") or "configs/credentials.yaml"


class RecommendSkillsRequest(BaseModel):
    """Request schema for `/v1/recommend-skills`."""
    query: str = Field(..., min_length=1, description="User query text, e.g. 'data scientist'")
    top_k: Optional[int] = Field(20, ge=1, le=100, description="Number of merged/context rows for LLM + max output join")
    debug: bool = Field(False, description="Include debug fields in response")
    require_judge_pass: bool = Field(True, description="If true, run judge and require PASS")
    top_k_vector: Optional[int] = Field(20, ge=1, le=200, description="Vector retrieval depth (3a) for 4a/4b/5")
    top_k_bm25: Optional[int] = Field(20, ge=1, le=200, description="BM25 retrieval depth (3b) for 4a/4b/5")
    require_all_meta: bool = Field(
        False,
        description="If true, fail when any recommended skill cannot be joined to retrieval meta.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace-only")
        return stripped


class RecommendSkillsResponse(BaseModel):
    """Response schema for `/v1/recommend-skills`."""
    payload: Dict[str, Any]
    meta: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application (used by ASGI server)."""
    app = FastAPI(title=APP_NAME, version=APP_VERSION)

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/recommend-skills", response_model=RecommendSkillsResponse)
    def recommend_skills(req: RecommendSkillsRequest) -> Dict[str, Any]:
        """
        Run the skill recommendation pipeline and return its output.

        Notes:
        - This is a thin HTTP wrapper; core logic lives in `run_pipeline_5_api_payload`.
        - Exceptions are logged with stack traces but a client-safe 500 is returned.
        """
        try:
            out = run_pipeline_5_api_payload(
                query=req.query,
                top_k=req.top_k,
                debug=req.debug,
                parameters_path=DEFAULT_PARAMETERS_PATH,
                credentials_path=DEFAULT_CREDENTIALS_PATH,
                require_judge_pass=req.require_judge_pass,
                top_k_vector=req.top_k_vector,
                top_k_bm25=req.top_k_bm25,
                require_all_meta=req.require_all_meta,
            )
            return out
        except Exception as e:
            # Log full details internally; keep the HTTP response contract client-safe.
            logger.error("Pipeline error in /v1/recommend-skills", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return app

# ASGI entrypoint (e.g., `uvicorn app.main:app`)
app = create_app()