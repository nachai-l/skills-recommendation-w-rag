# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from functions.online.pipeline_5_api_payload import run_pipeline_5_api_payload

APP_NAME = "skills_recommendation_api"
APP_VERSION = "0.1.0"

# Default config paths (override via env in prod)
DEFAULT_PARAMETERS_PATH = os.environ.get("PARAMETERS_PATH", "configs/parameters.yaml")
DEFAULT_CREDENTIALS_PATH = os.environ.get("CREDENTIALS_PATH", "configs/credentials.yaml")


class RecommendSkillsRequest(BaseModel):
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


class RecommendSkillsResponse(BaseModel):
    payload: Dict[str, Any]
    meta: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=APP_VERSION)

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/recommend-skills", response_model=RecommendSkillsResponse)
    def recommend_skills(req: RecommendSkillsRequest) -> Dict[str, Any]:
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
            # Keep error surface predictable for API clients
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()