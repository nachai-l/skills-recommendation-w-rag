# functions/online/pipeline_5_api_payload.py
from __future__ import annotations

"""
Pipeline 5 — API Payload Builder

Responsibilities:
- Call 4b (must include retrieval_results)
- Optionally call 4c and require PASS
- Build final API response payload:
  each recommended skill includes retrieval meta (skill_text + criteria) + LLM reasoning.
"""

from typing import Any, Dict, Optional

from functions.core.api_payload import build_api_payload_from_4b
from functions.online.pipeline_4b_generate import run_pipeline_4b_generate
from functions.online.pipeline_4c_judge import run_pipeline_4c_judge
from functions.utils.logging import get_logger

logger = get_logger(__name__)


def run_pipeline_5_api_payload(
    *,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    parameters_path: str = "configs/parameters.yaml",
    credentials_path: str = "configs/credentials.yaml",
    schema_model_name_generation: str = "LLMOutput",
    judge_model_name: str = "JudgeResult",
    require_judge_pass: bool = True,
    top_k_vector: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
    require_all_meta: bool = False,
) -> Dict[str, Any]:
    # 1) Run generation (must include retrieval context)
    p4b = run_pipeline_4b_generate(
        query=query,
        top_k=top_k,
        debug=debug,
        parameters_path=parameters_path,
        credentials_path=credentials_path,
        schema_model_name=schema_model_name_generation,
        include_retrieval_results=True,   # required for joining meta
        top_k_vector=top_k_vector,
        top_k_bm25=top_k_bm25,
    )

    # 2) Optional judge gating (reuse p4b to avoid re-running 4b)
    if require_judge_pass:
        p4c = run_pipeline_4c_judge(
            query=query,
            top_k=top_k,
            debug=debug,
            parameters_path=parameters_path,
            credentials_path=credentials_path,
            schema_model_name_generation=schema_model_name_generation,
            judge_model_name=judge_model_name,
            include_retrieval_results=True,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            p4b_payload=p4b,
        )
        verdict = (p4c.get("judge_validated") or {}).get("verdict")
        if verdict != "PASS":
            raise RuntimeError(f"Judge did not PASS (verdict={verdict!r}); refusing to build API payload.")

    llm_validated = p4b["llm_validated"]
    retrieval_results = p4b.get("retrieval_results") or []
    k_out = int(p4b.get("top_k") or (top_k or 0) or 0)

    api_payload, dbg = build_api_payload_from_4b(
        query=query,
        llm_validated=llm_validated,
        retrieval_results=retrieval_results,
        top_k=k_out,
        require_all_meta=require_all_meta,
    )

    out: Dict[str, Any] = {
        "payload": api_payload,
        "meta": {
            "generation_cache_id": p4b.get("cache_id"),
        },
    }

    if debug:
        out["debug"] = {
            "join": dbg.__dict__,
            "num_retrieval_results": len(retrieval_results),
        }

    return out


__all__ = ["run_pipeline_5_api_payload"]