# functions/online/pipeline_5_api_payload.py
"""
Pipeline 5 — API payload builder (final response shaping)

Intent
- Produce the final response returned by `/v1/recommend-skills`.

Responsibilities
- Run Pipeline 4b generation (must include retrieval_results + context for meta join)
- Optionally run Pipeline 4c judge and enforce PASS (gate output)
- Join LLM recommendations with retrieval metadata to enrich each skill with:
  - canonical skill_id/source
  - skill_text + criteria fields
  - LLM reasoning + evidence
- Return an API-friendly envelope: {"payload": ..., "meta": ..., "debug": ...?}

Core join logic lives in:
- functions/core/api_payload.py (build_api_payload_from_4b)

Responsibilities:
- Call 4b (must include retrieval_results)
- Optionally call 4c and require PASS, retrying up to llm.max_retries times on FAIL
  (each retry forces fresh P4b generation to avoid returning the same bad output)
- Build final API response payload:
  each recommended skill includes retrieval meta (skill_text + criteria) + LLM reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from functions.core.api_payload import build_api_payload_from_4b
from functions.online.pipeline_4b_generate import run_pipeline_4b_generate
from functions.online.pipeline_4c_judge import run_pipeline_4c_judge
from functions.utils.config import load_parameters
from functions.utils.config_access import cfg_get_path as _get_path
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
    """
    Build the final API response payload for a skill recommendation request.

    Args:
      query: user query string
      top_k: output size (None -> rag.vector_search.top_k_default)
      debug: include debug blocks
      parameters_path/credentials_path: config file paths
      schema_model_name_generation: Pydantic export for 4b generation validation
      judge_model_name: Pydantic export for 4c judge validation
      require_judge_pass: if True, refuse to return payload unless judge verdict == PASS
      top_k_vector/top_k_bm25: retrieval depths (optional)
      require_all_meta: strict join policy for meta enrichment

    Returns:
      dict with:
        - payload: final API payload (query, analysis_summary, recommended_skills[])
        - meta: generation_cache_id (+ future metadata)
        - debug: optional debug info (join stats, retrieval counts)
    """
    params = load_parameters(parameters_path)
    max_retries = int(_get_path(params, ["llm", "max_retries"]))

    # 1) Initial generation (must include retrieval context for payload join)
    p4b = run_pipeline_4b_generate(
        query=query,
        top_k=top_k,
        debug=debug,
        parameters_path=parameters_path,
        credentials_path=credentials_path,
        schema_model_name=schema_model_name_generation,
        include_retrieval_results=True,
        top_k_vector=top_k_vector,
        top_k_bm25=top_k_bm25,
    )

    # 2) Optional judge gating with retries
    judge_attempts = 0
    if require_judge_pass:
        last_verdict = None
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                logger.warning(
                    "Judge FAIL (attempt %d/%d) — regenerating P4b with fresh LLM call",
                    attempt - 1,
                    max_retries,
                )
                p4b = run_pipeline_4b_generate(
                    query=query,
                    top_k=top_k,
                    debug=debug,
                    parameters_path=parameters_path,
                    credentials_path=credentials_path,
                    schema_model_name=schema_model_name_generation,
                    include_retrieval_results=True,
                    top_k_vector=top_k_vector,
                    top_k_bm25=top_k_bm25,
                    force_regenerate=True,
                )

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
            judge_attempts = attempt
            last_verdict = (p4c.get("judge_validated") or {}).get("verdict")

            if last_verdict == "PASS":
                break

            logger.warning(
                "Judge FAIL (attempt %d/%d): verdict=%r",
                attempt,
                max_retries,
                last_verdict,
            )

        if last_verdict != "PASS":
            raise RuntimeError(
                f"Judge did not PASS after {max_retries} attempt(s): verdict={last_verdict!r}"
            )

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
            "judge_attempts": judge_attempts,
        }

    return out


__all__ = ["run_pipeline_5_api_payload"]
