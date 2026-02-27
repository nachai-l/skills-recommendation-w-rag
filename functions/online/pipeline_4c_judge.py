# functions/online/pipeline_4c_judge.py
"""
Pipeline 4c — judge wrapper (online)

Intent
- Provide a thin ONLINE wrapper for judge validation (pipeline 4c).
- Reuse (or produce) pipeline 4b output, then delegate all judge work to:
    functions/core/llm_judge.py

Responsibilities
- Load parameters.yaml (typed config)
- Resolve repo_root (CWD-independent)
- Obtain generation payload (p4b):
  - reuse caller-provided `p4b_payload` when available (avoids re-running 4b)
  - otherwise run pipeline_4b_generate(...)
- Extract (llm_validated, context) from p4b and call run_pipeline_4c_judge_core(...)
- Attach traceability fields (generation_cache_id) and debug flags

Notes
- Judge usually requires context, so include_retrieval_results defaults to True.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from functions.core.llm_judge import run_pipeline_4c_judge_core
from functions.online.pipeline_4b_generate import run_pipeline_4b_generate
from functions.utils.config import load_parameters
from functions.utils.logging import get_logger
from functions.utils.paths import repo_root_from_parameters_path

logger = get_logger(__name__)


def run_pipeline_4c_judge(
    *,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    parameters_path: str = "configs/parameters.yaml",
    credentials_path: str = "configs/credentials.yaml",
    schema_model_name_generation: str = "LLMOutput",
    judge_model_name: str = "JudgeResult",
    include_retrieval_results: bool = True,  # judge usually needs context
    top_k_vector: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
    # Allow caller to provide an existing 4b payload to avoid re-running 4b
    p4b_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run judge validation (pipeline 4c).

    Optimization:
    - If p4b_payload is provided, reuse it (avoids re-running pipeline 4b).
    - Otherwise, run pipeline 4b to obtain llm_validated + context for judging.
    """
    params = load_parameters(parameters_path)
    repo_root = repo_root_from_parameters_path(parameters_path)

    # 1) Get 4b payload (reuse if provided)
    p4b = p4b_payload
    if p4b is None:
        p4b = run_pipeline_4b_generate(
            query=query,
            top_k=top_k,
            debug=debug,
            parameters_path=parameters_path,
            credentials_path=credentials_path,
            schema_model_name=schema_model_name_generation,
            include_retrieval_results=include_retrieval_results,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
        )

    llm_validated = p4b.get("llm_validated")
    if not isinstance(llm_validated, dict):
        raise ValueError("Pipeline 4c requires p4b['llm_validated'] as a dict")

    context = p4b.get("context") or ""
    if not isinstance(context, str):
        context = str(context)

    # 2) Run judge core
    payload = run_pipeline_4c_judge_core(
        params=params,
        repo_root=repo_root,
        query=query,
        llm_validated=llm_validated,
        context=context,
        parameters_path=parameters_path,
        credentials_path=credentials_path,
        judge_model_name=judge_model_name,
        debug=debug,
    )

    # 3) Traceability
    payload["generation_cache_id"] = p4b.get("cache_id")

    # Optional: indicate whether we reused p4b
    if debug:
        payload.setdefault("debug", {})
        payload["debug"]["used_provided_p4b_payload"] = p4b_payload is not None

    return payload


__all__ = ["run_pipeline_4c_judge"]