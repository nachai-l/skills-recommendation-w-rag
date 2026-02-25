# functions/online/pipeline_4b_generate.py
from __future__ import annotations

"""
Pipeline 4b — Online (Thin) Wrapper

Responsibilities:
- Load parameters.yaml
- Resolve repo_root (CWD-independent)
- Delegate all real work to core: functions/core/llm_generate.py
"""

from typing import Any, Dict, Optional

from functions.core.llm_generate import run_pipeline_4b_generate_core
from functions.utils.config import load_parameters
from functions.utils.logging import get_logger
from functions.utils.paths import repo_root_from_parameters_path

logger = get_logger(__name__)


def run_pipeline_4b_generate(
    *,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    parameters_path: str = "configs/parameters.yaml",
    credentials_path: str = "configs/credentials.yaml",
    schema_model_name: str = "LLMOutput",
    include_retrieval_results: bool = False,
    top_k_vector: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper around core 4b generation.

    Args:
      query: user query string
      top_k: output top_k (None -> rag.vector_search.top_k_default)
      debug: include debug fields
      parameters_path: path to parameters.yaml
      credentials_path: path to credentials.yaml
      schema_model_name: exported Pydantic model name in schema/llm_schema.py (default: Output)
      include_retrieval_results: include retrieval_results + context in output payload
      top_k_vector/top_k_bm25: retrieval depths for 4a (optional)

    Returns:
      JSON-serializable dict payload.
    """
    params = load_parameters(parameters_path)
    repo_root = repo_root_from_parameters_path(parameters_path)

    if debug:
        logger.info(
            "P4b online wrapper start",
            extra={
                "query_len": len((query or "").strip()),
                "top_k": top_k,
                "parameters_path": parameters_path,
                "credentials_path": credentials_path,
                "schema_model_name": schema_model_name,
            },
        )

    payload = run_pipeline_4b_generate_core(
        params=params,
        repo_root=repo_root,
        query=query,
        top_k=top_k,
        debug=debug,
        parameters_path=parameters_path,
        credentials_path=credentials_path,
        schema_model_name=schema_model_name,
        include_retrieval_results=include_retrieval_results,
        top_k_vector=top_k_vector,
        top_k_bm25=top_k_bm25,
    )

    if debug:
        logger.info(
            "P4b online wrapper done",
            extra={
                "top_k": payload.get("top_k"),
                "cache_id": payload.get("cache_id"),
            },
        )

    return payload


__all__ = ["run_pipeline_4b_generate"]