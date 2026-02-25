# functions/online/pipeline_3a_vector_search.py
from __future__ import annotations

"""
Pipeline 3a — Online Vector Search (FAISS)

Thin orchestration layer.

Responsibilities:
- Load config (parameters.yaml)
- Load cached FAISS store (index + meta)
- Embed query online (Gemini embeddings)
- Call core vector search logic
- Return API-friendly payload

Core logic lives in:
    functions/core/vector_search.py
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import inspect

from functions.core.index_store import get_faiss_store_cached
from functions.core.vector_search import to_api_payload, vector_search_faiss
from functions.utils.config import load_parameters
from functions.utils.logging import get_logger
from functions.utils.paths import repo_root_from_parameters_path, resolve_path

logger = get_logger(__name__)


# -----------------------------
# Config access helpers
# -----------------------------
def _get(cfg: Any, key: str) -> Any:
    """Get value from dict-like or attribute-like config."""
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def _get_path(cfg: Any, path: Sequence[str]) -> Any:
    """Traverse nested config by keys/attrs."""
    cur = cfg
    for k in path:
        cur = _get(cur, k)
    return cur


# -----------------------------
# Embedding adapter (MVP)
# -----------------------------

def _build_embed_fn_from_repo(params: Any) -> Callable[[str], np.ndarray]:
    """
    Build an embed_fn(query:str)->np.ndarray adapter using your existing embeddings utilities.

    Robustness:
    - Supports both:
        1) functions.utils.text_embeddings.embed_query(...)
        2) functions.utils.text_embeddings.GoogleEmbeddingModel(...).embed_query(...)
    - Does NOT assume ctor or method supports task_type; it inspects signatures.
    """
    embeddings_cfg = _get(params, "embeddings")
    model_name = _get(embeddings_cfg, "model_name")
    task_type = (
        _get(embeddings_cfg, "task_type")
        if (isinstance(embeddings_cfg, dict) and "task_type" in embeddings_cfg) or hasattr(embeddings_cfg, "task_type")
        else None
    )

    # -------------------------
    # Option 1: functional helper embed_query(...)
    # -------------------------
    try:
        from functions.utils.text_embeddings import embed_query  # type: ignore

        sig = inspect.signature(embed_query)
        accepts_model_name = "model_name" in sig.parameters
        accepts_task_type = "task_type" in sig.parameters

        def _fn(q: str) -> np.ndarray:
            kwargs = {}
            if accepts_model_name:
                kwargs["model_name"] = model_name
            if accepts_task_type and task_type is not None:
                kwargs["task_type"] = task_type
            return embed_query(q, **kwargs)

        return _fn
    except Exception:
        pass

    # -------------------------
    # Option 2: class-based GoogleEmbeddingModel(...).embed_query(...)
    # -------------------------
    try:
        from functions.utils.text_embeddings import GoogleEmbeddingModel  # type: ignore

        # ctor: try model_name kw, then positional
        try:
            model = GoogleEmbeddingModel(model_name=model_name)
        except TypeError:
            model = GoogleEmbeddingModel(model_name)

        embed_method = getattr(model, "embed_query", None)
        if embed_method is None or not callable(embed_method):
            raise AttributeError("GoogleEmbeddingModel has no callable embed_query(...)")

        sig = inspect.signature(embed_method)
        accepts_task_type = "task_type" in sig.parameters

        def _fn(q: str) -> np.ndarray:
            if accepts_task_type and task_type is not None:
                return embed_method(q, task_type=task_type)
            return embed_method(q)

        return _fn

    except Exception as e:
        raise ImportError(
            "Could not construct embed_fn from functions.utils.text_embeddings.\n"
            "Supported patterns:\n"
            "  - embed_query(text, model_name=?, task_type=?)\n"
            "  - GoogleEmbeddingModel(model_name=?).embed_query(text, task_type=?)\n"
            f"Original error: {e}"
        ) from e


# -----------------------------
# Public entrypoint
# -----------------------------
def run_pipeline_3a_vector_search(
    *,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    parameters_path: str = "configs/parameters.yaml",
    include_meta: bool = True,
    include_internal_idx: bool = True,
) -> Dict[str, Any]:
    """
    Run online vector search against FAISS.

    Args:
      query: user query string
      top_k: override; if None use rag.vector_search.top_k_default
      debug: include debug fields in output
      parameters_path: path to parameters.yaml
      include_meta: include meta dict for each hit
      include_internal_idx: include FAISS internal idx for each hit

    Returns:
      JSON-serializable dict payload.
    """
    params = load_parameters(parameters_path)

    # Load config values (supports both dict and ParametersConfig object)
    index_path_raw = _get_path(params, ["index_store", "faiss", "index_path"])
    meta_path_raw = _get_path(params, ["index_store", "faiss", "meta_path"])
    top_k_default = int(_get_path(params, ["rag", "vector_search", "top_k_default"]))
    expected_dim = int(_get_path(params, ["embeddings", "dim"]))
    normalize = bool(_get_path(params, ["embeddings", "normalize"]))

    repo_root = repo_root_from_parameters_path(parameters_path)

    index_path = resolve_path(str(index_path_raw), base_dir=repo_root)
    meta_path = resolve_path(str(meta_path_raw), base_dir=repo_root)

    k = int(top_k) if top_k is not None else top_k_default

    # load cached store
    store = get_faiss_store_cached(index_path=str(index_path), meta_path=str(meta_path))

    # embedding fn
    embed_fn = _build_embed_fn_from_repo(params)

    # meta keys from config
    meta_skill_id_key = str(_get_path(params, ["rag", "corpus", "id_col"]))
    meta_skill_name_key = str(_get_path(params, ["rag", "corpus", "title_col"]))

    if debug:
        logger.info(
            "P3a vector search start",
            extra={
                "query_len": len((query or "").strip()),
                "top_k": k,
                "index_path": str(index_path),
                "meta_path": str(meta_path),
                "expected_dim": expected_dim,
                "store_dim": int(getattr(store, "dim", 0) or 0),
            },
        )

    result = vector_search_faiss(
        query=query,
        top_k=k,
        embed_fn=embed_fn,
        store=store,
        dim=expected_dim,
        normalize=normalize,
        debug=debug,
        meta_skill_id_key=meta_skill_id_key,
        meta_skill_name_key=meta_skill_name_key,
    )

    payload = to_api_payload(
        result,
        include_meta=include_meta,
        include_internal_idx=include_internal_idx,
    )

    if debug:
        logger.info(
            "P3a vector search done",
            extra={
                "top_k": k,
                "num_results": len(payload.get("results", [])),
                "score_vector_max": payload.get("debug", {}).get("score_vector_max"),
            },
        )

    return payload


__all__ = ["run_pipeline_3a_vector_search"]