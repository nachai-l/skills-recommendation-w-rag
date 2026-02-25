from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH so `import functions.*` works
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.online.pipeline_4b_generate import run_pipeline_4b_generate
from functions.online.pipeline_4c_judge import run_pipeline_4c_judge


if __name__ == "__main__":
    query = "data scientist"

    # 1) Run 4b once (include context for judge)
    p4b = run_pipeline_4b_generate(
        query=query,
        top_k=20,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        schema_model_name="LLMOutput",
        include_retrieval_results=True,   # judge needs context/output_json grounding
        top_k_vector=20,
        top_k_bm25=20,
    )

    # 2) Run 4c judge reusing 4b payload (avoids re-running 4b)
    payload = run_pipeline_4c_judge(
        query=query,
        top_k=20,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        schema_model_name_generation="LLMOutput",
        judge_model_name="JudgeResult",
        include_retrieval_results=True,
        top_k_vector=20,
        top_k_bm25=20,
        p4b_payload=p4b,
    )

    print(
        f"query={payload.get('query')!r} "
        f"gen_cache_id={payload.get('generation_cache_id')} "
        f"judge_cache_id={payload.get('cache_id')}"
    )

    j = payload.get("judge_validated") or {}
    print("judge keys:", list(j.keys()))
    print("judge preview:", str(j)[:800])

    if payload.get("debug"):
        print("debug:", payload["debug"])