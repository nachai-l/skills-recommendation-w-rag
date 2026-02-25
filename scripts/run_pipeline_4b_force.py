# scripts/run_pipeline_4b_force.py
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


if __name__ == "__main__":
    # Static query for smoke test (edit as needed)
    query = "data scientist"

    payload = run_pipeline_4b_generate(
        query=query,
        top_k=20,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        schema_model_name="LLMOutput",
        include_retrieval_results=False,      # keep console output smaller by default
        top_k_vector=40,
        top_k_bm25=40,
    )

    print(f"query={payload.get('query')!r} top_k={payload.get('top_k')} cache_id={payload.get('cache_id')}")
    llm_validated = payload.get("llm_validated") or {}
    print("validated keys:", list(llm_validated.keys()))
    if "analysis_summary" in llm_validated:
        print("\n--- analysis_summary ---")
        print(str(llm_validated["analysis_summary"])[:800])

    recs = llm_validated.get("recommended_skills") or []
    print(f"\nnum_recommended_skills={len(recs)}")
    for i, r in enumerate(recs[:10], start=1):
        print(
            f"{i:02d}. skill_name={r.get('skill_name')!r} | "
            f"score={r.get('relevance_score')} | "
            f"evidence_n={len(r.get('evidence') or [])}"
        )

    if payload.get("debug"):
        print("\ndebug:", payload["debug"])