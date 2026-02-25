from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.online.pipeline_4a_hybrid_context import run_pipeline_4a_hybrid_context

if __name__ == "__main__":
    query = "data scientist"

    payload = run_pipeline_4a_hybrid_context(
        query=query,
        top_k=10,
        debug=True,
        parameters_path="configs/parameters.yaml",
        include_meta=False,            # keep console output smaller
        include_internal_idx=True,
        top_k_vector=20,               # retrieve deeper than output (optional)
        top_k_bm25=20,
    )

    print(f"query={payload['query']!r} top_k={payload['top_k']} num_results={len(payload['results'])}")
    for i, r in enumerate(payload["results"][:10], start=1):
        print(
            f"{i:02d}. skill_id={r.get('skill_id')} | skill_name={r.get('skill_name')} | "
            f"hybrid={r.get('score_hybrid'):.4f} | vec={r.get('score_vector'):.4f} | bm25={r.get('score_bm25'):.4f}"
        )

    if "debug" in payload:
        print("debug:", payload["debug"])

    # Show a short preview of context
    ctx = payload.get("context", "")
    print("\n--- context preview (first 800 chars) ---")
    print(ctx[:800])