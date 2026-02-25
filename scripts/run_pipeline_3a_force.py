# scripts/run_pipeline_3a_force.py
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH so `import functions.*` works
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.online.pipeline_3a_vector_search import run_pipeline_3a_vector_search

if __name__ == "__main__":
    # Static query for smoke test (edit as needed)
    query = "data scientist"

    payload = run_pipeline_3a_vector_search(
        query=query,
        top_k=5,
        debug=True,
        parameters_path="configs/parameters.yaml",
        include_meta=False,          # set True if you want full meta dump
        include_internal_idx=True,
    )

    # Print a compact view (avoid dumping huge meta by default)
    print(f"query={payload['query']!r} top_k={payload['top_k']} num_results={len(payload['results'])}")
    for i, r in enumerate(payload["results"][:10], start=1):
        print(
            f"{i:02d}. skill_id={r.get('skill_id')} | skill_name={r.get('skill_name')} | score_vector={r.get('score_vector'):.4f}"
        )
    if "debug" in payload:
        print("debug:", payload["debug"])