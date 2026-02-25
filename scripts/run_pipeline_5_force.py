from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH so `import functions.*` works
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.online.pipeline_5_api_payload import run_pipeline_5_api_payload


def _clear_dir(path: Path) -> None:
    """
    Delete all contents under `path` (files + folders), keep the directory itself.
    """
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    # 0) Clear artifacts/outputs before run
    outputs_dir = REPO_ROOT / "artifacts" / "outputs"
    _clear_dir(outputs_dir)

    # Static query for smoke test (edit as needed)
    query = "data scientist"

    out = run_pipeline_5_api_payload(
        query=query,
        top_k=20,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        require_judge_pass=True,
        top_k_vector=20,
        top_k_bm25=20,
        require_all_meta=False,
    )

    payload = out["payload"]

    # 1) Save JSON payload to artifacts/outputs/
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outputs_dir / f"pipeline5_api_payload__{ts}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved payload to: {out_path}")

    # 2) Print a compact summary
    print("query:", payload["query"])
    print("analysis_summary:", str(payload.get("analysis_summary", ""))[:200])
    print("num skills:", len(payload.get("recommended_skills", [])))

    for i, r in enumerate(payload["recommended_skills"][:5], start=1):
        print(
            f"{i:02d}. {r.get('skill_name')} | id={r.get('skill_id')} | score={r.get('relevance_score')} | "
            f"has_criteria={bool(r.get('Foundational_Criteria'))}"
        )

    if out.get("debug"):
        print("debug:", out["debug"])