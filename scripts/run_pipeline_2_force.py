# scripts/run_pipeline_2_force.py
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH so `import functions.*` works
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.batch.pipeline_2_ingest_index import main

if __name__ == "__main__":
    raise SystemExit(main(parameters_path="configs/parameters.yaml"))