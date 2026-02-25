# scripts/run_pipeline_1_force.py
"""
Manual runner — Pipeline 1 (REAL execution)

This script:
- Ensures repo root is on PYTHONPATH
- Uses real configs and pipeline code
- Runs pipeline_1_schema_txt_ensure.main()
- Does NOT monkeypatch anything
- Does NOT clean up files (inspect outputs freely)

Usage:
    python scripts/run_pipeline_1_force.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH so `import functions.*` works
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------
# Imports AFTER path fix
# ---------------------------------------------------------------------
from functions.batch.pipeline_1_schema_txt_ensure import main as pipeline_1_main
from functions.utils.logging import get_logger


def main() -> int:
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("RUNNING PIPELINE 1 — REAL EXECUTION")
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Working directory: %s", Path.cwd())
    logger.info("=" * 80)

    # -----------------------------------------------------------------
    # Preconditions (explicit, fail fast)
    # -----------------------------------------------------------------
    required_files = [
        REPO_ROOT / "configs/parameters.yaml",
        REPO_ROOT / "configs/credentials.yaml",
        REPO_ROOT / "schema/llm_schema.py",  # Pipeline 1 depends on Pipeline 0
    ]

    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file missing for Pipeline 1:\n  {p}\n\n"
                "Did you run Pipeline 0 first?"
            )

    # -----------------------------------------------------------------
    # Run pipeline 1
    # -----------------------------------------------------------------
    logger.info("Invoking pipeline_1_schema_txt_ensure.main()")
    rc = pipeline_1_main()

    logger.info("Pipeline 1 finished with return code: %s", rc)
    logger.info("=" * 80)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
