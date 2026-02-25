"""
Manual runner — Pipeline 0 (REAL execution)

This script:
- Ensures repo root is on PYTHONPATH
- Uses real configs, prompts, and pipeline code
- Runs pipeline_0_schema_ensure.main()
- Does NOT monkeypatch LLM calls
- Does NOT clean up files (inspect outputs freely)

Usage:
    python scripts/run_pipeline_0_force.py
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
from functions.batch.pipeline_0_schema_ensure import main as pipeline_0_main
from functions.utils.logging import get_logger


def main() -> int:
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("RUNNING PIPELINE 0 — REAL EXECUTION")
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Working directory: %s", Path.cwd())
    logger.info("=" * 80)

    # -----------------------------------------------------------------
    # Preconditions (explicit, fail fast)
    # -----------------------------------------------------------------
    required_files = [
        REPO_ROOT / "configs/parameters.yaml",
        REPO_ROOT / "configs/credentials.yaml",
    ]

    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    # Prompt path is resolved inside pipeline 0 (default or forward-compat)
    # but we check existence early for clarity
    default_prompt = REPO_ROOT / "prompts/schema_auto_py_generation.yaml"
    if not default_prompt.exists():
        raise FileNotFoundError(
            "Missing schema auto-generation prompt:\n"
            f"  {default_prompt}\n\n"
            "Pipeline 0 cannot run without this file."
        )

    # -----------------------------------------------------------------
    # Run pipeline 0
    # -----------------------------------------------------------------
    logger.info("Invoking pipeline_0_schema_ensure.main()")
    rc = pipeline_0_main()

    logger.info("Pipeline 0 finished with return code: %s", rc)
    logger.info("=" * 80)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
