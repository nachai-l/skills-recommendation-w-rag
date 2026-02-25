# functions/utils/verbosity.py
"""
Verbosity Utilities — Thin, Consistent, Pipeline-Friendly Logging Gates

Intent
- Provide a small helper used by pipelines (esp. Pipeline 4) to:
  - clamp verbose to a safe range
  - decide a progress logging cadence
  - emit logs only when verbose >= vmin
  - format compact per-item prefixes

Design
- Keep it dependency-free and testable.
- Avoid "multiple * parameters" issues by using keyword-only + *args (no extra * markers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from functions.utils.logging import get_logger


def clamp_verbose(v: Optional[int]) -> int:
    """
    Clamp verbose into [0, 10]. None => 0.
    """
    if v is None:
        return 0
    try:
        iv = int(v)
    except Exception:
        return 0
    if iv < 0:
        return 0
    if iv > 10:
        return 10
    return iv


def item_log_every_n(verbose: int) -> Optional[int]:
    """
    Decide how often to log item progress based on verbose.
    Convention used by Pipeline 4:
      - verbose <= 0: None (no cadence-based progress logs)
      - verbose 1..10: every N items where N=verbose (clamped)
    """
    v = clamp_verbose(verbose)
    if v <= 0:
        return None
    return v


@dataclass(frozen=True)
class VerbositySpec:
    """
    Formatting helpers (kept small and stable).
    """
    short_len: int = 8

    def short(self, s: str) -> str:
        return str(s)[: self.short_len]

    def item_prefix(self, *, item_no: int, n_total: int, work_id: str, cache_id: str) -> str:
        return f"[{item_no:>3}/{n_total}] wid={self.short(work_id)} cid={self.short(cache_id)}"


class VerbosityLogger:
    """
    Small wrapper around a real logger that gates emission by `verbose >= vmin`.

    level: one of {"debug","info","warning","error"}
    """

    def __init__(self, logger, *, verbose: int) -> None:
        self.logger = logger
        self.verbose = clamp_verbose(verbose)

    def log(self, vmin: int, level: str, msg: str, *args) -> None:
        if self.verbose < int(vmin):
            return

        lv = (level or "info").lower().strip()
        if lv == "debug":
            self.logger.debug(msg, *args)
        elif lv in ("warning", "warn"):
            self.logger.warning(msg, *args)
        elif lv == "error":
            self.logger.error(msg, *args)
        else:
            self.logger.info(msg, *args)


__all__ = [
    "clamp_verbose",
    "item_log_every_n",
    "VerbositySpec",
    "VerbosityLogger",
]
