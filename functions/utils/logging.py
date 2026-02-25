# functions/utils/logging.py
"""
Logging Utilities — Consistent, Structured Logs + Optional Client Silencing

Intent
- Provide consistent logging across all pipelines/modules with a single configuration entrypoint.
- Support correlation via `run_id` (injected into LogRecord).
- Optionally reduce noise from verbose third-party client libraries (e.g., HTTP stacks, google genai SDK)
  using a *belt-and-suspenders* strategy: logger disabling + handler filters + root filter.

What this module guarantees
- **Idempotent root configuration:** `configure_logging()` avoids duplicating handlers across repeated calls.
- **Stable log format:** timestamps + level + logger name + message (and optional run_id in record).
- **Optional log-to-file:** add a FileHandler without breaking stream logging.
- **Robust client log silencing (when enabled):**
  - Disables noisy logger namespaces (prevents emission even if libraries attach handlers/reset levels),
  - Installs handler-level filters on root handlers (backstop),
  - Installs a root-logger filter (extra safety when propagation behaves unexpectedly).

Integration with parameters.yaml
- This module does NOT read YAML directly (kept reusable & testable).
- Pipelines should load `configs/parameters.yaml` and call:
    configure_logging_from_params(params, level="INFO", log_file=None)
  which applies:
    params.llm.silence_client_lv_logs

Primary API
- configure_logging(level="INFO", log_file=None, silence_client_lv_logs=False) -> None
- configure_logging_from_params(params, level="INFO", log_file=None) -> None
- get_logger(name, run_id=None) -> logging.Logger
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Internal state to avoid duplicating handlers / filters
_CONFIGURED = False
_CURRENT_LOG_FILE: Optional[str] = None
_SILENCE_CLIENT_LV_LOGS: Optional[bool] = None  # None = never set explicitly


# Keep this conservative and specific.
# Avoid disabling overly-broad namespaces like "google" which can hide useful warnings.
_NOISY_PREFIXES = [
    # HTTP stacks
    "httpx",
    "httpcore",
    "hpack",
    "h2",
    # Gemini SDK / genai libs (actual namespaces may vary by version)
    "google_genai",
    "google_genai.models",
    # Commonly noisy lower-level Google transport layers (keep narrow)
    "google.api_core",
    "google.auth",
    "googleapiclient",
]


class RunIdFilter(logging.Filter):
    """Inject run_id into log records (if desired)."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class NoisyLibFilter(logging.Filter):
    """
    Drop low-level logs from noisy third-party libraries.

    Behavior:
    - If enabled=True:
        - allow WARNING+ always
        - drop INFO/DEBUG for records whose logger name matches a noisy prefix
    - If enabled=False:
        - allow everything
    """

    def __init__(self, *, enabled: bool, prefixes: list[str], min_level: int) -> None:
        super().__init__()
        self.enabled = enabled
        self.prefixes = prefixes
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.enabled:
            return True

        # Always allow WARNING+ through
        if record.levelno >= self.min_level:
            return True

        # Drop INFO/DEBUG for noisy namespaces
        name = record.name or ""
        for p in self.prefixes:
            if name == p or name.startswith(p + "."):
                return False

        return True


def _is_noisy_logger_name(name: str) -> bool:
    for p in _NOISY_PREFIXES:
        if name == p or name.startswith(p + "."):
            return True
    return False


def _install_noisy_filter_on_root_handlers(*, enabled: bool) -> None:
    """
    Install (or refresh) a NoisyLibFilter on all root handlers.
    Idempotent: removes previous NoisyLibFilter(s) first, then adds one with current 'enabled' state.
    """
    root = logging.getLogger()
    for h in root.handlers:
        for f in list(getattr(h, "filters", [])):
            if isinstance(f, NoisyLibFilter):
                h.removeFilter(f)

        h.addFilter(
            NoisyLibFilter(
                enabled=enabled,
                prefixes=list(_NOISY_PREFIXES),
                min_level=logging.WARNING,
            )
        )


def _install_noisy_filter_on_root_logger(*, enabled: bool) -> None:
    """
    Extra safety: apply NoisyLibFilter at ROOT LOGGER level too.
    This helps if a library log propagates and is handled by something unexpected.
    """
    root = logging.getLogger()
    for f in list(root.filters):
        if isinstance(f, NoisyLibFilter):
            root.removeFilter(f)

    root.addFilter(
        NoisyLibFilter(
            enabled=enabled,
            prefixes=list(_NOISY_PREFIXES),
            min_level=logging.WARNING,
        )
    )


def _apply_client_log_silencing(silence_client_lv_logs: bool) -> None:
    """
    Silence noisy third-party client logs (INFO-level request traces).

    Robust strategy:
    - Disable noisy loggers entirely when enabled (kills logs even if they attach handlers/reset levels).
    - Still install handler/root filters as a backstop.
    - When disabled: re-enable and restore reasonable INFO levels & propagation.
    """
    level = logging.WARNING if silence_client_lv_logs else logging.INFO

    manager_dict = logging.Logger.manager.loggerDict  # type: ignore[attr-defined]

    names_to_touch = set(_NOISY_PREFIXES)
    for name in list(manager_dict.keys()):
        if _is_noisy_logger_name(name):
            names_to_touch.add(name)

    for name in sorted(names_to_touch):
        lg = logging.getLogger(name)

        if silence_client_lv_logs:
            # Hard stop: do not emit anything from these namespaces
            lg.disabled = True
            lg.propagate = False
            # Defensive: remove handlers the lib might have attached
            for h in list(lg.handlers):
                lg.removeHandler(h)
        else:
            # Re-enable
            lg.disabled = False
            lg.setLevel(level)
            lg.propagate = True  # allow normal bubbling

    # Backstops (in case libs create new child loggers later)
    _install_noisy_filter_on_root_handlers(enabled=silence_client_lv_logs)
    _install_noisy_filter_on_root_logger(enabled=silence_client_lv_logs)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    *,
    silence_client_lv_logs: bool = False,
) -> None:
    """
    Configure root logging (idempotent for handlers).

    - Avoids handler duplication across repeated imports / calls.
    - If log_file is provided, adds a FileHandler in addition to StreamHandler.
    - Applies client-level log silencing based on silence_client_lv_logs.
      (This can be re-applied on subsequent calls without duplicating handlers.)

    IMPORTANT:
    - Even if the flag didn't change, a new handler (FileHandler) may have been added,
      so we must re-install handler filters.
    """
    global _CONFIGURED, _CURRENT_LOG_FILE, _SILENCE_CLIENT_LV_LOGS

    root = logging.getLogger()
    root_level = getattr(logging, level.upper(), None)
    if root_level is None:
        raise ValueError(f"Invalid log level: {level}")
    root.setLevel(root_level)

    def _has_stream_handler() -> bool:
        return any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        )

    def _has_file_handler(path: str) -> bool:
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if Path(getattr(h, "baseFilename", "")).resolve() == Path(path).resolve():
                        return True
                except Exception:
                    continue
        return False

    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    if not _has_stream_handler():
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not _has_file_handler(log_file):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)
        _CURRENT_LOG_FILE = log_file

    # Apply/refresh third-party silencing.
    # If the flag changed OR never set OR we added new handlers, re-apply fully.
    if _SILENCE_CLIENT_LV_LOGS is None or _SILENCE_CLIENT_LV_LOGS != silence_client_lv_logs:
        _apply_client_log_silencing(silence_client_lv_logs)
        _SILENCE_CLIENT_LV_LOGS = silence_client_lv_logs
    else:
        # Ensure filters exist on newly added handlers
        _install_noisy_filter_on_root_handlers(enabled=bool(_SILENCE_CLIENT_LV_LOGS))
        _install_noisy_filter_on_root_logger(enabled=bool(_SILENCE_CLIENT_LV_LOGS))

    _CONFIGURED = True


def configure_logging_from_params(
    params: Any,
    *,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Convenience helper to integrate with configs/parameters.yaml without reading YAML here.

    Expected params shape:
      params.llm.silence_client_lv_logs: bool (optional; defaults to False)

    Usage (in pipelines):
      params = load_parameters(...)
      configure_logging_from_params(params, level="INFO")
    """
    silence = False
    try:
        silence = bool(getattr(getattr(params, "llm", None), "silence_client_lv_logs", False))
    except Exception:
        silence = False

    configure_logging(level=level, log_file=log_file, silence_client_lv_logs=silence)


def get_logger(name: str, run_id: Optional[str] = None) -> logging.Logger:
    """
    Get a module logger with consistent configuration.

    Notes:
    - We configure logging lazily with INFO level by default, unless configured already.
    - Pipelines SHOULD call configure_logging_from_params(params, ...) early to apply config.
    - We avoid adding per-logger handlers (handlers live on root).
    - If run_id is provided, attach a filter to this logger (idempotent per run_id).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        # default fallback; real pipelines should override via configure_logging_from_params(...)
        configure_logging(level="INFO", log_file=None, silence_client_lv_logs=False)

    logger = logging.getLogger(name)

    if run_id is not None:
        already = False
        for f in logger.filters:
            if isinstance(f, RunIdFilter) and f.run_id == run_id:
                already = True
                break
        if not already:
            logger.addFilter(RunIdFilter(run_id=run_id))

    return logger


__all__ = [
    "get_logger",
    "configure_logging",
    "configure_logging_from_params",
    "RunIdFilter",
    "NoisyLibFilter",
]
