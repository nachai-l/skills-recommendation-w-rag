# tests/test_logging.py
import logging
from pathlib import Path

import pytest

import functions.utils.logging as log_mod


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch):
    """
    Keep tests isolated while being compatible with pytest's own log capture handler.

    We do NOT try to remove pytest's internal handlers aggressively, because pytest
    may re-attach them. Instead we:
    - reset module globals
    - keep root handlers as-is, and test idempotency by comparing counts
    """
    monkeypatch.setattr(log_mod, "_CONFIGURED", False, raising=True)
    monkeypatch.setattr(log_mod, "_CURRENT_LOG_FILE", None, raising=True)
    monkeypatch.setattr(log_mod, "_SILENCE_CLIENT_LV_LOGS", None, raising=True)

    # Ensure noisy loggers are not left disabled across tests
    for name in list(logging.Logger.manager.loggerDict.keys()):  # type: ignore[attr-defined]
        lg = logging.getLogger(name)
        lg.disabled = False
        lg.propagate = True

    yield


def _count_file_handlers(root: logging.Logger) -> int:
    return sum(1 for h in root.handlers if isinstance(h, logging.FileHandler))


def test_configure_logging_idempotent_does_not_duplicate_handlers():
    root = logging.getLogger()

    before = len(root.handlers)
    log_mod.configure_logging(level="INFO", log_file=None, silence_client_lv_logs=False)
    after_first = len(root.handlers)

    # Must add at most 1 handler on first call (or add none if pytest already installed handlers)
    assert after_first >= before

    # Second call must not add more handlers
    log_mod.configure_logging(level="INFO", log_file=None, silence_client_lv_logs=False)
    after_second = len(root.handlers)
    assert after_second == after_first


def test_configure_logging_adds_file_handler_once(tmp_path: Path):
    root = logging.getLogger()
    log_file = tmp_path / "logs" / "app.log"

    before_total = len(root.handlers)
    before_files = _count_file_handlers(root)

    log_mod.configure_logging(level="INFO", log_file=str(log_file), silence_client_lv_logs=False)

    after_total = len(root.handlers)
    after_files = _count_file_handlers(root)

    # exactly one new FileHandler should appear
    assert after_files == before_files + 1
    assert after_total >= before_total
    assert log_file.parent.exists()

    # calling again with same file must not add another FileHandler
    log_mod.configure_logging(level="INFO", log_file=str(log_file), silence_client_lv_logs=False)
    assert _count_file_handlers(root) == after_files
    assert len(root.handlers) == after_total


def test_get_logger_lazy_configures():
    assert log_mod._CONFIGURED is False

    root = logging.getLogger()
    before = len(root.handlers)

    lg = log_mod.get_logger("x.y.z")
    assert isinstance(lg, logging.Logger)
    assert log_mod._CONFIGURED is True

    after = len(root.handlers)
    # get_logger should not spam handlers (may add one if none existed)
    assert after >= before

    # calling get_logger again should not add more handlers
    lg2 = log_mod.get_logger("x.y.z")
    assert lg2 is lg
    assert len(root.handlers) == after


def test_get_logger_run_id_filter_attached_idempotent():
    log_mod.configure_logging(level="INFO", log_file=None, silence_client_lv_logs=False)

    lg = log_mod.get_logger("mod", run_id="r1")
    assert any(isinstance(f, log_mod.RunIdFilter) and f.run_id == "r1" for f in lg.filters)

    # same run_id should not duplicate filter
    lg2 = log_mod.get_logger("mod", run_id="r1")
    assert lg2 is lg
    assert sum(isinstance(f, log_mod.RunIdFilter) and f.run_id == "r1" for f in lg.filters) == 1


def test_noisy_filter_blocks_info_when_enabled():
    log_mod.configure_logging(level="INFO", log_file=None, silence_client_lv_logs=True)

    root = logging.getLogger()

    noisy_logger = logging.getLogger("httpx")
    assert noisy_logger.disabled is True  # hard-disabled per implementation

    assert any(isinstance(f, log_mod.NoisyLibFilter) for f in root.filters)
    for h in root.handlers:
        assert any(isinstance(f, log_mod.NoisyLibFilter) for f in h.filters)


def test_noisy_filter_allows_warning_when_enabled():
    log_mod.configure_logging(level="INFO", log_file=None, silence_client_lv_logs=True)

    f = log_mod.NoisyLibFilter(enabled=True, prefixes=["httpx"], min_level=logging.WARNING)

    rec_warn = logging.LogRecord(
        name="httpx",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="warn",
        args=(),
        exc_info=None,
    )
    rec_info = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="info",
        args=(),
        exc_info=None,
    )
    rec_other_info = logging.LogRecord(
        name="myapp",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="info",
        args=(),
        exc_info=None,
    )

    assert f.filter(rec_warn) is True
    assert f.filter(rec_info) is False
    assert f.filter(rec_other_info) is True


def test_configure_logging_from_params_uses_llm_flag():
    class LLM:
        def __init__(self, silence_client_lv_logs: bool):
            self.silence_client_lv_logs = silence_client_lv_logs

    class Params:
        def __init__(self, silence: bool):
            self.llm = LLM(silence)

    p_true = Params(True)
    log_mod.configure_logging_from_params(p_true, level="INFO", log_file=None)
    assert log_mod._SILENCE_CLIENT_LV_LOGS is True

    p_false = Params(False)
    log_mod.configure_logging_from_params(p_false, level="INFO", log_file=None)
    assert log_mod._SILENCE_CLIENT_LV_LOGS is False


def test_configure_logging_invalid_level_raises():
    with pytest.raises(ValueError):
        log_mod.configure_logging(level="NOT_A_LEVEL", log_file=None, silence_client_lv_logs=False)
