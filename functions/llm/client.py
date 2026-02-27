# functions/llm/client.py
"""
Gemini client factory (google.genai)

Intent
- Centralize construction of a Gemini client context for this repository.
- Standardize resolution of:
  - model name
  - authentication (API key from environment variable)
  - config inputs (supports dicts and typed config objects)

Design principles
- No network calls are performed here; this module only prepares a client instance
  and resolves configuration values.
- Secrets are never read from files. The API key must be provided via an environment
  variable (e.g., `GEMINI_API_KEY`), whose name is declared in `credentials.yaml`.

Configuration inputs
- `credentials_config` may be:
  1) a full credentials object with `.gemini`
  2) an object/dict already representing the `gemini` section
  3) a dict like `{"gemini": {...}}`

Model name resolution (priority order)
1) `model_name_override` (typically from parameters.yaml)
2) `credentials_config.gemini.model_name` (if present)
3) environment variable `GEMINI_MODEL`

Primary API
- get_model_name(credentials_config, model_name_override=None) -> str
- build_gemini_client(credentials_config, model_name_override=None) -> dict
    Returns:
      {
        "client": genai.Client,
        "model_name": "<resolved model name>",
      }

Dependencies
- google.genai (Gemini SDK)
- os.environ for secret retrieval
- functions.utils.logging.get_logger for optional logging
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from google import genai

from functions.utils.logging import get_logger


def _get(obj: Any, key: str, default=None):
    """Best-effort getter supporting dicts and attribute-style config objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_gemini_config(credentials_config: Any) -> Any:
    """
    Accept:
    - full creds object that has `.gemini`
    - already gemini section object
    - dict with {"gemini": {...}} or direct {...}
    """
    if isinstance(credentials_config, dict):
        if isinstance(credentials_config.get("gemini"), dict):
            return credentials_config["gemini"]
        return credentials_config

    gemini_section = getattr(credentials_config, "gemini", None)
    return gemini_section if gemini_section is not None else credentials_config


def get_model_name(credentials_config: Any, *, model_name_override: Optional[str] = None) -> str:
    """
    Resolve model name.

    Priority:
    1) model_name_override (passed by pipeline from parameters.yaml)
    2) credentials_config.model_name (if present)
    3) env var GEMINI_MODEL
    """
    if model_name_override:
        return str(model_name_override)

    cfg = _resolve_gemini_config(credentials_config)

    cfg_model = _get(cfg, "model_name", None)
    if cfg_model:
        return str(cfg_model)

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model:
        return env_model

    raise ValueError("Gemini model name not found (override/credentials/env)")


def build_gemini_client(
    credentials_config: Any,
    *,
    model_name_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a Gemini client context (client + resolved model name).

    Notes:
    - This function does not make any network calls.
    - API key is retrieved from an environment variable specified by credentials config.
    """
    logger = get_logger(__name__)

    cfg = _resolve_gemini_config(credentials_config)

    api_key_env = _get(cfg, "api_key_env", None)
    if not api_key_env:
        raise ValueError("credentials_config.api_key_env is required")

    api_key = os.environ.get(str(api_key_env))
    if not api_key:
        raise EnvironmentError(f"Environment variable '{api_key_env}' not set for Gemini API key")

    model_name = get_model_name(credentials_config, model_name_override=model_name_override)

    # Keep logs non-sensitive; do not log API key or full credential blobs.
    # logger.info("Initializing Gemini client (model=%s)", model_name)

    client = genai.Client(api_key=api_key)

    return {
        "client": client,
        "model_name": model_name,
    }


__all__ = ["build_gemini_client", "get_model_name"]
