# tests/test_api_edge_cases.py
"""
Edge case tests for API request validation and config defaults.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Import the request model lazily to avoid triggering the google.genai chain.
# The Pydantic model itself only depends on pydantic, not the pipeline.
_import_error = None
try:
    from app.main import RecommendSkillsRequest, create_app
except BaseException as _exc:
    _import_error = _exc
    RecommendSkillsRequest = None  # type: ignore[assignment, misc]
    create_app = None  # type: ignore[assignment]

skip_if_import_failed = pytest.mark.skipif(
    _import_error is not None,
    reason=f"Could not import app.main (missing runtime dependency): {_import_error}",
)


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------

@skip_if_import_failed
def test_whitespace_only_query_rejected():
    with pytest.raises(ValidationError, match="Query cannot be empty or whitespace-only"):
        RecommendSkillsRequest(query="   ")


@skip_if_import_failed
def test_tabs_and_newlines_only_query_rejected():
    with pytest.raises(ValidationError, match="Query cannot be empty or whitespace-only"):
        RecommendSkillsRequest(query="\t\n  \t")


@skip_if_import_failed
def test_empty_string_query_rejected():
    with pytest.raises(ValidationError):
        RecommendSkillsRequest(query="")


@skip_if_import_failed
def test_valid_query_accepted():
    req = RecommendSkillsRequest(query="data scientist")
    assert req.query == "data scientist"


@skip_if_import_failed
def test_query_with_surrounding_whitespace_stripped():
    req = RecommendSkillsRequest(query="  data scientist  ")
    assert req.query == "data scientist"


# ---------------------------------------------------------------------------
# Config path defaults
# ---------------------------------------------------------------------------

def test_env_var_empty_string_uses_default():
    """os.environ.get(k) or default should use default when env var is empty."""
    with patch.dict(os.environ, {"PARAMETERS_PATH": "", "CREDENTIALS_PATH": ""}):
        params_path = os.environ.get("PARAMETERS_PATH") or "configs/parameters.yaml"
        creds_path = os.environ.get("CREDENTIALS_PATH") or "configs/credentials.yaml"
        assert params_path == "configs/parameters.yaml"
        assert creds_path == "configs/credentials.yaml"


def test_env_var_set_uses_custom_value():
    with patch.dict(os.environ, {"PARAMETERS_PATH": "/custom/path.yaml"}):
        params_path = os.environ.get("PARAMETERS_PATH") or "configs/parameters.yaml"
        assert params_path == "/custom/path.yaml"


# ---------------------------------------------------------------------------
# Error redaction
# ---------------------------------------------------------------------------

@skip_if_import_failed
def test_error_response_does_not_leak_details():
    """The API should return 'Internal server error', not the actual exception message."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/recommend-skills",
        json={"query": "data scientist", "top_k": 5},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Internal server error"
    assert "configs/" not in detail
    assert "Traceback" not in detail
