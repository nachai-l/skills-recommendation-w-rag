from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest


from functions.utils.text_embeddings import (
    GoogleEmbeddingModel,
    build_embedding_model,
    _load_gemini_api_key,
)


# -----------------------------
# Test helpers (fake SDK)
# -----------------------------

class _FakeEmbedding:
    def __init__(self, values: List[float]):
        self.values = values


class _FakeResp:
    def __init__(self, values: List[float]):
        self.embeddings = [_FakeEmbedding(values)]


class _FakeModels:
    def __init__(self, vectors_by_text: dict[str, List[float]] | None = None, default_vec: List[float] | None = None):
        self.vectors_by_text = vectors_by_text or {}
        self.default_vec = default_vec

    def embed_content(self, model: str, contents: str, config: dict):
        # Return deterministic vector based on exact content string
        if contents in self.vectors_by_text:
            return _FakeResp(self.vectors_by_text[contents])
        if self.default_vec is not None:
            return _FakeResp(self.default_vec)
        # fallback: vector from length of string (stable)
        n = len(contents)
        return _FakeResp([float(n), float(n + 1), float(n + 2), float(n + 3)])


class _FakeClient:
    def __init__(self, models: _FakeModels):
        self.models = models


def _patch_client(model: GoogleEmbeddingModel, fake_client: _FakeClient) -> None:
    # Monkeypatch method to avoid importing google.genai
    def _ensure_client_noop():
        model._client = fake_client
    model._ensure_client = _ensure_client_noop  # type: ignore[method-assign]


# -----------------------------
# Tests: embedding behavior
# -----------------------------

def test_build_embedding_model_wires_args():
    m = build_embedding_model(model_name="m1", credentials_path="c1", output_dim=123, uniqueness_guard_enabled=False)
    assert isinstance(m, GoogleEmbeddingModel)
    assert m.model_name == "m1"
    assert m.credentials_path == "c1"
    assert m.output_dim == 123
    assert m.uniqueness_guard_enabled is False


def test_embed_texts_returns_float32_l2_normalized_contiguous():
    # Two different raw texts -> deterministic fake vectors -> normalize
    fake = _FakeClient(_FakeModels(vectors_by_text={
        "hello": [1.0, 2.0, 3.0, 4.0],
        "world": [2.0, 3.0, 4.0, 5.0],
    }))
    m = GoogleEmbeddingModel(output_dim=None, uniqueness_guard_enabled=False)
    _patch_client(m, fake)

    mat = m.embed_texts(["hello", "world"], task_type="RETRIEVAL_DOCUMENT")
    assert mat.dtype == np.float32
    assert mat.flags["C_CONTIGUOUS"] is True
    assert mat.shape == (2, 4)

    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)


def test_embed_documents_uses_default_task_type():
    fake = _FakeClient(_FakeModels(default_vec=[1.0, 0.0, 0.0, 0.0]))
    m = GoogleEmbeddingModel(output_dim=None, uniqueness_guard_enabled=False)
    _patch_client(m, fake)

    mat = m.embed_documents(["a", "b"])
    assert mat.shape == (2, 4)
    # should already be normalized to unit vectors
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_embed_query_returns_1d_float32_normalized():
    fake = _FakeClient(_FakeModels(default_vec=[3.0, 4.0, 0.0, 0.0]))
    m = GoogleEmbeddingModel(output_dim=None, uniqueness_guard_enabled=False)
    _patch_client(m, fake)

    v = m.embed_query("q")
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
    assert v.ndim == 1
    assert v.shape == (4,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_output_dim_truncates_before_normalization():
    # vector length 4 -> truncate to 2 -> normalize in 2D space
    fake = _FakeClient(_FakeModels(default_vec=[3.0, 4.0, 100.0, 100.0]))
    m = GoogleEmbeddingModel(output_dim=2, uniqueness_guard_enabled=False)
    _patch_client(m, fake)

    mat = m.embed_documents(["x"])
    assert mat.shape == (1, 2)
    assert np.isclose(np.linalg.norm(mat[0]), 1.0, atol=1e-6)
    # Should be proportional to [3,4] normalized => [0.6,0.8]
    assert np.allclose(mat[0], np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)


def test_output_dim_exceeds_dim_raises():
    fake = _FakeClient(_FakeModels(default_vec=[1.0, 2.0, 3.0, 4.0]))
    m = GoogleEmbeddingModel(output_dim=10, uniqueness_guard_enabled=False)
    _patch_client(m, fake)

    with pytest.raises(ValueError) as e:
        m.embed_documents(["x"])
    assert "exceeds embedding dimension" in str(e.value).lower()


def test_uniqueness_guard_triggers_on_collapse():
    # Different inputs -> identical vectors => uniqueness ratio becomes low
    fake = _FakeClient(_FakeModels(default_vec=[1.0, 1.0, 1.0, 1.0]))
    m = GoogleEmbeddingModel(
        output_dim=None,
        uniqueness_guard_enabled=True,
        uniqueness_guard_min_unique_ratio=1.0,  # strict
    )
    _patch_client(m, fake)

    with pytest.raises(RuntimeError) as e:
        m.embed_documents(["a", "b", "c"])
    assert "uniqueness too low" in str(e.value).lower()


# -----------------------------
# Tests: credentials resolution
# -----------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_load_gemini_api_key_uses_env_from_credentials_yaml(monkeypatch, tmp_path: Path):
    creds_path = _write(
        tmp_path,
        "credentials.yaml",
        """
gemini:
  api_key_env: MY_GEMINI_KEY
""",
    )
    monkeypatch.setenv("MY_GEMINI_KEY", "abc123")
    # ensure fallback isn't used accidentally
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    key = _load_gemini_api_key(str(creds_path))
    assert key == "abc123"


def test_load_gemini_api_key_falls_back_to_default_env(monkeypatch, tmp_path: Path):
    creds_path = _write(
        tmp_path,
        "credentials.yaml",
        """
gemini:
  api_key_env: SOME_MISSING_ENV
""",
    )
    monkeypatch.delenv("SOME_MISSING_ENV", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fallback456")

    key = _load_gemini_api_key(str(creds_path))
    assert key == "fallback456"


def test_load_gemini_api_key_raises_if_missing(monkeypatch, tmp_path: Path):
    creds_path = _write(
        tmp_path,
        "credentials.yaml",
        """
gemini:
  api_key_env: SOME_MISSING_ENV
""",
    )
    monkeypatch.delenv("SOME_MISSING_ENV", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError) as e:
        _load_gemini_api_key(str(creds_path))
    assert "missing gemini api key" in str(e.value).lower()