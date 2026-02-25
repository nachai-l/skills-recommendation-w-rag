from __future__ import annotations

"""
functions.utils.text_embeddings

Embedding utilities for Skills Recommendation (Hybrid RAG: FAISS + BM25).
(Updated: batching + bounded parallelism + disk cache + progress logging + test mode)

Key features
------------
- Batch embedding using the Gemini embeddings API via `embed_content(contents=[...])`
- Optional on-disk cache per batch to resume/re-run cheaply
- Optional bounded parallelism across batches (ThreadPoolExecutor)
- Progress logging every N batches (configurable)
- Optional test_mode to truncate workload deterministically for fast dry runs
- Deterministic numeric post-processing:
  - float32
  - optional dimension truncation
  - optional L2 normalization
  - contiguous arrays (FAISS-friendly)

Design constraints
------------------
- Serving-safe imports: do NOT import google/genai at module import time.
  Import inside methods that actually call the SDK.
- Stable factory `build_embedding_model()` so tests can monkeypatch one symbol.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from functions.utils.config import load_credentials
from functions.utils.logging import get_logger


# -----------------------------------------------------------------------------
# Credentials
# -----------------------------------------------------------------------------
def _load_gemini_api_key(credentials_path: str = "configs/credentials.yaml") -> str:
    """
    Resolve Gemini API key using repo-standard credential config.

    Precedence:
    1) Env var specified by credentials.yaml (gemini.api_key_env)
    2) Env var GEMINI_API_KEY (fallback)
    """
    creds = load_credentials(credentials_path)
    env_name = (creds.gemini.api_key_env or "GEMINI_API_KEY").strip()

    key = os.getenv(env_name, "").strip()
    if key:
        return key

    key2 = os.getenv("GEMINI_API_KEY", "").strip()
    if key2:
        return key2

    raise ValueError(
        f"Missing Gemini API key: env var '{env_name}' not set (and GEMINI_API_KEY fallback not set). "
        f"Check {credentials_path} gemini.api_key_env and your environment."
    )


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    """
    Atomic write for .npy arrays (best-effort across platforms).

    IMPORTANT:
    - np.save() appends ".npy" if the given filename does not end with ".npy".
    - Therefore, the temp filename MUST end with ".npy" to keep paths consistent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure target is .npy (your cache paths are already .npy, but keep safe)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")

    # Temp file MUST also end with .npy so np.save does not add another suffix.
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npy")

    np.save(tmp, arr)
    os.replace(str(tmp), str(path))

def _try_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Embedding model wrapper
# -----------------------------------------------------------------------------
@dataclass
class GoogleEmbeddingModel:
    """
    Google embedding wrapper (Gemini embeddings).

    Notes
    -----
    - Lazily imports and instantiates the client to avoid import-time failures.
    - Supports batching + optional parallelism + optional disk cache for resume.
    - Supports progress logging + test mode truncation.
    """

    model_name: str = "gemini-embedding-001"
    credentials_path: str = "configs/credentials.yaml"

    # If set, truncate vectors to this dimension (e.g., for index compatibility)
    output_dim: Optional[int] = None

    # Performance knobs
    batch_size: int = 128
    max_workers: int = 1
    cache_dir: Optional[str] = None

    # Progress / test knobs
    report_log_per_n_batch: int = 10
    test_mode: bool = False
    test_max_iteration: int = 30  # number of batches

    # Numeric behavior
    normalize: bool = True

    # Debug/safety knobs
    uniqueness_guard_enabled: bool = True
    uniqueness_guard_min_unique_ratio: float = 0.85
    uniqueness_guard_round_decimals: int = 8

    # Internal client (typed loosely to avoid import-time dependency)
    _client: Optional[object] = None

    # -------------------------
    # Client + response helpers
    # -------------------------
    def _ensure_client(self) -> None:
        """
        Lazily create the google.genai client.

        Keeping the import here prevents tests/serving from failing if google-genai
        is not installed or not needed.
        """
        if self._client is not None:
            return

        from google import genai  # type: ignore

        api_key = _load_gemini_api_key(self.credentials_path)
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _as_nonempty_text(text: str) -> str:
        """
        Ensure we never send empty content to the embedding API.
        Some providers reject empty strings; we normalize to a single space.
        """
        t = (text or "").strip()
        return t if t else " "

    @staticmethod
    def _extract_vectors(resp: Any) -> List[List[float]]:
        """
        Extract embedding vectors from SDK response.

        Expected:
            resp.embeddings[i].values -> List[float]
        """
        if resp is None:
            raise RuntimeError("Embedding response is None")

        embeddings = getattr(resp, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Embedding response has no embeddings")

        out: List[List[float]] = []
        for i, emb in enumerate(embeddings):
            values = getattr(emb, "values", None)
            if values is None:
                raise RuntimeError(f"Embedding response embeddings[{i}] has no 'values'")
            out.append(list(values))
        return out

    # -------------------------
    # Vector post-processing
    # -------------------------
    def _maybe_truncate(self, mat: np.ndarray) -> np.ndarray:
        """
        Optionally truncate embedding dimension for index compatibility.

        Truncation is applied before normalization (if normalization is enabled),
        so similarity remains coherent in the truncated space.
        """
        if self.output_dim is None:
            return mat

        od = int(self.output_dim)
        if od <= 0:
            raise ValueError(f"output_dim must be positive or None, got {self.output_dim}")

        if mat.ndim != 2:
            raise ValueError("_maybe_truncate expects a 2D array")

        d = int(mat.shape[1])
        if od > d:
            raise ValueError(f"output_dim={od} exceeds embedding dimension d={d} for model={self.model_name}")
        if od == d:
            return mat
        return mat[:, :od]

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        """
        L2-normalize rows of a 2D matrix.

        - Avoids division by zero by treating zero-norm rows as norm=1.
        """
        if mat.ndim != 2:
            raise ValueError("_l2_normalize expects a 2D array")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return mat / norms

    def _vector_signatures(self, mat: np.ndarray) -> List[bytes]:
        """
        Create stable signatures for each row for uniqueness checks.
        """
        rounded = np.round(mat, decimals=int(self.uniqueness_guard_round_decimals))
        return [rounded[i].tobytes() for i in range(int(rounded.shape[0]))]

    def _maybe_raise_on_low_uniqueness(self, mat: np.ndarray, texts: List[str], *, task_type: str) -> None:
        """
        Detect suspicious embedding collapse: many different inputs -> identical/nearly-identical vectors.
        """
        if not self.uniqueness_guard_enabled:
            return
        if mat.ndim != 2 or mat.shape[0] <= 1:
            return
        if len(texts) != mat.shape[0]:
            raise RuntimeError(
                f"Uniqueness guard: input/output length mismatch: texts={len(texts)} mat_rows={mat.shape[0]}"
            )

        unique_texts = list(dict.fromkeys(texts))
        n_unique_texts = len(unique_texts)
        if n_unique_texts <= 1:
            return

        sigs = self._vector_signatures(mat)
        n_unique_vecs = len(set(sigs))
        ratio = n_unique_vecs / float(n_unique_texts)

        if ratio < float(self.uniqueness_guard_min_unique_ratio):
            preview_n = min(5, n_unique_texts)
            previews = [unique_texts[i][:120].replace("\n", " ") for i in range(preview_n)]
            raise RuntimeError(
                "Embedding uniqueness too low: "
                f"{n_unique_vecs}/{n_unique_texts} ({ratio:.2%}) "
                f"for task_type={task_type} model={self.model_name}. "
                "This MAY indicate embedding collapse or API misuse. "
                f"Examples (first {preview_n} unique inputs): {previews}"
            )

    # -------------------------
    # Cache helpers (per-batch)
    # -------------------------
    def _cache_path_for_batch(self, cleaned_texts: List[str], *, task_type: str) -> Optional[Path]:
        """
        Compute an order-sensitive cache path for a batch.

        Cache key includes:
        - model_name
        - task_type
        - output_dim
        - normalize flag
        - per-text hashes in order
        """
        if not self.cache_dir:
            return None

        parts = [
            f"model={self.model_name}",
            f"task={task_type}",
            f"outdim={self.output_dim or ''}",
            f"norm={int(bool(self.normalize))}",
        ]
        per_text = [_sha256_str(t) for t in cleaned_texts]
        batch_key = _sha256_str("\n".join(parts + per_text))

        return Path(self.cache_dir) / self.model_name / str(task_type) / f"batch_{batch_key}.npy"

    # -------------------------
    # Remote call (BATCH)
    # -------------------------
    def _embed_batch_remote(self, cleaned_texts: List[str], *, task_type: str) -> np.ndarray:
        """
        Embed a batch of cleaned texts.

        Primary path: batch call (contents=list[str]).
        Fallback path: per-item call (for test fakes / older clients that only accept contents=str).
        """
        self._ensure_client()

        # Try batch call first
        try:
            resp = self._client.models.embed_content(  # type: ignore[attr-defined]
                model=self.model_name,
                contents=cleaned_texts,
                config={"task_type": task_type},
            )
            vectors = self._extract_vectors(resp)
            mat = np.asarray(vectors, dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(cleaned_texts):
                raise RuntimeError(
                    f"Unexpected embedding shape: got {mat.shape}, expected ({len(cleaned_texts)}, D)"
                )
            return mat

        except TypeError:
            # Back-compat for tests / older fakes that only accept contents: str
            vectors: List[List[float]] = []
            for t in cleaned_texts:
                r = self._client.models.embed_content(  # type: ignore[attr-defined]
                    model=self.model_name,
                    contents=t,
                    config={"task_type": task_type},
                )
                vectors.extend(self._extract_vectors(r))
            mat = np.asarray(vectors, dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(cleaned_texts):
                raise RuntimeError(
                    f"Unexpected embedding shape (fallback): got {mat.shape}, expected ({len(cleaned_texts)}, D)"
                )
            return mat

    # -------------------------
    # Public API
    # -------------------------
    def embed_texts(self, texts: List[str], *, task_type: str) -> np.ndarray:
        """
        Embed a list of texts with an explicit task_type.

        Returns
        -------
        np.ndarray:
            Shape (N, D_out), float32, optionally L2-normalized, contiguous.
        """
        logger = get_logger(__name__)

        if not texts:
            raise ValueError("embed_texts: texts is empty")
        if not task_type or not str(task_type).strip():
            raise ValueError("embed_texts: task_type is required")

        bs = int(self.batch_size) if self.batch_size else 128
        if bs <= 0:
            raise ValueError("batch_size must be > 0")

        # Test mode: deterministically truncate workload
        if self.test_mode:
            max_iter = int(self.test_max_iteration) if self.test_max_iteration else 0
            if max_iter <= 0:
                raise ValueError("test_mode is true but test_max_iteration must be > 0")
            max_texts = min(len(texts), bs * max_iter)
            if max_texts < len(texts):
                logger.warning(
                    "TEST_MODE enabled: truncating embedding workload from %d texts to %d texts "
                    "(batch_size=%d, test_max_iteration=%d).",
                    len(texts),
                    max_texts,
                    bs,
                    max_iter,
                )
                texts = texts[:max_texts]

        # Clean once; preserves row alignment with the caller.
        cleaned_all = [self._as_nonempty_text(t) for t in texts]

        # Split into batches by index
        batches: List[Tuple[int, int]] = []
        n = len(cleaned_all)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batches.append((start, end))

        total_batches = len(batches)
        log_every = int(self.report_log_per_n_batch) if self.report_log_per_n_batch else 0
        if log_every < 1:
            log_every = 10

        def _run_one(start: int, end: int) -> Tuple[int, np.ndarray]:
            cleaned = cleaned_all[start:end]

            cache_path = self._cache_path_for_batch(cleaned, task_type=task_type)
            if cache_path is not None:
                cached = _try_load_npy(cache_path)
                if cached is not None:
                    return start, np.asarray(cached, dtype=np.float32)

            mat = self._embed_batch_remote(cleaned, task_type=task_type)

            if cache_path is not None:
                _atomic_write_npy(cache_path, mat)

            return start, mat

        max_workers = int(self.max_workers) if self.max_workers else 1
        mats_by_start: List[Tuple[int, np.ndarray]] = []

        # Execute (possibly parallel) across batches + coarse progress logs
        if max_workers <= 1 or total_batches == 1:
            for i, (s, e) in enumerate(batches, start=1):
                mats_by_start.append(_run_one(s, e))
                if i % log_every == 0 or i == total_batches:
                    logger.info(
                        "Embedding progress: %d/%d batches (batch_size=%d, cache=%s, task_type=%s, model=%s)",
                        i,
                        total_batches,
                        bs,
                        bool(self.cache_dir),
                        task_type,
                        self.model_name,
                    )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_run_one, s, e) for (s, e) in batches]
                done = 0
                for fut in as_completed(futs):
                    mats_by_start.append(fut.result())
                    done += 1
                    if done % log_every == 0 or done == total_batches:
                        logger.info(
                            "Embedding progress: %d/%d batches (batch_size=%d, workers=%d, cache=%s, task_type=%s, model=%s)",
                            done,
                            total_batches,
                            bs,
                            max_workers,
                            bool(self.cache_dir),
                            task_type,
                            self.model_name,
                        )

        # Reassemble in original order
        mats_by_start.sort(key=lambda x: x[0])
        mats = [m for _, m in mats_by_start]
        mat_all = np.vstack(mats).astype(np.float32, copy=False)

        # Post-process once (deterministic final space)
        mat_all = self._maybe_truncate(mat_all)
        if self.normalize:
            mat_all = self._l2_normalize(mat_all)
        mat_all = np.ascontiguousarray(mat_all, dtype=np.float32)

        self._maybe_raise_on_low_uniqueness(mat_all, cleaned_all, task_type=task_type)
        return mat_all

    def embed_documents(self, texts: List[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Convenience wrapper for document embeddings."""
        return self.embed_texts(texts, task_type=task_type)

    def embed_query(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
        """Embed a single query vector (shape (D_out,), float32)."""
        vec = self.embed_texts([text], task_type=task_type)
        return vec[0]


# -----------------------------------------------------------------------------
# Stable public factory (tests monkeypatch this symbol)
# -----------------------------------------------------------------------------
def build_embedding_model(
    *,
    model_name: str = "gemini-embedding-001",
    credentials_path: str = "configs/credentials.yaml",
    output_dim: Optional[int] = None,
    batch_size: int = 128,
    max_workers: int = 1,
    cache_dir: Optional[str] = None,
    report_log_per_n_batch: int = 10,
    test_mode: bool = False,
    test_max_iteration: int = 30,
    normalize: bool = True,
    uniqueness_guard_enabled: bool = True,
    uniqueness_guard_min_unique_ratio: float = 0.85,
    uniqueness_guard_round_decimals: int = 8,
) -> GoogleEmbeddingModel:
    """
    Stable factory for the embedding model.

    Why this exists
    ---------------
    - Tests can monkeypatch this symbol as the single "embedding entrypoint".
    - Avoids refactor churn when constructor args change.
    """
    return GoogleEmbeddingModel(
        model_name=model_name,
        credentials_path=credentials_path,
        output_dim=output_dim,
        batch_size=batch_size,
        max_workers=max_workers,
        cache_dir=cache_dir,
        report_log_per_n_batch=report_log_per_n_batch,
        test_mode=test_mode,
        test_max_iteration=test_max_iteration,
        normalize=normalize,
        uniqueness_guard_enabled=uniqueness_guard_enabled,
        uniqueness_guard_min_unique_ratio=uniqueness_guard_min_unique_ratio,
        uniqueness_guard_round_decimals=uniqueness_guard_round_decimals,
    )


__all__ = [
    "GoogleEmbeddingModel",
    "build_embedding_model",
    "_load_gemini_api_key",
]