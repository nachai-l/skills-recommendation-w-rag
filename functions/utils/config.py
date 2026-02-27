# functions/utils/config.py
"""
Config Loader — Universal LLM Batch Generation Framework (Typed YAML Configs)

Intent
- Load and validate YAML configs:
  - configs/parameters.yaml
  - configs/credentials.yaml
- Return typed configuration objects (Pydantic v2), used as the single source of truth
  across batch + online pipelines.

What this module guarantees
- Strict validation: invalid configs fail fast with actionable Pydantic errors.
- Unicode whitespace hardening BEFORE YAML parse: NBSP/BOM/narrow NBSP normalized.
- Minimal filesystem setup via ensure_dirs() for cache / outputs / reports / logs /
  schema archive / index_store (used by online serving).

Backward / forward compatibility
- Accepts older configs that used:
  - llm_schema.path (mapped to llm_schema.py_path)
  - cache.artifact_dir (mapped to cache.dir)
  - report.sample_per_role (mapped to report.sample_per_group)
- Missing blocks fall back to model defaults (run/context/artifacts/etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from functions.utils.logging import get_logger


# -----------------------------
# Parameter models (Universal)
# -----------------------------

InputFormat = Literal["csv", "tsv", "psv", "xlsx"]
GroupingMode = Literal["group_output", "row_output_with_group_context"]

ContextColumnsMode = Literal["all", "include", "exclude"]
KVOrder = Literal["input_order", "alpha"]

OutputFormat = Literal["psv", "jsonl"]


class RunConfig(BaseModel):
    """Top-level runtime metadata (logging/timezone/run_id)."""
    name: str = "universal_llm_batch_gen_framework"
    timezone: str = "Asia/Tokyo"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    run_id: Optional[str] = None


class InputConfig(BaseModel):
    """Input table settings for batch ingestion (pipeline 2)."""
    path: str = "raw_data/input.csv"
    format: InputFormat = "csv"
    encoding: str = "utf-8"
    sheet: Optional[str] = None  # for xlsx only
    required_columns: Optional[list[str]] = None


class GroupingConfig(BaseModel):
    """Optional grouping behavior for batch outputs (unrelated to Skills-RAG online flow)."""
    enabled: bool = False
    column: Optional[str] = None
    mode: GroupingMode = "group_output"
    max_rows_per_group: int = 50

    @field_validator("max_rows_per_group")
    @classmethod
    def _validate_max_rows_per_group(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("grouping.max_rows_per_group must be > 0")
        return v


class ContextColumnsConfig(BaseModel):
    """Column-selection controls for context building."""
    mode: ContextColumnsMode = "all"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class ContextConfig(BaseModel):
    """
    Context construction controls for LLM generation.

    Used by functions/core/context_builder.py
    """

    columns: ContextColumnsConfig = Field(default_factory=ContextColumnsConfig)

    row_template: str = "{__ROW_KV_BLOCK__}"
    auto_kv_block: bool = True
    kv_order: KVOrder = "input_order"

    max_context_chars: int = 12000
    truncate_field_chars: int = 2000

    group_header_template: Optional[str] = None
    group_footer_template: Optional[str] = None

    @field_validator("max_context_chars", "truncate_field_chars")
    @classmethod
    def _validate_nonnegative_int(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"context.{info.field_name} must be >= 0")
        return v


class GenerationPromptConfig(BaseModel):
    path: str = "prompts/generation.yaml"


class JudgePromptConfig(BaseModel):
    enabled: bool = False
    path: str = "prompts/judge.yaml"


class SchemaAutoPyPromptConfig(BaseModel):
    path: str = "prompts/schema_auto_py_generation.yaml"


class SchemaAutoJsonPromptConfig(BaseModel):
    path: str = "prompts/schema_auto_json_summarization.yaml"


class PromptsConfig(BaseModel):
    generation: GenerationPromptConfig = Field(default_factory=GenerationPromptConfig)
    judge: JudgePromptConfig = Field(default_factory=JudgePromptConfig)
    schema_auto_py_generation: SchemaAutoPyPromptConfig = Field(default_factory=SchemaAutoPyPromptConfig)
    schema_auto_json_summarization: SchemaAutoJsonPromptConfig = Field(default_factory=SchemaAutoJsonPromptConfig)


class SchemaConfig(BaseModel):
    """
    Schema paths and auto-generation behavior.

    Current (preferred):
      - py_path
      - txt_path

    Backward compatibility:
      - path -> py_path
    """

    py_path: str = "schema/llm_schema.py"
    txt_path: str = "schema/llm_schema.txt"
    auto_generate: bool = True
    force_regenerate: bool = False
    archive_dir: str = "archived/"

    # Back-compat input field (optional)
    path: Optional[str] = None

    @model_validator(mode="after")
    def _apply_backcompat(self) -> "SchemaConfig":
        if (not self.py_path or self.py_path == "schema/llm_schema.py") and self.path:
            self.py_path = self.path
        return self


class LLMConfig(BaseModel):
    """LLM execution settings used by online pipelines (4b/4c) and schema auto-gen."""
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 1.0
    max_retries: int = 3
    runner_max_retries: Optional[int] = None
    timeout_sec: int = 60
    max_workers: int = 10
    silence_client_lv_logs: bool = True

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError("llm.temperature must be within [0.0, 2.0]")
        return v

    @field_validator("max_retries", "timeout_sec", "max_workers")
    @classmethod
    def _validate_positive_int(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"llm.{info.field_name} must be > 0")
        return v

    @field_validator("runner_max_retries")
    @classmethod
    def _validate_runner_max_retries(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("llm.runner_max_retries must be > 0")
        return v


class CacheConfig(BaseModel):
    """
    Deterministic cache configuration (LLM calls + optional retrieval artifacts).

    Current (preferred):
      - dir

    Backward compatibility:
      - artifact_dir -> dir
    """

    enabled: bool = True
    force: bool = False
    dir: str = "artifacts/cache"
    dump_failures: bool = True
    verbose: int = 0  # pipeline verbosity (0..10)

    artifact_dir: Optional[str] = None

    @field_validator("verbose")
    @classmethod
    def _validate_verbose(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("cache.verbose must be within [0, 10]")
        return v

    @model_validator(mode="after")
    def _apply_backcompat(self) -> "CacheConfig":
        if self.artifact_dir and (not self.dir or self.dir == "artifacts/cache"):
            self.dir = self.artifact_dir
        return self


class ArtifactsConfig(BaseModel):
    """Convenience paths for common artifact folders."""
    dir: str = "artifacts"
    outputs_dir: str = "artifacts/outputs"
    reports_dir: str = "artifacts/reports"
    logs_dir: str = "artifacts/logs"


class OutputsConfig(BaseModel):
    """Batch output formats/paths (unrelated to API payload; used in batch jobs)."""
    formats: list[OutputFormat] = Field(default_factory=lambda: ["psv", "jsonl"])
    psv_path: str = "artifacts/outputs/output.psv"
    jsonl_path: str = "artifacts/outputs/output.jsonl"

    @field_validator("formats")
    @classmethod
    def _validate_formats_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("outputs.formats must contain at least one format (psv/jsonl)")
        return v


class ReportConfig(BaseModel):
    """Report output configuration (md/html)."""
    enabled: bool = True
    md_path: str = "artifacts/reports/report.md"
    html_path: str = "artifacts/reports/report.html"

    write_html: bool = True
    sample_per_group: int = 2
    include_full_examples: bool = False
    max_reason_examples: int = 5

    sample_per_role: Optional[int] = None

    @model_validator(mode="after")
    def _apply_backcompat(self) -> "ReportConfig":
        if self.sample_per_role is not None:
            self.sample_per_group = self.sample_per_role
        return self

    @field_validator("sample_per_group", "max_reason_examples")
    @classmethod
    def _validate_nonnegative_int(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"report.{info.field_name} must be >= 0")
        return v


# -----------------------------
# NEW: Skills-RAG blocks
# -----------------------------

EmbeddingTaskType = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"]
RagVectorBackend = Literal["faiss", "vertex"]
RagBm25Backend = Literal["local"]


class EmbeddingsConfig(BaseModel):
    """Embeddings settings for batch index build (pipeline 2) and online vector search (3a)."""
    model_name: str = "gemini-embedding-001"
    batch_size: int = 128
    dim: Optional[int] = None
    normalize: bool = True
    task_type: EmbeddingTaskType = "SEMANTIC_SIMILARITY"

    max_workers: int = 5
    cache_dir: Optional[str] = "artifacts/cache/embeddings"

    report_log_per_n_batch: int = 10
    test_mode: bool = False
    test_max_iteration: int = 30

    @field_validator("batch_size", "max_workers", "report_log_per_n_batch", "test_max_iteration")
    @classmethod
    def _validate_positive_ints(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"embeddings.{info.field_name} must be > 0")
        return v

    @field_validator("dim")
    @classmethod
    def _validate_dim(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("embeddings.dim must be > 0 if provided")
        return v


class RagBackendConfig(BaseModel):
    vector: RagVectorBackend = "faiss"
    bm25: RagBm25Backend = "local"


class RagCorpusConfig(BaseModel):
    """Column mapping for the skills corpus (must exist in input table)."""
    id_col: str = "skill_id"
    title_col: str = "skill_name"
    text_col: str = "skill_text"
    source_col: Optional[str] = "source"


class RagBm25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    tokenizer: str = "simple"  # keep as str to allow future custom modes
    lower: bool = True
    remove_punct: bool = True
    min_token_len: int = 2

    @field_validator("min_token_len")
    @classmethod
    def _validate_min_token_len(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rag.bm25.min_token_len must be >= 1")
        return v


class RagVectorSearchConfig(BaseModel):
    top_k_default: int = 20

    @field_validator("top_k_default")
    @classmethod
    def _validate_top_k_default(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rag.vector_search.top_k_default must be > 0")
        return v


class RagHybridConfig(BaseModel):
    alpha: float = 0.6
    normalize_scores: bool = True
    tie_break: list[str] = Field(default_factory=lambda: ["hybrid", "vector", "bm25", "id"])

    @field_validator("alpha")
    @classmethod
    def _validate_alpha(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("rag.hybrid.alpha must be within [0.0, 1.0]")
        return v


class RagConfig(BaseModel):
    backend: RagBackendConfig = Field(default_factory=RagBackendConfig)
    corpus: RagCorpusConfig = Field(default_factory=RagCorpusConfig)
    bm25: RagBm25Config = Field(default_factory=RagBm25Config)
    vector_search: RagVectorSearchConfig = Field(default_factory=RagVectorSearchConfig)
    hybrid: RagHybridConfig = Field(default_factory=RagHybridConfig)


class IndexStoreFaissConfig(BaseModel):
    index_path: str = "artifacts/index_store/faiss.index"
    meta_path: str = "artifacts/index_store/faiss_meta.jsonl"


class IndexStoreBm25Config(BaseModel):
    corpus_path: str = "artifacts/index_store/bm25_corpus.jsonl"
    stats_path: str = "artifacts/index_store/bm25_stats.json"


class IndexStoreConfig(BaseModel):
    """Where pipeline 2 writes indexes and where online pipelines read them from."""
    dir: str = "artifacts/index_store"
    faiss: IndexStoreFaissConfig = Field(default_factory=IndexStoreFaissConfig)
    bm25: IndexStoreBm25Config = Field(default_factory=IndexStoreBm25Config)


class ParametersConfig(BaseModel):
    """Top-level typed view of parameters.yaml."""
    run: RunConfig = Field(default_factory=RunConfig)

    input: InputConfig = Field(default_factory=InputConfig)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)

    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    llm_schema: SchemaConfig = Field(default_factory=SchemaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # Skills-RAG blocks
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    index_store: IndexStoreConfig = Field(default_factory=IndexStoreConfig)

    @model_validator(mode="after")
    def _validate_rag_columns_exist(self) -> "ParametersConfig":
        # If input.required_columns is set, ensure it covers the RAG corpus column names.
        req = set(self.input.required_columns or [])
        if req:
            missing = []
            for col in [self.rag.corpus.id_col, self.rag.corpus.title_col, self.rag.corpus.text_col]:
                if col and col not in req:
                    missing.append(col)
            # source_col is optional and may not be in required_columns
            if missing:
                raise ValueError(
                    f"input.required_columns is missing required rag.corpus columns: {missing}"
                )
        return self


# -----------------------------
# Credentials models
# -----------------------------


class CredentialsGeminiRequest(BaseModel):
    timeout_seconds: int = 60
    retry_backoff_seconds: int = 2
    max_retry_backoff_seconds: int = 20


class CredentialsGemini(BaseModel):
    api_key_env: str = "GEMINI_API_KEY"
    model_name: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    request: CredentialsGeminiRequest = Field(default_factory=CredentialsGeminiRequest)


class CredentialsConfig(BaseModel):
    gemini: CredentialsGemini = Field(default_factory=CredentialsGemini)


# -----------------------------
# YAML helpers
# -----------------------------

_BAD_WHITESPACE = ["\u00A0", "\u2007", "\u202F", "\uFEFF"]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {str(path)}")

    raw = p.read_text(encoding="utf-8")

    for ch in _BAD_WHITESPACE:
        raw = raw.replace(ch, " ")

    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/object: {str(path)}")
    return data


def load_parameters(path: str | Path = "configs/parameters.yaml") -> ParametersConfig:
    """Load parameters.yaml and return a validated typed ParametersConfig."""
    logger = get_logger(__name__)
    raw = _load_yaml(path)
    try:
        return ParametersConfig.model_validate(raw)
    except ValidationError as e:
        logger.error("Invalid parameters.yaml: %s", e)
        raise


def load_credentials(path: str | Path = "configs/credentials.yaml") -> CredentialsConfig:
    """Load credentials.yaml and return a validated typed CredentialsConfig."""
    logger = get_logger(__name__)
    raw = _load_yaml(path)
    try:
        return CredentialsConfig.model_validate(raw)
    except ValidationError as e:
        logger.error("Invalid credentials.yaml: %s", e)
        raise


def ensure_dirs(params: ParametersConfig) -> None:
    """
    Ensure configured output directories exist.

    Creates:
    - cache.dir
    - artifacts.outputs_dir / artifacts.reports_dir / artifacts.logs_dir
    - parent dir for outputs.psv_path and outputs.jsonl_path
    - report output directories
    - llm_schema.archive_dir
    - parent dir for llm_schema.py_path and llm_schema.txt_path
    - index_store.dir and parent dirs for faiss/bm25 paths
    """

    dirs = [
        params.cache.dir,
        params.artifacts.outputs_dir,
        params.artifacts.reports_dir,
        params.artifacts.logs_dir,
        str(Path(params.outputs.psv_path).parent),
        str(Path(params.outputs.jsonl_path).parent),
        str(Path(params.report.md_path).parent),
        str(Path(params.report.html_path).parent),
        params.llm_schema.archive_dir,
        str(Path(params.llm_schema.py_path).parent),
        str(Path(params.llm_schema.txt_path).parent),
        # NEW
        params.index_store.dir,
        str(Path(params.index_store.faiss.index_path).parent),
        str(Path(params.index_store.faiss.meta_path).parent),
        str(Path(params.index_store.bm25.corpus_path).parent),
        str(Path(params.index_store.bm25.stats_path).parent),
    ]
    if params.embeddings.cache_dir:
        dirs.append(params.embeddings.cache_dir)

    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


__all__ = [
    "ParametersConfig",
    "CredentialsConfig",
    "load_parameters",
    "load_credentials",
    "ensure_dirs",
]