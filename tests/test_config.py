from pathlib import Path

import pytest

from functions.utils.config import (
    CredentialsConfig,
    ParametersConfig,
    _load_yaml,
    load_credentials,
    load_parameters,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_load_yaml_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_root_not_mapping(tmp_path: Path):
    p = _write(tmp_path, "bad.yaml", "- a\n- b\n")
    with pytest.raises(ValueError) as e:
        _load_yaml(p)
    assert "yaml root must be a mapping" in str(e.value).lower()


def test_load_yaml_sanitizes_bad_whitespace(tmp_path: Path):
    # NBSP + BOM should not break YAML parsing
    content = "\ufeffa:\u00A0 1\n"
    p = _write(tmp_path, "ok.yaml", content)
    d = _load_yaml(p)
    assert d["a"] == 1


def test_load_parameters_ok_without_context_block(tmp_path: Path):
    """
    Forward-compat:
    - context block is optional and defaults should apply when omitted.

    Also validates backward compatibility:
    - llm_schema.path is accepted and mapped to llm_schema.py_path
    - cache.artifact_dir is accepted and mapped to cache.dir
    """
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
input:
  path: data/input.csv
  format: csv
  encoding: utf-8
  sheet: null

grouping:
  enabled: false
  column: null
  mode: group_output
  max_rows_per_group: 50

prompts:
  generation:
    path: prompts/generation.yaml
  judge:
    enabled: false
    path: prompts/judge.yaml
  schema_auto_py_generation:
    path: prompts/schema_auto_py_generation.yaml
  schema_auto_json_summarization:
    path: prompts/schema_auto_json_summarization.yaml

llm_schema:
  path: schema/llm_schema.py
  auto_generate: true
  force_regenerate: false
  archive_dir: archived/

llm:
  model_name: gemini-3-flash-preview
  temperature: 1.0
  max_retries: 3
  timeout_sec: 60
  max_workers: 10
  silence_client_lv_logs: true

cache:
  enabled: true
  force: false
  artifact_dir: artifacts/cache/
  dump_failures: true

outputs:
  formats: [psv, jsonl]
  psv_path: artifacts/outputs/output.psv
  jsonl_path: artifacts/outputs/output.jsonl

report:
  enabled: true
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html

artifacts:
  dir: artifacts
  outputs_dir: artifacts/outputs
  reports_dir: artifacts/reports
  logs_dir: artifacts/logs
""",
    )

    params = load_parameters(p)
    assert isinstance(params, ParametersConfig)
    assert params.input.format == "csv"
    assert params.grouping.enabled is False
    assert params.prompts.generation.path.endswith("generation.yaml")

    # ✅ new schema fields
    assert params.llm_schema.py_path.endswith("llm_schema.py")
    assert params.llm_schema.txt_path.endswith("llm_schema.txt")  # default exists

    # ✅ backward compatible mapping: cache.artifact_dir -> cache.dir
    assert params.cache.dir.rstrip("/").endswith("artifacts/cache")

    assert params.outputs.formats == ["psv", "jsonl"]
    assert params.outputs.psv_path.endswith("output.psv")
    assert params.outputs.jsonl_path.endswith("output.jsonl")

    assert params.report.enabled is True
    assert params.report.md_path.endswith("report.md")
    assert params.report.html_path.endswith("report.html")

    # context defaults should exist even if omitted
    assert params.context.columns.mode == "all"
    assert params.context.columns.include == []
    assert params.context.columns.exclude == []
    assert params.context.row_template == "{__ROW_KV_BLOCK__}"
    assert params.context.auto_kv_block is True
    assert params.context.kv_order == "input_order"
    assert params.context.max_context_chars == 12000
    assert params.context.truncate_field_chars == 2000

    # NEW blocks should exist with defaults even if omitted
    assert params.embeddings.model_name
    assert params.rag.corpus.id_col
    assert params.index_store.dir


def test_load_parameters_ok_with_context_block(tmp_path: Path):
    """
    Forward-compat:
    - when context block is present, it should validate and override defaults.
    """
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
grouping:
  enabled: false

context:
  columns:
    mode: exclude
    exclude: [raw_html]
  row_template: |
    === ROW ===
    {__ROW_KV_BLOCK__}
  auto_kv_block: true
  kv_order: alpha
  max_context_chars: 1234
  truncate_field_chars: 200
  group_header_template: "Group {group_key} ({n_rows} rows)\\n"
  group_footer_template: "END\\n"

prompts:
  generation:
    path: prompts/generation.yaml
""",
    )

    params = load_parameters(p)
    assert params.context.columns.mode == "exclude"
    assert params.context.columns.exclude == ["raw_html"]
    assert "=== ROW ===" in params.context.row_template
    assert params.context.kv_order == "alpha"
    assert params.context.max_context_chars == 1234
    assert params.context.truncate_field_chars == 200
    assert "Group" in (params.context.group_header_template or "")
    assert (params.context.group_footer_template or "").strip() == "END"


def test_load_parameters_grouping_enabled_allows_null_column_when_validator_removed(tmp_path: Path):
    """
    Since we removed the 'grouping.column is required when enabled=true' validator,
    this should load without raising.

    Note: runtime checks should be enforced in the pipeline that uses grouping.
    """
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
grouping:
  enabled: true
  column: null
""",
    )
    params = load_parameters(p)
    assert params.grouping.enabled is True
    assert params.grouping.column is None


def test_load_parameters_invalid_grouping_mode(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
grouping:
  enabled: false
  mode: nope
""",
    )
    with pytest.raises(Exception):
        load_parameters(p)


def test_load_parameters_invalid_context_columns_mode(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
context:
  columns:
    mode: nope
""",
    )
    with pytest.raises(Exception):
        load_parameters(p)


def test_load_parameters_invalid_context_nonnegative_ints(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
context:
  max_context_chars: -1
""",
    )
    with pytest.raises(Exception) as e:
        load_parameters(p)
    assert "context.max_context_chars" in str(e.value).lower()


def test_load_parameters_backcompat_llm_schema_path_maps_to_py_path(tmp_path: Path):
    """
    Backward-compat explicit test:
    - llm_schema.path should be accepted and mapped to llm_schema.py_path
    """
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
llm_schema:
  path: schema/legacy_llm_schema.py
""",
    )
    params = load_parameters(p)
    assert params.llm_schema.py_path.endswith("schema/legacy_llm_schema.py")


def test_load_parameters_backcompat_cache_artifact_dir_maps_to_dir(tmp_path: Path):
    """
    Backward-compat explicit test:
    - cache.artifact_dir should be accepted and mapped to cache.dir
    """
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
cache:
  artifact_dir: artifacts/cache_legacy/
""",
    )
    params = load_parameters(p)
    assert params.cache.dir.rstrip("/").endswith("artifacts/cache_legacy")


def test_load_parameters_new_rag_blocks_override(tmp_path: Path):
    """
    New blocks should validate and override defaults.
    """
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
embeddings:
  model_name: gemini-embedding-001
  batch_size: 16
  dim: 3072
  normalize: true
  task_type: SEMANTIC_SIMILARITY

rag:
  corpus:
    id_col: skill_id
    title_col: skill_name
    text_col: skill_text
    source_col: source
  hybrid:
    alpha: 0.7

index_store:
  dir: artifacts/index_store
  faiss:
    index_path: artifacts/index_store/faiss.index
    meta_path: artifacts/index_store/faiss_meta.jsonl
  bm25:
    corpus_path: artifacts/index_store/bm25_corpus.jsonl
    stats_path: artifacts/index_store/bm25_stats.json
""",
    )
    params = load_parameters(p)
    assert params.embeddings.batch_size == 16
    assert params.embeddings.dim == 3072
    assert params.rag.hybrid.alpha == 0.7
    assert params.rag.corpus.source_col == "source"
    assert params.index_store.dir.endswith("artifacts/index_store")
    assert params.index_store.faiss.index_path.endswith("faiss.index")


def test_load_parameters_rag_hybrid_alpha_rejects_out_of_range(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
rag:
  hybrid:
    alpha: 1.5
""",
    )
    with pytest.raises(Exception) as e:
        load_parameters(p)
    assert "alpha" in str(e.value).lower()


def test_load_credentials_ok(tmp_path: Path):
    p = _write(
        tmp_path,
        "credentials.yaml",
        """
gemini:
  api_key_env: GEMINI_API_KEY
  model_name: gemini-3-flash-preview
""",
    )
    creds = load_credentials(p)
    assert isinstance(creds, CredentialsConfig)
    assert creds.gemini.api_key_env == "GEMINI_API_KEY"
    assert creds.gemini.model_name == "gemini-3-flash-preview"


def test_llm_runner_max_retries_default_is_none(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
llm:
  model_name: m
  temperature: 1.0
  max_retries: 5
  timeout_sec: 60
  max_workers: 1
""",
    )
    params = load_parameters(p)
    assert params.llm.runner_max_retries is None
    assert params.llm.max_retries == 5


def test_llm_runner_max_retries_explicit(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
llm:
  model_name: m
  temperature: 1.0
  max_retries: 5
  runner_max_retries: 2
  timeout_sec: 60
  max_workers: 1
""",
    )
    params = load_parameters(p)
    assert params.llm.runner_max_retries == 2


def test_llm_runner_max_retries_rejects_zero(tmp_path: Path):
    p = _write(
        tmp_path,
        "parameters.yaml",
        """
llm:
  model_name: m
  temperature: 1.0
  max_retries: 3
  runner_max_retries: 0
  timeout_sec: 60
  max_workers: 1
""",
    )
    with pytest.raises(Exception) as e:
        load_parameters(p)
    assert "runner_max_retries" in str(e.value).lower()