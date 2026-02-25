# tests/test_pipeline_0_schema_ensure.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import functions.batch.pipeline_0_schema_ensure as p0
from functions.core.schema_postprocess import validate_schema_ast


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _mk_params(
    *,
    schema_py_path: str = "schema/llm_schema.py",
    archive_dir: str = "archived/",
    auto_generate: bool = True,
    force_regenerate: bool = False,
    prompt_path: str | None = None,
    cache_enabled: bool = True,
    cache_dir: str = "artifacts/cache",
    cache_force: bool = False,
    cache_dump_failures: bool = True,
):
    """
    Pipeline 0 reads:
    - params.llm_schema.{py_path,archive_dir,auto_generate,force_regenerate}
    - params.prompts.schema_auto_py_generation.path (optional, forward compat)
    - params.llm.{model_name,temperature,max_retries,timeout_sec}
    - params.cache.{enabled,dir,force,dump_failures}
    """
    params = SimpleNamespace(
        llm_schema=SimpleNamespace(
            py_path=schema_py_path,
            archive_dir=archive_dir,
            auto_generate=auto_generate,
            force_regenerate=force_regenerate,
        ),
        prompts=SimpleNamespace(
            schema_auto_py_generation=SimpleNamespace(path=prompt_path) if prompt_path else None,
            generation=SimpleNamespace(path="prompts/generation.yaml"),
        ),
        llm=SimpleNamespace(
            model_name="dummy",
            temperature=0.0,
            max_retries=1,
            timeout_sec=1,
        ),
        grouping=SimpleNamespace(enabled=False, column=None, mode="group_output", max_rows_per_group=50),
        cache=SimpleNamespace(
            enabled=cache_enabled,
            dir=cache_dir,
            force=cache_force,
            dump_failures=cache_dump_failures,
        ),
        outputs=SimpleNamespace(dir="artifacts/outputs/"),
        context=SimpleNamespace(),  # not used in pipeline 0
        input=SimpleNamespace(path="data/input.csv", format="csv", encoding="utf-8", sheet=None),
    )
    return params


def _mk_creds():
    return SimpleNamespace(gemini=SimpleNamespace(api_key_env="GEMINI_API_KEY", model_name=None))


# Minimal valid schema for import validation tests (BaseModel present)
VALID_SCHEMA_CODE = """
from pydantic import BaseModel

class Output(BaseModel):
    foo: str
"""


# Minimal schema that matches your intended contract (LLMOutput/JudgeResult) AND is strict-ready
VALID_SCHEMA_V2_LLMOUTPUT = """
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_name: str = Field(...)
    example_answer_good: str = Field(...)
    example_answer_mid: str = Field(...)
    example_answer_bad: str = Field(...)
    grading_rubrics: str = Field(...)

class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["PASS","FAIL"] = Field(...)
    score: int = Field(..., ge=0, le=100)
    reasons: List[str] = Field(default_factory=list)

__all__ = ["LLMOutput", "JudgeResult"]
"""


def test_extract_python_code_prefers_python_fence():
    raw = "hello\n```python\nprint('x')\n```\nbye\n```\nprint('y')\n```"
    code = p0._extract_python_code(raw)
    assert "print('x')" in code
    assert "print('y')" not in code


def test_extract_python_code_falls_back_to_any_fence():
    raw = "```\nprint('y')\n```"
    code = p0._extract_python_code(raw)
    assert "print('y')" in code


def test_extract_python_code_no_fence_returns_raw():
    raw = "print('z')"
    code = p0._extract_python_code(raw)
    assert code.strip() == raw


def test_archive_existing_creates_timestamped_copy(tmp_path: Path):
    schema = _write(tmp_path, "schema/llm_schema.py", "x=1\n")
    archive_dir = tmp_path / "archived"
    archived = p0._archive_existing(schema, archive_dir)
    assert archived is not None
    assert archived.exists()
    assert archived.read_text(encoding="utf-8") == "x=1\n"


def test_validate_importable_ok(tmp_path: Path):
    schema = _write(tmp_path, "schema/llm_schema.py", VALID_SCHEMA_CODE)
    p0._validate_importable(schema)  # should not raise


def test_validate_importable_raises_if_not_importable(tmp_path: Path):
    schema = _write(tmp_path, "schema/llm_schema.py", "this is not python\n")
    with pytest.raises(RuntimeError) as e:
        p0._validate_importable(schema)
    assert "not importable" in str(e.value).lower()


def test_validate_importable_raises_if_no_basemodel(tmp_path: Path):
    schema = _write(tmp_path, "schema/llm_schema.py", "x=1\n")
    with pytest.raises(RuntimeError) as e:
        p0._validate_importable(schema)
    assert "no pydantic basemodel" in str(e.value).lower()


def test_main_noop_when_schema_exists_and_not_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "schema/llm_schema.py", VALID_SCHEMA_CODE)

    params = _mk_params(force_regenerate=False)
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    monkeypatch.setattr(p0, "_call_llm_generate_schema", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    assert p0.main() == 0


def test_main_generates_when_missing_and_auto_generate_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Validate that:
    - fences are stripped
    - schema is written
    - schema remains importable
    """
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "prompts/schema_auto_py_generation.yaml", "prompt: make schema\n")

    params = _mk_params(auto_generate=True, force_regenerate=False, prompt_path=None)
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    # Return fenced python — pipeline must strip fences + write clean python
    monkeypatch.setattr(
        p0,
        "_call_llm_generate_schema",
        lambda *a, **k: f"```python\n{VALID_SCHEMA_CODE}\n```",
    )

    assert p0.main() == 0

    out_schema = tmp_path / "schema/llm_schema.py"
    assert out_schema.exists()
    out_text = out_schema.read_text(encoding="utf-8")
    assert "```" not in out_text  # ensure fences stripped

    # still importable (pipeline already validates, but keep as a strong assertion)
    p0._validate_importable(out_schema)


def test_main_postprocess_injects_model_config_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    If LLM returns a BaseModel class without strict ConfigDict, postprocess should inject it.
    """
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "prompts/schema_auto_py_generation.yaml", "prompt: make schema\n")

    params = _mk_params(auto_generate=True, force_regenerate=False, prompt_path=None)
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    # No ConfigDict, no model_config — should get injected
    raw = """
from pydantic import BaseModel

class Output(BaseModel):
    foo: str
"""
    monkeypatch.setattr(p0, "_call_llm_generate_schema", lambda *a, **k: raw)

    assert p0.main() == 0

    out_text = (tmp_path / "schema/llm_schema.py").read_text(encoding="utf-8")
    assert "from pydantic import ConfigDict" in out_text
    assert 'model_config = ConfigDict(extra="forbid")' in out_text


def test_main_archives_when_force_regenerate_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "schema/llm_schema.py", VALID_SCHEMA_CODE)
    _write(tmp_path, "prompts/schema_auto_py_generation.yaml", "prompt: make schema\n")

    params = _mk_params(auto_generate=True, force_regenerate=True, archive_dir="archived/")
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    monkeypatch.setattr(p0, "_call_llm_generate_schema", lambda *a, **k: VALID_SCHEMA_CODE)

    assert p0.main() == 0

    arch_dir = tmp_path / "archived"
    assert arch_dir.exists()
    archived_files = list(arch_dir.glob("llm_schema_*.py"))
    assert len(archived_files) >= 1


def test_main_raises_when_missing_and_auto_generate_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "prompts/schema_auto_py_generation.yaml", "prompt: make schema\n")

    params = _mk_params(auto_generate=False, force_regenerate=False)
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    with pytest.raises(RuntimeError) as e:
        p0.main()
    assert "auto_generate=false" in str(e.value).lower()


def test_main_uses_forward_compat_prompt_path_when_provided(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "prompts/custom_schema_prompt.yaml", "prompt: make schema\n")

    params = _mk_params(
        auto_generate=True,
        force_regenerate=False,
        prompt_path="prompts/custom_schema_prompt.yaml",
    )
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    monkeypatch.setattr(p0, "_call_llm_generate_schema", lambda *a, **k: VALID_SCHEMA_V2_LLMOUTPUT)

    assert p0.main() == 0
    assert (tmp_path / "schema/llm_schema.py").exists()


# ---------------------------------------------------------------------
# New tests for the cache-dir / cache-enabled fix inside _call_llm_generate_schema
# ---------------------------------------------------------------------

def test_call_llm_generate_schema_uses_cache_dir_and_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Ensure _call_llm_generate_schema passes:
    - cache_dir == params.cache.dir
    - write_cache == params.cache.enabled
    - dump_failures == params.cache.dump_failures
    """
    monkeypatch.chdir(tmp_path)

    params = _mk_params(cache_enabled=False, cache_dir="my_cache_dir", cache_dump_failures=False)
    creds = _mk_creds()

    # Avoid touching real prompts/generation.yaml on disk
    monkeypatch.setattr(p0, "_build_client_ctx", lambda *a, **k: {"client": object(), "model_name": "dummy"})
    monkeypatch.setattr(p0, "_render_generation_prompt_as_context", lambda *a, **k: "CTX")

    # Patch build_common_variables + runner at their source modules
    import functions.llm.prompts as prompts_mod
    import functions.llm.runner as runner_mod

    monkeypatch.setattr(prompts_mod, "build_common_variables", lambda **kw: kw)

    captured = {}

    def fake_run_prompt_yaml_text(**kwargs):
        captured.update(kwargs)
        return VALID_SCHEMA_CODE

    monkeypatch.setattr(runner_mod, "run_prompt_yaml_text", fake_run_prompt_yaml_text)

    # Prompt path can be anything (we don't read it due to fake runner)
    out = p0._call_llm_generate_schema(prompt_path="prompts/schema_auto_py_generation.yaml", params=params, creds=creds)
    assert "class Output" in out

    assert captured.get("cache_dir") == "my_cache_dir"
    assert captured.get("write_cache") is False
    assert captured.get("dump_failures") is False


def test_call_llm_generate_schema_write_cache_true_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Sanity check: when cache.enabled=True, write_cache should be True.
    """
    monkeypatch.chdir(tmp_path)

    params = _mk_params(cache_enabled=True, cache_dir="artifacts/cache", cache_dump_failures=True)
    creds = _mk_creds()

    monkeypatch.setattr(p0, "_build_client_ctx", lambda *a, **k: {"client": object(), "model_name": "dummy"})
    monkeypatch.setattr(p0, "_render_generation_prompt_as_context", lambda *a, **k: "CTX")

    import functions.llm.prompts as prompts_mod
    import functions.llm.runner as runner_mod

    monkeypatch.setattr(prompts_mod, "build_common_variables", lambda **kw: kw)

    captured = {}

    def fake_run_prompt_yaml_text(**kwargs):
        captured.update(kwargs)
        return VALID_SCHEMA_CODE

    monkeypatch.setattr(runner_mod, "run_prompt_yaml_text", fake_run_prompt_yaml_text)

    _ = p0._call_llm_generate_schema(prompt_path="prompts/schema_auto_py_generation.yaml", params=params, creds=creds)

    assert captured.get("write_cache") is True
    assert captured.get("cache_dir") == "artifacts/cache"
    assert captured.get("dump_failures") is True


# ---------------------------------------------------------------------------
# validate_schema_ast – unit tests
# ---------------------------------------------------------------------------

def test_validate_schema_ast_accepts_valid_pydantic_code():
    """Happy path: clean pydantic-only code should pass without raising."""
    validate_schema_ast(VALID_SCHEMA_V2_LLMOUTPUT)  # must not raise


def test_validate_schema_ast_rejects_disallowed_import_and_eval():
    """validate_schema_ast must reject disallowed imports and dangerous names."""
    # Disallowed import
    with pytest.raises(ValueError, match="Disallowed import"):
        validate_schema_ast("import os\nfrom pydantic import BaseModel\n")

    # Dangerous name inside a class body (exercises the AST-walk name check)
    with pytest.raises(ValueError, match="Disallowed name"):
        validate_schema_ast(
            "from pydantic import BaseModel\n"
            "class M(BaseModel):\n"
            "    x: str\n"
            "    y: str = eval('secret')\n"  # eval as a default value expression
        )

    # Bare non-docstring expression at module level (exercises top-level Expr guard)
    with pytest.raises(ValueError, match="Disallowed top-level construct"):
        validate_schema_ast(
            "from pydantic import BaseModel\n"
            "print('side-effect')\n"  # top-level call — not a docstring
        )


def test_main_rejects_schema_with_disallowed_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pipeline must raise RuntimeError (not silently write) when the LLM
    returns schema code that contains a disallowed import."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "prompts/schema_auto_py_generation.yaml", "prompt: make schema\n")

    params = _mk_params(auto_generate=True, force_regenerate=False)
    creds = _mk_creds()

    monkeypatch.setattr(p0, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p0, "load_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(p0, "ensure_dirs", lambda *a, **k: None)

    # Simulate an LLM that sneaks in 'import os'
    bad_schema = (
        "import os\n"
        "from pydantic import BaseModel\n"
        "class LLMOutput(BaseModel):\n"
        "    path: str = os.getcwd()\n"
    )
    monkeypatch.setattr(p0, "_call_llm_generate_schema", lambda *a, **k: bad_schema)

    with pytest.raises(RuntimeError, match="static safety check"):
        p0.main()

    # Schema file must NOT have been written
    assert not (tmp_path / "schema/llm_schema.py").exists()
