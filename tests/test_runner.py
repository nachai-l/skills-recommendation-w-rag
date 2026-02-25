import textwrap
from pathlib import Path

import pytest

import functions.llm.runner as runner_mod


# -------------------------
# Helpers (same style as test_runner.py)
# -------------------------

class DummyClient:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    class models:
        pass


def _make_client_ctx(responses):
    """
    Build a fake client_ctx with deterministic responses.
    Each call to Gemini returns the next response string.
    """
    client = DummyClient(responses)

    def fake_generate_content(*args, **kwargs):
        text = client._responses[client._i]
        client._i += 1

        class R:
            pass

        r = R()
        r.text = text
        return r

    client.models.generate_content = fake_generate_content

    return {
        "client": client,
        "model_name": "dummy-model",
    }


def _write_prompt(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "prompt.yaml"
    p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return p


# -------------------------
# Tests (text runner)
# -------------------------

def test_text_runner_basic_success(tmp_path):
    prompt_path = _write_prompt(
        tmp_path,
        """
        user: |
          Return plain text.
        """,
    )

    ctx = _make_client_ctx(["SOME TEXT OUTPUT"])

    out = runner_mod.run_prompt_yaml_text(
        prompt_path=prompt_path,
        variables={},
        client_ctx=ctx,
        cache_dir=tmp_path / "cache",
    )

    assert out == "SOME TEXT OUTPUT"


def test_text_runner_strips_code_fences(tmp_path):
    prompt_path = _write_prompt(
        tmp_path,
        """
        user: |
          Return python without fences.
        """,
    )

    ctx = _make_client_ctx(["```python\nclass A:\n    pass\n```"])

    out = runner_mod.run_prompt_yaml_text(
        prompt_path=prompt_path,
        variables={},
        client_ctx=ctx,
        cache_dir=tmp_path / "cache",
    )

    assert "class A" in out
    assert "```" not in out


def test_text_runner_cache_hit(tmp_path):
    prompt_path = _write_prompt(
        tmp_path,
        """
        user: |
          cache hit
        """,
    )

    cache_dir = tmp_path / "cache"
    cache_id = "text_unit_1"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # pre-write cached text
    (cache_dir / f"{cache_id}.txt").write_text("CACHED TEXT", encoding="utf-8")

    ctx = _make_client_ctx([])  # should not be called

    out = runner_mod.run_prompt_yaml_text(
        prompt_path=prompt_path,
        variables={},
        client_ctx=ctx,
        cache_dir=cache_dir,
        cache_id=cache_id,
    )

    assert out == "CACHED TEXT"


def test_text_runner_force_bypass_cache(tmp_path):
    prompt_path = _write_prompt(
        tmp_path,
        """
        user: |
          force bypass
        """,
    )

    cache_dir = tmp_path / "cache"
    cache_id = "text_unit_2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / f"{cache_id}.txt").write_text("OLD TEXT", encoding="utf-8")

    ctx = _make_client_ctx(["NEW TEXT"])

    out = runner_mod.run_prompt_yaml_text(
        prompt_path=prompt_path,
        variables={},
        client_ctx=ctx,
        cache_dir=cache_dir,
        cache_id=cache_id,
        force=True,
    )

    assert out == "NEW TEXT"


def test_text_runner_must_contain_retry_then_success(tmp_path):
    prompt_path = _write_prompt(
        tmp_path,
        """
        user: |
          must contain
        """,
    )

    ctx = _make_client_ctx(
        [
            "no required token here",
            "class LLMOutput:\n    pass",
        ]
    )

    out = runner_mod.run_prompt_yaml_text(
        prompt_path=prompt_path,
        variables={},
        client_ctx=ctx,
        cache_dir=tmp_path / "cache",
        max_retries=2,
        must_contain=["class LLMOutput"],
    )

    assert "class LLMOutput" in out


def test_text_runner_exhaust_retries(tmp_path):
    prompt_path = _write_prompt(
        tmp_path,
        """
        user: |
          always fail
        """,
    )

    ctx = _make_client_ctx(["x", "y"])

    with pytest.raises(RuntimeError):
        runner_mod.run_prompt_yaml_text(
            prompt_path=prompt_path,
            variables={},
            client_ctx=ctx,
            cache_dir=tmp_path / "cache",
            max_retries=2,
            must_contain=["NEVER_PRESENT"],
        )
