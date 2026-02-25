import pytest

from functions.llm.prompts import (
    load_prompt_file,
    render_prompt_text,
    render_prompt_blocks,
)


def test_load_prompt_file_requires_user(tmp_path):
    p = tmp_path / "bad_prompt.yaml"
    p.write_text("name: x\nsystem: hi\n", encoding="utf-8")

    with pytest.raises(ValueError) as e:
        load_prompt_file(p)

    assert "missing required 'user'" in str(e.value).lower()


def test_load_prompt_file_sanitizes_unicode_whitespace(tmp_path):
    # includes NBSP (\u00A0) and CRLF
    content = (
        "name: t\n"
        "system: |\r\n"
        "  Hello\u00A0World\r\n"
        "user: |\r\n"
        "  Context: {context}\r\n"
    )

    p = tmp_path / "prompt.yaml"
    p.write_text(content, encoding="utf-8")

    d = load_prompt_file(p)

    assert d["system"].startswith("Hello World")  # NBSP -> space
    assert "\r\n" not in d["system"]  # CRLF -> LF
    assert d["_path"].endswith("prompt.yaml")


def test_render_prompt_text_missing_variable_raises_clear_error():
    template = "Hello {name} {missing}"
    with pytest.raises(KeyError) as e:
        render_prompt_text(template, {"name": "A"})
    # custom error message includes the missing placeholder name
    assert "missing template variable" in str(e.value).lower()
    assert "missing" in str(e.value).lower()


def test_render_prompt_blocks_composition_system_user_prefix(tmp_path):
    p = tmp_path / "prompt.yaml"
    p.write_text(
        """
name: demo
system: |
  SYSTEM: {sys}
user: |
  USER: {context}
assistant_prefix: |
  ASSIST: start
""".strip(),
        encoding="utf-8",
    )

    prompt = load_prompt_file(p)
    rendered = render_prompt_blocks(prompt, {"sys": "S", "context": "C"})

    # system then user then assistant_prefix, separated by blank lines
    assert "SYSTEM: S" in rendered
    assert "USER: C" in rendered
    assert "ASSIST: start" in rendered
    # ends with newline by design
    assert rendered.endswith("\n")


def test_render_prompt_blocks_no_system_no_prefix(tmp_path):
    p = tmp_path / "prompt.yaml"
    p.write_text(
        """
name: demo
user: |
  USER: {context}
""".strip(),
        encoding="utf-8",
    )

    prompt = load_prompt_file(p)
    rendered = render_prompt_blocks(prompt, {"context": "C"})

    assert rendered.strip() == "USER: C"
    assert rendered.endswith("\n")


def test_load_prompt_file_not_found(tmp_path):
    p = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_prompt_file(p)


# ------------------------------------------------------------------
# NEW TESTS — brace escaping behavior (critical for llm_schema JSON)
# ------------------------------------------------------------------

def test_render_prompt_text_escapes_braces_in_variable_values():
    """
    Injecting JSON (with { and }) into a placeholder must not break str.format().
    The rendered output should contain the original JSON exactly.
    """
    template = "Schema:\n{llm_schema}"
    json_blob = '{"a": 1, "b": {"c": 2}}'

    rendered = render_prompt_text(template, {"llm_schema": json_blob})

    assert json_blob in rendered
    assert rendered.startswith("Schema:")


def test_render_prompt_text_escape_braces_can_be_disabled():
    """
    If escape_braces=False, braces are not escaped.
    This is mainly for completeness; normally we keep it enabled.
    """
    template = "Value: {x}"
    value = "{test}"

    rendered = render_prompt_text(template, {"x": value}, escape_braces=False)

    # Without escaping, literal braces will be processed by format engine.
    # In this case "{test}" has no placeholder in template, so it should
    # remain intact inside the formatted result.
    assert rendered == "Value: {test}"
