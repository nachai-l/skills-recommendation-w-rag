from functions.utils.text import (
    json_stringify_if_needed,
    sanitize_psv_value,
    safe_truncate,
    simple_tokenize,
)


class TestJsonStringifyIfNeeded:
    """Tests for json_stringify_if_needed."""

    def test_dict_stringified(self):
        data = {"b": 2, "a": 1}
        result = json_stringify_if_needed(data)
        assert isinstance(result, str)
        # sorted keys
        assert result == '{"a": 1, "b": 2}'

    def test_list_stringified(self):
        data = [3, 1, 2]
        result = json_stringify_if_needed(data)
        assert result == "[3, 1, 2]"

    def test_nested_structure(self):
        data = {"a": [1, {"b": 2}]}
        result = json_stringify_if_needed(data)
        assert isinstance(result, str)
        # ensure valid JSON
        import json

        parsed = json.loads(result)
        assert parsed["a"][1]["b"] == 2

    def test_string_passthrough(self):
        assert json_stringify_if_needed("hello") == "hello"

    def test_int_passthrough(self):
        assert json_stringify_if_needed(42) == 42

    def test_none_passthrough(self):
        assert json_stringify_if_needed(None) is None


class TestSanitizePSVValue:
    """Tests for sanitize_psv_value."""

    def test_none_to_empty_string(self):
        assert sanitize_psv_value(None) == ""

    def test_nan_to_empty_string(self):
        import math

        assert sanitize_psv_value(float("nan")) == ""
        assert sanitize_psv_value(math.nan) == ""

    def test_numeric_passthrough(self):
        assert sanitize_psv_value(42) == 42
        assert sanitize_psv_value(3.14) == 3.14
        assert sanitize_psv_value(True) is True

    def test_newline_escaped(self):
        result = sanitize_psv_value("hello\nworld")
        assert result == "hello\\nworld"

    def test_tab_escaped(self):
        result = sanitize_psv_value("hello\tworld")
        assert result == "hello\\tworld"

    def test_crlf_normalized(self):
        result = sanitize_psv_value("hello\r\nworld")
        assert result == "hello\\nworld"

    def test_pipe_delimiter_escaped(self):
        result = sanitize_psv_value("hello|world")
        assert result == "hello\\|world"

    def test_multiple_control_chars(self):
        text = "a\nb\tc|d\r\ne"
        result = sanitize_psv_value(text)
        assert result == "a\\nb\\tc\\|d\\ne"

    def test_idempotency_for_escaped_sequences(self):
        """
        Applying sanitize twice should not change already-sanitized output
        (i.e., no double-escaping of literal backslash sequences).
        """
        text = "hello\nworld"
        once = sanitize_psv_value(text)
        twice = sanitize_psv_value(once)
        assert once == twice

    def test_empty_string(self):
        assert sanitize_psv_value("") == ""

    def test_json_string_remains_valid(self):
        """
        Ensure JSON produced by json_stringify_if_needed
        remains valid after PSV sanitization.
        """
        import json

        data = {"a": 1, "b": 2}
        json_str = json_stringify_if_needed(data)
        sanitized = sanitize_psv_value(json_str)

        parsed = json.loads(sanitized)
        assert parsed["a"] == 1
        assert parsed["b"] == 2


class TestPSVIntegrationSafety:
    """Integration tests for PSV behavior."""

    def test_single_line_guarantee(self):
        """
        Sanitized values must never contain real newline characters.
        """
        text = "hello\nworld"
        result = sanitize_psv_value(text)
        assert "\n" not in result
        assert "\r" not in result

    def test_json_with_newlines(self):
        """
        JSON containing newlines should still become single-line safe.
        """
        import json

        data = {"text": "hello\nworld"}
        json_str = json.dumps(data)
        sanitized = sanitize_psv_value(json_str)

        assert "\n" not in sanitized
        parsed = json.loads(sanitized)
        assert parsed["text"] == "hello\nworld"

    def test_determinism(self):
        """
        Same input must always produce identical output.
        """
        text = "a\nb|c\t"
        r1 = sanitize_psv_value(text)
        r2 = sanitize_psv_value(text)
        assert r1 == r2


class TestSafeTruncate:
    def test_no_truncate_when_max_chars_zero(self):
        s, applied = safe_truncate("hello", 0)
        assert s == "hello"
        assert applied is False

    def test_truncate_applies(self):
        s, applied = safe_truncate("hello world", 5)
        assert applied is True
        assert s.endswith("…")
        assert s.startswith("hello")

    def test_accepts_non_string(self):
        s, applied = safe_truncate(12345, 3)
        assert applied is True
        assert s.startswith("123")
        assert s.endswith("…")

    def test_none_becomes_empty(self):
        s, applied = safe_truncate(None, 3)
        assert s == ""
        assert applied is False


class TestSimpleTokenize:
    def test_basic(self):
        assert simple_tokenize("Hello world") == ["hello", "world"]

    def test_remove_punct(self):
        assert simple_tokenize("hello, world!") == ["hello", "world"]

    def test_keep_case_when_lower_false(self):
        assert simple_tokenize("Hello", lower=False) == ["Hello"]

    def test_min_token_len(self):
        assert simple_tokenize("a an the", min_token_len=2) == ["an", "the"]

    def test_empty(self):
        assert simple_tokenize("") == []
        assert simple_tokenize(None) == []