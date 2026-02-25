# functions/llm/prompts.py
"""
Prompt Templates (YAML) — File-based Loading + Deterministic Rendering

Intent
- Load prompt templates from YAML files under /prompts (file-based, no registry keys).
- Render prompts deterministically with runtime variables.
- Sanitize problematic Unicode whitespace that can silently break YAML block scalars.

IMPORTANT (2026-02)
- We safely support injecting JSON/text blobs (e.g., {llm_schema}, {context}) that may contain
  literal "{" and "}" characters.
- Default rendering uses a SAFE placeholder replacer (not str.format) to avoid brace issues.

Supported placeholders
- Templates use tokens like: {context}, {llm_schema}, {output_json}, etc.
- Literal braces in templates can be written as {{ and }} (same as format convention),
  and will render as single { and }.

Primary APIs
- load_prompt_file(path) -> dict
- render_prompt_blocks(prompt_dict, variables) -> str
- render_prompt_text(text, variables) -> str

Notes
- Rendering is strict: missing placeholders raise KeyError with a clear message.
- Use stable JSON strings when injecting JSON: json.dumps(..., sort_keys=True).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Unicode spaces that commonly break YAML / indentation or silently alter prompts
_BAD_WHITESPACE = {
    "\u00A0",  # NO-BREAK SPACE
    "\u2007",  # FIGURE SPACE
    "\u202F",  # NARROW NO-BREAK SPACE
    "\uFEFF",  # BOM
}

# Safe placeholder pattern: {var_name} where var_name is alnum/underscore
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


def _sanitize_prompt_text(text: str) -> str:
    """
    Normalize problematic Unicode whitespace to regular ASCII spaces.
    Normalize CRLF -> LF.
    """
    if not isinstance(text, str):
        return text

    for ch in _BAD_WHITESPACE:
        text = text.replace(ch, " ")

    return text.replace("\r\n", "\n")


def _sanitize_prompt_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize all string fields in a prompt dict.
    """
    clean: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            clean[k] = _sanitize_prompt_text(v)
        else:
            clean[k] = v
    return clean


def load_prompt_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a single prompt YAML file and return a sanitized dict.

    Raises:
      FileNotFoundError if path doesn't exist
      ValueError if YAML doesn't parse to dict or doesn't contain required fields
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    raw = _sanitize_prompt_text(raw)

    obj = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Prompt YAML must be a mapping/dict: {p}")

    obj = _sanitize_prompt_dict(obj)

    # Require at least user block (system is optional)
    if not obj.get("user"):
        raise ValueError(f"Prompt YAML missing required 'user' field: {p}")

    # Normalize optional fields to strings if present
    for key in ("name", "purpose", "system", "assistant_prefix"):
        if key in obj and obj[key] is not None and not isinstance(obj[key], str):
            obj[key] = str(obj[key])

    # Attach path for traceability/debugging
    obj["_path"] = str(p)

    return obj


def _safe_render_with_brace_literals(template: str, variables: Dict[str, Any]) -> str:
    """
    Safe renderer for templates with {var} placeholders that must support injected
    JSON/text containing braces.

    - Replaces only tokens matching {A-Za-z0-9_}
    - Supports literal braces via {{ and }} -> { and }
    - Raises KeyError with a clear message if any placeholder is missing
    """
    if not isinstance(template, str):
        raise TypeError("_safe_render_with_brace_literals expects a string template")

    # Protect literal braces written as {{ and }} (format-style escape)
    L_SENT = "\uE000"  # private use
    R_SENT = "\uE001"
    tmp = template.replace("{{", L_SENT).replace("}}", R_SENT)

    # Find placeholders and validate presence
    missing: set[str] = set()
    for name in _PLACEHOLDER_RE.findall(tmp):
        if name not in variables:
            missing.add(name)
    if missing:
        # deterministic message: smallest missing first
        m = sorted(missing)[0]
        raise KeyError(f"Missing template variable: {m}")

    # Replace placeholders
    def _repl(m: re.Match) -> str:
        key = m.group(1)
        val = variables.get(key, "")
        # Keep deterministic: None -> ""
        if val is None:
            return ""
        return str(val)

    rendered = _PLACEHOLDER_RE.sub(_repl, tmp)

    # Restore literal braces
    rendered = rendered.replace(L_SENT, "{").replace(R_SENT, "}")
    return rendered


def render_prompt_text(text: str, variables: Dict[str, Any], *, escape_braces: bool = True) -> str:
    """
    Render a text template.

    Default (escape_braces=True):
      - Uses safe token replacement for {var} placeholders
      - Supports injected JSON/text containing '{' and '}' without corruption
      - Supports literal braces via {{ and }} in the template

    Optional legacy mode (escape_braces=False):
      - Uses Python str.format(**variables)
      - Use only if you need format features beyond simple {var} replacement
    """
    if not isinstance(text, str):
        raise TypeError("render_prompt_text expects a string template")

    if escape_braces:
        return _safe_render_with_brace_literals(text, variables)

    # Legacy behavior
    try:
        return text.format(**variables)
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(f"Missing template variable: {missing}") from e


def render_prompt_blocks(prompt: Dict[str, Any], variables: Dict[str, Any], *, escape_braces: bool = True) -> str:
    """
    Render a prompt dict and compose the final prompt text.

    Composition (if present):
      system
      blank line
      user
      blank line
      assistant_prefix

    Returns:
      final_prompt: str
    """
    system = prompt.get("system", "") or ""
    user = prompt.get("user", "") or ""
    assistant_prefix = prompt.get("assistant_prefix", "") or ""

    system_r = render_prompt_text(system, variables, escape_braces=escape_braces) if system else ""
    user_r = render_prompt_text(user, variables, escape_braces=escape_braces)

    parts = []
    if system_r.strip():
        parts.append(system_r.strip())
    parts.append(user_r.strip())
    if assistant_prefix.strip():
        parts.append(render_prompt_text(assistant_prefix, variables, escape_braces=escape_braces).strip())

    return "\n\n".join(parts).strip() + "\n"


def json_dumps_stable(obj: Any) -> str:
    """
    Stable JSON serializer for prompt injection:
    - sort_keys=True for deterministic ordering
    - ensure_ascii=False for Thai/Japanese text
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def build_common_variables(
    *,
    context: Optional[str] = None,
    llm_schema: Optional[str] = None,
    output_json: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience helper to build the common variables dict used by our prompt files.

    Common placeholders:
      - {context}
      - {llm_schema}
      - {output_json}  (used by judge prompts)

    extra: any additional placeholders needed for specialized prompts
    """
    v: Dict[str, Any] = {}
    if context is not None:
        v["context"] = context
    if llm_schema is not None:
        v["llm_schema"] = llm_schema
    if output_json is not None:
        v["output_json"] = output_json
    if extra:
        v.update(extra)
    return v


__all__ = [
    "load_prompt_file",
    "render_prompt_text",
    "render_prompt_blocks",
    "json_dumps_stable",
    "build_common_variables",
]
