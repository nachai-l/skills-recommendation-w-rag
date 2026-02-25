# functions/llm/runner.py
"""
Gemini Prompt Runner (File-based YAML Prompt + JSON-only + Schema Validation + Cache)

Intent
- Provide one high-level entrypoint to run a prompt YAML file with variables,
  call Gemini, and return **validated JSON** as a Pydantic model.

This runner enforces:
- JSON extraction (even if the model adds fences or trailing text)
- schema validation (Pydantic)
- bounded retries with corrective instructions
- deterministic, caller-supplied caching via `cache_id`
- optional failure dumps for debugging

Prompt format
- Loaded from a YAML file (e.g. prompts/generation.yaml) using functions.llm.prompts:
  - load_prompt_file(path)
  - render_prompt_blocks(prompt_dict, variables)
- Placeholders are rendered using str.format(**variables), e.g. {context}, {llm_schema}.

Caching semantics
- HIT: cached payload exists and validates against schema
- STALE: cached payload exists but fails schema validation -> re-call Gemini
- MISS: no cache found
- FORCE: bypass cache read via force=True
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError

from functions.llm.prompts import load_prompt_file, render_prompt_blocks
from functions.utils.logging import get_logger


# Matches ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


def _sanitize_cache_id(cache_id: str) -> str:
    """Make cache_id filesystem-safe without hashing."""
    s = cache_id.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s[:200]  # keep filename reasonable


def _cache_path(cache_dir: str | Path, cache_id: str) -> Path:
    cid = _sanitize_cache_id(cache_id)
    return Path(cache_dir) / f"{cid}.json"


def _try_read_cache(cache_dir: str | Path, cache_id: Optional[str]) -> Optional[dict]:
    if not cache_id:
        return None
    p = _cache_path(cache_dir, cache_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(cache_dir: str | Path, cache_id: str, payload: dict) -> None:
    p = _cache_path(cache_dir, cache_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _cache_path_text(cache_dir: str | Path, cache_id: str) -> Path:
    cid = _sanitize_cache_id(cache_id)
    return Path(cache_dir) / f"{cid}.txt"


def _try_read_cache_text(cache_dir: str | Path, cache_id: Optional[str]) -> Optional[str]:
    if not cache_id:
        return None
    p = _cache_path_text(cache_dir, cache_id)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_cache_text(cache_dir: str | Path, cache_id: str, text: str) -> None:
    p = _cache_path_text(cache_dir, cache_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _sanitize_for_filename(s: str) -> str:
    s = s.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s[:160]


def _write_failure_dump(
    cache_dir: str | Path,
    cache_id: Optional[str],
    prompt_path: str,
    attempt: int,
    out_text: str,
) -> None:
    """
    Best-effort: write raw model output for debugging when JSON parsing/validation fails.
    """
    try:
        root = Path(cache_dir) / "_failures"
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cid = _sanitize_cache_id(cache_id or "no_cache_id")
        ptag = _sanitize_for_filename(Path(prompt_path).name)
        p = root / f"{cid}__{ptag}__a{attempt}__{ts}.txt"
        p.write_text(out_text or "", encoding="utf-8")
    except Exception:
        return


def _strip_code_fences(text: str) -> str:
    """If response is wrapped in markdown code fences, extract inner content."""
    m = _CODE_FENCE_RE.search(text)
    if m:
        inner = m.group(1)
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return text.strip()


def _raw_decode_first_json(text: str) -> Tuple[Any, int]:
    """
    Robustly parse the first JSON value from a string using json.JSONDecoder.raw_decode().

    Finds the first '{' or '[' and decodes from there, ignoring trailing junk.
    """
    s = text.lstrip("\ufeff \t\r\n")  # handle BOM + whitespace

    i_obj = s.find("{")
    i_arr = s.find("[")
    if i_obj == -1 and i_arr == -1:
        raise ValueError("No JSON object/array start found in response text")

    start = min([i for i in (i_obj, i_arr) if i != -1])
    s2 = s[start:]

    dec = json.JSONDecoder()
    obj, end = dec.raw_decode(s2)
    return obj, end


def _extract_json(text: str) -> Any:
    """
    Parse JSON from model output.

    Strategy:
    1) Strip code fences if present (```json ... ```).
    2) Try json.loads(full_text).
    3) Fallback: raw_decode the FIRST JSON object/array and ignore trailing text.
    """
    t = _strip_code_fences(text)

    try:
        return json.loads(t)
    except Exception:
        pass

    obj, _end = _raw_decode_first_json(t)
    return obj


def _corrective_prefix(error_msg: str) -> str:
    """Short corrective prefix to improve retries without bloating tokens."""
    return (
        "CRITICAL: Your previous output was invalid.\n"
        "You MUST output valid JSON only with the exact required schema.\n"
        "Do NOT include markdown, code fences, comments, or extra text.\n"
        f"Error: {error_msg}\n\n"
    )


def _call_gemini_text(
    client: Any,
    model_name: str,
    prompt: str,
    temperature: float,
) -> str:
    """
    Call google.genai client and return response text.
    Isolated for test mocking.
    """
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"temperature": temperature},
    )

    if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
        return resp.text

    if hasattr(resp, "candidates") and resp.candidates:
        cand0 = resp.candidates[0]
        if hasattr(cand0, "content") and cand0.content:
            return str(cand0.content)

    return str(resp)


def run_prompt_yaml_json(
    prompt_path: str | Path,
    variables: Dict[str, Any],
    schema_model: Type[BaseModel],
    client_ctx: Dict[str, Any],
    *,
    temperature: float = 0.3,
    max_retries: int = 3,
    cache_dir: str | Path = "artifacts/cache",
    cache_id: Optional[str] = None,
    force: bool = False,        # bypass cache read
    write_cache: bool = True,   # allow disabling cache write
    dump_failures: bool = True, # write raw outputs to artifacts/cache/_failures
) -> BaseModel:
    """
    Load prompt YAML from file, render with variables, call Gemini,
    extract JSON, validate against schema_model, and optionally cache.

    Notes:
    - No hashing is used; cache_id must be deterministic and provided by caller.
    """
    logger = get_logger(__name__)
    prompt_path = str(prompt_path)

    cache_path: Optional[Path] = _cache_path(cache_dir, cache_id) if cache_id else None

    # 1) cache read (unless forced)
    if cache_id and not force:
        cached = _try_read_cache(cache_dir, cache_id)
        if cached is not None:
            try:
                parsed = schema_model.model_validate(cached)
                logger.debug("LLM cache=HIT | %s | prompt=%s", str(cache_path), prompt_path)
                return parsed
            except ValidationError:
                logger.debug("LLM cache=STALE | %s | prompt=%s", str(cache_path), prompt_path)
        else:
            logger.debug("LLM cache=MISS | %s | prompt=%s", str(cache_path), prompt_path)
    elif cache_id and force:
        logger.debug("LLM cache=FORCE | %s | prompt=%s", str(cache_path), prompt_path)

    # 2) load + render prompt (file-based)
    prompt_dict = load_prompt_file(prompt_path)
    prompt = render_prompt_blocks(prompt_dict, variables)

    client = client_ctx["client"]
    model_name = client_ctx["model_name"]

    last_err: Optional[Exception] = None
    out_text: str = ""

    for attempt in range(1, max_retries + 1):
        try:
            out_text = _call_gemini_text(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
            )

            payload = _extract_json(out_text)
            parsed = schema_model.model_validate(payload)

            if cache_id and write_cache:
                _write_cache(cache_dir, cache_id, payload)

            return parsed

        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            last_err = e
            logger.warning("LLM output invalid (attempt %d/%d): %s", attempt, max_retries, str(e))

            if dump_failures:
                _write_failure_dump(
                    cache_dir=cache_dir,
                    cache_id=cache_id,
                    prompt_path=prompt_path,
                    attempt=attempt,
                    out_text=out_text if "out_text" in locals() else "",
                )

            prompt = _corrective_prefix(str(e)) + prompt
            continue

    raise RuntimeError(f"Failed to produce valid JSON after {max_retries} attempts: {last_err}")


def run_prompt_yaml_text(
    prompt_path: str | Path,
    variables: Dict[str, Any],
    client_ctx: Dict[str, Any],
    *,
    temperature: float = 0.3,
    max_retries: int = 3,
    cache_dir: str | Path = "artifacts/cache",
    cache_id: Optional[str] = None,
    force: bool = False,
    write_cache: bool = True,
    dump_failures: bool = True,
    strip_code_fences: bool = True,
    min_chars: int = 1,
    must_contain: Optional[list[str]] = None,
) -> str:
    """
    Run a YAML prompt file and return plain text output.

    Use for:
    - schema/llm_schema.py generation (python code)
    - llm_schema.txt summarization (schema contract)

    Light validation:
    - min_chars: output length must be >= min_chars after trimming
    - must_contain: all substrings must appear in output (case-sensitive)
    """
    logger = get_logger(__name__)
    prompt_path = str(prompt_path)

    cache_path_txt: Optional[Path] = _cache_path_text(cache_dir, cache_id) if cache_id else None

    # cache read (unless forced)
    if cache_id and not force:
        cached = _try_read_cache_text(cache_dir, cache_id)
        if cached is not None:
            out = cached
            out2 = _strip_code_fences(out) if strip_code_fences else out
            out2 = out2.strip()

            ok = len(out2) >= max(min_chars, 0)
            if ok and must_contain:
                ok = all(s in out2 for s in must_contain)

            if ok:
                logger.debug("LLM cache=HIT | %s | prompt=%s", str(cache_path_txt), prompt_path)
                return out2

            logger.debug("LLM cache=STALE | %s | prompt=%s", str(cache_path_txt), prompt_path)

        else:
            logger.debug("LLM cache=MISS | %s | prompt=%s", str(cache_path_txt), prompt_path)
    elif cache_id and force:
        logger.debug("LLM cache=FORCE | %s | prompt=%s", str(cache_path_txt), prompt_path)

    prompt_dict = load_prompt_file(prompt_path)
    prompt = render_prompt_blocks(prompt_dict, variables)

    client = client_ctx["client"]
    model_name = client_ctx["model_name"]

    last_err: Optional[Exception] = None
    out_text: str = ""

    for attempt in range(1, max_retries + 1):
        try:
            out_text = _call_gemini_text(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
            )

            out = _strip_code_fences(out_text) if strip_code_fences else out_text
            out = out.strip()

            if len(out) < max(min_chars, 0):
                raise ValueError(f"Text output too short (len={len(out)} < {min_chars})")

            if must_contain:
                missing = [s for s in must_contain if s not in out]
                if missing:
                    raise ValueError(f"Text output missing required substrings: {missing}")

            if cache_id and write_cache:
                _write_cache_text(cache_dir, cache_id, out)

            return out

        except Exception as e:
            last_err = e
            logger.warning("LLM text output invalid (attempt %d/%d): %s", attempt, max_retries, str(e))

            if dump_failures:
                _write_failure_dump(
                    cache_dir=cache_dir,
                    cache_id=cache_id,
                    prompt_path=prompt_path,
                    attempt=attempt,
                    out_text=out_text if "out_text" in locals() else "",
                )

            # Same corrective prefix (still helpful for non-JSON outputs)
            prompt = _corrective_prefix(str(e)) + prompt
            continue

    raise RuntimeError(f"Failed to produce valid text after {max_retries} attempts: {last_err}")


__all__ = ["run_prompt_yaml_json", "run_prompt_yaml_text"]
