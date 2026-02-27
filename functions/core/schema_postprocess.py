# core/schema_postprocess.py
"""
Schema Post-Processing — deterministic hardening for schema/llm_schema.py

Intent
- Take raw LLM output (may include markdown fences / extra text)
- Extract Python code (best-effort)
- Enforce Pydantic v2 strictness:
    - ensure ConfigDict import
    - inject model_config = ConfigDict(extra="forbid") into BaseModel subclasses
- Ensure __all__ contains required exports

Design notes
- Conservative text surgery (line-based) for fence stripping / ConfigDict injection
- AST-based static safety check (validate_schema_ast) runs before execution
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Sequence

# Matches ```python ... ``` or ``` ... ```
_CODE_FENCE_PY_RE = re.compile(r"```python\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_ANY_RE = re.compile(r"```\s*(.*?)\s*```", flags=re.DOTALL)

# Detect import lines
_IMPORT_LINE_RE = re.compile(r"^\s*(from|import)\s+", flags=re.MULTILINE)

# Find class Foo(BaseModel):
_BASEMODEL_CLASS_RE = re.compile(
    r"(^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*BaseModel\s*\)\s*:\s*$)",
    flags=re.MULTILINE,
)


def extract_python_code(text: str) -> str:
    """
    Extract Python source code from a raw LLM response string.

    Tries extraction in priority order:
    1. Content inside a ```python ... ``` fenced block (case-insensitive).
    2. Content inside any ``` ... ``` fenced block (assumed Python).
    3. The raw text itself, stripped (fallback — assumes no prose wrapper).

    When multiple fenced blocks are present, the first matching block is used.

    Args:
        text: Raw LLM response that may contain prose, markdown fences, or bare code.

    Returns:
        Extracted and stripped Python source string, or ``""`` if ``text`` is empty.
    """
    if not text:
        return ""

    m = _CODE_FENCE_PY_RE.search(text)
    if m:
        return (m.group(1) or "").strip()

    m = _CODE_FENCE_ANY_RE.search(text)
    if m:
        return (m.group(1) or "").strip()

    return text.strip()


def _ensure_import(code: str, line: str) -> str:
    """
    Ensure a specific import line is present in the source code.

    If ``line`` already appears anywhere in ``code`` (substring match), the
    code is returned unchanged. Otherwise, ``line`` is inserted immediately
    after the last existing import statement. If no import statements exist,
    it is inserted at line 0. Existing imports are never reordered.

    Args:
        code: Python source code as a string.
        line: The exact import line to insert, e.g. ``"from pydantic import ConfigDict"``.

    Returns:
        Source code guaranteed to contain ``line``, with a trailing newline.
    """
    if line in code:
        return code

    lines = code.splitlines()
    last_import_idx = -1
    for i, ln in enumerate(lines):
        if _IMPORT_LINE_RE.match(ln):
            last_import_idx = i

    insert_idx = last_import_idx + 1 if last_import_idx >= 0 else 0
    lines.insert(insert_idx, line)
    out = "\n".join(lines)
    if not out.endswith("\n"):
        out += "\n"
    return out


def _insert_model_config_into_class_block(code: str, class_name: str) -> str:
    """
    Insert ``model_config = ConfigDict(extra="forbid")`` into a BaseModel subclass body.

    Targets the class named ``class_name`` with the signature
    ``class <name>(BaseModel):``. The line is inserted as the first body statement
    after any leading docstring. If ``model_config`` already appears as the first
    non-blank body statement (before or after a docstring), the code is returned
    unchanged.

    Heuristic (line-based, no AST):
    1. Locate the class header line by name.
    2. Skip leading blank lines in the body.
    3. If the first non-blank line is a triple-quoted docstring, skip past it.
    4. Skip any further blank lines.
    5. If ``model_config`` is already present at the insertion point, return unchanged.
    6. Otherwise insert the ``model_config`` line at the correct indentation.

    Args:
        code: Python source code as a string.
        class_name: Name of the ``BaseModel`` subclass to target.

    Returns:
        Source code with ``model_config = ConfigDict(extra="forbid")`` injected,
        or the original code unchanged if the class was not found or the config
        was already present.
    """
    lines = code.splitlines()

    header_idx: Optional[int] = None
    header_indent = ""
    header_pat = re.compile(
        rf"^(\s*)class\s+{re.escape(class_name)}\s*\(\s*BaseModel\s*\)\s*:\s*$"
    )

    for i, ln in enumerate(lines):
        m = header_pat.match(ln)
        if m:
            header_idx = i
            header_indent = m.group(1)
            break

    if header_idx is None:
        return code

    body_indent = header_indent + " " * 4

    j = header_idx + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    if j < len(lines) and lines[j].lstrip().startswith("model_config"):
        return code

    # Skip docstring if present
    if j < len(lines) and lines[j].lstrip().startswith(('"""', "'''")):
        quote = '"""' if lines[j].lstrip().startswith('"""') else "'''"

        # docstring ends on same line?
        if lines[j].count(quote) >= 2:
            j += 1
        else:
            j += 1
            while j < len(lines):
                if quote in lines[j]:
                    j += 1
                    break
                j += 1

        while j < len(lines) and lines[j].strip() == "":
            j += 1

        if j < len(lines) and lines[j].lstrip().startswith("model_config"):
            return code

    lines.insert(j, f'{body_indent}model_config = ConfigDict(extra="forbid")')
    out = "\n".join(lines)
    if not out.endswith("\n"):
        out += "\n"
    return out


def _ensure___all__(code: str, required: Sequence[str]) -> str:
    """
    Ensure ``__all__`` exists in the source and contains every name in ``required``.

    - If ``__all__`` is absent: appends ``__all__ = [<required names>]`` at the
      bottom of the file.
    - If ``__all__`` is present: preserves the existing order of names (scanning
      for both single- and double-quoted entries) and appends any ``required``
      names not already listed.

    Args:
        code: Python source code as a string.
        required: Sequence of names that must appear in ``__all__``.

    Returns:
        Source code with a ``__all__`` list that includes all ``required`` names.
    """
    m = re.search(r"^__all__\s*=\s*\[(.*?)\]\s*$", code, flags=re.MULTILINE | re.DOTALL)
    if not m:
        safe = ", ".join([f'"{n}"' for n in required])
        if not code.endswith("\n"):
            code += "\n"
        return code + f'\n__all__ = [{safe}]\n'

    inside = m.group(1) or ""
    present = set(re.findall(r'"([^"]+)"', inside)) | set(re.findall(r"'([^']+)'", inside))

    # preserve original order best-effort
    ordered: list[str] = []
    for n in re.findall(r'"([^"]+)"', inside) + re.findall(r"'([^']+)'", inside):
        if n not in ordered:
            ordered.append(n)

    for n in required:
        if n not in ordered:
            ordered.append(n)

    safe = ", ".join([f'"{n}"' for n in ordered])
    return re.sub(
        r"^__all__\s*=\s*\[.*?\]\s*$",
        f"__all__ = [{safe}]",
        code,
        flags=re.MULTILINE | re.DOTALL,
    )


# ---------------------------------------------------------------------------
# Static AST safety validator
# ---------------------------------------------------------------------------

_ALLOWED_IMPORT_MODULES: frozenset[str] = frozenset({"__future__", "typing", "pydantic"})

_DANGEROUS_NAMES: frozenset[str] = frozenset({
    "eval", "exec", "compile", "open", "__import__",
    "os", "sys", "subprocess", "socket", "shutil",
    "importlib", "builtins", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
})

_ALLOWED_TOP_LEVEL_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    # ast.Expr is handled separately: only allowed when wrapping a string constant
    # (i.e. a module-level docstring). Bare calls like print() or eval() are rejected.
)


def _is_docstring_expr(node: ast.AST) -> bool:
    """
    Return True if ``node`` is an ``ast.Expr`` wrapping a string constant.

    Used to distinguish module-level docstrings (permitted at the top level of
    schema files) from other bare expression statements such as ``print()`` or
    ``eval()`` (which are forbidden).

    Args:
        node: Any AST node from a parsed module.

    Returns:
        True if the node is an ``ast.Expr`` whose ``value`` is an
        ``ast.Constant`` holding a ``str``; False otherwise.
    """
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def validate_schema_ast(code: str) -> None:
    """
    Static AST safety check for LLM-generated schema code.

    Parses ``code`` and verifies it meets strict structural and content rules.
    Must be called *before* executing or importing the generated file via
    ``exec_module`` / ``importlib``.

    Top-level node rules:
        Permitted: ``import``, ``from ... import``, class definitions,
        simple assignments (``ast.Assign`` / ``ast.AnnAssign``), and
        ``ast.Expr`` nodes that wrap a bare string constant (i.e. module-level
        docstrings, as determined by :func:`_is_docstring_expr`).
        Forbidden: function definitions, decorated definitions, bare calls
        (e.g. ``print()``, ``eval()``), and all other statement types.

    Import allowlist:
        Only ``__future__``, ``typing``, and ``pydantic`` may be imported
        (checked by root module name).

    Name/attribute blocklist:
        The following names are forbidden *anywhere* in the AST, including
        inside class bodies: ``eval``, ``exec``, ``compile``, ``open``,
        ``__import__``, ``os``, ``sys``, ``subprocess``, ``socket``,
        ``shutil``, ``importlib``, ``builtins``, ``globals``, ``locals``,
        ``vars``, ``getattr``, ``setattr``, ``delattr``.

    Args:
        code: Python source code string to validate.

    Raises:
        ValueError: On the first violation found, with a message describing
            the disallowed construct, import, or name. Also raised if
            ``code`` contains a syntax error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Schema code has a syntax error: {exc}") from exc

    # --- top-level structure check ---
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, _ALLOWED_TOP_LEVEL_NODE_TYPES) and not _is_docstring_expr(node):
            raise ValueError(
                f"Disallowed top-level construct: {type(node).__name__}. "
                "Schema files may only contain imports, class definitions, "
                "assignments, and module-level docstrings."
            )

    # --- full AST walk for import allowlist and dangerous names ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_MODULES:
                    raise ValueError(
                        f"Disallowed import: 'import {alias.name}'. "
                        f"Only {sorted(_ALLOWED_IMPORT_MODULES)} are permitted."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in _ALLOWED_IMPORT_MODULES:
                raise ValueError(
                    f"Disallowed import: 'from {node.module} import ...'. "
                    f"Only {sorted(_ALLOWED_IMPORT_MODULES)} are permitted."
                )
        elif isinstance(node, ast.Name) and node.id in _DANGEROUS_NAMES:
            raise ValueError(
                f"Disallowed name: '{node.id}'. "
                "Schema code must not reference system-level builtins or modules."
            )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in _DANGEROUS_NAMES
        ):
            raise ValueError(
                f"Disallowed attribute access: '{node.value.id}.{node.attr}'."
            )


def postprocess_schema_py(
    raw_text: str,
    *,
    required_exports: Sequence[str] = ("LLMOutput", "JudgeResult"),
    enforce_forbid_extra: bool = True,
) -> str:
    """
    Deterministic post-processing and hardening pipeline for LLM-generated schema code.

    Applies the following transformations in order:

    1. **Fence stripping** — extracts Python source from markdown code blocks
       via :func:`extract_python_code`.
    2. **ConfigDict injection** (when ``enforce_forbid_extra=True``) — ensures
       ``from pydantic import ConfigDict`` is imported and inserts
       ``model_config = ConfigDict(extra="forbid")`` into every ``BaseModel``
       subclass body that does not already have it.
    3. **Literal import** — if the code references ``Literal[...]`` without an
       existing ``import Literal``, injects ``from typing import Literal``.
    4. **``__all__`` enforcement** — ensures ``__all__`` exists and contains
       every name in ``required_exports``.

    Args:
        raw_text: Raw LLM response, which may contain prose, markdown fences,
            or bare Python source.
        required_exports: Names that must appear in ``__all__``.
            Defaults to ``("LLMOutput", "JudgeResult")``.
        enforce_forbid_extra: When True, injects ``model_config`` into all
            ``BaseModel`` subclasses found in the code. Defaults to True.

    Returns:
        Hardened Python source code as a string, or ``""`` if no Python code
        could be extracted from ``raw_text``.
    """
    code = extract_python_code(raw_text).strip()
    if not code:
        return ""

    if not code.endswith("\n"):
        code += "\n"

    classes = [m.group(2) for m in _BASEMODEL_CLASS_RE.finditer(code)]

    if enforce_forbid_extra and classes:
        code = _ensure_import(code, "from pydantic import ConfigDict")
        for cname in classes:
            code = _insert_model_config_into_class_block(code, cname)

    # Ensure typing imports for common type constructs used by LLM-generated schemas
    if re.search(r"\bLiteral\s*\[", code) and "import Literal" not in code:
        code = _ensure_import(code, "from typing import Literal")

    code = _ensure___all__(code, required_exports)

    return code


__all__ = ["extract_python_code", "postprocess_schema_py", "validate_schema_ast"]
