# core/schema_postprocess.py
"""
Schema post-processing — deterministic hardening for schema/llm_schema.py

Intent
- Take raw LLM output (may include markdown fences / extra text)
- Extract Python code (best-effort)
- Enforce Pydantic v2 strictness:
    - ensure ConfigDict import
    - inject model_config = ConfigDict(extra="forbid") into BaseModel subclasses
- Ensure __all__ contains required exports

Design notes
- Conservative, line-based text surgery for fence stripping / ConfigDict injection.
- Static AST safety check (validate_schema_ast) runs BEFORE execution/import.
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
    Extract python code from an LLM response.
    Accepts:
    - raw python
    - ```python fenced blocks
    - ``` fenced blocks (assumed python)
    If multiple blocks exist, uses the first python-ish block.
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
    Ensure an import line exists. If missing, insert after the last import statement.
    Conservative: does not reorder existing imports.
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
    Insert `model_config = ConfigDict(extra="forbid")` as the first statement
    inside `class <name>(BaseModel):` block, if not already present.

    Heuristic:
    - Find class header
    - Skip blank lines
    - If first non-blank is a docstring, skip the docstring block
    - If next non-blank is model_config, do nothing
    - Else insert model_config line
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
    Ensure __all__ exists and contains required names.

    If missing: append __all__ at the bottom.
    If present: preserve existing order best-effort and append missing required names.
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
    """Return True iff node is an ast.Expr whose value is a string constant."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def validate_schema_ast(code: str) -> None:
    """
    Static AST safety check for LLM-generated schema code.

    Raises ValueError describing the first violation found. Must be called
    BEFORE exec_module / importlib execution of the generated file.

    Rules enforced:
    - Only imports from __future__, typing, pydantic are allowed.
    - Dangerous builtins and module names are forbidden anywhere in the AST.
    - Only imports, class definitions, simple assignments, and module-level
      docstrings are allowed at the top level (no function defs, no bare
      expression statements other than string literals).
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
    Deterministic post-LLM cleanup + hardening for schema/llm_schema.py.

    Guarantees (best-effort):
    - fences stripped
    - ConfigDict import present if we inject model_config
    - model_config inserted into BaseModel subclasses (if enforce_forbid_extra)
    - __all__ includes required exports
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
