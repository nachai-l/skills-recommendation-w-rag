# tests/test_schema_postprocess.py

from __future__ import annotations

import re

from functions.core.schema_postprocess import extract_python_code, postprocess_schema_py


def _count_model_config_in_class(code: str, class_name: str) -> int:
    """
    Count occurrences of 'model_config = ConfigDict(extra="forbid")' inside the given class block.
    Simple heuristic: scan from class header until next top-level class/def (no indent).
    """
    lines = code.splitlines()
    header_pat = re.compile(rf"^class\s+{re.escape(class_name)}\s*\(\s*BaseModel\s*\)\s*:\s*$")
    start = None
    for i, ln in enumerate(lines):
        if header_pat.match(ln):
            start = i + 1
            break
    if start is None:
        return 0

    # collect block lines while indented or blank/comment
    block = []
    for j in range(start, len(lines)):
        ln = lines[j]
        if ln.strip() == "":
            block.append(ln)
            continue
        # stop when next top-level statement starts
        if not ln.startswith((" ", "\t")):
            break
        block.append(ln)

    target = 'model_config = ConfigDict(extra="forbid")'
    return sum(1 for ln in block if target in ln)


def test_extract_python_code_prefers_python_fence():
    raw = "x\n```python\nprint('a')\n```\n``` \nprint('b')\n```\n"
    out = extract_python_code(raw)
    assert "print('a')" in out
    assert "print('b')" not in out


def test_extract_python_code_falls_back_to_any_fence():
    raw = "``` \nprint('b')\n```"
    out = extract_python_code(raw)
    assert "print('b')" in out


def test_extract_python_code_no_fence_returns_raw():
    raw = "print('z')\n"
    out = extract_python_code(raw)
    assert out.strip() == "print('z')"


def test_postprocess_strips_fences_and_adds_trailing_newline():
    raw = "```python\nfrom pydantic import BaseModel\n\nclass A(BaseModel):\n    x: int\n```"
    out = postprocess_schema_py(raw, required_exports=("A",))
    assert out.startswith("from pydantic import BaseModel")
    assert out.endswith("\n")
    assert "```" not in out


def test_postprocess_injects_configdict_import_when_needed():
    raw = """
from pydantic import BaseModel, Field

class LLMOutput(BaseModel):
    x: int = Field(...)
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput",))
    assert "from pydantic import ConfigDict" in out


def test_postprocess_injects_model_config_before_fields_when_no_docstring():
    raw = """
from pydantic import BaseModel

class LLMOutput(BaseModel):
    question_name: str
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput",))
    # inserted inside class
    assert _count_model_config_in_class(out, "LLMOutput") == 1

    # should appear before first field line
    cls_idx = out.splitlines().index("class LLMOutput(BaseModel):")
    after = out.splitlines()[cls_idx + 1 : cls_idx + 10]
    # first non-empty, indented line should be model_config
    first_stmt = next(ln for ln in after if ln.strip())
    assert 'model_config = ConfigDict(extra="forbid")' in first_stmt


def test_postprocess_injects_model_config_after_docstring_block():
    raw = '''
from pydantic import BaseModel

class LLMOutput(BaseModel):
    """
    doc
    """
    question_name: str
'''
    out = postprocess_schema_py(raw, required_exports=("LLMOutput",))
    assert _count_model_config_in_class(out, "LLMOutput") == 1

    # ensure it is after docstring end
    lines = out.splitlines()
    cls_i = lines.index("class LLMOutput(BaseModel):")
    # find docstring close line index (line containing only triple quotes or containing them)
    close_i = None
    for i in range(cls_i + 1, len(lines)):
        if '"""' in lines[i] or "'''" in lines[i]:
            # this finds both start and end; but our sample has start+end on separate lines.
            # the *end* is the second occurrence after the class.
            pass
    # stronger check: model_config line exists and comes before first field
    model_i = next(i for i in range(cls_i + 1, len(lines)) if "model_config = ConfigDict" in lines[i])
    field_i = next(i for i in range(cls_i + 1, len(lines)) if lines[i].strip().startswith("question_name"))
    assert model_i < field_i


def test_postprocess_does_not_duplicate_model_config_if_already_present():
    raw = """
from pydantic import BaseModel, ConfigDict

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_name: str
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput",))
    assert _count_model_config_in_class(out, "LLMOutput") == 1


def test_postprocess_handles_multiple_basemodel_classes():
    raw = """
from pydantic import BaseModel

class LLMOutput(BaseModel):
    question_name: str

class JudgeResult(BaseModel):
    verdict: str
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput", "JudgeResult"))
    assert _count_model_config_in_class(out, "LLMOutput") == 1
    assert _count_model_config_in_class(out, "JudgeResult") == 1
    assert "from pydantic import ConfigDict" in out


def test_postprocess_can_disable_forbid_extra_injection():
    raw = """
from pydantic import BaseModel

class LLMOutput(BaseModel):
    question_name: str
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput",), enforce_forbid_extra=False)
    assert "ConfigDict" not in out
    assert "model_config" not in out


def test_postprocess_appends___all___when_missing():
    raw = """
from pydantic import BaseModel

class LLMOutput(BaseModel):
    question_name: str
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput", "JudgeResult"))
    assert '__all__' in out
    assert '"LLMOutput"' in out
    assert '"JudgeResult"' in out


def test_postprocess_merges___all___preserving_existing_and_adds_required():
    raw = """
from pydantic import BaseModel

class LLMOutput(BaseModel):
    question_name: str

__all__ = ["LLMOutput"]
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput", "JudgeResult"))
    # keeps LLMOutput and adds JudgeResult
    assert '__all__' in out
    assert '"LLMOutput"' in out
    assert '"JudgeResult"' in out


def test_postprocess_noop_on_empty_input():
    assert postprocess_schema_py("") == ""
    assert postprocess_schema_py("   \n  ") == ""


def test_postprocess_injects_literal_import_when_used():
    raw = """
from pydantic import BaseModel
from typing import Optional

class JudgeResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    score: Optional[int] = None
"""
    out = postprocess_schema_py(raw, required_exports=("JudgeResult",))
    assert "from typing import Literal" in out


def test_postprocess_does_not_inject_literal_when_already_imported():
    raw = """
from pydantic import BaseModel
from typing import Literal, Optional

class JudgeResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    score: Optional[int] = None
"""
    out = postprocess_schema_py(raw, required_exports=("JudgeResult",))
    # Should not duplicate the import
    assert out.count("from typing import Literal") == 1


def test_postprocess_does_not_inject_literal_when_not_used():
    raw = """
from pydantic import BaseModel

class LLMOutput(BaseModel):
    name: str
"""
    out = postprocess_schema_py(raw, required_exports=("LLMOutput",))
    assert "Literal" not in out
