from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict
from typing import Literal

class SkillRecommendation(BaseModel):
    """Schema for a single recommended skill and its supporting evidence."""
    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(..., description="The canonical name of the skill.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score from 0.0 to 1.0.")
    reasoning: str = Field(..., description="Explanation of how the skill aligns with the query.")
    evidence: List[str] = Field(..., description="Direct snippets or phrases from the context.")

class LLMOutput(BaseModel):
    """Structured output for skill intelligence generation."""
    model_config = ConfigDict(extra="forbid")

    analysis_summary: str = Field(..., description="Concise summary of the matching analysis.")
    recommended_skills: List[SkillRecommendation] = Field(..., description="Sorted list of relevant skills.")

class JudgeResult(BaseModel):
    """Result of the quality evaluation phase."""
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(..., description="Final pass or fail determination.")
    score: int = Field(..., ge=0, le=100, description="Numeric quality score from 0 to 100.")
    reasons: List[str] = Field(default_factory=list, description="Specific reasons for the verdict.")

__all__ = ["LLMOutput", "JudgeResult", "SkillRecommendation"]
