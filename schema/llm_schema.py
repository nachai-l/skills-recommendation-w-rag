from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict
from typing import Literal

class RecommendedSkill(BaseModel):
    """Represents an individual skill recommendation with supporting evidence."""
    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        ..., 
        description="The canonical name of the recommended skill."
    )
    relevance_score: float = Field(
        ..., 
        description="Score between 0.0 and 1.0 indicating alignment with the query.",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why this skill was matched using the context."
    )
    evidence: List[str] = Field(
        ..., 
        description="Direct phrases or snippets from the context that support this recommendation."
    )

class LLMOutput(BaseModel):
    """The primary structured output for skill recommendations."""
    model_config = ConfigDict(extra="forbid")

    analysis_summary: str = Field(
        ..., 
        description="A concise summary of the overall matching process and findings."
    )
    recommended_skills: List[RecommendedSkill] = Field(
        ..., 
        description="A list of skills recommended based on the user query and context, sorted by relevance."
    )

class JudgeResult(BaseModel):
    """The structured output for evaluating the quality of the generated response."""
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass/fail status of the evaluation."
    )
    score: int = Field(
        ..., 
        description="A numerical quality score from 0 to 100.",
        ge=0,
        le=100
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific reasons or observations supporting the verdict."
    )

__all__ = ["LLMOutput", "JudgeResult"]
