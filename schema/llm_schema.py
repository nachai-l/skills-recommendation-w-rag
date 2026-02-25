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
        ge=0.0, 
        le=1.0, 
        description="A score between 0.0 and 1.0 indicating the strength of the match."
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why this skill is relevant based on the context."
    )
    evidence: List[str] = Field(
        ..., 
        description="Direct phrases or snippets from the context that support this recommendation."
    )

class LLMOutput(BaseModel):
    """The structured output for the skill intelligence engine generation."""
    model_config = ConfigDict(extra="forbid")

    analysis_summary: str = Field(
        ..., 
        description="A concise summary of the overall matching analysis (2-5 sentences)."
    )
    recommended_skills: List[RecommendedSkill] = Field(
        ..., 
        description="A list of skills recommended based on the user query and context, sorted by relevance."
    )

class JudgeResult(BaseModel):
    """The structured output for evaluating the quality of a generation."""
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass/fail judgment of the evaluation."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="A numeric score between 0 and 100 representing the quality of the output."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific justifications for the assigned verdict and score."
    )

__all__ = ["LLMOutput", "JudgeResult"]
