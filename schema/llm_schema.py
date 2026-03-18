from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict
from typing import Literal

class SkillRecommendation(BaseModel):
    """Represents an individual skill recommendation with supporting metadata."""
    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        ..., 
        description="The canonical name of the recommended skill."
    )
    relevance_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Score representing the alignment between the skill and the query."
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why this skill was recommended based on evidence."
    )
    evidence: List[str] = Field(
        ..., 
        description="Direct snippets from the context that support this recommendation."
    )

class LLMOutput(BaseModel):
    """Main output schema for skill intelligence generation."""
    model_config = ConfigDict(extra="forbid")

    analysis_summary: str = Field(
        ..., 
        description="A concise summary of the skill matching results and logic."
    )
    recommended_skills: List[SkillRecommendation] = Field(
        ..., 
        description="A list of skills recommended based on the user query and context."
    )

class JudgeResult(BaseModel):
    """Schema for the evaluation output of a judge model."""
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The binary evaluation result."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="The numerical assessment score from 0 to 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of justifications for the verdict and score."
    )

__all__ = ["LLMOutput", "JudgeResult", "SkillRecommendation"]
