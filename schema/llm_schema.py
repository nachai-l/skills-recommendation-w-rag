from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict

class SkillItem(BaseModel):
    """Represents an individual skill recommendation with supporting evidence."""
    model_config = ConfigDict(extra="forbid")
    
    skill_name: str = Field(
        ..., 
        description="The canonical name of the recommended skill."
    )
    relevance_score: float = Field(
        ..., 
        description="A score between 0.0 and 1.0 representing the strength of the match.",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        ..., 
        description="Justification for why this skill was selected based on the user query."
    )
    evidence: List[str] = Field(
        ..., 
        description="Direct snippets or phrases from the retrieved context that support the recommendation."
    )

class LLMOutput(BaseModel):
    """The structured response containing skill intelligence analysis and recommendations."""
    model_config = ConfigDict(extra="forbid")
    
    analysis_summary: str = Field(
        ..., 
        description="A concise summary of the skill matching analysis performed."
    )
    recommended_skills: List[SkillItem] = Field(
        ..., 
        description="A list of skills identified as relevant, sorted by relevance score."
    )

class JudgeResult(BaseModel):
    """The outcome of an automated evaluation of the LLM generation quality."""
    model_config = ConfigDict(extra="forbid")
    
    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The binary pass/fail result of the judge evaluation."
    )
    score: int = Field(
        ..., 
        description="A quality score from 0 to 100.",
        ge=0,
        le=100
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific justifications for the given verdict and score."
    )

__all__ = ["LLMOutput", "JudgeResult", "SkillItem"]
