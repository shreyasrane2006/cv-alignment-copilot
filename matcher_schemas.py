from typing import Optional, List
from pydantic import BaseModel, Field


class RequirementMatch(BaseModel):
    requirement: str
    match_label: str = Field(description="One of: Match, Partial Match, No Match")
    best_cv_evidence: Optional[str] = Field(
        default=None, description="The single most relevant CV snippet supporting this judgment, or null if none exists"
    )
    similarity_score: Optional[float] = Field(default=None, description="Cosine similarity of the best-matching CV chunk")
    reason: str = Field(description="One or two sentence explanation, referencing the evidence or its absence")


class MatchResult(BaseModel):
    overall_score: float = Field(description="0-100: (Match count*1 + Partial*0.5) / total requirements * 100")
    requirement_matches: List[RequirementMatch]