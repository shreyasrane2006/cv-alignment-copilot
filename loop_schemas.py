"""
Schemas for the interactive feedback loop: the Interviewer Agent's
probing questions, and the Critic Agent's verdict on the user's answer.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ProbingQuestion(BaseModel):
    requirement: str = Field(description="The JD requirement this question is trying to verify")
    question: str = Field(description="The open-ended question to ask the student")
    grounding_reference: Optional[str] = Field(
        default=None,
        description="The CV snippet this question is anchored to, if any (e.g. an adjacent project/role). Null if asking cold."
    )


class CriticVerdict(BaseModel):
    requirement: str
    updated_label: str = Field(description="One of: Match, Partial Match, No Match")
    is_grounded: bool = Field(description="True only if the student's answer contains concrete, checkable detail")
    reason: str = Field(description="1-2 sentences explaining the verdict")
    follow_up_needed: bool = Field(description="True if the answer was vague/ungrounded and needs another probing round")
    improvement_suggestion: str = Field(
        description="A concrete, actionable suggestion for what to actually ADD TO THE CV — a specific "
                    "bullet point or phrase the student could write, based on what they just said in chat. "
                    "This is the single most important field: it's what the student walks away and edits "
                    "their CV with. Always provide one, even for a full Match (suggest how to make it even "
                    "more explicit/quantified on the actual document)."
    )