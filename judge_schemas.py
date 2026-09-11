"""
Schema for the LLM-as-judge evaluation (llm_judge_eval.py).
"""
from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    question_quality_score: int = Field(
        description="1-5: how well-targeted was the Interviewer's probing question? "
                    "5 = specific, genuinely open-ended, anchored to the candidate's real CV context if any, "
                    "doesn't presume the answer. 1 = generic checklist question, yes/no phrased, or "
                    "disconnected from the actual requirement."
    )
    question_quality_reason: str = Field(description="1-2 sentences justifying the question score")
    suggestion_quality_score: int = Field(
        description="1-5: how concrete/actionable was the improvement_suggestion? "
                    "5 = a specific CV bullet built only from details the candidate actually gave, correctly "
                    "calibrated to the verdict. 1 = vague generic advice, or invents details the candidate "
                    "never said."
    )
    suggestion_quality_reason: str = Field(description="1-2 sentences justifying the suggestion score")
