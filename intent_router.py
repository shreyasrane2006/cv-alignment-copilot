"""
intent_router.py — classifies each student chat reply as either
GAP_EVIDENCE (an answer toward the requirement currently being probed ->
routes to the Critic, affects the score) or GENERAL_QUESTION (the student
asking something back -> routes to a grounded RAG answer, never affects
the score).

Replaces clarifier.py's old looks_like_a_question() keyword heuristic
(checked for "?" and a fixed phrase list). That heuristic had a real
failure mode: it couldn't distinguish "Does leading a 3-person team count
as leadership experience?" (a real answer, just phrased as a question)
from "What do you mean by leadership experience?" (an actual clarifying
question) — both contain "?" and question-shaped phrasing. An LLM
classification can use the actual content, not just surface phrasing.
"""
from pydantic import BaseModel, Field

from llm_client import generate_structured, PARSING_MODEL, PARSING_TEMPERATURE

SYSTEM_PROMPT = """You are a message router for a CV feedback interview loop. \
A student was just asked a specific question about a job requirement their \
CV doesn't yet clearly demonstrate. Classify their reply into exactly one \
category:

- "GAP_EVIDENCE": the student is answering the question — describing \
relevant experience, a project, a skill, or directly stating they don't \
have it ("no, I haven't done that"). This includes short, vague, or \
negative answers — judging whether an answer is GOOD evidence is the \
Critic's job, not yours. Your only job is: are they trying to answer?
- "GENERAL_QUESTION": the student is asking something back instead of \
answering — asking what a term means, why something matters for the role, \
for career advice, what they should improve in general, or anything else \
that isn't a direct answer to the question asked.

CRITICAL RULES:
- A reply phrased as a question can still be GAP_EVIDENCE if it's \
substantively answering (e.g. "Does leading a 3-person team on the auth \
migration count as leadership experience?" is GAP_EVIDENCE — they're \
describing real experience and just checking relevance).
- When genuinely ambiguous, prefer GAP_EVIDENCE — losing a follow-up \
attempt is a smaller cost than silently discarding a real answer.
"""


class IntentClassification(BaseModel):
    intent: str = Field(description="Exactly one of: GAP_EVIDENCE, GENERAL_QUESTION")
    reason: str = Field(description="One short sentence explaining the classification")


def classify_intent(question_asked: str, student_input: str) -> IntentClassification:
    result = generate_structured(
        model=PARSING_MODEL,
        temperature=PARSING_TEMPERATURE,
        response_model=IntentClassification,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question asked to the student: {question_asked}\n"
                f"Student's reply: {student_input}"
            )},
        ],
    )
    # Same defensive stance as the "prefer GAP_EVIDENCE when ambiguous" rule
    # above — if the model returns something outside the two known labels
    # (unconstrained JSON, so not impossible), don't silently drop a real
    # answer into a question-handling path.
    if result.intent not in ("GAP_EVIDENCE", "GENERAL_QUESTION"):
        result.intent = "GAP_EVIDENCE"
    return result
