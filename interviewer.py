"""
Interviewer Agent — for a given Partial/No Match requirement, generates
a single targeted, open-ended probing question. Anchors the question in
the candidate's real CV content where a plausible connection exists,
rather than asking cold ("Do you know Docker?").

Uses qwen3:8b (via DEV_MODEL) at temperature 0.2 — a little natural
variation is fine here since this agent only asks questions, it never
makes verdicts. (qwen3:4b was tried and rejected — see PROJECT_CONTEXT.md
§4.2 — DEV_MODEL has pointed at qwen3:8b since; this comment was stale.)
"""
from llm_client import generate_structured, DEV_MODEL, FEEDBACK_TEMPERATURE
import time
from loop_schemas import ProbingQuestion
from schemas import CVData

SYSTEM_PROMPT = """You are an interview co-pilot helping a student uncover \
hidden experience that might satisfy a job requirement their CV doesn't \
clearly demonstrate yet.

CRITICAL RULES:
- Ask exactly ONE open-ended question about the missing/weak requirement.
- If the candidate's CV contains a plausibly related project, role, or \
experience, anchor your question to it specifically (e.g. "You mentioned \
X — did that involve Y?"). This makes the question feel personalized, \
not like a generic checklist item.
- If nothing in the CV is even loosely related, ask a direct, honest \
question without inventing a false connection.
- Do NOT phrase the question in a way that assumes the answer is yes.
- Do NOT ask a yes/no question — phrase it to invite a real, detailed answer.
- Never fabricate a CV detail to reference. Only use what's actually there.
- If you are told this is a FOLLOW-UP (a previous answer + why it fell \
short is provided), you MUST NOT repeat the same question or ask for the \
same thing again in different words — this is a common failure mode for \
you specifically, so watch for it: rewording the original question with \
different verbs ("stored and handled safely" -> "stored and handled \
properly") is STILL a repeat, not a real follow-up. Pick ONE concrete, \
specific, checkable detail named in "why it fell short" (a standard's \
name, a certification, a number, a specific tool or checklist) and build \
your new question around asking for exactly that one thing.
- Example of a BAD follow-up: original question was "how did you ensure \
hygiene and compliance?", it fell short for "lacks specifics on food \
safety standards or measurable outcomes"; BAD follow-up: "Can you describe \
a time you ensured compliance with safety standards?" — same question, \
reworded, still broad.
- Example of a GOOD follow-up for that same case: "Was there a specific \
food safety standard or checklist you followed or were trained on — for \
example HACCP — and can you name it?" — asks for ONE concrete, checkable \
detail instead of repeating the broad original ask.
"""


def generate_probing_question(
    requirement: str,
    cv: CVData,
    previous_answer: str | None = None,
    previous_reason: str | None = None,
) -> ProbingQuestion:
    cv_context = (
        f"Summary: {cv.personal_summary}\n"
        f"Experience: {[(e.role, e.company, e.description) for e in cv.experience]}\n"
        f"Projects: {[(p.name, p.description) for p in cv.projects]}\n"
        f"Skills: {cv.skills}"
    )

    user_content = (
        f"Requirement the CV doesn't clearly demonstrate: {requirement}\n\n"
        f"Candidate's CV content:\n{cv_context}"
    )
    if previous_answer:
        user_content += (
            f"\n\nThis is a FOLLOW-UP question — the candidate already answered once:\n"
            f"Their previous answer: {previous_answer}\n"
            f"Why it fell short: {previous_reason}\n\n"
            f"Pick ONE specific, concrete detail out of \"why it fell short\" above "
            f"(a named standard, tool, certification, metric, or outcome) and write a "
            f"new question that asks ONLY for that one detail. A reworded version of "
            f"the original broad question is NOT acceptable — that is the exact "
            f"failure mode to avoid."
        )

    start = time.time()
    result = generate_structured(
        model=DEV_MODEL,
        temperature=FEEDBACK_TEMPERATURE,
        response_model=ProbingQuestion,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    print(f"[timing] Question generation took {time.time() - start:.1f}s")
    return result