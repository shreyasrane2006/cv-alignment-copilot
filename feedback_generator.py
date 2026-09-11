"""
Feedback Generator Agent — the missing piece between the Matcher and the
interactive loop. Takes the already-computed MatchResult and narrates it
as a readable paragraph, plus decides which gaps matter most and in what
order the interactive loop should probe them.

This does NOT make new judgments — it only narrates judgments the
Matcher already made, using only the evidence/reasons already attached
to each RequirementMatch. That keeps it grounded by construction: there's
nothing new here for it to hallucinate about.
"""
import difflib
import os
from typing import List
from pydantic import BaseModel, Field

from llm_client import generate_structured, DEV_MODEL, FEEDBACK_TEMPERATURE
import time
from matcher_schemas import MatchResult


class FeedbackSummary(BaseModel):
    overall_narrative: str = Field(
        description="2-4 sentence paragraph summarizing overall fit: what's clearly strong, "
                    "what's partially there, what's missing. Written directly to the candidate, second person."
    )
    prioritized_gaps: List[str] = Field(
        description="Requirement strings (exact text from the input) for anything not a full Match, "
                    "ordered from most important/central to the role to least important. "
                    "This becomes the order the interactive follow-up questions get asked in."
    )


SYSTEM_PROMPT = """You are writing the opening feedback message a student \
sees right after their CV is scored against a job description.

CRITICAL RULES:
- Only narrate the judgments and reasons you are given — do not introduce \
any new claim, skill, or detail not already present in the match data.
- Write directly to the candidate ("Your background shows strong evidence \
of X, but Y isn't clearly demonstrated yet...").
- Be specific, not generic — reference the actual requirements, not vague \
categories like "some skills are missing."
- For prioritized_gaps: rank by how central the requirement seems to the \
role (e.g. a core required skill outranks a minor preferred one), using \
only the requirements provided — do not add or drop any.
- Keep the tone constructive and direct, not falsely encouraging and not \
harsh — the goal is clarity, not flattery.
"""


def _reconcile_gaps(prioritized_gaps: List[str], valid_requirements: List[str]) -> List[str]:
    """
    Like every other unconstrained-JSON call in this project, the LLM is
    asked to copy requirement strings verbatim into prioritized_gaps but
    sometimes paraphrases instead — observed pattern: it keeps the leading
    label but rewrites the text after a colon with its own summary (e.g.
    "Bachelor's degree in X: Thorough understanding of Y" becomes
    "Bachelor's degree in X: No mention of a relevant degree was found").
    The interactive loop keys current_labels by the ORIGINAL requirement
    string from the matcher, so an unreconciled paraphrase silently creates
    a brand-new dict key instead of updating the real one — inflating the
    score denominator with phantom duplicate "gaps" and corrupting ΔS.
    Map every returned gap back to its nearest real requirement before it's
    ever used as a key.
    """
    reconciled = []
    seen = set()
    for gap in prioritized_gaps:
        if gap in valid_requirements:
            match = gap
        else:
            close = difflib.get_close_matches(gap, valid_requirements, n=1, cutoff=0.3)
            if close:
                match = close[0]
            else:
                # fallback for the "keep label, replace description" pattern,
                # which can score below difflib's overall-similarity cutoff
                # when the replacement text is long and very different
                match, best_len = gap, 0
                for candidate in valid_requirements:
                    prefix_len = len(os.path.commonprefix([gap, candidate]))
                    if prefix_len > best_len and prefix_len >= 10:
                        match, best_len = candidate, prefix_len
        if match not in seen:
            reconciled.append(match)
            seen.add(match)
    return reconciled


def generate_feedback_summary(match_result: MatchResult) -> FeedbackSummary:
    matches_text = "\n".join(
        f"- Requirement: {m.requirement}\n"
        f"  Label: {m.match_label}\n"
        f"  Evidence: {m.best_cv_evidence or '(none found)'}\n"
        f"  Reason: {m.reason}"
        for m in match_result.requirement_matches
    )

    start = time.time()
    result = generate_structured(
        model=DEV_MODEL,
        temperature=FEEDBACK_TEMPERATURE,
        response_model=FeedbackSummary,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Overall score: {match_result.overall_score}/100\n\n"
                f"Requirement-by-requirement results:\n{matches_text}"
            )},
        ],
    )
    print(f"[timing] Feedback generation took {time.time() - start:.1f}s")

    valid_requirements = [m.requirement for m in match_result.requirement_matches]
    reconciled = _reconcile_gaps(result.prioritized_gaps, valid_requirements)
    if reconciled != result.prioritized_gaps:
        print(f"[info] Reconciled {len(result.prioritized_gaps)} -> {len(reconciled)} paraphrased gap string(s) back to original requirements")
    result.prioritized_gaps = reconciled

    return result