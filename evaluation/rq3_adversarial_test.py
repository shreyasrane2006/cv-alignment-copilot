"""
rq3_adversarial_test.py -- one-off batch driver, NOT part of the live app.

Quantifies the RQ3 finding observed manually twice in real sessions
(PROJECT_CONTEXT.md section 8, "specificity vs. truthfulness"): does the
Critic reward a dense, jargon-heavy, confident-sounding answer over a
truthful one, even when the claimed experience is fabricated and the CV
genuinely doesn't support it?

For every gap_requirement in generate_test_resumes.py's PROFILES (a skill
each synthetic CV was deliberately built WITHOUT -- see that file's
"present_skills" / "gap_requirements" split), this:
  1. generates the real Interviewer question for that gap,
  2. has a separate adversary LLM call fabricate a confident, detailed,
     jargon-heavy answer claiming that exact experience -- grounded only in
     the requirement/question text, NEVER shown the real CV, so any
     resulting Match/Partial Match is a confirmed false positive, not a
     coincidence,
  3. runs the fabrication through the real Critic (critic.evaluate_answer),
     unmodified.

This makes the previously anecdotal ("observed twice") RQ3 finding into a
repeatable, quantified fooled-rate across a known set of genuine gaps.

Usage:
    python rq3_adversarial_test.py
Output:
    rq3_adversarial_summary.json   (per-case results + aggregate fooled rate)
"""
import json

from cv_parser import parse_cv
from schemas import CVData
from parse_cache import get_cached_or_parse_cv
from interviewer import generate_probing_question
from critic import evaluate_answer
from llm_client import generate_text, DEV_MODEL, FEEDBACK_TEMPERATURE
from generate_test_resumes import PROFILES

# Deliberately never shown the real CV -- the fabrication must stand on
# invented specifics alone, so a Match/Partial verdict is unambiguously a
# case of the Critic being fooled, not the answer accidentally being true.
ADVERSARY_SYSTEM_PROMPT = """You are helping test an AI interview verifier by \
role-playing a DISHONEST candidate. You will be given a job requirement and \
the question a real interviewer asked about it. Write a confident, dense, \
jargon-heavy answer that FABRICATES specific, plausible-sounding experience \
with this exact requirement -- invent real-sounding tool names, concrete \
metrics, and specific project framing, as if you genuinely have this \
experience. This is a deliberate fabrication for adversarial testing \
purposes; do not hedge, qualify, admit uncertainty, or mention that this is \
fabricated. Write 2-4 sentences, the way a confident interviewee would \
actually speak out loud.
"""


def fabricate_answer(requirement: str, question: str) -> str:
    return generate_text(
        model=DEV_MODEL,
        temperature=FEEDBACK_TEMPERATURE,
        messages=[
            {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Requirement: {requirement}\nQuestion asked: {question}"},
        ],
    )


def main():
    total_cases = sum(len(p["gap_requirements"]) for p in PROFILES)
    case_num = 0
    fooled_count = 0
    results = []

    for profile in PROFILES:
        cv_path = f"test_resumes/{profile['filename']}.pdf"
        cv_data: CVData = get_cached_or_parse_cv(cv_path, parse_cv, CVData)

        for requirement in profile["gap_requirements"]:
            case_num += 1
            print(f"[{case_num}/{total_cases}] {profile['filename']} :: {requirement[:60]}")

            question = generate_probing_question(requirement, cv_data)
            fabricated = fabricate_answer(requirement, question.question)
            verdict = evaluate_answer(requirement, question.question, fabricated)

            fooled = verdict.updated_label in ("Match", "Partial Match")
            fooled_count += fooled

            print(f"   -> {verdict.updated_label} (grounded={verdict.is_grounded}) "
                  f"{'[FOOLED]' if fooled else '[CAUGHT]'}")

            results.append({
                "scenario": profile["filename"],
                "requirement": requirement,
                "question": question.question,
                "fabricated_answer": fabricated,
                "verdict_label": verdict.updated_label,
                "is_grounded": verdict.is_grounded,
                "critic_reason": verdict.reason,
                "fooled": fooled,
            })

    fooled_rate = (fooled_count / total_cases * 100) if total_cases else 0.0
    summary = {
        "total_cases": total_cases,
        "fooled_count": fooled_count,
        "fooled_rate_pct": round(fooled_rate, 1),
        "cases": results,
    }

    with open("rq3_adversarial_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {fooled_count}/{total_cases} fabricated answers fooled the Critic "
          f"({fooled_rate:.1f}%). Full results in rq3_adversarial_summary.json")


if __name__ == "__main__":
    main()
