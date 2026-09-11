"""
generate_test_jds.py — one-off test-data prep tool, pairs with
generate_test_resumes.py. NOT part of the live app.

Generates one JD per synthetic resume profile in PROFILES, deliberately
built so the pair does NOT match 100%: the Required/Preferred skill
lists are NOT freely improvised by the LLM (that would risk drifting
away from what's actually in/out of the paired resume). Instead they
are taken directly from each profile's `present_skills` (what the
resume DOES have -> should score Match) and `gap_requirements` (what
the resume was explicitly told to leave out -> should score Partial/No
Match). The LLM is only used to write the surrounding company blurb
and responsibilities narrative, which doesn't affect matching.

Usage:
    python generate_test_jds.py
Output:
    test_jds/<slug>.txt  (one per profile, same slug as test_resumes/<slug>.pdf)
"""
import os

from llm_client import generate_text, DEV_MODEL
from generate_test_resumes import PROFILES

OUTPUT_DIR = "test_jds"
GENERATION_TEMPERATURE = 0.7

SYSTEM_PROMPT = """You are writing the opening narrative section of a realistic \
job posting for a FICTIONAL company, for software-testing purposes only.

CRITICAL RULES:
- Invent a plausible fictional company name.
- Write PLAIN TEXT only — no markdown, no asterisks.
- Output exactly two parts, in this order, each with its own header line \
in capitals:
  COMPANY BLURB (2-3 sentences: what the fictional company does, and a short \
"why join us" hook)
  RESPONSIBILITIES (4-6 bullet lines, each starting with "- ", describing \
day-to-day duties for the role — do NOT list specific tools/technologies \
here, keep it to duties/outcomes, since the required skills are supplied \
separately)
- Do NOT invent your own skills/requirements section — that is handled \
elsewhere. Just the company blurb and responsibilities.
"""


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_jd_text(profile: dict, narrative: str) -> str:
    required = profile["present_skills"] + profile["gap_requirements"]
    preferred = profile["preferred_extra"]

    return (
        f"{narrative.strip()}\n\n"
        f"Qualification\n\n"
        f"Required\n\n"
        f"{_bullets(required)}\n\n"
        f"Preferred\n\n"
        f"{_bullets(preferred)}\n"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, profile in enumerate(PROFILES, 1):
        out_path = os.path.join(OUTPUT_DIR, f"{profile['filename']}.txt")
        print(f"[{i}/{len(PROFILES)}] Generating JD for '{profile['filename']}'...")

        narrative = generate_text(
            model=DEV_MODEL,
            temperature=GENERATION_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Role: {profile['role_title']}\n"
                    f"Write the company blurb and responsibilities for this role."
                )},
            ],
        )

        jd_text = build_jd_text(profile, narrative)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(jd_text)

        n_required = len(profile["present_skills"]) + len(profile["gap_requirements"])
        print(f"   -> wrote {out_path} ({n_required} required, {len(profile['preferred_extra'])} preferred)")

    print(f"\nDone. {len(PROFILES)} synthetic test JDs in '{OUTPUT_DIR}/'.")
    print("Paired with test_resumes/ by matching filename — e.g.:")
    print("  python feedback_loop.py test_resumes\\ai_ml_engineer.pdf test_jds\\ai_ml_engineer.txt")


if __name__ == "__main__":
    main()
