"""
The Interactive Feedback Loop — ties together Matcher, Interviewer,
Clarifier, and Critic into a single CLI conversation. This is the
reconciled version combining everything built so far:

  - Concurrent, hash-based cached parsing (parse_cache.py): re-running
    the same CV/JD is near-instant instead of ~50s/~20s. Note: when BOTH
    are cache misses (first-ever run), true parallel speedup on a single
    6GB-VRAM GPU is limited — Ollama serializes requests to one loaded
    model — so concurrency mainly helps when at least one side is
    already cached, which is the common case after the first run.

  - Fatigue-reduction caps: MAX_GAPS_PER_SESSION limits how many gaps
    get probed at all (address the top 3 well, not 8 poorly).
    MAX_FOLLOW_UPS_PER_REQUIREMENT capped at 1 — a second follow-up
    rarely resolved anything in testing. "skip"/"stop" let the user
    bail out gracefully.

  - Two-way conversation: if the student's input looks like a question
    rather than an answer, it's routed to the Clarifier, answered, and
    the SAME original question re-asked — without consuming a follow-up
    attempt. Capped at MAX_CLARIFICATIONS_PER_REQUIREMENT so this can't
    become its own source of fatigue in the other direction.

Usage:
    python feedback_loop.py <cv.pdf> <jd.txt> [max_gaps]
"""
import sys
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from cv_parser import parse_cv
from jd_parser import parse_jd
from schemas import CVData, JDData
from parse_cache import get_cached_or_parse_cv, get_cached_or_parse_jd, get_cached_or_match
from matcher import match_cv_to_jd, dedup_requirements
from matcher_schemas import MatchResult
from ats_score import keyword_match_score, completeness_score
from feedback_generator import generate_feedback_summary
from interviewer import generate_probing_question
from critic import evaluate_answer
from clarifier import answer_general_question
from intent_router import classify_intent
from chroma_store import build_cv_collection

FINISH_WORDS = ("done", "stop", "finish", "exit", "quit")

MAX_GAPS_PER_SESSION = 3
MAX_FOLLOW_UPS_PER_REQUIREMENT = 1
MAX_CLARIFICATIONS_PER_REQUIREMENT = 2

LABEL_SCORE = {"Match": 1.0, "Partial Match": 0.5, "No Match": 0.0}


def recompute_score(requirement_matches: dict) -> float:
    if not requirement_matches:
        return 0.0
    total = sum(LABEL_SCORE.get(v, 0.0) for v in requirement_matches.values())
    return round((total / len(requirement_matches)) * 100, 1)


def handle_requirement(requirement: str, cv_data: CVData, jd_data: JDData, collection, transcript: list, suggestions: list) -> tuple[str | None, bool]:
    """
    Runs the question/clarify/answer cycle for one requirement.
    Returns (final_label_or_None, stop_requested). Appends any
    improvement suggestions generated to the shared `suggestions` list.
    """
    # attempts_used counts EVERY judged answer (the initial question plus any
    # follow-up), so max_attempts = 1 initial + MAX_FOLLOW_UPS_PER_REQUIREMENT
    # follow-ups. Previously this incremented on the initial answer itself and
    # was compared directly against MAX_FOLLOW_UPS_PER_REQUIREMENT, which (a)
    # meant the promised follow-up round never actually happened, and (b) hit
    # the cap by returning None instead of the verdict just computed — silently
    # discarding a real Partial Match verdict (with its suggestion already
    # shown to the student) and leaving current_labels/ΔS un-updated for it.
    attempts_used = 0
    max_attempts = 1 + MAX_FOLLOW_UPS_PER_REQUIREMENT
    clarifications_used = 0
    question_text = None
    grounding_reference = None
    last_label = None
    previous_answer = None
    previous_reason = None
    clar_history = []  # {"question", "answer"} per clarification on THIS
    # requirement only — same fix as qa_history (PROJECT_CONTEXT.md §4.18),
    # applied to the sibling in-gap clarification path it didn't cover: a
    # 2nd clarifying question about the same gap had zero memory of the 1st.

    while attempts_used < max_attempts:
        if question_text is None:
            question = generate_probing_question(
                requirement, cv_data,
                previous_answer=previous_answer,
                previous_reason=previous_reason,
            )
            question_text = question.question
            grounding_reference = question.grounding_reference

        print(f"\n[Interviewer] {question_text}")
        student_input = input("[You] ").strip()

        if student_input.lower() == "stop":
            print("\nEnding session early at your request.")
            return last_label, True
        if not student_input or student_input.lower() == "skip":
            print("(skipped)")
            return last_label, False

        intent = classify_intent(question_text, student_input)
        if intent.intent == "GENERAL_QUESTION" and clarifications_used < MAX_CLARIFICATIONS_PER_REQUIREMENT:
            clarifications_used += 1
            clarification, _evidence = answer_general_question(
                student_input, cv_data, jd_data, collection,
                current_requirement=requirement, current_question_asked=question_text,
                conversation_history=clar_history,
            )
            print(f"[Agent] {clarification}")
            clar_history.append({"question": student_input, "answer": clarification})
            transcript.append({
                "type": "clarification",
                "requirement": requirement,
                "student_question": student_input,
                "agent_clarification": clarification,
                "timestamp": datetime.now().isoformat(),
            })
            continue  # re-ask the SAME question, does not consume a follow-up

        attempts_used += 1
        verdict = evaluate_answer(requirement, question_text, student_input)
        transcript.append({
            "type": "answer",
            "requirement": requirement,
            "question": question_text,
            "grounding_reference": grounding_reference,
            "answer": student_input,
            "verdict_label": verdict.updated_label,
            "is_grounded": verdict.is_grounded,
            "reason": verdict.reason,
            "improvement_suggestion": verdict.improvement_suggestion,
            "turn_number": attempts_used,
            "clarifications_used_this_requirement": clarifications_used,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"[Critic] {verdict.updated_label} — {verdict.reason}")
        print(f"[Suggested CV edit] {verdict.improvement_suggestion}")
        suggestions.append({
            "requirement": requirement,
            "label": verdict.updated_label,
            "suggestion": verdict.improvement_suggestion,
        })
        last_label = verdict.updated_label

        if not verdict.follow_up_needed:
            return last_label, False

        previous_answer = student_input
        previous_reason = verdict.reason
        question_text = None  # a genuine follow-up needs a freshly generated question, informed by the above

    print(f"(reached max follow-ups for '{requirement}', moving on)")
    return last_label, False


def run_feedback_loop(cv_path: str, jd_path: str, max_gaps: int = MAX_GAPS_PER_SESSION):
    print("Parsing CV and JD concurrently...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        cv_future = executor.submit(get_cached_or_parse_cv, cv_path, parse_cv, CVData)
        jd_future = executor.submit(get_cached_or_parse_jd, jd_path, parse_jd, JDData)
        try:
            cv_data = cv_future.result()
        except ValueError:
            print(
                "\nThis CV appears to be scanned or image-only. "
                "Please upload a text-based PDF."
            )
            return
        jd_data = jd_future.result()

    # Deterministic, non-LLM checks — pure Python over data already parsed,
    # so this adds microseconds, not time, regardless of where it's called.
    ats_keywords = keyword_match_score(dedup_requirements(jd_data), cv_data)
    ats_completeness = completeness_score(cv_data)
    print(f"[info] Naive ATS keyword-match score: {ats_keywords['score']}/100 (contrast baseline, not used in dS)")
    print(f"[info] CV structural completeness score: {ats_completeness['score']}/100")

    print("Running static baseline match...")
    baseline_result = get_cached_or_match(cv_data, jd_data, match_cv_to_jd, MatchResult)
    initial_score = baseline_result.overall_score
    print(f"\n=== Baseline (static, one-shot) score: {initial_score}/100 ===\n")
    print(f"=== For comparison: naive keyword-match score: {ats_keywords['score']}/100 ===\n")

    current_labels = {m.requirement: m.match_label for m in baseline_result.requirement_matches}
    transcript = []
    suggestions = []

    print("Generating feedback summary...")
    feedback = generate_feedback_summary(baseline_result)
    print("\n--- Feedback ---")
    print(feedback.overall_narrative)
    print("----------------\n")

    all_gaps = feedback.prioritized_gaps
    gaps = all_gaps[:max_gaps]
    # Built once, reused for both gap-interrogation clarifications and the
    # open Q&A phase below — cheap (embeddings via nomic-embed-text, not an
    # LLM call), but no reason to rebuild it per question.
    collection = build_cv_collection(cv_data)

    stopped_early = False
    if not gaps:
        print("No gaps found — every requirement already matched on the static pass.")
    else:
        skipped_count = len(all_gaps) - len(gaps)
        print(f"Addressing the top {len(gaps)} requirement(s) out of {len(all_gaps)} total gaps"
              f"{f' ({skipped_count} lower-priority gaps skipped this session)' if skipped_count else ''}.")
        print("(Type 'skip' to move on, 'stop' to end early, or just ask a question back if something's unclear.)\n")

        for requirement in gaps:
            updated_label, stop_requested = handle_requirement(requirement, cv_data, jd_data, collection, transcript, suggestions)
            if updated_label:
                current_labels[requirement] = updated_label
            if stop_requested:
                stopped_early = True
                break

    # The MAX_GAPS_PER_SESSION cap (§7 of PROJECT_CONTEXT.md) is evidence-based
    # for being INTERROGATED — real testing showed answer quality degrading
    # after repeated probing rounds. It says nothing about student-initiated
    # Q&A, which has a different fatigue profile (the student is driving, not
    # being probed) — so unless they explicitly stopped early, open the floor
    # instead of ending the session the moment the capped gaps are done.
    if not stopped_early:
        print("\n" + "=" * 50)
        print("All priority gaps addressed. Ask anything else about your CV, this job,")
        print("or general advice — type 'done' whenever you're ready for your summary.")
        print("=" * 50)
        qa_history = []  # {"question", "answer"} per turn — without this, every
        # turn is answered in isolation with no memory of what was already said,
        # which is why "yes" / "which platforms" / "how" follow-ups just got the
        # same generic gap summary re-derived from scratch instead of an actual
        # answer to what was asked (confirmed bug, see PROJECT_CONTEXT.md §8/§4.18)
        while True:
            student_input = input("\n[You] ").strip()
            if not student_input or student_input.lower() in FINISH_WORDS:
                break
            answer, _evidence = answer_general_question(
                student_input, cv_data, jd_data, collection,
                conversation_history=qa_history,
            )
            print(f"[Agent] {answer}")
            qa_history.append({"question": student_input, "answer": answer})
            transcript.append({
                "type": "open_qa",
                "student_question": student_input,
                "agent_answer": answer,
                "timestamp": datetime.now().isoformat(),
            })

    final_score = recompute_score(current_labels)

    if suggestions:
        print("\n" + "=" * 50)
        print("SUGGESTED CV CHANGES — here's what to actually add or edit:")
        print("=" * 50)
        for i, s in enumerate(suggestions, 1):
            print(f"\n{i}. [{s['label']}] {s['requirement']}")
            print(f"   -> {s['suggestion']}")

    print("\n" + "=" * 50)
    print(f"Baseline score:  {initial_score}/100")
    print(f"Final score:     {final_score}/100")
    print(f"Score delta (dS): {round(final_score - initial_score, 1):+.1f}")
    if stopped_early:
        print("(Session ended early by user request)")
    print("=" * 50)

    log = {
        "timestamp": datetime.now().isoformat(),
        "cv_file": cv_path,
        "jd_file": jd_path,
        "initial_score": initial_score,
        "final_score": final_score,
        "score_delta": round(final_score - initial_score, 1),
        "total_gaps_found": len(all_gaps),
        "gaps_addressed_this_session": len(gaps),
        "stopped_early": stopped_early,
        "suggestions": suggestions,
        "transcript": transcript,
        "ats_naive_keyword_score": ats_keywords["score"],
        "ats_completeness_score": ats_completeness["score"],
    }
    out_path = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\nSession log saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feedback_loop.py <cv.pdf> <jd.txt> [max_gaps]")
        sys.exit(1)
    max_gaps_arg = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_GAPS_PER_SESSION
    run_feedback_loop(sys.argv[1], sys.argv[2], max_gaps_arg)