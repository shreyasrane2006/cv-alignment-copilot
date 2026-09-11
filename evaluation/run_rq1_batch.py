"""
run_rq1_batch.py — one-off batch driver, NOT part of the live app.

Runs feedback_loop.py against all 8 synthetic test_resumes/test_jds pairs to
collect a real RQ1 (ΔS) dataset now that the scoring bugs (§4.10/§4.11 in
PROJECT_CONTEXT.md) are fixed. Scripted as a realistic student would actually
type: short, brief answers (not long detailed paragraphs), with a diverging
clarifying question interjected before every gap to stress-test the
Clarifier's "re-ask the same original question, don't lose state" behavior
rather than just answering straight through.

Usage:
    python run_rq1_batch.py
Output:
    rq1_run_<scenario>.log   (full stdout per scenario)
    rq1_summary.json         (aggregated ΔS results across all 8)
"""
import glob
import json
import os
import subprocess
import time

# feedback_loop.py's own print() calls (e.g. em-dashes in "[Critic] Label —
# reason") default to Windows' console codepage (cp1252) as a *subprocess*
# with no real console attached, which is valid cp1252 but not valid UTF-8 —
# capturing with subprocess.run(..., encoding="utf-8") then crashes on
# decode. Forcing PYTHONIOENCODING=utf-8 on the child makes it emit real
# UTF-8 instead, matching what the parent expects.
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# Generic, topic-agnostic clarifying/divergent questions — cycled per gap so
# every gap gets at least one "confuse the agent" interjection before being
# answered, regardless of which specific requirement is being asked about.
CLARIFY_A = "Sorry, what exactly do you mean by that?"
CLARIFY_B = "Why does that matter for this role?"

# Short, brief, honest answers per scenario — matches how a real student
# would actually type (a sentence or two), not the long detailed paragraphs
# used in earlier ad-hoc testing this session. Two answers per scenario are
# enough since the pattern cycles them across however many gap/follow-up
# rounds actually occur.
SCENARIO_ANSWERS = {
    "ai_ml_engineer": [
        "No, I haven't used LangChain or any agent framework.",
        "Not yet, I've only called the OpenAI API directly.",
    ],
    "backend_developer": [
        "No, I've never used Docker or deployed to the cloud.",
        "Everything I've built has just run locally so far.",
    ],
    "data_scientist": [
        "No deep learning experience, just scikit-learn models.",
        "I haven't deployed any models to production yet.",
    ],
    "devops_engineer": [
        "No, I've never touched Kubernetes or Terraform.",
        "Just Docker and some bash scripts, nothing beyond that.",
    ],
    "react_frontend_developer": [
        "No, I've only written plain JavaScript, not TypeScript.",
        "I haven't written any automated tests for my projects.",
    ],
    "financial_accountant": [
        "No, I've only used Excel, never SAP or Oracle.",
        "I haven't managed a team, just handled my own tasks.",
    ],
    "mechanical_engineer": [
        "No, I haven't used ANSYS or done FEA work.",
        "I don't hold any Six Sigma certification.",
    ],
    "hr_generalist": [
        "No, I haven't handled compensation or benefits directly.",
        "I've mostly focused on recruiting, not performance reviews.",
    ],
}


def build_stdin_lines(answers: list[str]) -> list[str]:
    a1, a2 = answers
    # cycled pattern: clarify, answer, clarify, answer... covers up to
    # 3 gaps x (2 clarifications + 2 answer rounds) = 12 lines worst case
    # for the capped interrogation phase.
    pattern = [CLARIFY_A, a1, CLARIFY_B, a2] * 3
    # CONFIRMED, FIXED bug (2026-08-20, PROJECT_CONTEXT.md 4.26): once the
    # interrogation phase ends, feedback_loop.py opens an UNCAPPED open Q&A
    # phase that only exits on a FINISH_WORD ("done"/"stop"/...) -- the old
    # buffer here was just 4 filler lines with no FINISH_WORD in it at all,
    # so it relied entirely on the interrogation phase happening to consume
    # exactly the right number of lines to land past the buffer before
    # input() ever got called again. That held by luck in earlier runs; once
    # recent fixes changed how many turns a session actually takes, every
    # scenario crashed with EOFError instead of completing. Padded with a
    # generous number of harmless filler lines (covers even a long open Q&A
    # phase) and "done" is now GUARANTEED to be the final line, so the
    # session always terminates cleanly regardless of exact turn counts.
    buffer = [a1, a2, "no", "not really", "no thanks", "nothing else"] * 4
    return pattern + buffer + ["done"]


def main():
    scenarios = list(SCENARIO_ANSWERS.keys())
    results = []

    for i, name in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] Running {name}...")
        lines = build_stdin_lines(SCENARIO_ANSWERS[name])
        stdin_text = "\n".join(lines) + "\n"

        existing_logs = set(glob.glob("session_log_*.json"))
        start = time.time()

        proc = subprocess.run(
            ["python", "feedback_loop.py", f"test_resumes/{name}.pdf", f"test_jds/{name}.txt"],
            input=stdin_text, capture_output=True, text=True, encoding="utf-8",
            env=_SUBPROCESS_ENV,
        )
        elapsed = time.time() - start

        with open(f"rq1_run_{name}.log", "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            if proc.returncode != 0:
                f.write("\n--- STDERR ---\n" + proc.stderr)

        new_logs = set(glob.glob("session_log_*.json")) - existing_logs
        session_log_path = sorted(new_logs)[-1] if new_logs else None

        entry = {
            "scenario": name,
            "exit_code": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "session_log": session_log_path,
        }

        if session_log_path:
            with open(session_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            entry.update({
                "initial_score": log_data["initial_score"],
                "final_score": log_data["final_score"],
                "score_delta": log_data["score_delta"],
                "gaps_addressed": log_data["gaps_addressed_this_session"],
                "total_gaps_found": log_data["total_gaps_found"],
                "stopped_early": log_data["stopped_early"],
                "num_clarifications": sum(
                    1 for t in log_data["transcript"] if t["type"] == "clarification"
                ),
            })
            print(f"   -> dS = {entry['score_delta']:+.1f} "
                  f"({entry['initial_score']} -> {entry['final_score']}), "
                  f"{entry['num_clarifications']} clarification(s), {elapsed:.0f}s")
        else:
            print(f"   -> FAILED (exit {proc.returncode}), see rq1_run_{name}.log")

        results.append(entry)

    with open("rq1_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} scenarios run. Summary saved to rq1_summary.json")


if __name__ == "__main__":
    main()
