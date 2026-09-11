"""
web_session.py — session-state layer for the GUI backend (server.py).

feedback_loop.py's handle_requirement() blocks on input() in a while loop,
which works for a single CLI process but can't serve concurrent HTTP
requests. This module reimplements the SAME state machine (identical caps,
identical transitions, identical agent calls) as explicit steps driven by
HTTP requests instead of blocking stdin — every actual AI-agent call
(parsing, matching, feedback narrative, interviewer, critic, clarifier) is
imported and reused unchanged from the existing modules. Only the
control-flow shell is rewritten, because it has to be.
"""
import os
import uuid
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from schemas import CVData, JDData
from cv_parser import parse_cv
from jd_parser import parse_jd
from parse_cache import (
    get_cached_or_parse_cv, get_cached_or_parse_jd, check_match_cache, write_match_cache,
    is_cv_cached, is_jd_cached,
)
from matcher import match_cv_to_jd_stream, dedup_requirements
from matcher_schemas import MatchResult
from ats_score import keyword_match_score, completeness_score, matched_keywords
from feedback_generator import generate_feedback_summary
from interviewer import generate_probing_question
from critic import evaluate_answer
from clarifier import answer_general_question
from intent_router import classify_intent
from chroma_store import build_cv_collection
from feedback_loop import (
    FINISH_WORDS,
    MAX_GAPS_PER_SESSION,
    MAX_FOLLOW_UPS_PER_REQUIREMENT,
    MAX_CLARIFICATIONS_PER_REQUIREMENT,
    recompute_score,
)

UPLOAD_DIR = "uploads"


class ScannedPdfError(Exception):
    """Raised when the uploaded CV PDF has no extractable text (scanned/image-only)."""


@dataclass
class Session:
    session_id: str
    cv_data: CVData
    jd_data: JDData
    cv_filename: str
    # Set later by run_analysis_stream() once matching/feedback complete —
    # empty/zero defaults here so create_session_pending() can construct the
    # Session right after parsing, before analysis has run.
    baseline_result: Optional[MatchResult] = None
    current_labels: dict = field(default_factory=dict)
    initial_score: float = 0.0
    ats_keyword_score: float = 0.0
    ats_completeness_score: float = 0.0
    # Revised-CV re-check (student re-uploads their edited CV against the
    # SAME jd_data) — set by add_revised_cv()/run_revised_analysis_stream()
    revised_cv_data: Optional[CVData] = None
    revised_result: Optional[MatchResult] = None
    feedback_narrative: str = ""
    gaps: list = field(default_factory=list)
    total_gaps: int = 0
    gap_index: int = -1
    transcript: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    stopped_early: bool = False
    done: bool = False
    current_requirement: Optional[str] = None
    current_question: Optional[str] = None
    current_grounding: Optional[str] = None
    attempts_used: int = 0
    max_attempts: int = 0
    clarifications_used: int = 0
    previous_answer: Optional[str] = None
    previous_reason: Optional[str] = None
    # True once the capped gap-interrogation phase is done (or there were no
    # gaps at all) and hasn't been explicitly stopped — every message from
    # here is a student-initiated question, not a Critic-scored answer.
    open_qa: bool = False
    _qa_collection: object = field(default=None, repr=False)


_sessions: dict[str, Session] = {}

# Each request against a session mutates shared, in-memory state (current
# gap, attempts_used, current_labels, ...) across several seconds of LLM
# calls. Nothing previously stopped two concurrent requests for the SAME
# session from both reading the pre-mutation state and both applying their
# own mutation — confirmed in practice: a double-submitted answer produced
# two identical verdicts and two identical "gaps addressed" transitions back
# to back. A per-session lock, acquired non-blocking, turns a genuine
# duplicate into a rejected request instead of a silently duplicated effect.
_session_locks: dict[str, threading.Lock] = {}


class SessionBusyError(Exception):
    """Raised when a second request arrives for a session that's still
    processing a previous one — the caller should reject it, not queue it
    (queueing would still double-process the same input once the lock frees)."""


def _get_lock(session_id: str) -> threading.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = threading.Lock()
        _session_locks[session_id] = lock
    return lock


def try_acquire_session_lock(session_id: str) -> bool:
    """For the two SSE streaming endpoints (server.py) — StreamingResponse
    commits its HTTP status before the generator body ever runs, so raising
    SessionBusyError *inside* the generator (as submit_answer does) can't be
    turned into a clean 409 anymore by the time it happens. The route handler
    must check this BEFORE constructing the StreamingResponse instead."""
    return _get_lock(session_id).acquire(blocking=False)


def release_session_lock(session_id: str) -> None:
    _get_lock(session_id).release()


def _get_qa_collection(session: Session):
    """Built once per session, reused for every clarification/open-Q&A
    question — cheap (embeddings via nomic-embed-text, not an LLM call), but
    no reason to rebuild it on every message."""
    if session._qa_collection is None:
        session._qa_collection = build_cv_collection(session.cv_data)
    return session._qa_collection


def _requirements_table(baseline_result: MatchResult) -> list[dict]:
    return [
        {
            "requirement": m.requirement,
            "match_label": m.match_label,
            "best_cv_evidence": m.best_cv_evidence,
            "reason": m.reason,
        }
        for m in baseline_result.requirement_matches
    ]


def _session_summary(session: Session) -> dict:
    final_score = recompute_score(session.current_labels)
    return {
        "initial_score": session.initial_score,
        "final_score": final_score,
        "score_delta": round(final_score - session.initial_score, 1),
        "total_gaps_found": session.total_gaps,
        "gaps_addressed_this_session": len(session.gaps),
        "stopped_early": session.stopped_early,
        "suggestions": session.suggestions,
        "ats_naive_keyword_score": session.ats_keyword_score,
        "ats_completeness_score": session.ats_completeness_score,
    }


def _finalize_session(session: Session) -> dict:
    session.done = True
    summary = _session_summary(session)
    log = {
        "timestamp": datetime.now().isoformat(),
        "cv_file": session.cv_filename,
        "initial_score": session.initial_score,
        "final_score": summary["final_score"],
        "score_delta": summary["score_delta"],
        "total_gaps_found": session.total_gaps,
        "gaps_addressed_this_session": len(session.gaps),
        "stopped_early": session.stopped_early,
        "suggestions": session.suggestions,
        "transcript": session.transcript,
        "ats_naive_keyword_score": session.ats_keyword_score,
        "ats_completeness_score": session.ats_completeness_score,
    }
    out_path = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return summary


def _advance_to_next_gap(session: Session) -> dict:
    """Moves to the next gap in session.gaps, generates its opening question,
    resets per-requirement state. If no gaps remain, opens the floor for
    student-initiated Q&A instead of ending the session — the
    MAX_GAPS_PER_SESSION cap is evidence-based for being INTERROGATED, not
    for how long a student-driven conversation should be allowed to run."""
    session.gap_index += 1
    if session.gap_index >= len(session.gaps):
        session.open_qa = True
        return {
            "type": "open_qa",
            "message": (
                "All priority gaps addressed. Ask anything else about your CV, "
                "this job, or general advice — type 'done' whenever you're ready "
                "for your summary."
            ),
        }

    requirement = session.gaps[session.gap_index]
    session.current_requirement = requirement
    session.attempts_used = 0
    session.max_attempts = 1 + MAX_FOLLOW_UPS_PER_REQUIREMENT
    session.clarifications_used = 0
    session.previous_answer = None
    session.previous_reason = None

    question = generate_probing_question(requirement, session.cv_data)
    session.current_question = question.question
    session.current_grounding = question.grounding_reference

    return {
        "type": "question",
        "requirement": requirement,
        "question": session.current_question,
        "gap_number": session.gap_index + 1,
        "total_gaps_this_session": len(session.gaps),
    }


_pending_uploads: dict[str, dict] = {}


def save_pending_upload(cv_path: str, jd_text: str, cv_filename: str) -> str:
    """Just persists the uploaded file + JD text and hands back a pending_id —
    no parsing yet. Split out from the actual parsing (run_upload_stream)
    because browser EventSource can only do GET requests, not a multipart
    file upload, so the file has to land on disk via a quick POST first and
    the slow part (LLM parsing) streams separately over GET."""
    jd_path = os.path.join(UPLOAD_DIR, f"jd_{uuid.uuid4().hex}.txt")
    with open(jd_path, "w", encoding="utf-8") as f:
        f.write(jd_text)

    pending_id = uuid.uuid4().hex
    _pending_uploads[pending_id] = {"cv_path": cv_path, "jd_path": jd_path, "cv_filename": cv_filename}
    return pending_id


def run_upload_stream(pending_id: str):
    """Generator yielding real progress events for the CV+JD parsing phase —
    same underlying work create_session_pending used to do in one blocking
    call, now surfaced step by step (mirrors run_analysis_stream's approach
    to the matching phase below) instead of the GUI showing a single static
    "this can take a minute" message for the whole thing. Cache-hit status
    is checked BEFORE each parse starts so the message is honest about
    whether it'll be instant or genuinely running the local LLM — not a
    simulated/timed animation."""
    pending = _pending_uploads.pop(pending_id, None)
    if pending is None:
        yield ("upload_failed", {"message": "Upload session expired or already used — please try again."})
        return

    cv_path, jd_path, cv_filename = pending["cv_path"], pending["jd_path"], pending["cv_filename"]

    yield ("parsing_started", {
        "cv_cached": is_cv_cached(cv_path),
        "jd_cached": is_jd_cached(jd_path),
    })

    cv_data = None
    jd_data = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(get_cached_or_parse_cv, cv_path, parse_cv, CVData): "cv",
                executor.submit(get_cached_or_parse_jd, jd_path, parse_jd, JDData): "jd",
            }
            for future in as_completed(futures):
                which = futures[future]
                if which == "cv":
                    try:
                        cv_data = future.result()
                    except ValueError as e:
                        raise ScannedPdfError(
                            "This CV appears to be scanned or image-only. Please upload a text-based PDF."
                        ) from e
                    yield ("cv_parsed", {})
                else:
                    jd_data = future.result()
                    yield ("jd_parsed", {})
    except ScannedPdfError as e:
        yield ("upload_failed", {"message": str(e)})
        return
    except Exception as e:
        # Anything else (Ollama not running, a retry-exhausted LLM parse,
        # ...) must still become a clean event, not a silently crashed
        # stream -- EventSource's onerror alone gives zero information about
        # WHY, and "Ollama isn't running" is a real, likely failure mode for
        # an offline-first app that depends on a local service being up.
        yield ("upload_failed", {"message": f"Parsing failed: {e}"})
        return

    # Deterministic, non-LLM checks — pure Python over data already parsed
    # above, so this adds microseconds, not time, to the request.
    ats_keywords = keyword_match_score(dedup_requirements(jd_data), cv_data)
    ats_completeness = completeness_score(cv_data)

    session = Session(
        session_id=uuid.uuid4().hex,
        cv_data=cv_data,
        jd_data=jd_data,
        cv_filename=cv_filename,
        ats_keyword_score=ats_keywords["score"],
        ats_completeness_score=ats_completeness["score"],
    )
    _sessions[session.session_id] = session

    yield ("upload_complete", {
        "session_id": session.session_id,
        "ats_naive_keyword_score": session.ats_keyword_score,
        "ats_completeness_score": session.ats_completeness_score,
    })


def _stream_match(cv_data: CVData, jd_data: JDData):
    """
    Shared by run_analysis_stream (baseline) and run_revised_analysis_stream
    (revised-CV re-check) — yields ("requirement_matched", data) events, one
    per requirement, then returns the final MatchResult (via generator return,
    retrieved with `yield from`).

    On a cache MISS, each event's "retrieved_evidence" carries the REAL
    retrieval candidates (code-computed cosine similarity, never touched by
    the LLM) plus literal keyword overlap, for the GUI's transparency panel.
    On a cache HIT, raw retrieval candidates were never stored (only the
    final judged MatchResult is cached), so "retrieved_evidence" is empty —
    re-deriving them would mean re-running embeddings just for display,
    which costs real time on what's supposed to be the instant path. The
    tradeoff: transparency detail is only available on a fresh match.
    """
    score_map = {"Match": 1.0, "Partial Match": 0.5, "No Match": 0.0}
    cached = check_match_cache(cv_data, jd_data, MatchResult)

    if cached is not None:
        matches = cached.requirement_matches
        for i, m in enumerate(matches, 1):
            running_score = round(
                sum(score_map.get(x.match_label, 0.0) for x in matches[:i]) / i * 100, 1
            )
            yield ("requirement_matched", {
                "requirement": m.requirement, "match_label": m.match_label,
                "best_cv_evidence": m.best_cv_evidence, "reason": m.reason,
                "progress": i, "total": len(matches), "running_score": running_score,
                "retrieved_evidence": [],
            })
        return cached

    matches = []
    for i, total, m, evidence in match_cv_to_jd_stream(cv_data, jd_data):
        matches.append(m)
        running_score = round(
            sum(score_map.get(x.match_label, 0.0) for x in matches) / len(matches) * 100, 1
        )
        yield ("requirement_matched", {
            "requirement": m.requirement, "match_label": m.match_label,
            "best_cv_evidence": m.best_cv_evidence, "reason": m.reason,
            "progress": i, "total": total, "running_score": running_score,
            "retrieved_evidence": [
                {
                    "text": e["text"], "source": e["source"],
                    "similarity": round(e["similarity"], 3),
                    "matched_terms": matched_keywords(m.requirement, e["text"]),
                }
                for e in evidence
            ],
        })

    overall_score = round(
        sum(score_map.get(m.match_label, 0.0) for m in matches) / len(matches) * 100, 1
    ) if matches else 0.0
    result = MatchResult(overall_score=overall_score, requirement_matches=matches)
    write_match_cache(cv_data, jd_data, result)
    return result


def run_analysis_stream(session: Session):
    """
    Generator yielding (event_name, data) tuples: baseline match (streamed
    per-requirement so the frontend can fill in the score ring/table live),
    then the feedback narrative, then the first gap's question. Mirrors
    run_feedback_loop's setup phase, just broken into progressively-emitted
    steps instead of one blocking call.
    """
    baseline_result = yield from _stream_match(session.cv_data, session.jd_data)
    session.baseline_result = baseline_result
    session.initial_score = baseline_result.overall_score
    session.current_labels = {m.requirement: m.match_label for m in baseline_result.requirement_matches}

    yield ("analysis_complete", {
        "initial_score": session.initial_score,
        "requirements": _requirements_table(baseline_result),
    })

    feedback = generate_feedback_summary(baseline_result)
    all_gaps = feedback.prioritized_gaps
    gaps = all_gaps[:MAX_GAPS_PER_SESSION]
    session.feedback_narrative = feedback.overall_narrative
    session.gaps = gaps
    session.total_gaps = len(all_gaps)

    yield ("narrative_ready", {
        "narrative": feedback.overall_narrative,
        "total_gaps": len(all_gaps),
        "gaps_this_session": len(gaps),
    })

    if not gaps:
        session.open_qa = True
        chat_event = {
            "type": "open_qa",
            "message": (
                "No gaps found — every requirement already matched on the static "
                "pass. Ask anything about your CV, this job, or general advice — "
                "type 'done' whenever you're ready for your summary."
            ),
        }
    else:
        chat_event = _advance_to_next_gap(session)

    yield ("chat_ready", chat_event)


def add_revised_cv(session: Session, cv_path: str) -> None:
    """Parses the student's re-uploaded, manually-edited CV (cached like any
    other CV parse). Does NOT touch session.cv_data/current_labels/etc — this
    is a separate, independent re-check against the same jd_data, not a
    continuation of the original chat session."""
    try:
        cv_data = get_cached_or_parse_cv(cv_path, parse_cv, CVData)
    except ValueError as e:
        raise ScannedPdfError(
            "This CV appears to be scanned or image-only. Please upload a text-based PDF."
        ) from e
    session.revised_cv_data = cv_data


def run_revised_analysis_stream(session: Session):
    """
    Generator yielding (event_name, data) tuples for the revised-CV re-check:
    matches the student's edited CV against the SAME jd_data (streamed like
    the baseline pass), then a final event with the three-way comparison —
    Baseline (original CV, static pass) / Co-Pilot Simulated (what the chat
    session's verdicts implied) / True Revised (an independent, real re-match
    of the actual edited document — not simulated).
    """
    revised_result = yield from _stream_match(session.revised_cv_data, session.jd_data)
    session.revised_result = revised_result

    simulated_score = recompute_score(session.current_labels) if session.current_labels else session.initial_score

    yield ("revised_complete", {
        "baseline_score": session.initial_score,
        "simulated_score": simulated_score,
        "true_revised_score": revised_result.overall_score,
        "requirements": _requirements_table(revised_result),
    })


def get_session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Unknown session_id: {session_id}")
    return session


def submit_answer(session: Session, student_input: str) -> dict:
    """Public entry point — rejects a genuinely concurrent duplicate request
    for the same session instead of letting it silently double-process (see
    _session_locks above). Real logic is _submit_answer_impl()."""
    lock = _get_lock(session.session_id)
    if not lock.acquire(blocking=False):
        raise SessionBusyError(f"Session {session.session_id} is already processing a message.")
    try:
        return _submit_answer_impl(session, student_input)
    finally:
        lock.release()


def _submit_answer_impl(session: Session, student_input: str) -> dict:
    """Mirrors handle_requirement()'s per-turn logic in feedback_loop.py, one
    HTTP round-trip per turn instead of one input() call per turn."""
    if session.done:
        return {"type": "done", "session_summary": _session_summary(session)}

    student_input = student_input.strip()

    if session.open_qa:
        if not student_input or student_input.lower() in FINISH_WORDS:
            return {"type": "done", "session_summary": _finalize_session(session)}
        # Without this, every open-Q&A turn is answered in isolation with no
        # memory of what was already said — confirmed bug (PROJECT_CONTEXT.md
        # §4.18): "yes"/"which platforms"/"how" follow-ups just got the same
        # generic gap summary re-derived from scratch each time.
        history = [
            {"question": t["student_question"], "answer": t["agent_answer"]}
            for t in session.transcript if t["type"] == "open_qa"
        ]
        answer, evidence = answer_general_question(
            student_input, session.cv_data, session.jd_data, _get_qa_collection(session),
            conversation_history=history,
        )
        session.transcript.append({
            "type": "open_qa",
            "student_question": student_input,
            "agent_answer": answer,
            "timestamp": datetime.now().isoformat(),
        })
        return {"type": "general_answer", "answer": answer, "evidence": evidence}

    requirement = session.current_requirement

    if student_input.lower() == "stop":
        session.stopped_early = True
        return {"type": "done", "session_summary": _finalize_session(session)}

    if not student_input or student_input.lower() == "skip":
        return _advance_to_next_gap(session)

    intent = classify_intent(session.current_question, student_input)
    if intent.intent == "GENERAL_QUESTION" and session.clarifications_used < MAX_CLARIFICATIONS_PER_REQUIREMENT:
        session.clarifications_used += 1
        # Same fix as the open_qa history above (PROJECT_CONTEXT.md §4.18),
        # applied to the sibling in-gap clarification path: clarifications_used
        # resets to 0 per new requirement (see gap-init below), so filtering
        # the transcript by requirement scopes this to just the current gap's
        # clarifications, mirroring qa_history's per-requirement scope.
        clar_history = [
            {"question": t["student_question"], "answer": t["agent_clarification"]}
            for t in session.transcript
            if t["type"] == "clarification" and t["requirement"] == requirement
        ]
        clarification, evidence = answer_general_question(
            student_input, session.cv_data, session.jd_data, _get_qa_collection(session),
            current_requirement=requirement, current_question_asked=session.current_question,
            conversation_history=clar_history,
        )
        session.transcript.append({
            "type": "clarification",
            "requirement": requirement,
            "student_question": student_input,
            "agent_clarification": clarification,
            "timestamp": datetime.now().isoformat(),
        })
        return {
            "type": "clarification",
            "clarification": clarification,
            "evidence": evidence,
            "question": session.current_question,  # same question re-shown, no follow-up consumed
        }

    session.attempts_used += 1
    verdict = evaluate_answer(requirement, session.current_question, student_input)
    session.transcript.append({
        "type": "answer",
        "requirement": requirement,
        "question": session.current_question,
        "grounding_reference": session.current_grounding,
        "answer": student_input,
        "verdict_label": verdict.updated_label,
        "is_grounded": verdict.is_grounded,
        "reason": verdict.reason,
        "improvement_suggestion": verdict.improvement_suggestion,
        "turn_number": session.attempts_used,
        "clarifications_used_this_requirement": session.clarifications_used,
        "timestamp": datetime.now().isoformat(),
    })
    session.suggestions.append({
        "requirement": requirement,
        "label": verdict.updated_label,
        "suggestion": verdict.improvement_suggestion,
    })
    session.current_labels[requirement] = verdict.updated_label

    verdict_payload = {
        "requirement": requirement,
        "label": verdict.updated_label,
        "reason": verdict.reason,
        "suggestion": verdict.improvement_suggestion,
        "current_score": recompute_score(session.current_labels),
    }

    if not verdict.follow_up_needed or session.attempts_used >= session.max_attempts:
        next_event = _advance_to_next_gap(session)
        return {"type": "verdict", **verdict_payload, "next": next_event}

    session.previous_answer = student_input
    session.previous_reason = verdict.reason
    follow_up = generate_probing_question(
        requirement, session.cv_data,
        previous_answer=session.previous_answer,
        previous_reason=session.previous_reason,
    )
    session.current_question = follow_up.question
    session.current_grounding = follow_up.grounding_reference

    return {
        "type": "verdict",
        **verdict_payload,
        "next": {
            "type": "question",
            "requirement": requirement,
            "question": session.current_question,
            "gap_number": session.gap_index + 1,
            "total_gaps_this_session": len(session.gaps),
        },
    }
