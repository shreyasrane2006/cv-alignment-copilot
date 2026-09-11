"""
Semantic Matcher Agent — for each JD requirement, retrieves the most
relevant CV evidence via ChromaDB, then asks the local LLM to judge
Match/Partial/No Match grounded only in that retrieved evidence.

NOTE ON DESIGN: this was briefly batched into a single call returning a
list of RequirementMatch objects, to cut down on call count. In practice
that nested-list schema caused Ollama's grammar-constrained JSON decoder
to hang indefinitely (confirmed: got stuck for 10+ minutes on a single
requirement, versus ~9s for a simple unconstrained call). Reverted to one
call per requirement with a SIMPLE (non-list) schema — slower in call
count, but each call is small, fast (~10-15s), and predictable. 15
requirements at ~12s each is ~3 minutes total, which is far better than
an indefinite hang.
"""
from llm_client import generate_structured, PARSING_MODEL, PARSING_TEMPERATURE
from matcher_schemas import RequirementMatch, MatchResult
from schemas import CVData, JDData
from chroma_store import build_cv_collection, query_evidence
import time

SYSTEM_PROMPT = """You are a precise skill-entailment judge. Given a job \
requirement and a small set of candidate CV evidence snippets (already \
retrieved as the most semantically similar parts of the CV), decide \
whether the CV evidence logically ENTAILS the requirement.

Labels:
- "Match": the evidence clearly and directly demonstrates the requirement.
- "Partial Match": related but doesn't fully/explicitly demonstrate it \
(adjacent skill, different stack achieving a similar outcome, academic \
exposure without hands-on proof).
- "No Match": unrelated, or no relevant evidence among the candidates.

CRITICAL RULES:
- Only use the provided CV evidence snippets. Do not invent experience.
- If none of the candidates are relevant, label "No Match" and set \
best_cv_evidence to null — do not force a weak snippet into Partial Match.
- Keep your reason to 1-2 sentences, referencing the evidence directly.
"""


def match_requirement(requirement: str, evidence_candidates: list[dict]) -> RequirementMatch:
    evidence_text = "\n".join(
        f"- ({c['source']}, similarity={c['similarity']:.2f}) {c['text']}"
        for c in evidence_candidates
    ) or "(no candidate evidence retrieved)"

    return generate_structured(
        model=PARSING_MODEL,
        temperature=PARSING_TEMPERATURE,
        response_model=RequirementMatch,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Job requirement: {requirement}\n\nCandidate CV evidence:\n{evidence_text}"},
        ],
    )


def dedup_requirements(jd: JDData) -> list[str]:
    """Shared with ats_score.py's naive keyword baseline, so both it and the
    semantic matcher score literally the same requirement list — an
    apples-to-apples comparison, not two different lists that happen to be
    similar."""
    raw_requirements = jd.required_skills + jd.preferred_skills + jd.qualifications
    # dedupe while preserving order — the JD parser sometimes puts the same
    # requirement in two fields (e.g. a "0-2 years" line in both
    # required_skills and qualifications), which was silently doubling
    # matching time for no benefit
    requirements = list(dict.fromkeys(raw_requirements))
    if len(requirements) < len(raw_requirements):
        print(f"[info] Deduplicated {len(raw_requirements)} -> {len(requirements)} requirements")
    return requirements


def match_cv_to_jd_stream(cv: CVData, jd: JDData, top_k: int = 3):
    """
    Same matching logic as match_cv_to_jd, but yields (index, total, match,
    evidence) as each RequirementMatch completes instead of only returning
    the final MatchResult — lets a caller (the GUI's SSE endpoint) show
    progress during the ~1-2 min this normally runs blind. total is known
    upfront (post-dedup requirement count), so the consumer doesn't need to
    duplicate the dedup logic just to render "3 of 11". match_cv_to_jd()
    below is just this generator drained into a list, so both share one
    implementation.

    `evidence` is the RAW retrieval candidate list from query_evidence() —
    real, code-computed cosine similarities, not anything the LLM touched.
    Exposed for the GUI's transparency panel: RequirementMatch.similarity_score
    is LLM-*self-reported* (part of its structured JSON output) and shouldn't
    be trusted as the real number — this is the real number.
    """
    collection = build_cv_collection(cv)
    requirements = dedup_requirements(jd)

    print(f"[info] {len(requirements)} requirement(s) to match: {requirements}")

    total = len(requirements)
    overall_start = time.time()
    for i, req in enumerate(requirements, 1):
        evidence = query_evidence(collection, req, top_k)
        start = time.time()
        match = match_requirement(req, evidence)
        elapsed = time.time() - start
        print(f"[timing] ({i}/{total}) '{req[:50]}...' -> {match.match_label} ({elapsed:.1f}s)")
        yield i, total, match, evidence

    print(f"[timing] Total matching time: {time.time() - overall_start:.1f}s for {total} requirement(s)")


def match_cv_to_jd(cv: CVData, jd: JDData, top_k: int = 3) -> MatchResult:
    matches = [match for _, _, match, _ in match_cv_to_jd_stream(cv, jd, top_k)]

    score_map = {"Match": 1.0, "Partial Match": 0.5, "No Match": 0.0}
    overall_score = round(
        (sum(score_map.get(m.match_label, 0.0) for m in matches) / len(matches)) * 100, 1
    ) if matches else 0.0

    return MatchResult(overall_score=overall_score, requirement_matches=matches)


if __name__ == "__main__":
    import sys, json
    from cv_parser import parse_cv
    from jd_parser import parse_jd

    if len(sys.argv) < 3:
        print("Usage: python matcher.py <cv.pdf> <jd.txt>")
        sys.exit(1)

    print("Parsing CV and JD...")
    cv_data = parse_cv(sys.argv[1])
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        jd_data = parse_jd(f.read())

    print("Matching...")
    result = match_cv_to_jd(cv_data, jd_data)
    print(json.dumps(result.model_dump(), indent=2))