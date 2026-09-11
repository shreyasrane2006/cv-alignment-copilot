"""
ats_score.py — two deterministic, non-LLM scores that run alongside (not
instead of) the semantic matcher: a naive literal keyword-overlap score, and
a CV structural-completeness score. Both are plain Python over data already
in memory (parsed CVData/JDData) — no new I/O, no new model calls, so they
add microseconds, not time, wherever they're called from.

Inspired by (not copied from) the "AI Powered CV Checker" dissertation's
description of CareerSet-style resume checkers — but built as our own
mechanism: the keyword score exists specifically to CONTRAST against the
semantic matcher (naive ATS keyword-matching vs. grounded semantic
entailment), and the completeness score reuses the CVData object our own
cv_parser.py already produces rather than re-extracting via a separate NER
pass. Categories, weights, and word lists here are our own, not CareerSet's.
"""
import re

from schemas import CVData
from matcher_schemas import RequirementMatch

_STOPWORDS = {
    "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "be", "this", "that", "as", "by", "at", "from", "will", "your", "you",
    "the", "if", "it", "its", "their", "such", "etc", "e", "g", "i", "including",
}
# Words that are technically content words (not grammatical stopwords) but
# are so generic in job-posting/resume boilerplate that counting them as a
# "keyword hit" produces false positives — e.g. "Docker experience" and
# "AWS experience" both contain "experience", which appears in nearly every
# resume regardless of whether Docker or AWS actually do. Excluding these
# keeps the naive score naive-but-honest rather than trivially inflated.
_GENERIC_FILLER = {
    "experience", "experienced", "application", "applications", "role", "roles",
    "requirement", "requirements", "years", "year", "work", "working", "worked",
    "ability", "knowledge", "understanding", "familiarity", "familiar",
    "skill", "skills", "environment", "environments", "strong", "solid",
    "proven", "excellent", "good", "team", "teams", "including", "related",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS and w not in _GENERIC_FILLER}


def matched_keywords(text_a: str, text_b: str) -> list[str]:
    """Shared with the GUI's RAG-transparency panel — literal token overlap
    between a JD requirement and a retrieved CV evidence chunk, for keyword
    highlighting. Same tokenizer/filler-exclusion as keyword_match_score()."""
    return sorted(_tokenize(text_a) & _tokenize(text_b))


def _flatten_cv_text(cv: CVData) -> str:
    parts = [cv.personal_summary or ""]
    parts += cv.skills
    for e in cv.experience:
        parts.append(f"{e.role} {e.company}")
        parts += e.description
    for p in cv.projects:
        parts.append(f"{p.name} {p.description}")
    parts += cv.certifications
    parts += cv.achievements
    return " ".join(parts)


def keyword_match_score(requirements: list[str], cv: CVData) -> dict:
    """
    Naive literal ATS-style keyword overlap — no embeddings, no LLM,
    deliberately simple. This is the naive baseline the semantic matcher is
    being contrasted against, not a replacement for it: it just checks
    whether any significant word from a requirement appears literally
    anywhere in the CV text, with no understanding of synonyms or context.
    """
    cv_tokens = _tokenize(_flatten_cv_text(cv))
    per_requirement = []
    hits = 0
    for req in requirements:
        matched = sorted(_tokenize(req) & cv_tokens)
        found = len(matched) > 0
        hits += found
        per_requirement.append({"requirement": req, "keyword_hit": found, "matched_terms": matched})

    score = round(hits / len(requirements) * 100, 1) if requirements else 0.0
    return {"score": score, "per_requirement": per_requirement}


def completeness_score(cv: CVData) -> dict:
    """
    Structural completeness — does the CV have the sections/fields an ATS or
    recruiter expects to find at all? Reuses the CVData our own cv_parser.py
    already produced; no new extraction pass. Presence/absence only, not a
    judgment of quality or truthfulness — stays inside the deterministic,
    non-LLM half of this feature by construction.
    """
    checks = [
        ("Professional summary present", bool(cv.personal_summary and len(cv.personal_summary.strip()) > 20), 15),
        ("At least one education entry", len(cv.education) >= 1, 10),
        ("At least 5 listed skills", len(cv.skills) >= 5, 15),
        ("At least one work experience entry", len(cv.experience) >= 1, 20),
        ("Experience entries include dates", bool(cv.experience) and all(e.dates for e in cv.experience), 10),
        ("Experience entries have detail bullets", bool(cv.experience) and all(len(e.description) >= 1 for e in cv.experience), 15),
        ("At least one contact/profile link (LinkedIn, GitHub, portfolio)", len(cv.links) >= 1, 10),
        ("Projects or certifications listed", (len(cv.projects) + len(cv.certifications)) >= 1, 5),
    ]
    total_weight = sum(w for _, _, w in checks)
    earned = sum(w for _, passed, w in checks if passed)
    score = round(earned / total_weight * 100, 1) if total_weight else 0.0
    return {
        "score": score,
        "checks": [{"name": name, "passed": passed, "weight": weight} for name, passed, weight in checks],
    }
