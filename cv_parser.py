"""
CV Parser Agent — converts raw CV text into structured CVData.

Key design choice, straight from a hard lesson learned earlier in this
project: the system prompt explicitly forbids inventing information.
This matters because an ungrounded LLM will happily "fill in" plausible
details (a technology, a metric, a tool) that were never actually in
the document — exactly the failure mode this whole project's Critic
module exists to catch. Better to prevent it here at the source too.
"""
from schemas import CVData
from llm_client import generate_structured, PARSING_MODEL, PARSING_TEMPERATURE
from pdf_utils import extract_text_from_pdf
import time

SYSTEM_PROMPT = """You are a precise CV/resume parser. Extract structured \
information from the CV text into the given schema.

CRITICAL RULES:
- Only extract information that is EXPLICITLY present in the text.
- Do NOT infer, assume, or invent any skill, tool, metric, or detail \
that is not literally written in the CV.
- If a section is not present, leave the corresponding field empty — \
do not guess or fabricate a plausible-sounding substitute.
- The raw text may have sections in a slightly scrambled order due to \
PDF layout extraction (e.g. a header appearing after its own content). \
Use your judgment to correctly group content under the right section \
regardless of raw text order — but do not use this as license to \
invent content that isn't there.
- For the "skills" field: extract every individual skill, tool, \
language, or technology mentioned ANYWHERE in the CV (summary, \
experience bullets, projects, dedicated skills section), as a flat \
deduplicated list.
"""


def parse_cv(pdf_path: str) -> CVData:
    """Extract raw text from a CV PDF and structure it via the local LLM."""
    raw_text = extract_text_from_pdf(pdf_path)

    start = time.time()
    result = generate_structured(
        model=PARSING_MODEL,
        temperature=PARSING_TEMPERATURE,
        response_model=CVData,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CV text to parse:\n\n{raw_text}"},
        ],
    )
    print(f"[timing] CV parsing took {time.time() - start:.1f}s")
    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python cv_parser.py <path_to_cv.pdf>")
        sys.exit(1)

    parsed = parse_cv(sys.argv[1])
    print(json.dumps(parsed.model_dump(), indent=2))