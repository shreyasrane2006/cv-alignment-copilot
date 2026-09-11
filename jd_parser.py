"""
Job Description Parser Agent — converts raw pasted JD text into
structured JDData (required skills, preferred skills, responsibilities).
"""
from schemas import JDData
from llm_client import generate_structured, PARSING_MODEL, PARSING_TEMPERATURE
import time

SYSTEM_PROMPT = """You are a precise job description parser. Extract \
structured information from the job posting text into the given schema.

CRITICAL RULES:
- Only extract requirements/responsibilities EXPLICITLY stated in the text.
- Distinguish carefully between "required" (must-have, "you should have", \
"we require") and "preferred" (nice-to-have, "desirable", "a plus") — \
do not put a preferred skill under required or vice versa.
- Do not invent or infer skills that aren't mentioned.
- Break multi-skill sentences into individual list items where sensible \
(e.g. "Python, SQL and Git" becomes three separate entries).
"""


def parse_jd(jd_text: str) -> JDData:
    """Structure raw job description text via the local LLM."""
    start = time.time()
    result = generate_structured(
        model=PARSING_MODEL,
        temperature=PARSING_TEMPERATURE,
        response_model=JDData,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Job description text to parse:\n\n{jd_text}"},
        ],
    )
    print(f"[timing] JD parsing took {time.time() - start:.1f}s")
    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python jd_parser.py <path_to_jd.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        jd_text = f.read()

    parsed = parse_jd(jd_text)
    print(json.dumps(parsed.model_dump(), indent=2))