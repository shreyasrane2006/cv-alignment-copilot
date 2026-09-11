# CV Alignment Co-Pilot

A fully local, offline, multi-agent tool that replaces a one-shot CV-to-job-description score with a short interactive conversation. It parses a CV and job description, retrieves the most relevant CV evidence for each requirement using a Retrieval-Augmented Generation (RAG) style pipeline, judges each requirement as **Match**, **Partial Match**, or **No Match** against that evidence only, and then interviews the candidate on its lowest-confidence gaps — crediting real, previously unsurfaced experience where it's provided, and declining to reward vague or ungrounded claims.

Built as the implementation for an MSc dissertation, *"Evaluating an Interactive LLM-Based Feedback Loop for Student CV-to-Job Alignment,"* University of Manchester, 2026.

## Why

Existing CV-checking tools (ATS scanners, Jobscan, CareerSet) score a CV against a job description in a single automated pass. A candidate may hold relevant experience that simply isn't written down in the words the matcher is looking for, and a single-pass tool has no way to ask whether it exists. This project tests whether a grounded, interactive conversation can recover that missing evidence — without just trading one failure mode (missed evidence) for another (rewarding unverifiable confidence).

## How it works

A five-stage pipeline, running entirely on a single consumer GPU (6GB VRAM):

1. **Parse** — CV and job description are parsed into structured data (PyMuPDF + LLM extraction), constrained to never infer content not literally present in either document.
2. **Baseline match** — every job requirement is scored against the CV twice: a deterministic keyword/completeness score (no LLM), and a semantic match where the top-3 most relevant CV passages are retrieved per requirement (ChromaDB + `nomic-embed-text`) and judged for entailment by the LLM — the model never sees any CV text retrieval didn't surface for that specific requirement.
3. **Narrate** — the per-requirement verdicts are narrated into a plain-English summary with a prioritised list of gaps.
4. **Interrogate** — for up to three prioritised gaps, an Interviewer agent asks a targeted question, a Critic agent judges the reply against the requirement (crediting real detail, declining vague answers), and a concrete CV-bullet suggestion is produced.
5. **Open chat** — once the capped gaps are exhausted, the conversation opens into free-form Q&A.

All LLM calls use `qwen3:8b` served locally through [Ollama](https://ollama.com).

## Running it

```bash
pip install -r requirements.txt
ollama pull qwen3:8b
ollama pull nomic-embed-text
python server.py
```

Then open `http://127.0.0.1:8000` in a browser. Or launch it via `.claude/launch.json` if you're using Claude Code's dev-server preview.

## Repository layout

```
server.py, web_session.py       FastAPI backend / session orchestration
matcher.py, critic.py,          The core agent pipeline (retrieval-matcher,
clarifier.py, interviewer.py,   verifying critic, open-Q&A clarifier,
intent_router.py                intent classification)
feedback_loop.py,               Gap prioritisation and narrated feedback
feedback_generator.py
cv_parser.py, jd_parser.py,     Document parsing and caching
pdf_utils.py, parse_cache.py
embeddings.py, chroma_store.py  Retrieval index
ats_score.py                    Deterministic keyword/completeness baseline
llm_client.py                   Ollama client wrapper
static/                         Frontend (single-page HTML/JS)
test_data/                      8 synthetic CV/job-description profiles used
                                 for the RQ1 and RQ3 evaluations
evaluation/                     Scripts and result data for the dissertation's
                                 RQ1 (alignment gain), RQ2 (entailment accuracy
                                 against a golden set), and RQ3 (grounding /
                                 adversarial testing) evaluations
```

## Evaluation

The `evaluation/` folder contains the scripts and output data behind the dissertation's three research questions:

- **RQ1** (does the interactive loop improve alignment?) — `run_rq1_batch.py`, results in `data/rq1_summary.json`.
- **RQ2** (entailment accuracy against human-labelled ground truth) — `golden_set_check.py`, `export_disagreements_for_adjudication.py`, `adjudication_report.py`, data in `data/golden_dataset_gemini.json`, `data/adjudication_*.json`.
- **RQ3** (does grounding reduce hallucination, and can it be gamed?) — `rq3_adversarial_test.py`, `llm_judge_eval.py`, data in `data/rq3_adversarial_summary.json`, `data/llm_judge_summary.json`.

Note: `generate_golden_set_gemini.py` expects a local `./Resume` directory of source CVs used to build the golden set — that directory isn't included in this repo since it's a third-party resume dataset, not original material. `data/golden_dataset_gemini.json` (the generated output) is included.

## Constraints

Zero-cost and fully offline by design — no paid API, no candidate data leaves the machine, and every claim the system makes about a candidate is traceable to either the literal CV text or a specific retrieved passage, never the model's unconstrained recall.
