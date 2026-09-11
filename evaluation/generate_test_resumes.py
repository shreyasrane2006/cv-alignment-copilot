"""
generate_test_resumes.py — one-off test-data prep tool, NOT part of the
live app. Uses the same local qwen3:8b (already running via Ollama, zero
cost, offline) to write realistic-but-fictional resumes across a spread
of sectors, then renders them as digital-text PDFs with fpdf2.

Why this exists: the CVs in Resume/ (Kaggle "Resume Dataset") are all
scanned-image PDFs with zero extractable text — confirmed via PyMuPDF,
not usable by the local pipeline, which deliberately has no OCR support
(see PROJECT_CONTEXT.md §6). Rather than add OCR to the live app, we
generate synthetic resumes instead: guaranteed real digital text, zero
new dependency on the live pipeline, no privacy concern (fictional
people), and we get to deliberately calibrate each one to have a mix of
clear matches, partial matches, and clean gaps against a target JD —
useful for exercising the interactive loop's probing behavior.

Usage:
    python generate_test_resumes.py
Output:
    test_resumes/<slug>.pdf  (one per profile below)
"""
import os
import re
import unicodedata

from fpdf import FPDF

from llm_client import generate_text, DEV_MODEL
from pdf_utils import extract_text_from_pdf

OUTPUT_DIR = "test_resumes"
GENERATION_TEMPERATURE = 0.8  # creative writing, not a judgment task — higher variety is fine here

SYSTEM_PROMPT = """You are writing a realistic, complete, single-page resume \
for a FICTIONAL candidate, for software-testing purposes only.

CRITICAL RULES:
- Invent a plausible fictional name, and realistic-sounding (but fictional) \
company names, project names, and dates.
- Write PLAIN TEXT only — no markdown, no asterisks, no tables. Use simple \
section headers in capitals (e.g. "SKILLS", "EXPERIENCE") each on its own line.
- Use plain ASCII punctuation only: a hyphen "-" for bullets and dashes, \
straight quotes, no em-dashes, no smart quotes, no bullet-point characters.
- Include these sections, in this order: a short professional summary, \
EDUCATION, SKILLS (a flat comma or line separated list of specific named \
tools/technologies/languages), EXPERIENCE (2-3 roles, each with 2-4 bullet \
lines describing concrete tasks/outcomes), PROJECTS (1-2), and CERTIFICATIONS \
if relevant to the field (omit the section entirely if none fit naturally).
- Be specific: name real tools/technologies/methods for the given field \
rather than vague descriptions, so the content is checkable.
- Keep total length roughly 400-600 words — realistic for one page.
"""


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _sanitize_for_core_font(text: str) -> str:
    """fpdf2's built-in core fonts (Helvetica) only support latin-1. Normalize
    common 'smart' punctuation the model might slip in despite instructions,
    then drop anything else outside latin-1 rather than letting it crash the PDF write."""
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "•": "-",
        " ": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1")


def write_resume_pdf(resume_text: str, out_path: str) -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    clean_text = _sanitize_for_core_font(resume_text)
    for line in clean_text.split("\n"):
        # w=0 ("use remaining width") doesn't reliably reset across successive
        # multi_cell calls in fpdf2 — explicit width + x-reset avoids a spurious
        # "not enough horizontal space" exception on later lines.
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 6, line)
    pdf.output(out_path)


PROFILES = [
    {
        "filename": "ai_ml_engineer",
        "role_title": "AI/ML Engineer",
        "brief": (
            "Entry-level AI/ML Engineer candidate, 0-1 years experience, recent CS "
            "graduate. Strong in Python, has built a couple of personal LLM/chatbot "
            "projects using OpenAI's API and basic prompt engineering, comfortable "
            "with Git and REST APIs. Do NOT mention LangChain, CrewAI, AutoGPT, "
            "vector databases, or RAG anywhere — those should be genuine gaps for "
            "this candidate, not just weakly implied."
        ),
        # present_skills / gap_requirements feed generate_test_jds.py — kept in
        # sync with the "Do NOT mention" clauses above so the paired JD's
        # requirements are guaranteed (not just likely) to produce a mix of
        # Match and No/Partial Match, rather than trusting free LLM generation
        # on both sides to happen to line up.
        "present_skills": ["Python", "OpenAI API integration", "Git version control", "REST APIs", "Prompt engineering fundamentals"],
        "gap_requirements": ["Experience with LangChain, CrewAI, or AutoGPT agent frameworks", "Experience with vector databases and semantic search (e.g. Pinecone, Chroma, FAISS)", "Practical experience building Retrieval-Augmented Generation (RAG) pipelines"],
        "preferred_extra": ["Familiarity with asynchronous Python programming", "Experience fine-tuning open-source LLMs"],
    },
    {
        "filename": "backend_developer",
        "role_title": "Backend Developer",
        "brief": (
            "Backend Developer, 2 years experience, strong in Python/Django, REST "
            "API design, PostgreSQL, and Git. Do NOT mention Docker, Kubernetes, "
            "AWS/Azure/GCP, or any containerization/cloud deployment — this "
            "candidate has never touched deployment infrastructure."
        ),
        "present_skills": ["Python", "Django framework", "REST API design", "PostgreSQL", "Git"],
        "gap_requirements": ["Containerization experience with Docker", "Experience deploying applications on AWS, Azure, or GCP", "Familiarity with Kubernetes orchestration"],
        "preferred_extra": ["Experience with CI/CD pipelines", "Message queue experience (RabbitMQ or Kafka)"],
    },
    {
        "filename": "data_scientist",
        "role_title": "Data Scientist",
        "brief": (
            "Data Scientist, 3 years experience, strong in Python, pandas, "
            "scikit-learn, SQL, and building classical ML models (regression, "
            "random forests). Only briefly, vaguely mentions 'some exposure to "
            "neural networks in coursework' — do not give real deep learning "
            "project experience or mention PyTorch/TensorFlow by name."
        ),
        "present_skills": ["Python", "pandas", "scikit-learn", "SQL", "classical ML models (regression, random forests)"],
        "gap_requirements": ["Hands-on deep learning experience with PyTorch or TensorFlow", "Experience deploying ML models to production", "Big data tools such as Spark or Hadoop"],
        "preferred_extra": ["Experience with cloud ML platforms (SageMaker or Vertex AI)", "A/B testing and experimentation design"],
    },
    {
        "filename": "devops_engineer",
        "role_title": "DevOps Engineer",
        "brief": (
            "DevOps Engineer, 3 years experience, strong in Docker, CI/CD pipelines "
            "(GitHub Actions), Linux administration, and bash scripting. Has used "
            "AWS EC2 and S3 directly but has NOT used Kubernetes or Terraform — "
            "leave those out entirely as clean gaps."
        ),
        "present_skills": ["Docker", "CI/CD pipelines (GitHub Actions)", "Linux administration", "Bash scripting", "AWS EC2 and S3"],
        "gap_requirements": ["Kubernetes cluster management", "Infrastructure-as-Code with Terraform", "Configuration management with Ansible or Chef"],
        "preferred_extra": ["Monitoring/observability tools (Prometheus, Grafana)", "Experience with multi-cloud environments"],
    },
    {
        "filename": "react_frontend_developer",
        "role_title": "Frontend Developer",
        "brief": (
            "Frontend Developer, 2 years experience, strong in JavaScript, React, "
            "HTML/CSS, and REST API integration. Do NOT mention TypeScript, Next.js, "
            "or any testing frameworks (Jest, Cypress) — genuine gaps."
        ),
        "present_skills": ["JavaScript", "React", "HTML/CSS", "REST API integration"],
        "gap_requirements": ["TypeScript", "Automated testing with Jest or Cypress", "Experience with Next.js or server-side rendering"],
        "preferred_extra": ["State management libraries (Redux or Zustand)", "Accessibility (WCAG) best practices"],
    },
    {
        "filename": "financial_accountant",
        "role_title": "Staff Accountant",
        "brief": (
            "Staff Accountant, 4 years experience, strong in GAAP compliance, "
            "account reconciliation, month-end close, and advanced Excel. Do NOT "
            "mention SAP, Oracle Financials, or any team management/supervisory "
            "experience — this candidate is strictly an individual contributor."
        ),
        "present_skills": ["GAAP compliance", "account reconciliation", "month-end close processes", "advanced Excel"],
        "gap_requirements": ["Hands-on experience with SAP Financial Accounting (FI) modules or Oracle Financials", "Experience managing or supervising a team of accountants", "Experience with financial forecasting/budgeting for departmental operations"],
        "preferred_extra": ["Professional certification (CPA or equivalent)", "Experience supporting external audits"],
    },
    {
        "filename": "mechanical_engineer",
        "role_title": "Mechanical Design Engineer",
        "brief": (
            "Mechanical Design Engineer, 3 years experience, strong in SolidWorks, "
            "AutoCAD, and tolerance stack-up analysis for manufactured parts. Do NOT "
            "mention Six Sigma, Lean Manufacturing certification, or FEA/simulation "
            "software (ANSYS) — leave those as clean gaps."
        ),
        "present_skills": ["SolidWorks", "AutoCAD", "tolerance stack-up analysis", "manufacturing drawings"],
        "gap_requirements": ["FEA/simulation experience with ANSYS or similar", "Six Sigma or Lean Manufacturing certification", "Experience with DFM/DFA for high-volume production"],
        "preferred_extra": ["GD&T certification", "Experience with PLM systems"],
    },
    {
        "filename": "hr_generalist",
        "role_title": "HR Generalist",
        "brief": (
            "HR Generalist, 3 years experience, strong in full-cycle recruiting, "
            "employee onboarding, and running an HRIS system (e.g. Workday) for "
            "record-keeping. Do NOT mention compensation & benefits administration "
            "or labor law/compliance auditing experience — genuine gaps."
        ),
        "present_skills": ["full-cycle recruiting", "employee onboarding", "HRIS administration (e.g. Workday)"],
        "gap_requirements": ["Compensation and benefits administration experience", "Labor law compliance and audit experience", "Experience designing performance management programs"],
        "preferred_extra": ["SHRM-CP or PHR certification", "Experience with DEI initiatives"],
    },
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, profile in enumerate(PROFILES, 1):
        out_path = os.path.join(OUTPUT_DIR, f"{profile['filename']}.pdf")
        print(f"[{i}/{len(PROFILES)}] Generating '{profile['filename']}'...")

        resume_text = generate_text(
            model=DEV_MODEL,
            temperature=GENERATION_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Candidate profile to write as a resume:\n{profile['brief']}"},
            ],
        )

        write_resume_pdf(resume_text, out_path)

        # verify it round-trips through the SAME extraction the live app uses
        extracted_len = len(extract_text_from_pdf(out_path).strip())
        print(f"   -> wrote {out_path} ({extracted_len} extractable chars)")

    print(f"\nDone. {len(PROFILES)} synthetic test resumes in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
