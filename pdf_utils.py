"""
Raw text extraction from a CV PDF.
Kept deliberately simple: PyMuPDF handles standard digital PDFs well. 
No layout reconstruction attempted here — the LLM
parsing step handles minor section-order scrambling from the raw text.
"""
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file, page by page, concatenated."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts)

    if len(full_text.strip()) < 50:
        raise ValueError(
            f"Extracted text is suspiciously short ({len(full_text)} chars) — "
            f"this PDF may be a scanned image rather than digital text. "
            f"PyMuPDF cannot extract text from scanned/image-only PDFs."
        )
    return full_text
