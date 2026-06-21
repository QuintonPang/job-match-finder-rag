"""
parse_resume.py

Purpose:
    Extract plain text from a PDF resume file, so it can be used as a
    search query against our job postings in Qdrant.

Why we don't need an LLM for this step:
    Unlike Lesson 2's job-cleaning task (which needed to interpret
    messy, unpredictable text into a precise structure), here we just
    need the resume's raw text. The embedding model itself captures
    meaning from natural, unstructured text -- we don't need to force
    it into a rigid schema first. This sidesteps the exact kind of
    small-model formatting unreliability we fought through earlier.
"""

import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF file and extracts all readable text from every page,
    joined together into one block of text.
    """
    text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def clean_resume_text(text: str) -> str:
    """
    Light cleanup -- collapse excessive blank lines/whitespace that
    PDFs sometimes produce, without trying to restructure the content.
    We deliberately keep this minimal: unlike job postings, resumes
    don't need heavy noise-stripping, they just need to be readable
    plain text for the embedding model.
    """
    lines = [line.strip() for line in text.split("\n")]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def parse_resume(pdf_path: str) -> str:
    """
    Full pipeline: PDF file -> clean, ready-to-embed text.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    return clean_resume_text(raw_text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_resume.py <path_to_resume.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    resume_text = parse_resume(pdf_path)

    print("--- EXTRACTED RESUME TEXT ---")
    print(resume_text)
    print()
    print(f"Total characters extracted: {len(resume_text)}")