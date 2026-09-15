"""
Input ingestion.

Handles the three input types the system accepts (per the project spec):
1. A simple product description ("Outdoor LED street light, 120W, IP66, highway installation")
2. Detailed technical specifications (multi-line spec text)
3. An entire tender PDF

Both (1) and (2) are just text and flow through `clean_text` directly.
(3) goes through `extract_text_from_pdf` first. Language detection is run
on everything so the pipeline knows whether it needs to route through
translation before extraction/embedding (see ranking/llm_ranker.py, which
doubles as the translation layer when an LLM is configured).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

try:
    from langdetect import detect as _detect_lang
    from langdetect import LangDetectException
except ImportError:  # pragma: no cover - optional dependency
    _detect_lang = None
    LangDetectException = Exception

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class IngestedDocument:
    text: str
    language: str  # ISO 639-1 code, e.g. "en", "hi"; "unknown" if undetectable
    source_type: str  # "text" | "pdf"
    page_count: int = 1

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "source_type": self.source_type,
            "page_count": self.page_count,
            "char_count": len(self.text),
        }


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Best-effort language detection. Returns 'en' if detection is
    unavailable or the text is too short to detect reliably — plain
    English descriptions/specs are the common case and should never be
    blocked on this."""
    if not text or len(text.strip()) < 8 or _detect_lang is None:
        return "en"
    try:
        return _detect_lang(text)
    except LangDetectException:
        return "unknown"


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> IngestedDocument:
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install it with "
            "`pip install pypdf`."
        )
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = clean_text("\n\n".join(pages_text))
    return IngestedDocument(
        text=text,
        language=detect_language(text),
        source_type="pdf",
        page_count=len(reader.pages),
    )


def ingest(input_text: str = None, pdf_path: Union[str, Path] = None) -> IngestedDocument:
    """Single entry point for all three input types. Pass either
    `input_text` (product description or detailed spec) or `pdf_path`
    (tender PDF) — exactly one of the two."""
    if pdf_path is not None:
        return extract_text_from_pdf(pdf_path)
    if input_text is not None:
        text = clean_text(input_text)
        return IngestedDocument(text=text, language=detect_language(text), source_type="text")
    raise ValueError("Provide either input_text or pdf_path")
