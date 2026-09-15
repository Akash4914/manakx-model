"""
RAG ranking / explanation layer, and multilingual translation.

Per the spec: semantic search retrieves the top 10-20 candidate
standards; an LLM then analyzes them together with the tender/spec text
and produces the final ranked, explained recommendations ("1. IS XXXX -
Highly Relevant. Reason: Main product standard for LED luminaires.").

This module also does double duty as the translation step for
multilingual input ("the system translates ... and performs the same
standards search") — translation is just another LLM call, so it lives
next to the other LLM usage rather than as a separate service.

Both capabilities require an LLM and are OPTIONAL: if no
`ANTHROPIC_API_KEY` is configured (or the `anthropic` package isn't
installed), the pipeline falls back automatically —
`RuleBasedRanker` for ranking/explanation, and a pass-through (with a
warning) for translation. This keeps the whole pipeline runnable and
demoable with zero API cost/setup, while the LLM path is a drop-in
upgrade once you have API access.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ANTHROPIC_AVAILABLE = False

from ..retrieval.standards_repository import Standard

DEFAULT_MODEL = os.environ.get("MANAKX_LLM_MODEL", "claude-sonnet-4-5")


@dataclass
class RankedStandard:
    standard: Standard
    relevance: str  # "Highly Relevant" | "Relevant" | "Possibly Relevant"
    reason: str
    semantic_score: float

    def to_dict(self) -> dict:
        return {
            **self.standard.to_dict(include_relations=False),
            "relevance": self.relevance,
            "reason": self.reason,
            "match_confidence_pct": round(self.semantic_score * 100, 1),
        }


class BaseRanker(ABC):
    @abstractmethod
    def rank(self, query_text: str, candidates: List[tuple]) -> List[RankedStandard]:
        """`candidates` is a list of (Standard, semantic_score) tuples,
        already ordered by the vector search. Returns the final ranked,
        explained list (typically top 3-5 of the input candidates)."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class RuleBasedRanker(BaseRanker):
    """Offline fallback: labels relevance from the semantic score's own
    distribution and writes the reason from the strongest keyword/title
    overlap between the query and the standard. No network call, no API
    key — always available."""

    HIGH_THRESHOLD = 0.35
    MED_THRESHOLD = 0.15

    def _label(self, score: float) -> str:
        if score >= self.HIGH_THRESHOLD:
            return "Highly Relevant"
        if score >= self.MED_THRESHOLD:
            return "Relevant"
        return "Possibly Relevant"

    def _reason(self, query_text: str, standard: Standard) -> str:
        query_lower = query_text.lower()
        hit_keywords = [k for k in standard.keywords if k.lower() in query_lower]
        if hit_keywords:
            return f"Matches on: {', '.join(hit_keywords[:3])}."
        return f"Scope: {standard.scope.split('.')[0]}."

    def rank(self, query_text: str, candidates: List[tuple]) -> List[RankedStandard]:
        return [
            RankedStandard(
                standard=standard,
                relevance=self._label(score),
                reason=self._reason(query_text, standard),
                semantic_score=score,
            )
            for standard, score in candidates
        ]

    @property
    def name(self) -> str:
        return "rule-based-fallback"


class LLMRanker(BaseRanker):
    """Real RAG ranking: sends the query + candidate standards to Claude
    and asks it to rank/explain them. Requires `pip install anthropic`
    and an `ANTHROPIC_API_KEY` environment variable."""

    def __init__(self, model: str = DEFAULT_MODEL):
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic package is not installed. Run `pip install anthropic`."
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._model = model
        self._client = anthropic.Anthropic()

    def rank(self, query_text: str, candidates: List[tuple]) -> List[RankedStandard]:
        if not candidates:
            return []
        candidate_block = "\n".join(
            f"- id: {s.id}\n  title: {s.title}\n  scope: {s.scope}"
            for s, _ in candidates
        )
        prompt = (
            "You are ranking Indian Standards (IS) for relevance to a "
            "procurement specification. Given the specification and a list "
            "of candidate standards retrieved by semantic search, return a "
            "JSON array (only JSON, no other text) of objects "
            '{"id": "<standard id>", "relevance": "Highly Relevant"|'
            '"Relevant"|"Possibly Relevant", "reason": "<one short sentence>"}, '
            "ordered from most to least relevant. Only include standards "
            "that are genuinely relevant.\n\n"
            f"Specification:\n{query_text}\n\nCandidate standards:\n{candidate_block}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        parsed = json.loads(cleaned)

        by_id = {s.id: (s, score) for s, score in candidates}
        ranked: List[RankedStandard] = []
        for item in parsed:
            match = by_id.get(item.get("id"))
            if match is None:
                continue
            standard, score = match
            ranked.append(
                RankedStandard(
                    standard=standard,
                    relevance=item.get("relevance", "Relevant"),
                    reason=item.get("reason", ""),
                    semantic_score=score,
                )
            )
        return ranked

    def translate_to_english(self, text: str, source_language: str) -> str:
        prompt = (
            f"Translate the following procurement specification from "
            f"language code '{source_language}' to English. Return only the "
            f"translation, nothing else.\n\n{text}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if hasattr(b, "text")).strip()

    @property
    def name(self) -> str:
        return f"llm:{self._model}"


def get_default_ranker() -> BaseRanker:
    """Prefer the real LLM ranker; fall back to the rule-based one if no
    API key/package is configured."""
    if _ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMRanker()
        except Exception:
            pass
    return RuleBasedRanker()


def translate_if_needed(text: str, language: str, ranker: Optional[BaseRanker] = None) -> str:
    """Translate non-English input to English if an LLM ranker is
    available; otherwise return the text unchanged (the rest of the
    pipeline will then run in the original language with reduced
    accuracy — flagged via `metadata.translation_applied` in the
    pipeline output)."""
    if language in ("en", "unknown") or not text.strip():
        return text
    if isinstance(ranker, LLMRanker):
        return ranker.translate_to_english(text, language)
    return text
