"""
Revision checking.

The spec: "the system should also prevent officials from using outdated
standards" — given an IS number (typed in directly, or found as a
"recommended_standards"/"referenced_standards" entry elsewhere in the
pipeline), say whether it's current, and if not, what supersedes it.

Also recognises free-form references like "IS 9137:2010" or "IS9137"
typed by a user, and maps them back to the closest catalogue entry —
useful both for a direct "check this standard" lookup and for the
tender analyzer, which extracts these patterns straight out of tender
text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..retrieval.standards_repository import Standard, StandardsRepository

# Matches "IS 9137:2010", "IS-9137", "IS9137:2010", "IS 9137" etc.
IS_REFERENCE_RE = re.compile(r"\bIS[\s\-]?(\d{2,6})(?:[\s:\-](\d{4}))?\b", re.IGNORECASE)


@dataclass
class RevisionStatus:
    query: str
    matched_standard: Optional[Standard]
    is_current: bool
    message: str
    current_version_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "matched_standard_id": self.matched_standard.id if self.matched_standard else None,
            "is_current": self.is_current,
            "message": self.message,
            "current_version_id": self.current_version_id,
        }


def extract_standard_references(text: str) -> List[str]:
    """Pull every "IS <number>[:<year>]"-style reference out of free text
    (used by the tender analyzer to find standards a tender already
    cites)."""
    refs = []
    for match in IS_REFERENCE_RE.finditer(text):
        number, year = match.group(1), match.group(2)
        refs.append(f"IS-{number}:{year}" if year else f"IS-{number}")
    return refs


class RevisionChecker:
    def __init__(self, repository: StandardsRepository):
        self._repo = repository
        # Index by bare number (e.g. "9137") so "IS 9137" and "IS-9137:2010"
        # both resolve even if the catalogue id includes a year suffix.
        self._by_number: dict[str, List[Standard]] = {}
        for s in repository.all():
            number = re.sub(r"[^\d]", "", s.id.split(":")[0])
            self._by_number.setdefault(number, []).append(s)

    def _resolve(self, reference: str) -> Optional[Standard]:
        exact = self._repo.get(reference)
        if exact:
            return exact
        number = re.sub(r"[^\d]", "", reference.split(":")[0])
        candidates = self._by_number.get(number, [])
        if not candidates:
            return None
        # Prefer an exact id match if the reference included a year that
        # matches a catalogue entry; otherwise fall back to the first
        # (base) entry for that number, whatever its status.
        for c in candidates:
            if c.id == reference:
                return c
        return candidates[0]

    def check(self, reference: str) -> RevisionStatus:
        standard = self._resolve(reference)
        if standard is None:
            return RevisionStatus(
                query=reference,
                matched_standard=None,
                is_current=False,
                message=f"'{reference}' was not found in the standards catalogue.",
            )
        if standard.is_current:
            return RevisionStatus(
                query=reference,
                matched_standard=standard,
                is_current=True,
                message=f"{standard.id} ({standard.year}) is the current version.",
            )
        return RevisionStatus(
            query=reference,
            matched_standard=standard,
            is_current=False,
            message=(
                f"{standard.id} ({standard.year}) is outdated. "
                f"Superseded by {standard.superseded_by}."
            ),
            current_version_id=standard.superseded_by,
        )
