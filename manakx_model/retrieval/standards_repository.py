"""
Standards repository.

Loads the catalogue of Indian Standards the system searches over,
including the extra fields the spec calls for beyond plain
title/scope: allied-standard relations (for the knowledge graph),
revision/supersession status, and certification requirements.

Swap the bundled mock dataset for the real, teammate-collected one by
pointing `StandardsRepository.load()` at a different path — every other
module only depends on the `Standard` schema below, not on the file
itself.

Expected record schema (JSON, list under a top-level "standards" key):
    {
      "id": "IS-10322", "title": "...", "category": "...", "scope": "...",
      "keywords": [...], "year": 2019, "status": "current"|"superseded",
      "superseded_by": null | "IS-XXXX:2024",
      "relations": {"testing": [...], "safety": [...], "terminology": [...],
                     "installation": [...], "material": [...]},
      "certification": {"bis_mandatory": true|false, "scheme": "..."|null}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "standards_dataset.json"


@dataclass
class Standard:
    id: str
    title: str
    category: str
    scope: str
    keywords: List[str]
    year: Optional[int] = None
    status: str = "current"
    superseded_by: Optional[str] = None
    relations: Dict[str, List[str]] = field(default_factory=dict)
    certification: Dict = field(default_factory=dict)

    @property
    def corpus_text(self) -> str:
        """Text blob embedded/matched against requirement text."""
        return " ".join([self.title, self.scope, " ".join(self.keywords)])

    @property
    def is_current(self) -> bool:
        return self.status == "current"

    def to_dict(self, include_relations: bool = True) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "scope": self.scope,
            "keywords": self.keywords,
            "year": self.year,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "certification": self.certification,
        }
        if include_relations:
            d["relations"] = self.relations
        return d


class StandardsRepository:
    def __init__(self, standards: List[Standard]):
        self._standards = standards
        self._by_id = {s.id: s for s in standards}

    @classmethod
    def load(cls, path: Optional[str] = None) -> "StandardsRepository":
        source = Path(path) if path else _DEFAULT_PATH
        with open(source, "r", encoding="utf-8") as f:
            raw = json.load(f)
        standards = [
            Standard(
                id=item["id"],
                title=item["title"],
                category=item.get("category", ""),
                scope=item.get("scope", ""),
                keywords=item.get("keywords", []),
                year=item.get("year"),
                status=item.get("status", "current"),
                superseded_by=item.get("superseded_by"),
                relations=item.get("relations", {}),
                certification=item.get("certification", {}),
            )
            for item in raw["standards"]
        ]
        return cls(standards)

    def all(self) -> List[Standard]:
        return list(self._standards)

    def get(self, standard_id: str) -> Optional[Standard]:
        return self._by_id.get(standard_id)

    def __len__(self) -> int:
        return len(self._standards)
