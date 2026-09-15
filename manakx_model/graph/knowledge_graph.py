"""
Standards knowledge graph — allied standards.

The spec is explicit that the problem doesn't only ask for the main
standard: a main product standard typically references several allied
standards (test method, safety, terminology, installation, material),
and the system should surface that whole cluster, not just the single
best match.

Implemented as a lightweight in-memory graph built directly from each
Standard's `relations` field (already loaded by StandardsRepository) —
enough to demonstrate and reason about for an SIH prototype, per the
spec's own note that "PostgreSQL can also handle the relationships for
an SIH prototype." A real deployment with a much larger, richer
standards catalogue could swap this for Neo4j behind the same
`get_allied_standards` interface without touching the rest of the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..retrieval.standards_repository import Standard, StandardsRepository

RELATION_TYPES = ["testing", "safety", "terminology", "installation", "material"]


@dataclass
class AlliedStandard:
    relation: str  # one of RELATION_TYPES
    standard: Standard

    def to_dict(self) -> dict:
        return {"relation": self.relation, **self.standard.to_dict(include_relations=False)}


class StandardsKnowledgeGraph:
    def __init__(self, repository: StandardsRepository):
        self._repo = repository

    def _resolve_current(self, standard: Standard) -> Standard:
        """Follow `superseded_by` links so an allied-standard relation that
        was recorded against an old edition still surfaces the current one
        — relation data shouldn't need manual updates every time a linked
        standard gets revised."""
        seen = set()
        current = standard
        while not current.is_current and current.superseded_by and current.id not in seen:
            seen.add(current.id)
            newer = self._repo.get(current.superseded_by)
            if newer is None:
                break
            current = newer
        return current

    def get_allied_standards(self, main_standard_id: str) -> List[AlliedStandard]:
        """Return every standard related to `main_standard_id`, tagged with
        its relation type (testing / safety / terminology / installation /
        material) — the "Main Product Standard -> allied standards" tree
        from the spec, flattened to a list. Each is resolved to its current
        version even if the stored relation points at a superseded one."""
        main = self._repo.get(main_standard_id)
        if main is None:
            return []
        allied: List[AlliedStandard] = []
        seen_ids = set()
        for relation in RELATION_TYPES:
            for related_id in main.relations.get(relation, []):
                related_standard = self._repo.get(related_id)
                if related_standard is None:
                    continue
                related_standard = self._resolve_current(related_standard)
                if related_standard.id in seen_ids:
                    continue
                seen_ids.add(related_standard.id)
                allied.append(AlliedStandard(relation=relation, standard=related_standard))
        return allied

    def get_relation_tree(self, main_standard_id: str) -> Optional[Dict]:
        """Same data as `get_allied_standards`, shaped as a tree matching
        the spec's diagram (Main Product Standard with branches per
        relation type) — convenient for a UI that wants to render it
        that way directly."""
        main = self._repo.get(main_standard_id)
        if main is None:
            return None
        tree: Dict = {"main": main.to_dict(include_relations=False), "allied": {}}
        for relation in RELATION_TYPES:
            related_ids = main.relations.get(relation, [])
            tree["allied"][relation] = [
                self._repo.get(rid).to_dict(include_relations=False)
                for rid in related_ids
                if self._repo.get(rid) is not None
            ]
        return tree
