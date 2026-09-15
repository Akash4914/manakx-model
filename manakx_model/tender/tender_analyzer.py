"""
Tender document analyzer.

The spec's "strongest feature to demonstrate": given a full tender
(already containing its own referenced standards), audit it rather than
just recommend from scratch —

    Tender Standards Audit
    Detected Product: Centrifugal Water Pump
    Referenced Standards: IS XXXX:2010
    Issues: IS XXXX:2010 is outdated.
    Missing Standards: + IS YYYY - Performance testing, + IS ZZZZ - Electrical safety
    Recommended Current Standard: IS XXXX:2025

Reuses every other module rather than reimplementing anything: attribute
extraction for the detected product, revision checking for the
outdated-reference audit, and semantic search + the knowledge graph for
what the tender *should* reference but doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..extraction.attribute_extractor import extract_attributes
from ..graph.knowledge_graph import StandardsKnowledgeGraph
from ..retrieval.semantic_search import SemanticSearch
from ..retrieval.standards_repository import Standard
from ..revision.revision_checker import RevisionChecker, extract_standard_references


@dataclass
class TenderAudit:
    detected_product: Optional[str]
    referenced_standards: List[dict]
    issues: List[str]
    missing_standards: List[dict]
    recommended_current_standards: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detected_product": self.detected_product,
            "referenced_standards": self.referenced_standards,
            "issues": self.issues,
            "missing_standards": self.missing_standards,
            "recommended_current_standards": self.recommended_current_standards,
        }


def analyze_tender(
    tender_text: str,
    semantic_search: SemanticSearch,
    revision_checker: RevisionChecker,
    knowledge_graph: StandardsKnowledgeGraph,
    top_k: int = 10,
) -> TenderAudit:
    attributes = extract_attributes(tender_text)

    # Standards the tender already cites, with their current-vs-outdated status.
    raw_refs = extract_standard_references(tender_text)
    referenced_standards = []
    referenced_ids_current = set()
    issues: List[str] = []
    recommended_current: List[str] = []

    for ref in raw_refs:
        status = revision_checker.check(ref)
        referenced_standards.append(status.to_dict())
        if status.matched_standard is not None:
            referenced_ids_current.add(status.matched_standard.id)
            if not status.is_current:
                issues.append(status.message)
                if status.current_version_id:
                    recommended_current.append(status.current_version_id)

    # What the tender *should* reference, based on semantic search over the
    # tender text itself (falls back to matching against detected product
    # type if the free text is too sparse to embed well).
    query = tender_text if len(tender_text.split()) > 3 else (attributes.product or tender_text)
    candidates = semantic_search.search(query, top_k=top_k)

    missing_standards: List[dict] = []
    seen_missing_ids = set()

    def _add_missing(standard: Standard, why: str):
        if standard.id in referenced_ids_current or standard.id in seen_missing_ids:
            return
        seen_missing_ids.add(standard.id)
        missing_standards.append({**standard.to_dict(include_relations=False), "why": why})

    if candidates:
        top_standard, _ = candidates[0]
        _add_missing(top_standard, "Main applicable product standard not referenced.")
        for allied in knowledge_graph.get_allied_standards(top_standard.id):
            _add_missing(allied.standard, f"Allied {allied.relation} standard not referenced.")

    return TenderAudit(
        detected_product=attributes.product,
        referenced_standards=referenced_standards,
        issues=issues,
        missing_standards=missing_standards,
        recommended_current_standards=sorted(set(recommended_current)),
    )
