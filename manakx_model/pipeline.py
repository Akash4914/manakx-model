"""
End-to-end MANAKX model pipeline — the module your teammates integrate
against.

    from manakx_model import ManakxPipeline
    pipeline = ManakxPipeline()                     # build once, reuse
    result = pipeline.analyze("Outdoor LED street light, 120W, IP66, highway installation")

Implements the exact flow from the spec:

    Input (description / spec / tender PDF)
        -> Text extraction & cleaning              (ingestion)
        -> NLP attribute extraction                 (extraction)
        -> Embedding + vector search (top 10)        (embeddings, retrieval)
        -> LLM/RAG ranking + explanation (top 3-5)    (ranking)
        -> Allied standards (knowledge graph)         (graph)
        -> Revision / outdated check                  (revision)
        -> Certification recommendation                (certification)

plus a second entry point, `analyze_tender`, for the tender-document-audit
mode described separately in the spec.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from .certification.certification_rules import get_certification_requirement
from .extraction.attribute_extractor import extract_attributes
from .graph.knowledge_graph import StandardsKnowledgeGraph
from .ingestion.text_extractor import ingest
from .ranking.llm_ranker import get_default_ranker, translate_if_needed
from .retrieval.semantic_search import SemanticSearch
from .retrieval.standards_repository import StandardsRepository
from .revision.revision_checker import RevisionChecker
from .tender.tender_analyzer import analyze_tender as _analyze_tender

# How many candidates the vector search retrieves before the RAG layer
# narrows/explains them, per the spec ("Top 10 relevant standards").
DEFAULT_RETRIEVAL_TOP_K = 10
# How many of those the RAG layer keeps in the final, explained output.
DEFAULT_FINAL_TOP_K = 5


class ManakxPipeline:
    """Holds everything expensive to build (embeddings index, LLM/rule
    ranker) so it's constructed once and reused across many `analyze()`
    calls — build one instance per process, not one per request."""

    def __init__(self, standards_dataset_path: Optional[str] = None):
        self.repository = StandardsRepository.load(standards_dataset_path)
        self.semantic_search = SemanticSearch(self.repository)
        self.ranker = get_default_ranker()
        self.knowledge_graph = StandardsKnowledgeGraph(self.repository)
        self.revision_checker = RevisionChecker(self.repository)

    def analyze(
        self,
        input_text: Optional[str] = None,
        pdf_path: Optional[Union[str, Path]] = None,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> dict:
        """Main entry point: analyze a product description, a detailed
        spec, or a tender PDF (pass exactly one of `input_text`/`pdf_path`)
        and recommend applicable standards."""
        doc = ingest(input_text=input_text, pdf_path=pdf_path)
        if not doc.text.strip():
            raise ValueError("No text found in the input.")

        working_text = translate_if_needed(doc.text, doc.language, self.ranker)
        translation_applied = working_text != doc.text

        attributes = extract_attributes(working_text)
        candidates = self.semantic_search.search(working_text, top_k=retrieval_top_k)
        ranked = self.ranker.rank(working_text, candidates)[:final_top_k]

        allied_standards = []
        certification = None
        latest_revision = None
        if ranked:
            main_id = ranked[0].standard.id
            allied_standards = [a.to_dict() for a in self.knowledge_graph.get_allied_standards(main_id)]
            certification = get_certification_requirement(ranked[0].standard).to_dict()
            latest_revision = self.revision_checker.check(main_id).to_dict()

        return {
            "input": {
                "language_detected": doc.language,
                "translation_applied": translation_applied,
                "source_type": doc.source_type,
            },
            "detected_attributes": attributes.to_dict(),
            "recommended_standards": [r.to_dict() for r in ranked],
            "allied_standards": allied_standards,
            "certification": certification,
            "latest_revision": latest_revision,
            "engine": {
                "embedding_backend": self.semantic_search.backend_name,
                "vector_index_backend": self.semantic_search.index_backend_name,
                "ranking_backend": self.ranker.name,
            },
        }

    def analyze_tender(
        self,
        input_text: Optional[str] = None,
        pdf_path: Optional[Union[str, Path]] = None,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> dict:
        """Tender-document-audit mode: given a full tender's text (or PDF),
        report which standards it already cites, whether those are
        outdated, and which applicable standards it's missing."""
        doc = ingest(input_text=input_text, pdf_path=pdf_path)
        if not doc.text.strip():
            raise ValueError("No text found in the input.")
        working_text = translate_if_needed(doc.text, doc.language, self.ranker)

        audit = _analyze_tender(
            working_text,
            semantic_search=self.semantic_search,
            revision_checker=self.revision_checker,
            knowledge_graph=self.knowledge_graph,
            top_k=retrieval_top_k,
        )
        return {
            "input": {"language_detected": doc.language, "source_type": doc.source_type},
            **audit.to_dict(),
        }

    def check_revision(self, standard_reference: str) -> dict:
        """Direct lookup: is this specific IS number current or outdated?"""
        return self.revision_checker.check(standard_reference).to_dict()
