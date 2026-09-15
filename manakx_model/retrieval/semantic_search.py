"""
Semantic search over the standards catalogue.

    Tender description -> Embedding Model -> Vector -> Search Indian
    Standards Vector Database -> Top 10 relevant standards

exactly as laid out in the spec. Wires `EmbeddingBackend` (embeddings/backend.py)
to `VectorIndex` (retrieval/vector_index.py) and the loaded `StandardsRepository`.
"""

from __future__ import annotations

from typing import List, Tuple

from ..embeddings.backend import EmbeddingBackend, get_default_backend
from .standards_repository import Standard, StandardsRepository
from .vector_index import VectorIndex


class SemanticSearch:
    def __init__(
        self,
        repository: StandardsRepository,
        backend: EmbeddingBackend = None,
    ):
        self._repo = repository
        self._standards = repository.all()
        self._backend = backend or get_default_backend()

        corpus = [s.corpus_text for s in self._standards]
        self._backend.fit(corpus)
        vectors = self._backend.encode(corpus)

        self._index = VectorIndex(dim=vectors.shape[1])
        self._index.add(vectors)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def index_backend_name(self) -> str:
        return self._index.backend_name

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[Standard, float]]:
        """Returns up to `top_k` (Standard, similarity_score) pairs, best first."""
        if not self._standards:
            return []
        query_vector = self._backend.encode([query_text])
        scores, indices = self._index.search(query_vector, top_k=min(top_k, len(self._standards)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._standards[idx], float(score)))
        return results
