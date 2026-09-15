"""
Vector index for the standards corpus.

Wraps FAISS (the spec explicitly suggests FAISS/Qdrant/Pinecone/pgvector)
behind a tiny interface. If FAISS isn't installed, falls back to a plain
NumPy cosine-similarity search — functionally identical for a standards
catalogue of this size (thousands of records), since FAISS's real payoff
is at a much larger scale. Swap in Qdrant/Pinecone/pgvector later by
implementing the same two methods if the real deployment needs a
persistent/remote vector DB instead of an in-process one.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _FAISS_AVAILABLE = False


class VectorIndex:
    def __init__(self, dim: int):
        self._dim = dim
        self._backend_name = "faiss" if _FAISS_AVAILABLE else "numpy-fallback"
        if _FAISS_AVAILABLE:
            # Inner product on normalised vectors == cosine similarity.
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = None
            self._vectors: np.ndarray | None = None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if _FAISS_AVAILABLE:
            self._index.add(vectors)
        else:
            self._vectors = vectors

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices), each shape (n_queries, top_k)."""
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")
        if _FAISS_AVAILABLE:
            scores, indices = self._index.search(query_vectors, top_k)
            return scores, indices

        # NumPy fallback: brute-force cosine similarity (vectors are
        # already normalised by the embedding backend).
        sims = query_vectors @ self._vectors.T
        top_k = min(top_k, sims.shape[1])
        indices = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, indices, axis=1)
        return scores, indices

    def __len__(self) -> int:
        if _FAISS_AVAILABLE:
            return self._index.ntotal
        return 0 if self._vectors is None else len(self._vectors)
