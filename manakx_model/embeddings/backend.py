"""
Embedding backends.

The spec explicitly calls for embeddings + vector search rather than
keyword search, so that "protective headgear for construction workers"
retrieves "industrial safety helmets" even though they share almost no
words. `SentenceTransformerBackend` is the real semantic embedder this
is meant to run on.

`SentenceTransformerBackend` downloads its model weights from the
Hugging Face Hub the first time it runs, so it needs outbound internet
access on whatever machine actually runs the service. If that's
unavailable (or the package isn't installed), `TfidfBackend` is used
instead automatically — it keeps the whole pipeline runnable and
testable offline, at the cost of falling back to lexical rather than
true semantic similarity for that run. Everything downstream only talks
to the `EmbeddingBackend` interface, so this swap is invisible to the
rest of the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy dependency
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(ABC):
    """Common interface: text in, dense vectors out."""

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit on the standards corpus (no-op for models that don't need it)."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of embeddings."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SentenceTransformerBackend(EmbeddingBackend):
    """True semantic embeddings via a pretrained sentence-transformer model.

    Requires `pip install sentence-transformers` and internet access to
    download the model the first time it runs (weights are cached
    locally after that).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` to use real semantic "
                "embeddings, or use TfidfBackend as a fallback."
            )
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: List[str]) -> None:
        pass  # pretrained model needs no fitting

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"


class TfidfBackend(EmbeddingBackend):
    """Offline fallback: TF-IDF vectors, cosine-normalised so they behave
    like the sentence-transformer output for downstream cosine-similarity
    search. Lexical rather than semantic — it will still miss synonym
    cases like "headgear" vs "helmet" that have no shared word stem — but
    it needs no download and never fails offline, which is why the
    pipeline falls back to it automatically when embeddings aren't
    available.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfBackend.fit(corpus) must be called before encode()")
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @property
    def name(self) -> str:
        return "tfidf-fallback"


def get_default_backend() -> EmbeddingBackend:
    """Prefer real semantic embeddings; fall back to TF-IDF if the model
    can't be loaded (not installed, or no internet to download weights)."""
    if _SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return SentenceTransformerBackend()
        except Exception:
            pass
    return TfidfBackend()
