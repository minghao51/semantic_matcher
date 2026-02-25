"""Blocking strategies for efficient candidate filtering."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


class BlockingStrategy(ABC):
    """Abstract base class for blocking strategies."""

    @abstractmethod
    def block(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Return top_k candidate entities for the query.

        Args:
            query: Query text
            entities: List of all entities
            top_k: Maximum number of candidates to return

        Returns:
            List of candidate entities (top_k or fewer)
        """
        pass


class NoOpBlocking(BlockingStrategy):
    """
    Pass-through blocking for small datasets.

    Returns all entities up to top_k without any filtering.
    """

    def block(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Return all entities or top_k if smaller."""
        if len(entities) <= top_k:
            return entities
        return entities[:top_k]


class BM25Blocking(BlockingStrategy):
    """
    Fast lexical blocking using BM25.

    Uses BM25 algorithm for efficient lexical matching.
    Good for keyword-heavy queries and proper nouns.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 blocking.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.cached_entities = None
        self._entity_hash = None

    def build_index(self, entities: List[Dict[str, Any]]):
        """Build BM25 index from entities."""
        self.cached_entities = entities

        # Compute hash for efficient comparison
        entity_tuples = sorted(
            (e['id'], e.get('text', e.get('name', ''))) for e in entities
        )
        self._entity_hash = hash(tuple(entity_tuples))

        tokenized_corpus = [
            self._tokenize(e.get('text', e.get('name', '')))
            for e in entities
        ]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def block(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Return top_k candidates using BM25 scores."""
        # Compute hash for current entities
        entity_tuples = sorted(
            (e['id'], e.get('text', e.get('name', ''))) for e in entities
        )
        current_hash = hash(tuple(entity_tuples))

        # Rebuild index if entities changed
        if self.bm25 is None or self._entity_hash != current_hash:
            self.build_index(entities)

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [entities[i] for i in top_indices]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization - lowercase and split."""
        return text.lower().split()


class TFIDFBlocking(BlockingStrategy):
    """
    TF-IDF based blocking.

    Uses TF-IDF vectorization for lexical matching.
    Good for document-level similarity.
    """

    def __init__(self):
        """Initialize TF-IDF blocking."""
        self.vectorizer = None
        self.matrix = None
        self.cached_entities = None
        self._entity_hash = None

    def build_index(self, entities: List[Dict[str, Any]]):
        """Build TF-IDF index from entities."""
        self.cached_entities = entities

        # Compute hash for efficient comparison
        entity_tuples = sorted(
            (e['id'], e.get('text', e.get('name', ''))) for e in entities
        )
        self._entity_hash = hash(tuple(entity_tuples))

        texts = [e.get('text', e.get('name', '')) for e in entities]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)

    def block(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Return top_k candidates using TF-IDF scores."""
        # Compute hash for current entities
        entity_tuples = sorted(
            (e['id'], e.get('text', e.get('name', ''))) for e in entities
        )
        current_hash = hash(tuple(entity_tuples))

        # Rebuild index if entities changed
        if self.vectorizer is None or self._entity_hash != current_hash:
            self.build_index(entities)

        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().flatten()

        # Get top_k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [entities[i] for i in top_indices]


class FuzzyBlocking(BlockingStrategy):
    """
    Fuzzy string matching blocking.

    Uses RapidFuzz for approximate string matching.
    Good for catching typos and variations.
    """

    def __init__(self, score_cutoff: int = 70):
        """
        Initialize fuzzy blocking.

        Args:
            score_cutoff: Minimum similarity score (0-100)
        """
        self.score_cutoff = score_cutoff

    def block(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Return top_k candidates using fuzzy matching."""
        texts = [e.get('text', e.get('name', '')) for e in entities]

        # Extract top matches with indices
        # process.extract returns list of (match, score, index) tuples
        results = process.extract(
            query,
            texts,
            scorer=fuzz.token_sort_ratio,
            limit=top_k
        )

        # Filter by score cutoff, preserving indices
        filtered = [(text, score, idx) for text, score, idx in results if score >= self.score_cutoff]

        # Return matching entities using correct indices
        return [entities[idx] for _, _, idx in filtered]
