from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass


class RerankerBackend(ABC):
    """Abstract base class for reranker backends."""

    @abstractmethod
    def score(self, query: str, docs: list[str]) -> List[float]:
        """Score query-document pairs."""
        pass

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return top_k.

        Default implementation using score(). Subclasses can override for optimization.
        """
        # Extract texts from candidates
        texts = [cand.get(text_field, cand.get("name", "")) for cand in candidates]

        # Score all pairs
        scores = self.score(query, texts)

        # Add scores and sort
        scored = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["cross_encoder_score"] = float(score)
            scored.append(item)

        # Return top_k
        reranked = sorted(scored, key=lambda x: x["cross_encoder_score"], reverse=True)[
            :top_k
        ]

        return reranked
