"""Hybrid matching pipeline with blocking, retrieval, and reranking."""

from typing import Dict, Any, List, Optional

from .matcher import EmbeddingMatcher
from .reranker import CrossEncoderReranker
from .blocking import BlockingStrategy, NoOpBlocking


class HybridMatcher:
    """
    Three-stage waterfall pipeline for semantic entity matching.

    Combines fast blocking, semantic retrieval, and precise reranking
    for accurate and efficient matching.

    Pipeline Stages:
        1. Blocking (BM25/TF-IDF/Fuzzy) - Fast lexical filtering
        2. Bi-Encoder Retrieval - Semantic similarity search
        3. Cross-Encoder Reranking - Precise cross-attention scoring

    Example:
        >>> from semanticmatcher import HybridMatcher
        >>> from semanticmatcher.core.blocking import BM25Blocking
        >>>
        >>> matcher = HybridMatcher(
        ...     entities=products,
        ...     blocking_strategy=BM25Blocking(),
        ...     retriever_model="bge-base",
        ...     reranker_model="bge-m3"
        ... )
        >>>
        >>> results = matcher.match(
        ...     "iPhone 15 case",
        ...     blocking_top_k=1000,
        ...     retrieval_top_k=50,
        ...     final_top_k=5
        ... )
    """

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        blocking_strategy: Optional[BlockingStrategy] = None,
        retriever_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        normalize: bool = True,
    ):
        """
        Initialize the hybrid matcher.

        Args:
            entities: List of entity dictionaries
            blocking_strategy: Blocking strategy (defaults to NoOpBlocking)
            retriever_model: Model name for bi-encoder retrieval
            reranker_model: Model name for cross-encoder reranking
            normalize: Whether to normalize text (lowercase, remove accents, etc.)
        """
        # Stage 1: Blocking
        self.blocker = blocking_strategy or NoOpBlocking()

        # Stage 2: Bi-Encoder Retrieval
        self.retriever = EmbeddingMatcher(
            entities=entities,
            model_name=retriever_model,
            normalize=normalize,
        )
        self.retriever.build_index()

        # Stage 3: Cross-Encoder Reranking
        self.reranker = CrossEncoderReranker(model=reranker_model)

    def match(
        self,
        query: str,
        blocking_top_k: int = 1000,
        retrieval_top_k: int = 50,
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Match query using three-stage waterfall pipeline.

        Args:
            query: Search query
            blocking_top_k: Number of candidates after blocking stage
            retrieval_top_k: Number of candidates after retrieval stage
            final_top_k: Number of final results after reranking

        Returns:
            List of matched entities with scores (bi-encoder and cross-encoder)
        """
        # Stage 1: Blocking - Fast lexical filtering
        candidates = self.blocker.block(
            query,
            self.retriever.entities,
            top_k=blocking_top_k
        )

        # Early exit if no candidates from blocking
        if not candidates:
            return []

        # Stage 2: Bi-Encoder Retrieval - Semantic similarity
        retrieved = self.retriever.match(
            query,
            candidates=candidates,
            top_k=retrieval_top_k,
        )

        # Ensure retrieved is a list (handle single result case)
        if retrieved is None:
            return []
        if not isinstance(retrieved, list):
            retrieved = [retrieved]

        # Filter out None results
        retrieved = [r for r in retrieved if r is not None]

        # Stage 3: Cross-Encoder Reranking - Precise scoring
        if not retrieved:
            return []

        final = self.reranker.rerank(query, retrieved, top_k=final_top_k)

        return final

    def match_bulk(
        self,
        queries: List[str],
        blocking_top_k: int = 1000,
        retrieval_top_k: int = 50,
        final_top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch matching for multiple queries.

        Args:
            queries: List of search queries
            blocking_top_k: Number of candidates after blocking stage
            retrieval_top_k: Number of candidates after retrieval stage
            final_top_k: Number of final results after reranking

        Returns:
            List of matched entity lists (one per query)
        """
        return [
            self.match(
                q,
                blocking_top_k,
                retrieval_top_k,
                final_top_k
            )
            for q in queries
        ]
