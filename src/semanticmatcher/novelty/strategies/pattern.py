"""
Pattern-based novelty detection strategy (refactored).

Adapts the existing PatternBasedNoveltyStrategy to implement
the NoveltyStrategy protocol.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import PatternConfig
from .pattern_strategy import PatternBasedNoveltyStrategy as PatternScorer


@StrategyRegistry.register
class PatternStrategy(NoveltyStrategy):
    """
    Pattern-based strategy for novelty detection.

    Extracts orthographic and linguistic patterns from known entities
    and flags novel samples based on pattern violations.
    """

    strategy_id = "pattern"

    def __init__(self):
        self._config: PatternConfig = None
        self._pattern_scorer: PatternScorer = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: PatternConfig,
    ) -> None:
        """
        Initialize the pattern strategy.

        Args:
            reference_embeddings: Embeddings of known samples (not used)
            reference_labels: Labels of known samples
            config: PatternConfig with threshold
        """
        self._config = config or PatternConfig()

        self._pattern_scorer = PatternScorer(known_entities=reference_labels)

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using pattern analysis.

        Args:
            texts: Input texts (entity names)
            embeddings: Text embeddings (not used)
            predicted_classes: Predicted classes (not used)
            confidences: Prediction confidences (not used)
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        for idx, text in enumerate(texts):
            novelty_score = self._pattern_scorer.score_novelty(text)

            is_novel = novelty_score >= self._config.threshold

            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "pattern_novelty_score": novelty_score,
                "pattern_is_novel": is_novel,
                "pattern_text": text,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return PatternConfig as the config schema."""
        return PatternConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Pattern is a complementary signal
        return 0.2
