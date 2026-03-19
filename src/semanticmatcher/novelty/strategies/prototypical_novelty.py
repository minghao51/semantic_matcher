"""
Prototypical networks novelty detection strategy (refactored).

Adapts the existing PrototypicalNoveltyDetector to implement
the NoveltyStrategy protocol.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import PrototypicalConfig
from .prototypical_strategy import PrototypicalNoveltyDetector as PrototypicalDetectorImpl


@StrategyRegistry.register
class PrototypicalStrategy(NoveltyStrategy):
    """
    Prototypical networks strategy for novelty detection.

    Computes class prototypes from known samples and flags
    samples far from their predicted class prototype as novel.
    """

    strategy_id = "prototypical"

    def __init__(self):
        self._config: PrototypicalConfig = None
        self._detector_impl: PrototypicalDetectorImpl = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: PrototypicalConfig,
    ) -> None:
        """
        Initialize the prototypical strategy.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Labels of known samples
            config: PrototypicalConfig with distance_threshold, model_name
        """
        self._config = config or PrototypicalConfig()

        self._detector_impl = PrototypicalDetectorImpl(
            distance_threshold=self._config.distance_threshold,
            model_name=self._config.model_name,
        )

        if hasattr(self._detector_impl, 'train'):
            # Store reference data for later use
            self._reference_embeddings = reference_embeddings
            self._reference_labels = reference_labels

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using prototypical networks.

        Args:
            texts: Input texts
            embeddings: Text embeddings
            predicted_classes: Predicted classes
            confidences: Prediction confidences
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        if hasattr(self._detector_impl, 'predict_batch'):
            results = self._detector_impl.predict_batch(texts, embeddings)

            for idx, (is_novel, distance) in enumerate(results):
                if is_novel:
                    flags.add(idx)

                metrics[idx] = {
                    "prototypical_is_novel": is_novel,
                    "prototypical_distance": distance,
                    "prototypical_novelty_score": min(distance / self._config.distance_threshold, 1.0),
                }
        else:
            # Fallback: compute distances manually
            for idx in range(len(texts)):
                # Placeholder: use embedding norm as distance
                distance = float(np.linalg.norm(embeddings[idx]))

                is_novel = distance > self._config.distance_threshold

                if is_novel:
                    flags.add(idx)

                metrics[idx] = {
                    "prototypical_is_novel": is_novel,
                    "prototypical_distance": distance,
                    "prototypical_novelty_score": min(distance / self._config.distance_threshold, 1.0),
                }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return PrototypicalConfig as the config schema."""
        return PrototypicalConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Prototypical networks are experimental
        return 0.1
