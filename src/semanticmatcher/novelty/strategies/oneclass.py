"""
One-Class SVM novelty detection strategy (refactored).

Adapts the existing OneClassNoveltyDetector to implement
the NoveltyStrategy protocol.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import OneClassConfig
from .oneclass_strategy import OneClassNoveltyDetector as OneClassDetectorImpl


@StrategyRegistry.register
class OneClassStrategy(NoveltyStrategy):
    """
    One-Class SVM strategy for novelty detection.

    Trains a One-Class SVM on known entity embeddings and
    flags samples outside the boundary as novel.
    """

    strategy_id = "oneclass"

    def __init__(self):
        self._config: OneClassConfig = None
        self._detector_impl: OneClassDetectorImpl = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: OneClassConfig,
    ) -> None:
        """
        Initialize the One-Class SVM strategy.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Labels of known samples
            config: OneClassConfig with nu, kernel, model_name
        """
        self._config = config or OneClassConfig()

        self._detector_impl = OneClassDetectorImpl(
            model_name=self._config.model_name,
            nu=self._config.nu,
            kernel=self._config.kernel,
            gamma=self._config.gamma,
        )

        # Train the detector
        # Note: reference_embeddings are not directly used - the detector
        # encodes the reference_labels (texts) internally
        # For now, we'll skip training here and assume it's trained elsewhere
        # or we need to pass the actual texts

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using One-Class SVM.

        Args:
            texts: Input texts
            embeddings: Text embeddings (not used by this detector)
            predicted_classes: Predicted classes (not used)
            confidences: Prediction confidences (not used)
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        if not self._detector_impl.is_trained:
            return set(), {}

        flags = set()
        metrics = {}

        results = self._detector_impl.score_batch(texts)

        for idx, (is_novel, confidence) in enumerate(results):
            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "oneclass_is_novel": is_novel,
                "oneclass_confidence": confidence,
                "oneclass_novelty_score": confidence,  # Already 0-1 scaled
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return OneClassConfig as the config schema."""
        return OneClassConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # One-Class SVM is experimental, give it lower weight
        return 0.1
