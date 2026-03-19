"""
SetFit contrastive learning novelty detection strategy (refactored).

Adapts the existing SetFitNoveltyDetector to implement
the NoveltyStrategy protocol.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import SetFitConfig
from .setfit_novelty import SetFitNoveltyDetector as SetFitDetectorImpl


@StrategyRegistry.register
class SetFitStrategy(NoveltyStrategy):
    """
    SetFit contrastive learning strategy for novelty detection.

    Uses contrastive learning to train a classifier on known samples
    and flags low-confidence predictions as novel.
    """

    strategy_id = "setfit"

    def __init__(self):
        self._config: SetFitConfig = None
        self._detector_impl: SetFitDetectorImpl = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: SetFitConfig,
    ) -> None:
        """
        Initialize the SetFit strategy.

        Args:
            reference_embeddings: Embeddings of known samples (not used directly)
            reference_labels: Labels of known samples
            config: SetFitConfig with margin, model_name, epochs, etc.
        """
        self._config = config or SetFitConfig()

        self._detector_impl = SetFitDetectorImpl(
            known_entities=reference_labels,
            model_name=self._config.model_name,
            margin=self._config.margin,
            num_epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
        )
        self._detector_impl.train(show_progress=False)

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using SetFit.

        Args:
            texts: Input texts
            embeddings: Text embeddings (not used by SetFit)
            predicted_classes: Predicted classes (not used)
            confidences: Prediction confidences (not used)
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        if hasattr(self._detector_impl, 'predict_batch'):
            results = self._detector_impl.predict_batch(texts)

            for idx, (is_novel, confidence) in enumerate(results):
                if is_novel:
                    flags.add(idx)

                metrics[idx] = {
                    "setfit_is_novel": is_novel,
                    "setfit_confidence": confidence,
                    "setfit_novelty_score": 1.0 - confidence,  # Invert: low confidence = novel
                }
        else:
            # Fallback: use confidence scores
            for idx, confidence in enumerate(confidences):
                is_novel = confidence < self._config.threshold

                if is_novel:
                    flags.add(idx)

                metrics[idx] = {
                    "setfit_is_novel": is_novel,
                    "setfit_confidence": confidence,
                    "setfit_novelty_score": 1.0 - confidence,
                }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return SetFitConfig as the config schema."""
        return SetFitConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # SetFit is experimental
        return 0.1
