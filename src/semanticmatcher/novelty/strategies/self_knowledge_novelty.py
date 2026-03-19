"""
Self-knowledge (sparse autoencoder) novelty detection strategy (refactored).

Adapts the existing SelfKnowledgeDetector to implement
the NoveltyStrategy protocol.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import SelfKnowledgeConfig
from .self_knowledge import SelfKnowledgeDetector


@StrategyRegistry.register
class SelfKnowledgeStrategy(NoveltyStrategy):
    """
    Self-knowledge strategy for novelty detection.

    Uses a sparse autoencoder to learn representations of known
    samples and flags high reconstruction error as novel.
    """

    strategy_id = "self_knowledge"

    def __init__(self):
        self._config: SelfKnowledgeConfig = None
        self._detector: SelfKnowledgeDetector = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: SelfKnowledgeConfig,
    ) -> None:
        """
        Initialize the self-knowledge strategy.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Labels of known samples (not used)
            config: SelfKnowledgeConfig with hidden_dim, threshold, epochs
        """
        self._config = config or SelfKnowledgeConfig()

        # Initialize detector
        self._detector = SelfKnowledgeDetector(
            input_dim=reference_embeddings.shape[1],
            hidden_dim=self._config.hidden_dim,
        )

        # Train the autoencoder
        if hasattr(self._detector, 'train'):
            self._detector.train(
                reference_embeddings,
                epochs=self._config.epochs,
                batch_size=self._config.batch_size,
            )

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using self-knowledge (reconstruction error).

        Args:
            texts: Input texts (not used)
            embeddings: Text embeddings
            predicted_classes: Predicted classes (not used)
            confidences: Prediction confidences (not used)
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        # Compute reconstruction errors
        if hasattr(self._detector, 'compute_reconstruction_error'):
            errors = self._detector.compute_reconstruction_error(embeddings)
        elif hasattr(self._detector, 'predict'):
            # Some implementations might use predict
            errors = self._detector.predict(embeddings)
        else:
            # Fallback: compute manually
            errors = np.linalg.norm(
                embeddings - self._detector.decode(self._detector.encode(embeddings)),
                axis=1
            )

        # Normalize errors to 0-1 range
        if len(errors) > 0:
            max_error = np.max(errors)
            if max_error > 0:
                normalized_errors = errors / max_error
            else:
                normalized_errors = errors
        else:
            normalized_errors = errors

        for idx, error in enumerate(normalized_errors):
            is_novel = error > self._config.threshold

            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "self_knowledge_reconstruction_error": float(error),
                "self_knowledge_novelty_score": float(error),
                "self_knowledge_is_novel": is_novel,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return SelfKnowledgeConfig as the config schema."""
        return SelfKnowledgeConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Self-knowledge is experimental but can be useful
        return 0.15
