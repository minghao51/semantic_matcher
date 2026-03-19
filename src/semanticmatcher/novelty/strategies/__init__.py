"""
Novelty detection strategies.

This module contains the base strategy protocol and all concrete
strategy implementations for detecting novel entities.

Core Strategies:
- confidence: Confidence threshold
- knn_distance: kNN distance-based
- uncertainty: Margin/entropy uncertainty
- clustering: Clustering-based

Advanced Strategies:
- self_knowledge: Sparse autoencoder
- pattern: Pattern-based
- oneclass: One-Class SVM
- prototypical: Prototypical networks
- setfit: SetFit contrastive learning

Usage:
    from semanticmatcher.novelty import StrategyRegistry, DetectionConfig

    # List available strategies
    strategies = StrategyRegistry.list_strategies()

    # Use in configuration
    config = DetectionConfig(
        strategies=["confidence", "knn_distance", "clustering"],
    )

Importing Strategies:
    # Import specific strategies directly
    from semanticmatcher.novelty.strategies.confidence import ConfidenceStrategy
    from semanticmatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
"""

# Base protocol
from .base import NoveltyStrategy

# Import low-level strategy helpers that are still useful directly.
from .pattern_strategy import PatternBasedNoveltyStrategy, score_batch_novelty
from .oneclass_strategy import OneClassNoveltyDetector
from .prototypical_strategy import PrototypicalNoveltyDetector
from .setfit_novelty import SetFitNoveltyDetector

__all__ = [
    # Base
    "NoveltyStrategy",
    # Low-level strategy helpers
    "PatternBasedNoveltyStrategy",
    "score_batch_novelty",
    "OneClassNoveltyDetector",
    "PrototypicalNoveltyDetector",
    "SetFitNoveltyDetector",
]

# Note: New strategies (ConfidenceStrategy, KNNDistanceStrategy, etc.)
# are auto-registered via @StrategyRegistry.register decorator.
# Import them directly from their submodules when needed:
# from semanticmatcher.novelty.strategies.confidence import ConfidenceStrategy
