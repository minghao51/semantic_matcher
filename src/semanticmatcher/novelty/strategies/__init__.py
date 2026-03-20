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
# Note: Wrapper classes (PatternStrategy, OneClassStrategy, etc.) are NOT imported
# here to avoid circular imports. They are imported lazily in config/base.py.
from .pattern_impl import PatternScorer, score_batch_novelty
from .oneclass_impl import OneClassSVMDetector
from .prototypical_impl import PrototypicalDetector
from .setfit_impl import SetFitDetector
from .self_knowledge_impl import SelfKnowledgeDetector, SparseAutoencoder

__all__ = [
    # Base
    "NoveltyStrategy",
    # Low-level strategy helpers
    "PatternScorer",
    "score_batch_novelty",
    "OneClassSVMDetector",
    "PrototypicalDetector",
    "SetFitDetector",
    "SelfKnowledgeDetector",
    "SparseAutoencoder",
]

# Note: New strategies (ConfidenceStrategy, KNNDistanceStrategy, etc.)
# are auto-registered via @StrategyRegistry.register decorator.
# Import them directly from their submodules when needed:
# from semanticmatcher.novelty.strategies.confidence import ConfidenceStrategy
