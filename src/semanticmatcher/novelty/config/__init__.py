"""
Configuration for novelty detection.

This module contains all configuration classes for novelty detection,
including the main DetectionConfig and per-strategy configurations.
"""

from .base import DetectionConfig
from .strategies import (
    ConfidenceConfig,
    KNNConfig,
    ClusteringConfig,
    SelfKnowledgeConfig,
    PatternConfig,
    OneClassConfig,
    PrototypicalConfig,
    SetFitConfig,
)
from .weights import WeightConfig

__all__ = [
    "DetectionConfig",
    "ConfidenceConfig",
    "KNNConfig",
    "ClusteringConfig",
    "SelfKnowledgeConfig",
    "PatternConfig",
    "OneClassConfig",
    "PrototypicalConfig",
    "SetFitConfig",
    "WeightConfig",
]
