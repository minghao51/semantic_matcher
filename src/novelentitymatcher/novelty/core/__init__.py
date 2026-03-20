"""
Core novelty detection functionality.

This module contains the main detector orchestration, strategy registry,
and signal combination logic.
"""

from .detector import NoveltyDetector
from .strategies import StrategyRegistry
from .signal_combiner import SignalCombiner
from .metadata import MetadataBuilder

__all__ = [
    "NoveltyDetector",
    "StrategyRegistry",
    "SignalCombiner",
    "MetadataBuilder",
]
