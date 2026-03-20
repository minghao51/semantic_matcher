"""
Backward-compatible exports for the prototypical novelty detector.
"""

from .prototypical_impl import PrototypicalDetector

PrototypicalNoveltyDetector = PrototypicalDetector

__all__ = ["PrototypicalNoveltyDetector"]
