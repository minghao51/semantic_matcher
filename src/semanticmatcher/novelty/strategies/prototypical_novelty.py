"""
Backward-compatible exports for the prototypical novelty detector.
"""

from .prototypical_impl import PrototypicalDetector

PrototypicalStrategy = PrototypicalDetector
PrototypicalNoveltyDetector = PrototypicalDetector

__all__ = ["PrototypicalStrategy", "PrototypicalNoveltyDetector"]
