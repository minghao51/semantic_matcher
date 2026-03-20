"""
Backward-compatible exports for the SetFit novelty detector.
"""

from .setfit_impl import SetFitDetector

SetFitNoveltyDetector = SetFitDetector

__all__ = ["SetFitNoveltyDetector"]
