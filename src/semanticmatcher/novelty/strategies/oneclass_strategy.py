"""
Backward-compatible exports for the one-class novelty detector.
"""

from .oneclass_impl import OneClassSVMDetector

OneClassNoveltyDetector = OneClassSVMDetector

__all__ = ["OneClassNoveltyDetector"]
