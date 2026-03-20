"""
Utility functions for novelty detection.

This module contains shared utility functions used across
the novelty detection subsystem.
"""

from .scoring import normalize_score, compute_similarity

__all__ = [
    "normalize_score",
    "compute_similarity",
]
