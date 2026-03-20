"""
Report dataclasses for novelty detection.

This module re-exports the main report classes for convenience.
"""

from .results import (
    NovelSampleReport,
    DetectionReport,
    EvaluationReport,
)

__all__ = [
    "NovelSampleReport",
    "DetectionReport",
    "EvaluationReport",
]
