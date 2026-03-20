"""
Backward-compatible exports for the pattern novelty scorer.
"""

from .pattern_impl import PatternScorer, score_batch_novelty

PatternBasedNoveltyStrategy = PatternScorer

__all__ = ["PatternBasedNoveltyStrategy", "score_batch_novelty"]
