"""
Novel class detection and proposal system.

This module provides functionality for detecting novel classes in text data
and proposing meaningful new category names using LLMs.
"""

from .detector import NoveltyDetector, DetectionStrategy
from .entity_matcher import (
    NovelEntityMatcher,
    NovelEntityMatchResult,
    NoveltyMatchResult,
    create_novel_entity_matcher,
)
from .detector_api import NovelClassDetector
from .llm_proposer import LLMClassProposer
from .schemas import (
    ClassProposal,
    NovelClassAnalysis,
    NovelSampleReport,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
)
from .storage import save_proposals, load_proposals, list_proposals
from .ann_index import ANNIndex, ANNBackend

__all__ = [
    # Detector
    "NoveltyDetector",
    "DetectionStrategy",
    "NovelEntityMatcher",
    "NovelEntityMatchResult",
    "NoveltyMatchResult",
    "create_novel_entity_matcher",
    # Unified API
    "NovelClassDetector",
    # LLM Proposer
    "LLMClassProposer",
    # Schemas
    "ClassProposal",
    "NovelClassAnalysis",
    "NovelSampleReport",
    "NovelClassDiscoveryReport",
    "NovelSampleMetadata",
    # Storage
    "save_proposals",
    "load_proposals",
    "list_proposals",
    # ANN Index
    "ANNIndex",
    "ANNBackend",
]
