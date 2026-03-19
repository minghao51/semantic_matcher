"""Compatibility shim for the former benchmark-local NovelEntityMatcher."""

from ..novelty.entity_matcher import (
    NovelEntityMatcher,
    NovelEntityMatchResult,
    NoveltyMatchResult,
    create_novel_entity_matcher,
)

__all__ = [
    "NovelEntityMatcher",
    "NovelEntityMatchResult",
    "NoveltyMatchResult",
    "create_novel_entity_matcher",
]
