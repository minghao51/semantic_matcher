"""
Storage functionality for novelty detection.

This module contains persistence and indexing utilities
for storing proposals and searching embeddings.
"""

from .persistence import save_proposals, load_proposals, list_proposals
from .index import ANNBackend, ANNIndex

__all__ = [
    "save_proposals",
    "load_proposals",
    "list_proposals",
    "ANNBackend",
    "ANNIndex",
]
