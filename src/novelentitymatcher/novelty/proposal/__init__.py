"""
Class proposal functionality for novel samples.

This module contains proposer implementations that suggest
new classes for novel samples.
"""

from .llm import LLMClassProposer
from .retrieval import RetrievalAugmentedProposer

__all__ = [
    "LLMClassProposer",
    "RetrievalAugmentedProposer",
]
