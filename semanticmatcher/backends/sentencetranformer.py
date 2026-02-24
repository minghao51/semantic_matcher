"""Backward-compatible import shim for the misspelled module name."""

from .sentencetransformer import HFReranker, HFEmbedding

__all__ = ["HFEmbedding", "HFReranker"]
