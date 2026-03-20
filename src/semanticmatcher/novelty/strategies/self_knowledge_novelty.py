"""
Backward-compatible exports for the self-knowledge novelty detector.
"""

from .self_knowledge_impl import SelfKnowledgeDetector, SparseAutoencoder

__all__ = ["SelfKnowledgeDetector", "SparseAutoencoder"]
