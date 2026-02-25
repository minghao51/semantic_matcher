from .core.matcher import EntityMatcher, EmbeddingMatcher
from .core.classifier import SetFitClassifier
from .core.normalizer import TextNormalizer
from .core.reranker import CrossEncoderReranker
from .core.hybrid import HybridMatcher
from .core.blocking import (
    BlockingStrategy,
    BM25Blocking,
    TFIDFBlocking,
    FuzzyBlocking,
    NoOpBlocking,
)

__version__ = "0.1.0"

__all__ = [
    "EntityMatcher",
    "EmbeddingMatcher",
    "SetFitClassifier",
    "TextNormalizer",
    "CrossEncoderReranker",
    "HybridMatcher",
    "BlockingStrategy",
    "BM25Blocking",
    "TFIDFBlocking",
    "FuzzyBlocking",
    "NoOpBlocking",
]
