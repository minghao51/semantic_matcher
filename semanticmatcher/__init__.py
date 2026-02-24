from .core.matcher import EntityMatcher, EmbeddingMatcher
from .core.classifier import SetFitClassifier
from .core.normalizer import TextNormalizer

__version__ = "0.1.0"

__all__ = [
    "EntityMatcher",
    "EmbeddingMatcher",
    "SetFitClassifier",
    "TextNormalizer",
]
