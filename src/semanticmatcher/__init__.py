from importlib import import_module

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("semantic-matcher")
except Exception:  # pragma: no cover - fallback for local source usage
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

_EXPORTS = {
    "EntityMatcher": (".core.matcher", "EntityMatcher"),
    "EmbeddingMatcher": (".core.matcher", "EmbeddingMatcher"),
    "SetFitClassifier": (".core.classifier", "SetFitClassifier"),
    "TextNormalizer": (".core.normalizer", "TextNormalizer"),
    "CrossEncoderReranker": (".core.reranker", "CrossEncoderReranker"),
    "HybridMatcher": (".core.hybrid", "HybridMatcher"),
    "BlockingStrategy": (".core.blocking", "BlockingStrategy"),
    "BM25Blocking": (".core.blocking", "BM25Blocking"),
    "TFIDFBlocking": (".core.blocking", "TFIDFBlocking"),
    "FuzzyBlocking": (".core.blocking", "FuzzyBlocking"),
    "NoOpBlocking": (".core.blocking", "NoOpBlocking"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
