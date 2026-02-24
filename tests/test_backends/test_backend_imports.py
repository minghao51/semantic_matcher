from semanticmatcher.backends import HFReranker, HFEmbedding
from semanticmatcher.backends.sentencetranformer import (  # backward-compatible shim
    HFReranker as ShimHFReranker,
    HFEmbedding as ShimHFEmbedding,
)
from semanticmatcher.backends.sentencetransformer import (
    HFReranker as NewHFReranker,
    HFEmbedding as NewHFEmbedding,
)


def test_backend_exports_use_canonical_module():
    assert HFEmbedding is NewHFEmbedding
    assert HFReranker is NewHFReranker


def test_misspelled_backend_module_is_compat_shim():
    assert ShimHFEmbedding is NewHFEmbedding
    assert ShimHFReranker is NewHFReranker
