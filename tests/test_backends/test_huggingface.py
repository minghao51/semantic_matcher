import pytest

from semanticmatcher.backends.sentencetransformer import HFEmbedding, HFReranker


pytestmark = [pytest.mark.hf, pytest.mark.integration, pytest.mark.slow]


class TestHFEmbedding:
    """Tests for HuggingFace SentenceTransformer embedding backend."""

    def test_hf_embedding_init(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        assert backend.model is not None

    def test_hf_embedding_encode(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = backend.encode(["hello world"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0

    def test_hf_embedding_encode_multiple(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = backend.encode(["hello", "world", "foo"])
        assert len(embeddings) == 3

    def test_hf_embedding_similarity(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = backend.encode(["cat", "dog"])
        results = backend.similarity(embeddings[:1], embeddings, top_k=2)
        assert len(results) == 1
        assert len(results[0]) == 2


class TestHFReranker:
    """Tests for HuggingFace CrossEncoder reranker backend."""

    def test_hf_reranker_init(self):
        backend = HFReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert backend.model is not None

    def test_hf_reranker_score(self):
        backend = HFReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = backend.score("What is the capital of France?", ["Paris", "London", "Berlin"])
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(isinstance(s, (int, float)) for s in scores)
