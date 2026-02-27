from .base import EmbeddingBackend, RerankerBackend
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import semantic_search


class HFEmbedding(EmbeddingBackend):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts).tolist()

    def similarity(self, query_embeddings, corpus_embeddings, top_k):
        # Hook for future ANN backends (FAISS/Annoy) if needed.
        return semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)


class HFReranker(RerankerBackend):
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def score(self, query, docs):
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(score) for score in scores]
