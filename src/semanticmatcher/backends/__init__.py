from .base import EmbeddingBackend, RerankerBackend
from .sentencetransformer import HFEmbedding, HFReranker
from .reranker_st import STReranker
# from .ollama import OllamaEmbedding, OllamaReranker
# from .litellm import LiteLLMEmbedding, LiteLLMReranker

__all__ = [
    "EmbeddingBackend",
    "RerankerBackend",
    "HFEmbedding",
    "HFReranker",
    "STReranker",
    "get_embedding_backend",
    "get_reranker_backend",
]

def get_embedding_backend(provider: str, model: str, **kwargs) -> EmbeddingBackend:
    if provider == "huggingface":
        return HFEmbedding(model)
    # elif provider == "ollama":
    #     return OllamaEmbedding(model, api_base=kwargs.get("api_base", "http://localhost:11434"))
    # elif provider == "litellm":
    #     return LiteLLMEmbedding(model, api_key=kwargs.get("api_key"))
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

def get_reranker_backend(provider: str, model: str, **kwargs) -> RerankerBackend:
    if provider == "huggingface":
        return HFReranker(model)
    # elif provider == "ollama":
    #     return OllamaReranker(model, api_base=kwargs.get("api_base", "http://localhost:11434"))
    # elif provider == "litellm":
    #     return LiteLLMReranker(model, api_key=kwargs.get("api_key"))
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
