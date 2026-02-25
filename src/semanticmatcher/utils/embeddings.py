from typing import List, Iterator
import numpy as np
from sentence_transformers import SentenceTransformer

__all__ = ["compute_embeddings", "cosine_sim", "batch_encode"]


def compute_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"
) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    return model.encode(texts)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def batch_encode(
    texts: List[str],
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    batch_size: int = 32
) -> Iterator[np.ndarray]:
    """Encode texts in batches."""
    model = SentenceTransformer(model_name)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        yield model.encode(batch)
