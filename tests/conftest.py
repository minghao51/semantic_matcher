import pytest
from semanticmatcher.utils.embeddings import get_default_cache


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the global model cache before each test."""
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()
