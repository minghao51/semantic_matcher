# SemanticMatcher Integrations

## External APIs

### Hugging Face Hub
- **Purpose**: Model hosting and downloads
- **Models**:
  - sentence-transformers/paraphrase-mpnet-base-v2
  - BAAI/bge-m3
- **Auth**: None required for public models
- **Usage**: Automatic model download on first use
- **Location**: `semanticmatcher/core/classifier.py`

### LiteLLM (Optional)
- **Purpose**: Unified API for multiple LLM providers
- **Providers**: OpenAI, Anthropic, Cohere, and 100+ others
- **Auth**: API key via environment variable or parameter
- **Env Var**: `LITELLM_API_KEY`
- **Implementation**: `semanticmatcher/backends/litellm.py`
- **Features**:
  - Alternative embedding backend
  - Reranking capabilities

## Data Storage

### Local Storage
- **Models**: Saved via `save()` method
- **Config**: YAML file (`config.yaml`)
- **No Database**: All processing in-memory

## Authentication

### LiteLLM API Key
```python
# Via environment variable
os.environ["LITELLM_API_KEY"] = "your-key"

# Via parameter
from semanticmatcher.backends import LiteLLMEmbedding
backend = LiteLLMEmbedding(model="openai/embeddings", api_key="your-key")
```

## Backend Architecture

### Embedding Backends
1. **SentenceTransformer** (default)
   - Local execution
   - No API required
   - Models from Hugging Face

2. **LiteLLM** (optional)
   - Cloud API
   - Multiple providers
   - Requires API key

### Reranker Backend
- **LiteLLM Reranker** (optional)
  - Cloud-based reranking
  - Returns relevance scores

## Integration Points

### Public API
```python
# Main exports
from semanticmatcher import EntityMatcher, EmbeddingMatcher

# Components
from semanticmatcher import SetFitClassifier, TextNormalizer
from semanticmatcher.config import Config

# Optional backends
from semanticmatcher.backends import LiteLLMEmbedding, LiteLLMReranker
```

### Example Integrations
- `examples/country_matching.py`: Basic usage
- `examples/advanced_matching.py`: Custom backend usage

## External Services Summary
| Service | Required | Purpose | Auth |
|---------|----------|---------|------|
| Hugging Face Hub | Yes | Model downloads | None (public models) |
| LiteLLM | Optional | Embedding/reranking | API key |
