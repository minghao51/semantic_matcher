# Integrations

## External APIs & Services

### HuggingFace (Primary Integration)

**Purpose**: Model hub for pre-trained embeddings and rerankers

**Models Used**:

**Sentence Transformers (Embeddings)**:
- `BAAI/bge-m3` - Multilingual embeddings (8192 dimensions)
- `BAAI/bge-base-en-v1.5` - English embeddings (768 dimensions)
- `nomic-ai/nomic-embed-text-v1` - Long-context embeddings (768 dimensions)

**Cross-Encoders (Rerankers)**:
- `BAAI/bge-reranker-v2-m3` - Multilingual reranking
- `ms-marco-MiniLM-L-6-v2` - English reranking

**Integration Details**:
- Models downloaded and cached locally on first use
- Thread-safe model loading with `ModelCache` class
- No API keys required (models are public)
- Fallback to alternative models on download failure

**Configuration**:
```yaml
embeddings:
  default: BAAI/bge-m3
  models:
    - name: BAAI/bge-m3
      dimensions: 8192
```

---

### LiteLLM (Optional Integration)

**Purpose**: Multi-provider LLM API integration for embeddings and reranking

**Providers Supported** (via LiteLLM):
- OpenAI (`openai/`)
- Anthropic (`anthropic/`)
- Cohere (`cohere/`)
- Azure OpenAI (`azure/`)
- 100+ other providers

**Usage**:
```python
from semanticmatcher.backends import LiteLLMEmbeddingBackend

backend = LiteLLMEmbeddingBackend(
    model="openai/text-embedding-3-small",
    api_key="sk-..."
)
```

**API Key Handling**:
- **Current Implementation**: Sets `LITELLM_API_KEY` in `os.environ`
- **Security Concern**: Affects entire process (see CONCERNS.md)
- **Configuration**: Via environment variable or parameter

**Installation**:
```bash
uv pip install semanticmatcher[litellm]
```

---

## Public Data Sources (Ingestion Pipeline)

The project includes a data ingestion pipeline (`src/semanticmatcher/ingest/`) that downloads reference data from public sources.

### Data Sources

| Source | URL | Data | Purpose |
|--------|-----|------|---------|
| **GitHub Raw** | github.com | Industry codes, UNSPSC products, MCC codes | Entity classification |
| **BLS.gov** | bls.gov | SIC industry titles, SOC occupations | Occupation matching |
| **WorldTimeAPI** | worldtimeapi.org | Timezone offsets | Location data |
| **Datahub.io** | datahub.io | Language codes, currency codes | Validation data |
| **O*NET Center** | onetcenter.org | Occupation descriptions | Job matching |
| **UN Statistics** | unstats.un.org | Product/service codes (CPC) | Classification |
| **Wikidata SPARQL** | wikidata.org | University data | Institution matching |
| **Wikipedia** | wikipedia.org | Oldest universities list | Historical data |

### Ingestion Details

**Storage**: Data saved as CSV/JSON files in `data/` directory

**CLI Command**:
```bash
semanticmatcher-ingest
```

**Rate Limiting**:
- Respectful scraping with delays
- Error handling for network failures
- Retry logic for transient failures

**No Authentication Required**:
- All sources are publicly accessible
- No API keys needed
- No rate limits imposed

---

## Database Integrations

**Status**: None

**Details**:
- No SQL/NoSQL databases detected
- No ORM or database drivers
- Data stored as CSV/JSON files
- In-memory processing with pandas

**Future Considerations**:
- For large-scale deployments, consider:
  - PostgreSQL with pgvector for vector similarity
  - Qdrant/Milvus for vector databases
  - Redis for caching

---

## Authentication & Authorization

**Status**: Not Applicable

**Details**:
- No OAuth, JWT, or auth libraries
- No user management system
- No authentication required for HuggingFace models
- API keys only for optional LiteLLM integration

**Security**: N/A (library/tool, not a service)

---

## Webhooks & Real-time

**Status**: None

**Details**:
- No webhook handlers
- No WebSocket connections
- No message queue systems (RabbitMQ, Kafka, etc.)
- Batch processing only

---

## Payment & Billing

**Status**: None

**Details**:
- No payment processing (Stripe, PayPal, etc.)
- No billing integration
- Free HuggingFace models
- LiteLLM costs paid by user to their provider

---

## Third-Party Services Summary

| Service | Type | Auth Required | Purpose |
|---------|------|---------------|---------|
| HuggingFace Model Hub | Model Registry | No | Download models |
| LiteLLM | API Aggregator | Yes (optional) | Embedding/reranking APIs |
| Public Data Sources | Static Data | No | Reference datasets |

---

## Integration Patterns

### Model Loading Pattern
```python
# Thread-safe model cache
from semanticmatcher.backends.huggingface import ModelCache

cache = ModelCache()
model = cache.get_model("BAAI/bge-m3")
```

### Backend Selection Pattern
```python
# Auto-select backend based on configuration
from semanticmatcher import Matcher

matcher = Matcher()  # Automatically picks best backend
```

### Error Handling Pattern
```python
# Graceful fallback on model failures
try:
    model = load_model(primary_model)
except Exception:
    model = load_model(fallback_model)
```

---

## Security Considerations

### API Key Management
- **Current**: Environment variables (see CONCERNS.md for issues)
- **Recommendation**: Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
- **Never Commit**: API keys in code or config files

### Model Downloads
- Verify model checksums (not currently implemented)
- Download from official HuggingFace hub only
- Cache models to avoid repeated downloads

### Data Ingestion
- Validate all downloaded data
- Sanitize inputs to prevent injection attacks
- Use HTTPS for all network requests

---

## Monitoring & Observability

**Current**: None

**Recommendations**:
- Add structured logging (replace print statements)
- Metrics for model loading times
- Track embedding cache hit rates
- Monitor API usage for LiteLLM backend

---

## Future Integration Opportunities

1. **Vector Databases**: Qdrant, Milvus, pgvector for large-scale similarity search
2. **Message Queues**: Celery/RQ for async processing
3. **Object Storage**: S3/GCS for model and data storage
4. **Secrets Management**: HashiCorp Vault, AWS Secrets Manager
5. **Observability**: OpenTelemetry, Prometheus, Grafana
6. **Feature Flags**: LaunchDarkly, Unleash for A/B testing models
