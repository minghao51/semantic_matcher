# Integrations

## External APIs

### LLM Providers

#### OpenRouter
- **API Key**: `OPENROUTER_API_KEY`
- **Base URL**: `https://openrouter.ai/api/v1`
- **Models**: Claude Sonnet-4, GPT-4o, Gemini Pro 1.5
- **Usage**: Novel class naming via LLM proposer

#### Anthropic
- **API Key**: `ANTHROPIC_API_KEY`
- **Base URL**: `https://api.anthropic.com`
- **Models**: Claude Sonnet-4, Claude Opus-4, Claude Haiku
- **Usage**: Primary Claude integration

#### OpenAI
- **API Key**: `OPENAI_API_KEY`
- **Base URL**: `https://api.openai.com/v1`
- **Models**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- **Usage**: GPT model support

### LiteLLM Integration
- **Unified API** - Multi-provider LLM access
- **Embeddings** - Text embedding generation
- **Re-ranking** - Cross-encoding reranking
- **API key management** - Environment variable support

## Database & Storage

### Vector Databases
- **FAISS** - Local vector similarity search
- **HNSWlib** - Hierarchical Navigable Small World graphs
- **In-memory storage** - Temporary embeddings and indices

### File Storage
- **YAML** - Configuration files (config.yml, model registry)
- **JSON** - Entity data persistence, ingestion datasets
- **Parquet** - Cached embeddings storage

### Model Storage
- **Hugging Face Hub** - Model repository access and caching
- **Local cache** - `~/.cache/huggingface/` for downloaded models

## External Services

### Data Sources
- **Hugging Face Hub** - Model repository
- **GitHub** - External data fetching (raw.githubusercontent.com)
- **Government APIs** - Country data, timezone information
- **Public datasets** - Industries, languages, currencies

### CI/CD
- **GitHub Actions** - Multi-version testing matrix (Python 3.9-3.12)
- **PyPI** - Package distribution
- **Hatch** - Build and release automation

## Authentication & Security

### Environment Variables
- `ANTHROPIC_API_KEY` - Anthropic Claude access
- `OPENAI_API_KEY` - OpenAI GPT access
- `OPENROUTER_API_KEY` - OpenRouter multi-provider access
- `SEMANTIC_MATCHER_VERBOSE` - Logging verbosity control

### Secret Management
- Environment variable references in config
- No hardcoded credentials in source code
- API key rotation support via multiple providers

## Data Ingestion Sources

### Pre-built Datasets
- **Industries** - Industry classification data
- **Languages** - Language codes and names
- **Timezones** - Global timezone database
- **Currencies** - Currency codes and symbols

### External APIs (20+ URLs)
- GitHub raw content for datasets
- Government databases for country/currency data
- Public APIs for reference data

## Testing & Monitoring

### Testing Framework
- **PyTest** - Unit and integration tests
- **pytest-asyncio** - Async test support
- **pytest-cov** - Coverage reporting

### CI/CD Pipeline
- **GitHub Actions** - Automated testing
- **Multi-version matrix** - Python 3.9, 3.10, 3.11, 3.12
- **Integration markers** - `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.hf`

### Performance Monitoring
- **Benchmarking utilities** - `utils/benchmarks.py`
- **Memory profiling** - Test coverage for memory usage
- **Timing metrics** - Performance validation

## Model Management

### Model Registry
- **13+ pre-configured models** - Multi-language support
- **Auto-recommendation** - Use case-based model selection
- **Version management** - Model compatibility tracking
- **Dynamic loading** - Lazy model initialization

### Model Categories
- **English models** - MiniLM, MPNet, all-MiniLM-L12-v2
- **Multilingual** - paraphrase-multilingual, labse
- **Static embeddings** - Model2Vec distilbert
- **Specialized** - SetFit for few-shot learning
