# Tech Stack

## Languages & Runtime

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.9-3.12 | Primary language |
| **Package Manager** | uv | Modern Python package manager |
| **Build System** | Hatchling | PEP 517 build backend |
| **CI/CD** | GitHub Actions | Lint, test, publish workflows |

## Core Dependencies

### Machine Learning & NLP

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | >=3.0.0 | Text embeddings |
| `setfit` | >=1.0.0 | Few-shot learning classification |
| `torch` | >=2.0.0 | Deep learning framework |
| `datasets` | >=2.14.0 | HuggingFace datasets |
| `scikit-learn` | >=1.3.0 | ML utilities (cosine similarity, TF-IDF) |
| `numpy` | >=2.0.0 | Numerical computing |
| `nltk` | >=3.9.2 | Natural language processing |

### Data Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=2.0.0 | Data manipulation |
| `networkx` | >=3.0,<4.0 | Graph operations (hierarchical matching) |
| `rapidfuzz` | >=3.0.0 | Fast fuzzy string matching |
| `rank-bm25` | >=0.2.2 | BM25 ranking algorithm |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | >=2.31.0 | HTTP client |
| `pyyaml` | >=6.0.0 | YAML configuration |

### Development Tools

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.4.2 | Testing framework |
| `pytest-cov` | latest | Coverage reporting |
| `ruff` | >=0.1.0 | Python linter |
| `black` | >=23.0.0 | Code formatter |
| `patchright` | >=1.58.0 | Browser automation (Playwright fork) |
| `beautifulsoup4` | >=4.14.3 | HTML parsing |
| `requests-mock` | latest | HTTP mocking in tests |

## Configuration

### Build Configuration
- **File**: `pyproject.toml`
- **Dependencies**: Managed via `dependencies` and `optional-dependencies`
- **Python versions**: 3.9, 3.10, 3.11, 3.12

### Linting & Formatting
- **Formatter**: Black (line length 100)
- **Linter**: Ruff (selects E, F, I, UP, B, C4, DTZ, TID, SIM, PT, RSE)
- **Type checking**: Type hints mandatory (via ruff)

### CI/CD Configuration
- **Workflows**: `.github/workflows/`
  - `lint.yml` - Code quality checks
  - `test.yml` - Test suite execution
  - `publish.yml` - PyPI publishing

### Application Configuration
- **File**: `config.yaml`
- **Model Registries**: Embeddings and rerankers configuration
- **CLI Tool**: `semanticmatcher-ingest`

## Entry Points

### Library Entry Point
```python
from semanticmatcher import Matcher
```

### CLI Entry Point
```bash
semanticmatcher-ingest
```

## Optional Dependencies

### LiteLLM Integration
- **Package**: `litellm` (optional)
- **Purpose**: Multi-provider LLM API integration
- **Usage**: Embedding and reranking via API
- **Installation**: `uv pip install semanticmatcher[litellm]`

## Development Tools

### Package Management
```bash
uv sync              # Install/sync dependencies
uv run <command>     # Execute within managed environment
uv add <package>     # Add dependency to pyproject.toml
```

### Code Quality
```bash
uv run ruff check    # Lint code
uv run black .       # Format code
uv run pytest        # Run tests
```

## Technology Choices

### Why uv?
- 10-100x faster than pip
- Modern Python package manager
- Better dependency resolution

### Why SetFit?
- Few-shot learning without fine-tuning
- Efficient for entity classification tasks
- Good performance with minimal examples

### Why NetworkX?
- Robust graph algorithms
- Well-maintained library
- Ideal for hierarchical matching

### Why Multiple Embedding Models?
- Task-specific model selection
- Performance vs accuracy tradeoffs
- Multilingual support requirements
