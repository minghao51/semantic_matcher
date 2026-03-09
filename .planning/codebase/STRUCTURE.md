# Semantic Matcher Codebase Structure

## Directory Layout

```
semantic_matcher/
├── .github/                   # GitHub workflows & CI/CD
│   └── workflows/
│       └── lint.yml          # Linting workflow
│
├── .planning/                 # Planning and analysis documents
│   └── codebase/
│       ├── ARCHITECTURE.md   # This file
│       └── STRUCTURE.md      # Project structure documentation
│
├── checkpoints/               # Model training checkpoints
│
├── data/                      # Data storage
│   ├── raw/                  # Raw downloaded datasets
│   │   ├── currencies/
│   │   ├── industries/
│   │   ├── languages/
│   │   ├── occupations/
│   │   ├── products/
│   │   ├── timezones/
│   │   └── universities/
│   └── processed/            # Processed/ingested data
│       ├── currencies/
│       ├── industries/
│       ├── languages/
│       ├── occupations/
│       ├── products/
│       ├── timezones/
│       └── universities/
│
├── docs/                      # Documentation
│   ├── architecture/         # Architecture documentation
│   │   └── hierarchical-matching.md
│   ├── plans/                # Project plans and designs
│   │   ├── 2026-03-04-hierarchical-entity-categorization.md
│   │   └── 2026-03-04-hierarchical-entity-categorization-design.md
│   ├── architecture.md       # Main architecture docs
│   ├── examples.md           # Usage examples
│   ├── index.md              # Documentation index
│   ├── migration-guide.md    # API migration guide
│   ├── notebooks.md          # Notebook documentation
│   ├── quickstart.md         # Quick start guide
│   ├── troubleshooting.md    # Troubleshooting guide
│   └── *.md                  # Other documentation files
│
├── examples/                  # Usage examples and demos
│   ├── basic_usage.py
│   ├── embedding_matcher_demo.py
│   ├── hybrid_matching_demo.py
│   ├── zero_shot_classification.py
│   ├── threshold_tuning.py
│   ├── entity_matcher_demo.py
│   ├── matcher_comparison.py
│   ├── model_persistence.py
│   ├── batch_processing.py
│   ├── country_matching.py
│   ├── hierarchical_matching_example.py
│   └── custom_backend.py
│
├── experiments/               # Experimental scripts and notebooks
│   └── country_classifier/
│       ├── country_classifier.py
│       ├── country_classifier_advanced.py
│       └── country_classifier_quick.py
│
├── notebooks/                 # Jupyter notebooks
│
├── src/semanticmatcher/      # Main package source
│   ├── __init__.py           # Public API (lazy exports)
│   ├── config.py             # Configuration management
│   ├── exceptions.py         # Custom exception classes
│   │
│   ├── backends/             # ML backend abstractions
│   │   ├── __init__.py       # Backend factory functions
│   │   ├── base.py           # Abstract base classes
│   │   ├── litellm.py        # LiteLLM backend
│   │   ├── reranker_st.py    # SentenceTransformer reranker
│   │   └── sentencetransformer.py  # HF sentence-transformer backend
│   │
│   ├── core/                 # Core matching logic
│   │   ├── __init__.py       # Core module exports
│   │   ├── blocking.py       # Blocking strategies (BM25, TF-IDF, Fuzzy)
│   │   ├── classifier.py     # SetFit classifier wrapper
│   │   ├── hierarchy.py      # Hierarchical matching (DAG-based)
│   │   ├── hybrid.py         # Hybrid matcher (3-stage pipeline)
│   │   ├── matcher.py        # Main matcher classes (Matcher, EntityMatcher, EmbeddingMatcher)
│   │   ├── monitoring.py     # Performance monitoring
│   │   ├── normalizer.py     # Text normalization
│   │   └── reranker.py       # Cross-encoder reranker
│   │
│   ├── data/                 # Package data
│   │   └── __init__.py
│   │
│   ├── ingestion/            # Data ingestion scripts
│   │   ├── __init__.py       # Ingestion module exports
│   │   ├── base.py           # Base ingestion classes
│   │   ├── cli.py            # CLI for data ingestion
│   │   ├── currencies.py     # Currency data ingestion
│   │   ├── industries.py     # Industry data ingestion
│   │   ├── languages.py      # Language data ingestion
│   │   ├── occupations.py    # Occupation data ingestion
│   │   ├── products.py       # Product data ingestion
│   │   ├── timezones.py      # Timezone data ingestion
│   │   └── universities.py   # University data ingestion
│   │
│   └── utils/                # Utility functions
│       ├── __init__.py       # Utils module exports
│       ├── benchmarks.py     # Performance benchmarking
│       ├── embeddings.py     # Embedding utilities & caching
│       ├── preprocessing.py  # Text preprocessing
│       └── validation.py     # Input validation
│
├── tests/                     # Test suite
│   ├── conftest.py           # Pytest configuration & fixtures
│   ├── test_config.py        # Config tests
│   ├── test_packaging.py     # Packaging tests
│   │
│   ├── fixtures/             # Test fixtures and data
│   │
│   ├── test_backends/        # Backend tests
│   │   ├── test_backend_imports.py
│   │   ├── test_huggingface.py
│   │   ├── test_litellm.py
│   │   └── test_reranker_contracts.py
│   │
│   ├── test_core/            # Core matcher tests
│   │   ├── test_classifier.py
│   │   ├── test_hierarchy.py
│   │   ├── test_matcher.py
│   │   └── test_normalizer.py
│   │
│   ├── test_ingestion/       # Ingestion tests
│   │   ├── test_cli.py
│   │   └── test_timezones.py
│   │
│   └── test_utils/           # Utility tests
│       ├── test_embeddings.py
│       ├── test_preprocessing.py
│       └── test_validation.py
│
├── .claude/                   # Claude-specific configuration
├── .gitignore                 # Git ignore rules
├── .python-version            # Python version (3.13)
├── .ruff_cache/               # Ruff linting cache
├── .pytest_cache/             # Pytest cache
├── .venv/                     # Virtual environment
├── CLAUDE.md                  # Project guidelines for Claude
├── config.yaml                # Default configuration
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT License
├── pyproject.toml             # Project configuration & dependencies
├── README.md                  # Project README
└── uv.lock                    # UV lock file for dependencies
```

## Key Locations Summary

### Entry Points

| Location | Purpose |
|----------|---------|
| `src/semanticmatcher/__init__.py` | Main package entry point (lazy exports) |
| `src/semanticmatcher/ingestion/cli.py` | CLI entry point (`semanticmatcher-ingest`) |
| `pyproject.toml` | Package configuration and entry points |

### Core Business Logic

| Location | Purpose |
|----------|---------|
| `src/semanticmatcher/core/matcher.py` | Main matcher classes (3650+ lines) |
| `src/semanticmatcher/core/classifier.py` | SetFit classifier wrapper |
| `src/semanticmatcher/core/hybrid.py` | Hybrid matching pipeline |
| `src/semanticmatcher/core/hierarchy.py` | Hierarchical entity matching |
| `src/semanticmatcher/core/blocking.py` | Blocking strategies |
| `src/semanticmatcher/core/reranker.py` | Cross-encoder reranking |

### Backend Abstractions

| Location | Purpose |
|----------|---------|
| `src/semanticmatcher/backends/base.py` | Abstract backend interfaces |
| `src/semanticmatcher/backends/sentencetransformer.py` | HF sentence-transformer backend |
| `src/semanticmatcher/backends/reranker_st.py` | SentenceTransformer reranker |
| `src/semanticmatcher/backends/litellm.py` | LiteLLM integration |

### Configuration & Utilities

| Location | Purpose |
|----------|---------|
| `src/semanticmatcher/config.py` | Configuration management & model registries |
| `src/semanticmatcher/utils/validation.py` | Input validation with helpful errors |
| `src/semanticmatcher/utils/embeddings.py` | Embedding utilities & model caching |
| `src/semanticmatcher/utils/preprocessing.py` | Text preprocessing utilities |
| `src/semanticmatcher/exceptions.py` | Custom exception hierarchy |

### Data Ingestion

| Location | Purpose |
|----------|---------|
| `src/semanticmatcher/ingestion/cli.py` | CLI for data ingestion |
| `src/semanticmatcher/ingestion/*.py` | Domain-specific ingestion scripts |

### Testing

| Location | Purpose |
|----------|---------|
| `tests/test_core/` | Core matcher tests |
| `tests/test_backends/` | Backend contract tests |
| `tests/test_utils/` | Utility function tests |
| `tests/test_ingestion/` | Data ingestion tests |

### Documentation

| Location | Purpose |
|----------|---------|
| `docs/architecture.md` | Architecture overview |
| `docs/quickstart.md` | Quick start guide |
| `docs/migration-guide.md` | API migration guide |
| `docs/examples.md` | Usage examples |
| `docs/troubleshooting.md` | Troubleshooting guide |

### Examples & Demos

| Location | Purpose |
|----------|---------|
| `examples/basic_usage.py` | Basic usage example |
| `examples/hybrid_matching_demo.py` | Hybrid matching demo |
| `examples/hierarchical_matching_example.py` | Hierarchical matching demo |
| `examples/*.py` | Other usage examples |

### Configuration Files

| Location | Purpose |
|----------|---------|
| `pyproject.toml` | Project metadata, dependencies, build config |
| `config.yaml` | Default configuration values |
| `uv.lock` | Dependency lock file |
| `.github/workflows/lint.yml` | CI/CD linting workflow |

## File Naming Conventions

### Python Files
- **Modules**: `lowercase_with_underscores.py` (e.g., `matcher.py`, `blocking.py`)
- **Classes**: `CapitalizedWords` (e.g., `Matcher`, `EmbeddingMatcher`)
- **Functions/Methods**: `lowercase_with_underscores` (e.g., `fit()`, `match()`)
- **Constants**: `UPPERCASE_WITH_UNDERSCORES` (e.g., `MODEL_REGISTRY`)

### Documentation Files
- **Format**: `YYYYMMDD-filename.md` (e.g., `20260228-examples-fixes-report.md`)
- **Exception**: Core docs use simple names (e.g., `architecture.md`, `quickstart.md`)

### Test Files
- **Format**: `test_<module>.py` (e.g., `test_matcher.py`, `test_validation.py`)
- **Location**: Mirror source structure in `tests/` directory

## Import Patterns

### Public API (User-Facing)
```python
from semanticmatcher import Matcher  # Recommended
from semanticmatcher import EntityMatcher  # Deprecated but available
from semanticmatcher import EmbeddingMatcher  # Deprecated but available
from semanticmatcher import SetFitClassifier
from semanticmatcher import HierarchicalMatcher
```

### Internal Imports
```python
from semanticmatcher.core.matcher import Matcher, EntityMatcher, EmbeddingMatcher
from semanticmatcher.core.classifier import SetFitClassifier
from semanticmatcher.utils.validation import validate_entities
from semanticmatcher.config import resolve_model_alias
```

### Backend Imports
```python
from semanticmatcher.backends import get_embedding_backend, get_reranker_backend
from semanticmatcher.backends.base import EmbeddingBackend, RerankerBackend
```

## Module Dependencies

### Core Dependencies
```
matcher.py
  ├── classifier.py (SetFitClassifier)
  ├── normalizer.py (TextNormalizer)
  ├── utils/validation.py (validation functions)
  ├── utils/embeddings.py (ModelCache)
  └── config.py (model aliases)

hybrid.py
  ├── matcher.py (EmbeddingMatcher)
  ├── reranker.py (CrossEncoderReranker)
  └── blocking.py (BlockingStrategy)

hierarchy.py
  ├── matcher.py (EmbeddingMatcher)
  └── normalizer.py (TextNormalizer)
```

### Backend Dependencies
```
backends/base.py (abstract interfaces)
  ├── backends/sentencetransformer.py (concrete implementation)
  ├── backends/reranker_st.py (concrete implementation)
  └── backends/litellm.py (concrete implementation)
```

## Important Patterns

### Lazy Loading Pattern (`__init__.py`)
- Defers imports until first access
- Reduces startup time
- Enables circular dependency resolution
- Provides deprecation warnings

### Factory Pattern (`backends/__init__.py`)
```python
def get_embedding_backend(provider, model, **kwargs) -> EmbeddingBackend:
    if provider == "huggingface":
        return HFEmbedding(model)
```

### Strategy Pattern (`core/blocking.py`)
```python
class BlockingStrategy(ABC):
    @abstractmethod
    def block(self, query, entities, top_k):
        pass

class BM25Blocking(BlockingStrategy):
    def block(self, query, entities, top_k):
        # BM25 implementation
```

### Registry Pattern (`config.py`)
```python
MODEL_REGISTRY = {
    "default": "sentence-transformers/all-mpnet-base-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
}
```

## Testing Structure

### Test Organization
- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Contract Tests**: Test backend interface compliance
- **Marker System**: `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.hf`

### Test Fixtures
- Located in `tests/fixtures/`
- Shared test data and utilities
- Configured in `tests/conftest.py`

## Build & Packaging

### Build System
- **Tool**: Hatchling
- **Source**: `src/semanticmatcher/`
- **Entry Points**: Defined in `pyproject.toml`

### Distribution
- **Wheel**: `semantic_matcher-*.whl`
- **Source**: `semantic_matcher-*.tar.gz`
- **Python Versions**: 3.9, 3.10, 3.11, 3.12

## Development Workflow

### Adding New Features
1. Implement in `src/semanticmatcher/core/` or appropriate module
2. Add exports to `src/semanticmatcher/__init__.py`
3. Write tests in `tests/test_*/`
4. Update documentation in `docs/`
5. Add examples in `examples/`

### Adding New Backends
1. Create class inheriting from `EmbeddingBackend` or `RerankerBackend`
2. Implement required abstract methods
3. Add factory function in `backends/__init__.py`
4. Add tests in `tests/test_backends/`

### Adding New Ingestion Sources
1. Create module in `ingestion/` following existing pattern
2. Implement `run_*()` function
3. Add to `INGESTORS` dict in `ingestion/cli.py`
4. Add tests in `tests/test_ingestion/`

## Performance Considerations

### Caching
- **Model Cache**: `utils/embeddings.py` - Thread-safe LRU cache
- **Backend Caching**: Models cached to reduce loading overhead

### Lazy Initialization
- **Matcher Classes**: Matchers created only when needed
- **Module Imports**: Defers imports via `__getattr__`

### Batch Processing
- **EmbeddingMatcher**: Supports `batch_size` parameter
- **HybridMatcher**: Parallel bulk matching with `n_jobs`

## Configuration Locations

### Runtime Configuration
- **Default**: `config.yaml` (repo root)
- **Package**: `src/semanticmatcher/data/default_config.json`
- **CWD**: `./config.yaml`
- **Custom**: Via `Config(custom_path=path)`

### Model Registries
- **Embedding Models**: `config.py` - `MODEL_REGISTRY`
- **Reranker Models**: `config.py` - `RERANKER_REGISTRY`
- **Matcher Modes**: `config.py` - `MATCHER_MODE_REGISTRY`

## Documentation Structure

### User Documentation
- **Quick Start**: `docs/quickstart.md`
- **Examples**: `docs/examples.md`, `examples/*.py`
- **Migration**: `docs/migration-guide.md`
- **Troubleshooting**: `docs/troubleshooting.md`

### Developer Documentation
- **Architecture**: `docs/architecture.md`, `docs/architecture/*.md`
- **Planning**: `docs/plans/*.md`
- **Analysis**: `.planning/codebase/*.md`

### API Documentation
- **Public API**: `src/semanticmatcher/__init__.py`
- **Internal APIs**: Docstrings in source files
- **Type Hints**: Throughout codebase

## Security & Best Practices

### Input Validation
- **Location**: `utils/validation.py`
- **Pattern**: Validate early, fail fast with helpful errors
- **Coverage**: Entities, thresholds, model names, training data

### Error Handling
- **Location**: `exceptions.py`
- **Pattern**: Rich exceptions with context and suggestions
- **Hierarchy**: Base exception with specialized subclasses

### Dependency Management
- **Tool**: `uv`
- **Lock File**: `uv.lock`
- **Python**: `.python-version` (3.13)

## Key Files to Understand

### For New Contributors
1. `src/semanticmatcher/__init__.py` - Public API
2. `src/semanticmatcher/core/matcher.py` - Main matcher logic
3. `docs/quickstart.md` - How to use
4. `examples/basic_usage.py` - Working examples

### For Architecture Understanding
1. `docs/architecture.md` - Architecture overview
2. `src/semanticmatcher/core/hybrid.py` - Pipeline architecture
3. `src/semanticmatcher/backends/base.py` - Backend abstractions
4. `src/semanticmatcher/config.py` - Configuration system

### For Extending Functionality
1. `src/semanticmatcher/backends/base.py` - Backend interfaces
2. `src/semanticmatcher/core/blocking.py` - Strategy pattern example
3. `src/semanticmatcher/exceptions.py` - Exception patterns
4. `tests/test_backends/test_backend_imports.py` - Backend contract tests
