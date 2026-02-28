# Structure

## Directory Layout

```
semantic_matcher/
├── src/
│   └── semanticmatcher/          # Main package (import: `semanticmatcher`)
│       ├── __init__.py           # Lazy exports, package metadata
│       ├── config.py             # Model registries, config management
│       ├── backends/             # Model backend implementations
│       ├── core/                 # Core matching algorithms
│       ├── ingestion/            # Data ingestion scripts
│       └── utils/                # Shared utilities
├── tests/                        # Test suite
├── examples/                     # Usage examples
├── experiments/                  # Exploratory scripts
├── notebooks/                    # Jupyter notebooks (.ipynb only)
├── docs/                         # User documentation
├── data/                         # Datasets
│   ├── raw/                      # Downloaded source files
│   └── processed/                # Processed entity datasets
├── .planning/                    # Planning documents
└── pyproject.toml                # Package configuration
```

## Source Code Structure (`src/semanticmatcher/`)

### Package Root

**`__init__.py`** (47 lines)
- Lazy import system via `_EXPORTS` mapping
- Package version from metadata
- Public API exports

**`config.py`** (178 lines)
- `MODEL_REGISTRY` - Model alias mappings
- `RERANKER_REGISTRY` - Reranker model aliases
- `resolve_model_alias()` - Resolve short names to full HuggingFace IDs
- `resolve_reranker_alias()` - Resolve reranker aliases
- `recommend_model()` - Model recommendations based on use case
- `Config` class - YAML configuration loading

### Backends (`src/semanticmatcher/backends/`)

**`base.py`**
- `BaseBackend` (ABC) - Abstract interface for model backends

**`sentencetransformer.py`**
- `SentenceTransformerBackend` - Standard sentence transformer wrapper

**`reranker_st.py`**
- `STReranker` - Sentence-transformer cross-encoder reranker

**`litellm.py`**
- `LiteLLMBackend` - OpenAI-compatible API backend (stub, not active)

**`__init__.py`**
- Backend exports and initialization

### Core (`src/semanticmatcher/core/`)

**`matcher.py`** (320 lines) - **Largest core file**
- `EntityMatcher` - Few-shot matching with SetFit training
- `EmbeddingMatcher` - Pure embedding similarity matching
- `ModelCache` - Thread-safe model caching (moved to utils/embeddings in refactor)

**`hybrid.py`** (176 lines)
- `HybridMatcher` - Three-stage pipeline (blocking → retrieval → reranking)

**`blocking.py`** (205 lines)
- `BlockingStrategy` (ABC) - Abstract blocking interface
- `BM25Blocking` - BM25 lexical blocking
- `TFIDFBlocking` - TF-IDF vectorization blocking
- `FuzzyBlocking` - RapidFuzz approximate matching
- `NoOpBlocking` - Pass-through for small datasets

**`reranker.py`** (121 lines)
- `CrossEncoderReranker` - Cross-encoder reranking for precision

**`classifier.py`** (92 lines)
- `SetFitClassifier` - Few-shot classification wrapper

**`normalizer.py`** (small file)
- `TextNormalizer` - Text preprocessing and normalization

**`monitoring.py`** (137 lines)
- Performance monitoring and metrics

**`__init__.py`**
- Core module exports

### Ingestion (`src/semanticmatcher/ingestion/`)

**`base.py`** (70 lines)
- `BaseIngestor` (ABC) - Base class for data ingestion

**`cli.py`** (86 lines)
- CLI entry point: `semanticmatcher-ingest`
- Datasets: languages, currencies, industries, timezones, occupations, products, universities

**Individual Ingestion Modules:**
| File | Lines | Dataset |
|------|-------|---------|
| `languages.py` | 83 | ISO language codes |
| `currencies.py` | 78 | Currency codes |
| `timezones.py` | 135 | IANA timezones |
| `occupations.py` | 188 | O*NET and SOC codes |
| `industries.py` | 247 | Industry codes (NAICS, SIC) |
| `products.py` | 356 | UNSPSC product codes |
| `universities.py` | 371 | University world rankings |

**Note**: Products and Universities are the largest ingestion files with hardcoded fallback data.

### Utils (`src/semanticmatcher/utils/`)

**`embeddings.py`** (154 lines)
- `ModelCache` - Thread-safe model caching with TTL
- `get_default_cache()` - Singleton cache instance
- `compute_embeddings()` - Batch embedding computation

**`preprocessing.py`** (81 lines)
- Text preprocessing utilities

**`validation.py`** (small file)
- Input validation functions

**`benchmarks.py`** (242 lines)
- Performance benchmarking tools

**`__init__.py`**
- Utility exports

## Test Structure (`tests/`)

```
tests/
├── conftest.py                   # Pytest fixtures
├── fixtures/                     # Test data fixtures
│   ├── entities.json
│   └── training_data.json
├── test_backends/                # Backend tests
├── test_core/                    # Core matcher tests
├── test_ingestion/               # Ingestion tests
├── test_utils/                   # Utility tests
└── test_config.py                # Configuration tests
```

**Pytest Markers**:
- `integration` - Tests requiring network/external services
- `slow` - Expensive tests (disabled in default CI)
- `hf` - HuggingFace model-backed tests

## Documentation (`docs/`)

**User Guides:**
- `index.md` - Documentation index
- `quickstart.md` - Getting started guide
- `examples.md` - Example scripts overview
- `notebooks.md` - Jupyter notebook index
- `troubleshooting.md` - Common issues and fixes

**Reference:**
- `architecture.md` - System architecture documentation
- `country-classifier-scripts.md` - Country classifier workflow

## Examples (`examples/`)

| File | Purpose |
|------|---------|
| `basic_usage.py` | Raw SetFit training example |
| `country_matching.py` | Country-code matching |
| `custom_backend.py` | Backend model comparison |
| `hybrid_matching_demo.py` | Three-stage pipeline demo |
| `zero_shot_classification.py` | SetFit classification examples |

## Key File Locations

**Entry Points:**
- CLI: `src/semanticmatcher/ingestion/cli.py:main()`
- Package: `src/semanticmatcher/__init__.py`

**Core Logic:**
- Matching: `src/semanticmatcher/core/matcher.py`
- Hybrid Pipeline: `src/semanticmatcher/core/hybrid.py`
- Reranking: `src/semanticmatcher/core/reranker.py`

**Configuration:**
- Model Registries: `src/semanticmatcher/config.py`
- Package Config: `pyproject.toml`

**Utilities:**
- Model Cache: `src/semanticmatcher/utils/embeddings.py`
- Validation: `src/semanticmatcher/utils/validation.py`

## Naming Conventions

**Files:**
- `snake_case.py` for all Python files
- Module names: `matcher.py`, `blocking.py`, `hybrid.py`

**Classes:**
- `CamelCase` for all classes
- Examples: `EntityMatcher`, `BM25Blocking`, `CrossEncoderReranker`

**Functions/Methods:**
- `snake_case` for all functions and methods
- Examples: `match()`, `build_index()`, `resolve_model_alias()`

**Constants:**
- `UPPER_CASE` for constants
- Examples: `MODEL_REGISTRY`, `SOURCE_URL`, `DEFAULT_THRESHOLD`

**Private Members:**
- `_leading_underscore` for internal/private
- Examples: `_EXPORTS`, `_tokenize()`, `__getattr__()`

## Import Patterns

**Standard Layout:**
```python
# 1. Standard library
from typing import List, Dict, Any, Optional

# 2. Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# 3. Local imports
from .classifier import SetFitClassifier
from ..utils.validation import validate_entities
```

**Lazy Imports:**
- Package-level imports use `__getattr__` for lazy loading
- Avoids circular dependencies
- Faster initial import time

## Data Flow Through Files

**Ingestion**:
```
ingestion/cli.py → ingestion/[dataset].py → data/processed/
```

**Matching**:
```
core/matcher.py → backends/sentencetransformer.py → utils/embeddings.py (cache)
```

**Hybrid**:
```
core/hybrid.py → core/blocking.py → core/matcher.py → core/reranker.py
```
