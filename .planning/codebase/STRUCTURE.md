# Structure

## Directory Layout

```
semantic_matcher/
├── src/
│   └── semanticmatcher/
│       ├── __init__.py              # Package entry with unified API
│       ├── config.py                # Model registry and configuration
│       ├── cli.py                   # Command-line interface (if exists)
│       │
│       ├── core/                    # Core matching logic
│       │   ├── matcher.py           # Main Matcher class (1,869 lines)
│       │   ├── classifier.py        # Entity classification
│       │   ├── normalizer.py        # Text preprocessing
│       │   └── blocking.py          # Candidate selection strategies
│       │
│       ├── backends/                # Backend implementations
│       │   ├── __init__.py          # Backend factory
│       │   ├── sentence_transformers.py  # Dynamic embeddings
│       │   ├── static_embeddings.py      # Model2Vec integration
│       │   ├── litellm.py              # LLM/embedding API
│       │   └── reranking.py            # Cross-encoder reranking
│       │
│       ├── ingestion/               # Data ingestion modules
│       │   ├── cli.py               # Ingestion CLI tool
│       │   ├── industries.py        # Industry data
│       │   ├── languages.py         # Language codes
│       │   ├── currencies.py        # Currency data
│       │   └── timezones.py         # Timezone database
│       │
│       ├── novelty/                 # Novelty detection system
│       │   ├── __init__.py          # Package exports
│       │   ├── detector_api.py      # NovelClassDetector API
│       │   ├── detector.py          # Detection strategies
│       │   ├── llm_proposer.py      # LLM-based class naming
│       │   └── schemas.py           # Pydantic configuration schemas
│       │
│       ├── utils/                   # Utilities
│       │   ├── __init__.py          # Utility exports
│       │   ├── validation.py        # Input validation
│       │   ├── embeddings.py        # Embedding utilities
│       │   ├── preprocessing.py     # Text preprocessing
│       │   ├── caching.py           # Caching utilities
│       │   └── benchmarks.py        # Performance benchmarks (1,000 lines)
│       │
│       └── data/                    # Static assets
│           ├── country_codes.py     # Country code mappings
│           └── default_config.yml   # Default configuration
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_matcher.py              # Matcher tests
│   ├── test_classifier.py           # Classifier tests
│   ├── test_normalizer.py           # Normalizer tests
│   ├── test_backends/               # Backend tests
│   ├── test_novelty/                # Novelty detection tests
│   ├── test_integration.py          # Integration tests
│   └── conftest.py                  # Test fixtures
│
├── examples/                        # Usage examples
│   ├── basic_matching.py
│   ├── novelty_detection.py
│   └── async_api.py
│
├── docs/                            # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── CHANGELOG.md
│
├── scripts/                         # Utility scripts
│   ├── setup_llm.sh
│   └── benchmark.py
│
├── pyproject.toml                   # Project configuration
├── ruff.toml                        # Linting rules
├── README.md                        # Project documentation
├── CHANGELOG.md                     # Version history
└── CLAUDE.md                        # Claude Code instructions
```

## Key Locations

### Entry Points
- **Main API**: `src/semanticmatcher/__init__.py` - Exports `Matcher` class
- **Core Logic**: `src/semanticmatcher/core/matcher.py` - Main implementation
- **CLI**: `src/semanticmatcher/ingestion/cli.py` - Ingestion commands
- **Novelty Detection**: `src/semanticmatcher/novelty/detector_api.py` - NovelClassDetector

### Configuration
- **Model Registry**: `src/semanticmatcher/config.py` - 13+ pre-configured models
- **Default Config**: `src/semanticmatcher/data/default_config.yml`
- **Project Config**: `pyproject.toml` - Dependencies and project metadata
- **Linting**: `ruff.toml` - Code style rules

### Data Storage
- **Static Data**: `src/semanticmatcher/data/` - Country codes, defaults
- **Ingestion Data**: JSON files in project root (industries, languages, etc.)
- **Model Cache**: `~/.cache/huggingface/` - Downloaded models
- **Embedding Cache**: `.parquet` files for pre-computed embeddings

### Testing
- **Test Suite**: `tests/` - Comprehensive test coverage
- **Fixtures**: `tests/conftest.py` - Shared test data and configurations
- **Integration Tests**: `tests/test_integration.py` - End-to-end tests

## Naming Conventions

### Files and Directories
- **Snake case** for all files and directories
- **Descriptive names** - `sentence_transformers.py`, `novel_class_detector.py`
- **Module grouping** - Related files in subdirectories
- **Test files** - `test_<module>.py` pattern

### Classes
- **PascalCase** for all classes
- **Descriptive names** - `TextNormalizer`, `BM25Blocking`, `NovelClassDetector`
- **Base classes** - Prefixed with `Base` (e.g., `BaseMatcher`, `BaseBackend`)

### Functions and Methods
- **snake_case** for all functions and methods
- **Verb-based** - `compute_similarity`, `normalize_text`, `detect_novelty`
- **Async variants** - `async_match`, `async_fit`, `async_batch_match`

### Variables
- **snake_case** for all variables
- **Descriptive names** - `embedding_matrix`, `candidate_entities`, `similarity_scores`
- **Constants** - `UPPER_SNAKE_CASE` (e.g., `MODEL_SPECS`, `DEFAULT_THRESHOLD`)

### Type Variables
- **PascalCase** with `_T` suffix for type variables
- **Generic types** - `Entity_T`, `Embedding_T`, `Model_T`

## Import Organization

### Import Order
1. Standard library imports
2. Third-party imports
3. Local imports (relative imports within package)

### Example
```python
# Standard library
import os
from typing import Optional, List

# Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# Local
from .classifier import EntityClassifier
from .utils import validate_entities
```

### Public API
- **`__all__` exports** defined in `__init__.py` files
- **Explicit exports** for public interfaces
- **Internal modules** prefixed with underscore (`_internal.py`)

## Module Dependencies

### Core Dependencies
- `matcher.py` depends on: classifier, normalizer, blocking, backends
- `classifier.py` depends on: backends, utils
- `normalizer.py` depends on: utils
- `blocking.py` depends on: backends, utils

### Backend Dependencies
- `sentence_transformers.py` depends on: utils, caching
- `static_embeddings.py` depends on: utils, caching
- `litellm.py` depends on: utils, caching

### Novelty Detection Dependencies
- `detector_api.py` depends on: detector, llm_proposer, schemas
- `detector.py` depends on: backends, utils
- `llm_proposer.py` depends on: litellm backend, schemas

### Utils Dependencies
- `validation.py` - Minimal dependencies (pydantic)
- `embeddings.py` - Minimal dependencies (numpy)
- `preprocessing.py` - Minimal dependencies (nltk)
- `benchmarks.py` - Depends on: core, backends

## File Size Indicators

### Large Files (>500 lines)
- `core/matcher.py` - 1,869 lines (main implementation)
- `utils/benchmarks.py` - 1,000 lines (performance testing)
- `config.py` - 502 lines (model registry)

### Medium Files (100-500 lines)
- Most backend implementations
- Core service modules
- Test files

### Small Files (<100 lines)
- Utility modules
- Schema definitions
- CLI entry points

## Test Structure

### Test Organization
- Mirrors source directory structure
- `test_<module>.py` for each source module
- `conftest.py` for shared fixtures
- Integration tests in `test_integration.py`

### Test Categories
- **Unit tests** - Individual component testing
- **Integration tests** - End-to-end pipeline testing
- **Performance tests** - Benchmark and timing tests
- **Markers** - `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.hf`

## Documentation Locations

### Code Documentation
- **Docstrings** - All public classes and methods
- **Type hints** - Comprehensive type annotations
- **Comments** - Complex logic explanation

### External Documentation
- **README.md** - Project overview and quick start
- **docs/** - Detailed documentation
- **CHANGELOG.md** - Version history and changes
- **CLAUDE.md** - Claude Code specific instructions
