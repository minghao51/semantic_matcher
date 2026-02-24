# SemanticMatcher Structure

## Directory Layout

```
SemanticMatcher/
├── .github/
│   └── workflows/           # CI/CD configurations (minimal)
├── .planning/
│   └── codebase/           # This documentation
├── docs/                   # Additional documentation
├── examples/               # Usage examples
│   ├── advanced_matching.py
│   └── country_matching.py
├── notebooks/              # Jupyter notebooks
├── semanticmatcher/        # Main package
│   ├── __init__.py        # Public API exports
│   ├── config.py          # Configuration management
│   ├── backends/          # Backend implementations
│   │   ├── base.py
│   │   ├── litellm.py
│   │   └── sentencetranformer.py
│   ├── core/              # Core functionality
│   │   ├── classifier.py
│   │   ├── matcher.py
│   │   └── normalizer.py
│   └── utils/             # Utilities
│       ├── embeddings.py
│       ├── preprocessing.py
│       └── validation.py
├── tests/                 # Test suite
│   ├── test_core/
│   │   ├── test_matcher.py
│   │   └── test_classifier.py
│   ├── test_utils/
│   │   ├── test_validation.py
│   │   └── test_preprocessing.py
│   └── test_backends/
│       └── test_backends.py
├── config.yaml            # Default configuration
├── pyproject.toml         # Project metadata & dependencies
├── setup.py              # Installation script
├── uv.lock              # Dependency lock file
├── hello.py             # Simple demo
├── README.md            # Project overview
├── LICENSE              # MIT License
└── CLAUDE.md            # Project guidelines
```

## Key File Locations

### Configuration
- **Main Config**: `config.yaml` - Default settings
- **Config Class**: `semanticmatcher/config.py` - Config management

### Core Logic
- **Matcher**: `semanticmatcher/core/matcher.py` - Main matching classes
- **Classifier**: `semanticmatcher/core/classifier.py` - SetFit wrapper
- **Normalizer**: `semanticmatcher/core/normalizer.py` - Text preprocessing

### Backends
- **Base Interface**: `semanticmatcher/backends/base.py`
- **SentenceTransformer**: `semanticmatcher/backends/sentencetranformer.py`
- **LiteLLM**: `semanticmatcher/backends/litellm.py`

### Utilities
- **Validation**: `semanticmatcher/utils/validation.py`
- **Embeddings**: `semanticmatcher/utils/embeddings.py`
- **Preprocessing**: `semanticmatcher/utils/preprocessing.py`

### Tests
- **Core Tests**: `tests/test_core/`
- **Utils Tests**: `tests/test_utils/`
- **Backend Tests**: `tests/test_backends/`

### Examples
- **Basic**: `hello.py`
- **Country Matching**: `examples/country_matching.py`
- **Advanced**: `examples/advanced_matching.py`

## Naming Conventions

### Python Files
- **Modules**: `lowercase_with_underscores.py` (e.g., `matcher.py`)
- **Tests**: `test_<module>.py` (e.g., `test_matcher.py`)
- **Package**: `semanticmatcher/` (single word)

### Classes
- **Public**: `CamelCase` (e.g., `EntityMatcher`, `SetFitClassifier`)
- **Abstract**: `CamelCase` with "Base" suffix (e.g., `BackendBase`)

### Functions/Methods
- **Public**: `lowercase_with_underscores` (e.g., `match()`, `train()`)
- **Private**: `_leading_underscore` (e.g., `_load_config()`)

### Constants
- **All caps**: `UPPER_CASE` (e.g., `DEFAULT_MODEL`)

### Configuration Keys
- **Snake case**: `lowercase_with_underscores` (e.g., `default_model`)

## Import Structure

### Public API (`__init__.py`)
```python
from semanticmatcher import (
    EntityMatcher,
    EmbeddingMatcher,
    SetFitClassifier,
    TextNormalizer
)
```

### Internal Imports
```python
# Relative imports for package modules
from semanticmatcher.core import matcher, classifier
from semanticmatcher.backends import base
```

## File Sizes
- **matcher.py**: 158 lines (largest core file)
- **config.py**: 118 lines
- **Total Source**: ~679 lines
- **Total Tests**: ~485 lines
- **Test Ratio**: 71% (good coverage)
