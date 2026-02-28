# Conventions

## Code Style

### Formatting

**Tool**: Black (via `ruff format` or standalone)

**Configuration**:
```toml
[tool.ruff]
line-length = 88  # Black default
```

**Usage**:
```bash
# Format code
ruff format src/ tests/

# Check formatting
ruff format --check src/ tests/
```

### Linting

**Tool**: Ruff (fast Python linter)

**Configuration** in `pyproject.toml`:
- Uses Ruff defaults
- Line length: 88 characters
- Python 3.9+ compatibility

**Usage**:
```bash
# Lint code
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

## Type Hints

### Standard Practices

**Extensive Type Annotations**:
- All public APIs have type hints
- Use `Optional` for nullable returns
- Use `List`, `Dict`, `Any` from `typing` module

**Examples**:
```python
from typing import List, Dict, Any, Optional, Union

def match(
    self,
    query: str,
    top_k: int = 10,
    threshold: Optional[float] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Match query against entities."""
    pass
```

**Return Types**:
- Explicit return types on all functions
- `-> None` for void functions
- `-> List[Dict[str, Any]]` for structured results

## Naming Patterns

### Classes

**Pattern**: `CamelCase`

**Suffixes indicate type**:
- `*Matcher` - Matching classes (EntityMatcher, EmbeddingMatcher)
- `*Blocking` - Blocking strategies (BM25Blocking, TFIDFBlocking)
- `*Reranker` - Reranking classes (CrossEncoderReranker)
- `*Normalizer` - Text normalization (TextNormalizer)
- `*Classifier` - Classification (SetFitClassifier)

### Functions/Methods

**Pattern**: `snake_case`

**Common prefixes**:
- `get_*` - Retrieve cached values (get_default_cache)
- `resolve_*` - Resolve aliases (resolve_model_alias)
- `recommend_*` - Recommendations (recommend_model)
- `build_*` - Construct/initialize (build_index)
- `train` - Model training
- `predict` / `match` - Inference methods

### Constants

**Pattern**: `UPPER_CASE`

**Examples**:
```python
MODEL_REGISTRY = {...}
RERANKER_REGISTRY = {...}
SOURCE_URL = "https://..."
DEFAULT_THRESHOLD = 0.7
```

### Private Members

**Pattern**: `_leading_underscore`

**Levels of privacy**:
- `_protected` - Internal but not strict (single underscore)
- `__private__` - Dunder methods (Python protocol)
- `__mangled` - Name mangling (rarely used)

## Error Handling

### Validation Strategy

**Input Validation**:
- Located in `src/semanticmatcher/utils/validation.py`
- Validates entities, queries, training data
- Type checking with defensive programming

**Example Pattern**:
```python
def validate_entities(entities: List[Dict[str, Any]]) -> None:
    if not isinstance(entities, list):
        raise TypeError("entities must be a list")
    for entity in entities:
        if "id" not in entity:
            raise ValueError("entity missing 'id' field")
```

### Exception Propagation

**No Custom Exception Hierarchy**:
- Let ML framework exceptions bubble up
- Users handle `sentence-transformers` and `SetFit` errors
- No wrapping in custom exceptions

**Common Exceptions**:
- `ValueError` - Invalid input parameters
- `TypeError` - Wrong type passed
- `RuntimeError` - Model loading failures

### Error Messages

**Descriptive Messages**:
```python
raise ValueError(f"entities must have 'text' or 'name' field, got: {entity}")
```

## Documentation

### Docstring Format

**Google Style (preferred)**:
```python
def match(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Match query against entities.

    Args:
        query: Query text to match
        top_k: Number of results to return

    Returns:
        List of matched entities with scores

    Raises:
        ValueError: If query is empty
    """
    pass
```

### Module Docstrings

**All modules have descriptive docstrings**:
```python
"""Hybrid matching pipeline with blocking, retrieval, and reranking."""
```

### Example Code in Docstrings

**Doctest-style examples**:
```python
def resolve_model_alias(model_name: str) -> str:
    """
    Resolve model alias to full model name.

    Example:
        >>> resolve_model_alias("bge-base")
        'BAAI/bge-base-en-v1.5'
    """
    return MODEL_REGISTRY.get(model_name, model_name)
```

## Import Conventions

### Import Order

**Standard layout**:
1. Standard library imports
2. Third-party imports
3. Local imports

```python
# 1. Standard library
from pathlib import Path
from typing import List, Dict, Any

# 2. Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# 3. Local
from .classifier import SetFitClassifier
from ..utils.validation import validate_entities
```

### Import Style

**Prefer explicit imports**:
```python
from typing import List, Dict, Any, Optional

# Avoid:
from typing import *
```

**Relative imports in package**:
```python
from .classifier import SetFitClassifier  # sibling
from ..utils import validation  # parent/child
```

## Code Organization

### File Length Guidelines

**Preferred**: < 200 lines per file
**Acceptable**: 200-300 lines
**Review needed**: > 300 lines

**Current largest files**:
- `ingestion/universities.py` (371 lines) - Contains hardcoded data
- `ingestion/products.py` (356 lines) - Contains hardcoded data
- `core/matcher.py` (320 lines) - Core logic, could be modularized

### Class Organization

**Standard structure**:
```python
class MyClass:
    """Docstring."""

    def __init__(self, ...):
        """Initialize."""

    # Public methods
    def public_method(self):
        """Public API."""

    # Private methods
    def _private_method(self):
        """Internal helper."""

    # Properties
    @property
    def my_property(self):
        """Property getter."""
```

## Logging

**Current Status**: Minimal logging
- Progress bars via `tqdm` for long operations
- No structured logging framework
- Print statements for debugging (should be replaced with `logging`)

**Recommended Enhancement**:
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Loading model...")
logger.debug(f"Candidates after blocking: {len(candidates)}")
```

## Testing Patterns

### Test Structure

**Mirror Source Structure**:
```
tests/test_core/        → src/semanticmatcher/core/
tests/test_backends/    → src/semanticmatcher/backends/
tests/test_utils/       → src/semanticmatcher/utils/
```

### Test Naming

**Pattern**: `test_<function>_<scenario>`

```python
def test_match_returns_top_k_results():
    """Test that match returns exactly top_k results."""
    pass

def test_match_filters_below_threshold():
    """Test that match filters results below threshold."""
    pass
```

### AAA Pattern

**Arrange-Act-Assert**:
```python
def test_match_with_valid_query():
    # Arrange
    matcher = EmbeddingMatcher(entities=TEST_ENTITIES)
    matcher.build_index()
    query = "test query"

    # Act
    results = matcher.match(query, top_k=3)

    # Assert
    assert len(results) == 3
    assert "score" in results[0]
```

## Performance Patterns

### Caching

**ModelCache** (`utils/embeddings.py`):
- Thread-safe singleton pattern
- TTL-based expiration
- Memory limits
- Shared across instances

### Lazy Loading

**Package imports** (`__init__.py`):
- Lazy import via `__getattr__`
- Reduces initial import time
- Optional dependencies not loaded until needed

### Parallel Processing

**ThreadPoolExecutor**:
- Used in `HybridMatcher` for bulk operations
- Concurrent blocking/retrieval/reranking
- Configurable worker count

## Configuration

### YAML Configuration

**Config Class** (`config.py`):
- Loads YAML files
- Model registries
- Path resolution

### Environment Variables

**Currently**: Not used
**Future consideration**: API keys for LiteLLM backend

## Git Conventions

**Commit Format**:
- Conventional commits (feat:, fix:, docs:, refactor:)
- Descriptive body for major changes

**Branch Structure**:
- `main` - Stable production code
- Feature branches - `feature/` or `codex/` prefixes

## Code Review Priorities

When reviewing code, check:
1. Type hints present and accurate
2. Docstrings for public APIs
3. Input validation
4. Error handling
5. No hardcoded secrets
6. Test coverage
7. Adherence to naming conventions
