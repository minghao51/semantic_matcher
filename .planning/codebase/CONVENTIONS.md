# Conventions

## Code Style

### Linting and Formatting
- **Ruff** - Primary linter and formatter
- **Configuration** - `ruff.toml` for project-specific rules
- **Auto-formatting** - Enforced via pre-commit hooks
- **Line length** - Standard Python limits (88 characters default)

### Python Version
- **Minimum**: Python 3.9
- **Supported**: 3.9, 3.10, 3.11, 3.12
- **Testing**: Matrix testing on all supported versions
- **Modern features** - Type hints, async/await, pattern matching where appropriate

### Type Hints
- **Mandatory** for all public interfaces
- **Optional** for private methods (recommended)
- **Strict mode** - Use `mypy` or similar for type checking
- **Generic types** - Use TypeVar for generic classes/methods

## Naming Conventions

### Classes
- **PascalCase** for all class names
- **Descriptive names** - Avoid abbreviations
- **Base classes** - Prefixed with `Base`
- **Exceptions** - Suffixed with `Error`

```python
class TextNormalizer:
    pass

class BaseMatcher:
    pass

class ValidationError(Exception):
    pass
```

### Functions and Methods
- **snake_case** for all function names
- **Verb-based** - Start with action verb
- **Async variants** - Use `async_` prefix if needed

```python
def compute_similarity(text1: str, text2: str) -> float:
    pass

async def async_match(query: str) -> List[Match]:
    pass
```

### Variables
- **snake_case** for all variables
- **Descriptive names** - Avoid single letters except loop variables
- **Constants** - `UPPER_SNAKE_CASE`

```python
similarity_score = 0.95
DEFAULT_THRESHOLD = 0.7
MAX_CANDIDATES = 100
```

### Type Variables
- **PascalCase** with `_T` suffix
- **Bound types** - Specify base class when applicable

```python
from typing import TypeVar

Entity_T = TypeVar("Entity_T", bound=dict)
Model_T = TypeVar("Model_T", bound=SentenceTransformer)
```

## Code Patterns

### Error Handling

#### Exception Hierarchy
```python
class SemanticMatcherError(Exception):
    """Base exception for all semantic matcher errors."""
    pass

class ValidationError(SemanticMatcherError):
    """Raised when input validation fails."""
    def __init__(self, entity: str, field: str, message: str, suggestion: str = None):
        self.entity = entity
        self.field = field
        self.suggestion = suggestion
        super().__init__(message)
```

#### Validation Pattern
```python
def validate_entities(entities: List[Entity]) -> List[Entity]:
    """Validate and filter entities."""
    validated = []
    for entity in entities:
        try:
            validate_entity_schema(entity)
            validated.append(entity)
        except ValidationError as e:
            logger.warning(f"Skipping invalid entity: {e}")
    return validated
```

### Async Patterns

#### Async/Await Usage
```python
async def async_batch_match(
    self,
    queries: List[str],
    batch_size: int = 32
) -> List[List[Match]]:
    """Async batch matching with concurrency control."""
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[self.async_match(q) for q in batch]
        )
        results.extend(batch_results)
    return results
```

### Caching Patterns

#### LRU Cache with Thread Safety
```python
from functools import lru_cache
import threading

@lru_cache(maxsize=128)
def get_model(model_name: str) -> SentenceTransformer:
    """Get or load model with caching."""
    # Model loading logic
    pass

# Thread-safe variant for concurrent access
_model_cache = {}
_model_cache_lock = threading.Lock()

def get_model_threadsafe(model_name: str) -> SentenceTransformer:
    """Thread-safe model loading."""
    if model_name not in _model_cache:
        with _model_cache_lock:
            if model_name not in _model_cache:
                _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]
```

### Import Organization

#### Import Order
```python
# Standard library
import os
from typing import List, Optional

# Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# Local
from .classifier import EntityClassifier
from .utils import validate_input
```

#### Public API Exports
```python
# In __init__.py
from .matcher import Matcher
from .novelty.detector_api import NovelClassDetector

__all__ = [
    "Matcher",
    "NovelClassDetector",
]
```

## Documentation Standards

### Docstring Format
```python
def match(
    self,
    query: str,
    threshold: float = 0.7,
    top_k: int = 10
) -> List[Match]:
    """
    Find matching entities for a query.

    Args:
        query: The search query text
        threshold: Minimum similarity score (0-1)
        top_k: Maximum number of results to return

    Returns:
        List of matches sorted by similarity score

    Raises:
        ValidationError: If query is invalid
        TrainingError: If matcher not trained

    Example:
        >>> matcher = Matcher(mode="entity")
        >>> matcher.fit(entities)
        >>> results = matcher.match("python developer")
    """
```

### Comment Guidelines
- **What, not why** - Code should be self-explanatory
- **Complex logic** - Explain non-obvious algorithms
- **TODO markers** - Use for known issues
- **FIXME markers** - Use for bugs that need fixing

## Logging Conventions

### Logging Levels
```python
import logging

logger = logging.getLogger(__name__)

# DEBUG - Detailed diagnostic information
logger.debug(f"Embedding shape: {embeddings.shape}")

# INFO - General informational messages
logger.info(f"Loaded {len(entities)} entities")

# WARNING - Something unexpected but recoverable
logger.warning(f"Low confidence match: {score:.2f}")

# ERROR - Error occurred but execution continues
logger.error(f"Failed to load model: {model_name}")

# CRITICAL - Serious error, execution may stop
logger.critical("Failed to initialize matcher")
```

### Verbosity Control
```python
import os

VERBOSE = os.getenv("SEMANTIC_MATCHER_VERBOSE", "false").lower() == "true"

if VERBOSE:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
```

## Testing Conventions

### Test Structure
```python
class TestEntityMatcher:
    """Test suite for EntityMatcher."""

    def test_match_single_query(self):
        """Test matching a single query."""
        matcher = EntityMatcher()
        matcher.fit(entities)
        results = matcher.match("test query")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_async_match(self):
        """Test async matching."""
        matcher = EntityMatcher()
        await matcher.async_fit(entities)
        results = await matcher.async_match("test query")
        assert len(results) > 0
```

### Test Markers
```python
# Integration test (requires network/external services)
@pytest.mark.integration
def test_external_api():
    pass

# Slow test (long execution time)
@pytest.mark.slow
def test_large_dataset():
    pass

# Hugging Face model test
@pytest.mark.hf
def test_sentence_transformer():
    pass
```

### Fixture Usage
```python
@pytest.fixture
def sample_entities():
    """Provide sample entities for testing."""
    return [
        {"name": "Python Developer", "category": "engineering"},
        {"name": "Data Scientist", "category": "data"},
    ]

@pytest.fixture
def trained_matcher(sample_entities):
    """Provide a trained matcher instance."""
    matcher = EntityMatcher()
    matcher.fit(sample_entities)
    return matcher
```

## Configuration Patterns

### Model Registry
```python
MODEL_SPECS = {
    "miniLM": {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_length": 256,
        "language": "en",
    },
    "multilingual": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimensions": 384,
        "max_length": 512,
        "language": "multilingual",
    },
}
```

### Configuration Validation
```python
from pydantic import BaseModel, Field, validator

class MatcherConfig(BaseModel):
    """Matcher configuration with validation."""

    model_name: str = Field(..., min_length=1)
    threshold: float = Field(..., ge=0.0, le=1.0)
    top_k: int = Field(..., gt=0)

    @validator("model_name")
    def validate_model(cls, v):
        if v not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {v}")
        return v
```

## Performance Patterns

### Batch Processing
```python
def process_batch(items: List[str], batch_size: int = 32):
    """Process items in batches for efficiency."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_single_batch(batch)
        results.extend(batch_results)
    return results
```

### Memory Efficiency
```python
def stream_entities(filepath: str):
    """Stream entities from file to reduce memory usage."""
    with open(filepath) as f:
        for line in f:
            entity = json.loads(line)
            yield entity
```

### Early Exit
```python
def find_match(query: str, entities: List[Entity], threshold: float):
    """Find first match above threshold."""
    for entity in entities:
        score = compute_similarity(query, entity)
        if score >= threshold:
            return entity, score
    return None, 0.0
```

## Security Conventions

### API Key Management
```python
import os

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
```

### Input Sanitization
```python
def sanitize_input(text: str) -> str:
    """Remove potentially harmful content."""
    # Remove null bytes, control characters
    text = text.replace("\x00", "")
    text = "".join(char for char in text if char.isprintable() or char.isspace())
    return text.strip()
```

### Validation First
```python
def process_entity(entity: dict) -> Entity:
    """Validate and process entity."""
    # Always validate before processing
    validated = validate_entity_schema(entity)
    # Continue with validated data
    return create_entity(validated)
```
