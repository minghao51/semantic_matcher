# SemanticMatcher Code Conventions

## Code Style

### Formatting
- **Tool**: black
- **Line Length**: 88 characters (black default)
- **Config**: `pyproject.toml`

### Linting
- **Tool**: ruff
- **Config**: `pyproject.toml`
- **Minimum Version**: ruff≥0.1.0

## Type Hints

### Required Annotations
- **Function Parameters**: Always typed
- **Return Types**: Always specified
- **Class Attributes**: Typed where appropriate

### Common Type Patterns
```python
from typing import Optional, Union, List, Dict, Any

def match(
    self,
    texts: Union[str, List[str]],
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    pass
```

### Type Imports
```python
# Consolidated imports
from typing import Dict, List, Optional, Union
```

## Naming Conventions

### Classes
- **Pattern**: `CamelCase`
- **Examples**: `EntityMatcher`, `SetFitClassifier`, `TextNormalizer`

### Functions & Methods
- **Pattern**: `lowercase_with_underscores`
- **Public**: `match()`, `train()`, `normalize()`
- **Private**: `_load_model()`, `_validate_input()`

### Constants
- **Pattern**: `UPPER_CASE_WITH_UNDERSCORES`
- **Examples**: `DEFAULT_MODEL`, `DEFAULT_THRESHOLD`

### Variables
- **Pattern**: `lowercase_with_underscores`
- **Descriptive**: `entity_id` not `eid`

## Error Handling

### Exception Types
```python
# Specific exceptions
raise ValueError("Entity must have 'id' field")
raise RuntimeError("Model not trained. Call train() first.")

# Error checking before operation
if not self.is_trained or self.classifier is None:
    raise RuntimeError("Model not trained")
```

### Validation Pattern
```python
# Input validation utilities
from semanticmatcher.utils import validation

if not validation.is_valid_entity(entity):
    raise ValueError("Invalid entity format")
```

### Exception Catching
- **Specific**: Catch specific exception types
- **Generic**: Use `Exception` sparingly
- **Location**: `matcher.py:77` (one generic catch)

## Code Patterns

### Optional Dependencies
```python
# Pattern: Check availability
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Usage with fallback
if NLTK_AVAILABLE:
    nltk.word_tokenize(text)
else:
    text.split()
```

### Configuration Access
```python
# Singleton pattern
from semanticmatcher.config import Config

config = Config()  # Returns single instance
model = config.get("default_model")
```

### Model Loading
```python
# Pattern: Lazy load with caching
if self.model is None:
    self.model = SentenceTransformer(model_name)
```

## Docstrings

### Function Docstrings
```python
def match(self, texts: Union[str, List[str]]) -> List[Dict]:
    """
    Match texts to trained entities.

    Args:
        texts: Single text or list of texts to match

    Returns:
        List of matches with entity IDs and scores
    """
    pass
```

### Class Docstrings
```python
class EntityMatcher:
    """
    Few-shot semantic entity matcher using SetFit.

    Trains on 8-16 examples per entity.
    """
```

## Import Organization

### Import Order
1. Standard library
2. Third-party imports
3. Local imports

### Example
```python
# Standard library
import os
from typing import List, Optional

# Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# Local
from semanticmatcher.backends import base
from semanticmatcher.utils import validation
```

## Constants & Defaults

### Default Values
```python
DEFAULT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
DEFAULT_THRESHOLD = 0.7
DEFAULT_EPOCHS = 4
DEFAULT_BATCH_SIZE = 16
```

### Configuration Fallback
```python
# Pattern: Config -> Default
threshold = threshold or config.get("threshold", DEFAULT_THRESHOLD)
```

## Testing Conventions

### Test Organization
- **Path**: `tests/test_<module>/`
- **File**: `test_<module>.py`
- **Class**: `Test<ClassName>`

### Test Patterns
```python
# Fixture pattern
@pytest.fixture
def sample_entities():
    return [{"id": "us", "examples": ["USA", "United States"]}]

# Exception testing
with pytest.raises(ValueError):
    matcher.train(invalid_data)

# Float comparison
assert score == pytest.approx(0.85)
```

## Logging

### Current State
- **Implementation**: Minimal
- **Usage**: Mostly in example files
- **Pattern**: `print()` statements (to be improved)

### Recommended Pattern
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Training started")
logger.error("Model loading failed")
```
