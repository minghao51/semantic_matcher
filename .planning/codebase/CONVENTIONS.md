# Code Conventions

## Project Overview
Semantic Matcher is a text-to-entity matching library using SetFit few-shot learning and sentence transformers. The codebase follows Python best practices with emphasis on type safety, validation, and clear error messages.

## Code Style

### Formatting & Linting
- **Formatter**: Black (code style enforcement)
- **Linter**: Ruff (fast Python linter)
- **Python Version**: 3.9+ (supports 3.9, 3.10, 3.11, 3.12)
- **Line Length**: Black default (88 characters)
- **Import Style**: Absolute imports preferred

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `EntityMatcher`, `SetFitClassifier`)
- **Functions/Methods**: `snake_case` (e.g., `validate_entity`, `compute_embeddings`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MODEL_REGISTRY`, `SETFIT_AVAILABLE`)
- **Private/Internal**: Leading underscore (e.g., `_cache`, `_build_graph`)
- **Type Aliases**: `PascalCase` with "Type" suffix (e.g., `TextInput`, `PathLike`)

### Module Organization
```
src/semanticmatcher/
├── __init__.py          # Public API exports with lazy loading
├── exceptions.py        # Custom exception hierarchy
├── config.py            # Configuration management & model registries
├── core/                # Core matching logic
│   ├── matcher.py       # Main Matcher classes
│   ├── classifier.py    # SetFit wrapper
│   ├── normalizer.py    # Text normalization
│   └── hierarchy.py     # Hierarchical matching
├── backends/            # Backend abstractions
│   ├── base.py          # Abstract base classes
│   └── sentencetransformer.py  # HuggingFace integration
├── utils/               # Utilities
│   ├── validation.py    # Input validation
│   ├── embeddings.py    # Embedding computation & caching
│   └── preprocessing.py # Text preprocessing
└── ingestion/           # Data ingestion modules
```

## Type System

### Type Annotations
- **Mandatory**: All public functions/methods must have type hints
- **Style**: Use from `typing` module for compatibility
- **Common Types**:
  - `List[str]`, `Dict[str, Any]`, `Optional[str]`
  - `Union[str, List[str]]` for flexible inputs
  - Custom type aliases for complex types

```python
def validate_entity(entity: Dict[str, Any]) -> bool:
    """Validate a single entity dictionary."""
    if "id" not in entity:
        raise ValidationError(
            "Entity must have 'id' field",
            entity=entity,
            field="id",
        )
    return True
```

### Type Aliases
```python
TextInput = Union[str, List[str]]
PathLike = Union[str, Path]
```

## Error Handling

### Exception Hierarchy
All custom exceptions inherit from `SemanticMatcherError`:

```python
SemanticMatcherError (Exception)
├── ValidationError (ValueError, SemanticMatcherError)
├── TrainingError (RuntimeError, SemanticMatcherError)
├── MatchingError (RuntimeError, SemanticMatcherError)
└── ModeError (ValueError, SemanticMatcherError)
```

### Error Design Patterns

1. **Context-Rich Exceptions**: Include helpful diagnostic information
```python
raise ValidationError(
    "Entity must have 'id' field",
    entity=entity,
    field="id",
    suggestion="Add 'id' field: {'id': 'unique_id', 'name': 'Entity Name'}",
)
```

2. **Attribute-based Context**: Store context as instance attributes
```python
class ValidationError(ValueError, SemanticMatcherError):
    def __init__(self, message, *, entity=None, field=None, suggestion=None):
        self.raw_message = message
        self.entity = entity
        self.field = field
        self.suggestion = suggestion
```

3. **Formatted Messages**: Use `_format_message()` for consistent error display
```python
def _format_message(self) -> str:
    msg = self.raw_message
    if self.field:
        msg += f"\n  Problem field: {self.field}"
    if self.suggestion:
        msg += f"\n  💡 Suggestion: {self.suggestion}"
    return msg
```

### Error Handling Patterns

- **Validation Errors**: Use `ValidationError` for input validation failures
- **Training Errors**: Use `TrainingError` for model training failures
- **Import Errors**: Use `ImportError` with helpful install instructions
```python
if not SETFIT_AVAILABLE:
    raise ImportError("setfit is required. Install with: pip install setfit")
```

## Validation

### Validation Functions
Centralized in `utils/validation.py`:

- `validate_entity()` - Single entity validation
- `validate_entities()` - Batch validation with duplicate detection
- `validate_threshold()` - Threshold range checking (0-1)
- `validate_model_name()` - Model name validation

### Validation Pattern
```python
def validate_entities(entities: List[Dict[str, Any]]) -> bool:
    """Validate a list of entities."""
    if not entities:
        raise ValidationError(
            "entities list cannot be empty",
            suggestion="Provide at least one entity",
        )

    for entity in entities:
        validate_entity(entity)

    # Check for duplicate IDs
    ids = [e["id"] for e in entities]
    if len(ids) != len(set(ids)):
        duplicates = [eid for eid in ids if ids.count(eid) > 1]
        raise ValidationError(
            f"Entity IDs must be unique. Found duplicates: {duplicates}",
        )

    return True
```

## Configuration & Constants

### Model Registries
Centralized in `config.py` for easy model selection:

```python
MODEL_REGISTRY = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "default": "sentence-transformers/all-mpnet-base-v2",
}

def resolve_model_alias(model_name: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_REGISTRY.get(model_name, model_name)
```

### Configuration Loading
- **File Formats**: YAML (`.yaml`) or JSON (`.json`)
- **Search Order**: repo root → package default → current directory
- **Merging**: Deep merge for custom overrides
- **Access**: Dot-notation access (`config.get("model.name")`)

## Caching Patterns

### Model Cache
Thread-safe LRU cache in `utils/embeddings.py`:

```python
class ModelCache:
    def __init__(self, max_memory_gb: float = 4.0, ttl_seconds: Optional[float] = None):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get_or_load(self, model_name: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]
            model = factory()
            self._cache[model_name] = model
            return model
```

### Cache Clearing
Use `autouse` fixture in tests:
```python
@pytest.fixture(autouse=True)
def clear_model_cache():
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()
```

## Public API Design

### Lazy Loading
Use `__getattr__` for lazy imports in `__init__.py`:

```python
_EXPORTS = {
    "Matcher": (".core.matcher", "Matcher"),
    "SetFitClassifier": (".core.classifier", "SetFitClassifier"),
}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
```

### Deprecation Warnings
Use `warnings.warn()` for deprecated APIs:

```python
_DEPRECATED_CLASSES = {
    "EntityMatcher": "Matcher",
    "EmbeddingMatcher": "Matcher",
}

def __getattr__(name):
    if name in _DEPRECATED_CLASSES:
        replacement = _DEPRECATED_CLASSES[name]
        warnings.warn(
            f"{name} is deprecated. Use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
```

## Class Design Patterns

### Initialization Pattern
```python
class Matcher:
    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
    ):
        self.entities = entities
        self.model_name = model_name
        self.threshold = threshold
        self.normalize = normalize
```

### Method Chaining
Return `self` for fluent interface:

```python
def set_threshold(self, threshold: float) -> "Matcher":
    """Update threshold and return self for chaining."""
    self.threshold = validate_threshold(threshold)
    return self
```

## Utility Functions

### Helper Functions
Keep helper functions private (underscore prefix):

```python
def _coerce_texts(texts: TextInput) -> Tuple[List[str], bool]:
    """Coerce single string to list for uniform processing."""
    if isinstance(texts, str):
        return [texts], True
    return texts, False

def _unwrap_single(results: List[Any], single_input: bool) -> Any:
    """Unwrap single result if input was a single string."""
    if single_input:
        return results[0]
    return results
```

### Data Processing
```python
def _flatten_entity_texts(entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract all entity texts and their IDs."""
    entity_texts = []
    entity_ids = []
    for entity in entities:
        entity_texts.append(entity["name"])
        entity_ids.append(entity["id"])
        for alias in entity.get("aliases", []):
            entity_texts.append(alias)
            entity_ids.append(entity["id"])
    return entity_texts, entity_ids
```

## Documentation Standards

### Docstrings
- **Format**: Google-style docstrings
- **Required**: All public classes/methods/functions
- **Content**: Purpose, args, returns, raises, examples

```python
class Matcher:
    """
    Unified entity matcher with smart auto-selection.

    Automatically chooses the best matching strategy:
    - No training data → zero-shot (embedding similarity)
    - < 3 examples/entity → head-only training (~30s)
    - ≥ 3 examples/entity → full training (~3min)

    Example:
        matcher = Matcher(entities=[
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        ])
        matcher.fit()
        result = matcher.match("America")  # {"id": "US", "score": 0.95}
    """
```

### Comments
- Use inline comments for complex logic
- Explain "why", not "what"
- Keep comments up-to-date with code changes

## Testing Patterns

See `TESTING.md` for comprehensive testing guidelines.
