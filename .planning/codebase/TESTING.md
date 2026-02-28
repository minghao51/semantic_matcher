# Testing

## Test Framework

**pytest** (≥8.4.2) - Primary testing framework

**Configuration**: `pyproject.toml`
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: tests that depend on external services or network access",
    "slow: tests that are expensive to run in default CI",
    "hf: Hugging Face model-backed tests",
]
```

## Test Structure

### Directory Layout

```
tests/
├── conftest.py                   # Shared fixtures
├── fixtures/                     # Test data
│   ├── entities.json
│   └── training_data.json
├── test_backends/                # Backend tests
├── test_core/                    # Core matcher tests
├── test_ingestion/               # Ingestion tests
├── test_utils/                   # Utility tests
└── test_config.py                # Configuration tests
```

**Mirrors Source Structure**:
- `tests/test_core/` mirrors `src/semanticmatcher/core/`
- `tests/test_backends/` mirrors `src/semanticmatcher/backends/`
- `tests/test_utils/` mirrors `src/semanticmatcher/utils/`

## Test Markers

### `integration`

**Purpose**: Tests requiring external services or network access

**Usage**:
```python
import pytest

@pytest.mark.integration
def test_ingestion_download_from_external_api():
    """Test ingestion from actual API."""
    pass
```

**Run only integration tests**:
```bash
pytest -m integration
```

**Skip integration tests**:
```bash
pytest -m "not integration"
```

### `slow`

**Purpose**: Expensive tests (disabled in fast CI)

**Usage**:
```python
@pytest.mark.slow
def test_training_with_large_dataset():
    """Test full training pipeline."""
    pass
```

**Run slow tests**:
```bash
pytest -m slow
```

**Skip slow tests (default)**:
```bash
pytest -m "not slow"
```

### `hf`

**Purpose**: Tests that download/use Hugging Face models

**Usage**:
```python
@pytest.mark.hf
def test_sentence_transformer_backend():
    """Test actual sentence transformer model."""
    pass
```

**Run HF tests**:
```bash
pytest -m hf
```

**Skip HF tests**:
```bash
pytest -m "not hf"
```

## Fixtures

### conftest.py

**Auto-clearing Model Cache**:
```python
@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear model cache before each test."""
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()
```

**Test Data Fixtures**:
```python
@pytest.fixture
def sample_entities():
    """Provide sample entity data for tests."""
    return [
        {"id": "1", "name": "Entity One", "text": "..."},
        # ...
    ]
```

### Reusable Fixtures

**Common test entities**:
- Simple entity lists (2-5 items)
- Realistic entity lists (50+ items)
- Training data pairs (text, label)

**Models**:
- Mock models (no HuggingFace download)
- Real models (marked with `@pytest.mark.hf`)

## Test Patterns

### AAA Pattern

**Arrange-Act-Assert**:

```python
def test_match_returns_top_k_results():
    # Arrange
    entities = [
        {"id": "1", "name": "Apple", "text": "Apple Inc"},
        {"id": "2", "name": "Banana", "text": "Banana fruit"},
    ]
    matcher = EmbeddingMatcher(entities=entities)
    matcher.build_index()

    # Act
    results = matcher.match("Apple", top_k=1)

    # Assert
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["score"] > 0.8
```

### Descriptive Test Names

**Pattern**: `test_<function>_<scenario>_<expected_result>`

```python
def test_match_with_empty_query_returns_none():
    """Test that match returns None for empty query."""
    pass

def test_match_with_threshold_filters_low_scores():
    """Test that threshold filters low-scoring results."""
    pass

def test_train_with_insufficient_data_raises_error():
    """Test that training with too few examples raises ValueError."""
    pass
```

### Edge Case Testing

**Common edge cases to test**:
- Empty inputs (empty query, empty entity list)
- None/null values
- Single-item lists
- Very large `top_k` values
- Threshold at boundaries (0.0, 1.0)
- Unicode characters and accents
- Very long text inputs

```python
@pytest.mark.parametrize("query,expected", [
    ("", None),              # Empty string
    ("   ", None),           # Whitespace only
    ("日本語", "japanese"),   # Unicode
    ("a" * 10000, "long"),   # Very long
])
def test_match_edge_cases(query, expected):
    """Test match with various edge case inputs."""
    pass
```

## Test Organization

### Unit Tests

**Purpose**: Test individual functions/classes in isolation

**Characteristics**:
- No external dependencies (network, filesystem)
- Fast execution (< 1 second each)
- Mock external services

**Example**:
```python
def test_text_normalizer_lowercase():
    """Test that TextNormalizer converts to lowercase."""
    normalizer = TextNormalizer(lowercase=True)
    result = normalizer.normalize("HELLO World")
    assert result == "hello world"
```

### Integration Tests

**Purpose**: Test interactions between components

**Characteristics**:
- May use real HuggingFace models (marked `@pytest.mark.hf`)
- May download data (marked `@pytest.mark.integration`)
- Slower but more realistic

**Example**:
```python
@pytest.mark.hf
def test_end_to_end_matching_pipeline():
    """Test full pipeline with real model."""
    entities = load_real_entities()
    matcher = EmbeddingMatcher(entities)
    matcher.build_index()
    results = matcher.match("test query")
    assert len(results) > 0
```

## Mocking

### Hugging Face Models

**Avoid downloading in unit tests**:
```python
from unittest.mock import Mock, patch

def test_match_with_mocked_model():
    """Test match without downloading real model."""
    # Mock the sentence transformer
    with patch('semanticmatcher.backends.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        matcher = EmbeddingMatcher(entities)
        matcher.build_index()
        results = matcher.match("test")

        assert mock_model.encode.called
```

### HTTP Requests

**Mock external APIs**:
```python
from unittest.mock import patch

@patch('requests.get')
def test_ingestion_with_mocked_api(mock_get):
    """Test ingestion without real HTTP call."""
    mock_get.return_value.json.return_value = {"data": "fake"}
    # Run ingestion
    # Assert results
```

## Coverage

### Current Status

**Tools**: No coverage reporting configured yet

**Recommended Addition**:
```bash
# Install coverage.py
uv add --dev pytest-cov

# Run tests with coverage
pytest --cov=src/semanticmatcher --cov-report=html
```

### Target Coverage Goals

- **Overall**: ≥ 80%
- **Core modules**: ≥ 90%
- **Ingestion**: ≥ 70% (external dependencies)
- **Utils**: ≥ 85%

## Running Tests

### Standard Test Run

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_core/test_matcher.py

# Run specific test function
uv run pytest tests/test_core/test_matcher.py::test_match_returns_top_k
```

### Marker-Based Runs

```bash
# Skip integration and slow tests (fast CI)
uv run pytest -m "not integration and not slow"

# Only unit tests
uv run pytest -m "not integration and not hf"

# Full test suite (including slow)
uv run pytest -m "not integration"
```

### Development Workflow

```bash
# Watch mode (rerun on file changes)
uv run pytest-watch

# Stop on first failure
uv run pytest -x

# Drop into debugger on failure
uv run pytest -pdb

# Show local variables on failure
uv run pytest -l
```

## Test Data Management

### Fixtures Directory

**Location**: `tests/fixtures/`

**Files**:
- `entities.json` - Standard test entities
- `training_data.json` - Training pairs

### Synthetic Data

**Generate on-the-fly**:
```python
@pytest.fixture
def synthetic_entities(count=100):
    """Generate synthetic entity data."""
    return [
        {
            "id": str(i),
            "name": f"Entity {i}",
            "text": f"Test text for entity {i}",
        }
        for i in range(count)
    ]
```

## CI/CD Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
      - run: uv sync --group dev
      - run: uv run pytest -m "not integration and not slow"
```

### Fast CI Pipeline

1. Lint: `ruff check`
2. Format check: `ruff format --check`
3. Unit tests: `pytest -m "not integration and not hf"`
4. Type check (optional): `mypy src/`

## Performance Testing

### Benchmarking

**Tool**: `tests/benchmarks/` or `src/semanticmatcher/utils/benchmarks.py`

**Metrics**:
- Embedding computation time
- Matching latency (p50, p95, p99)
- Memory usage
- Model load time

### Example Benchmark

```python
def test_match_performance_benchmark():
    """Benchmark matching performance."""
    import time

    matcher = EmbeddingMatcher(large_entity_set)
    matcher.build_index()

    start = time.time()
    for _ in range(100):
        matcher.match("test query", top_k=10)
    duration = time.time() - start

    assert duration < 5.0  # 100 queries in < 5 seconds
```

## Test Maintenance

### Keeping Tests Updated

**When adding features**:
1. Write test first (TDD)
2. Implement feature
3. Verify test passes
4. Add integration test

**When refactoring**:
1. Run tests before refactoring
2. Refactor in small steps
3. Run tests after each change
4. Add tests for edge cases discovered

### Flaky Tests

**Common causes**:
- Network dependency in unit tests
- Non-deterministic ordering
- Shared state between tests
- Timing-dependent tests

**Prevention**:
- Use markers (`@pytest.mark.integration`)
- Clear fixtures between tests
- Avoid shared state
- Mock external dependencies
