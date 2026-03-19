# Testing

## Test Framework

### Core Framework
- **pytest** 8.4.2+ - Primary testing framework
- **pytest-asyncio** - Async test support with `asyncio_mode = "auto"`
- **pytest-cov** - Coverage reporting

### Test Configuration
```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session", autouse=True)
def clear_model_cache():
    """Clear model cache before all tests."""
    # Ensure clean state for tests
    pass

@pytest.fixture
def sample_entities():
    """Provide sample test data."""
    return [
        {"name": "Python Developer", "category": "engineering"},
        {"name": "Data Scientist", "category": "data"},
    ]
```

### Async Test Configuration
```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
```

## Test Structure

### Directory Organization
```
tests/
├── __init__.py
├── conftest.py                  # Shared fixtures
├── test_matcher.py              # Matcher tests
├── test_classifier.py           # Classifier tests
├── test_normalizer.py           # Normalizer tests
├── test_backends/               # Backend tests
│   ├── __init__.py
│   ├── test_sentence_transformers.py
│   ├── test_static_embeddings.py
│   └── test_litellm.py
├── test_novelty/                # Novelty detection tests
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_llm_proposer.py
└── test_integration.py          # Integration tests
```

### Test Class Organization
```python
class TestEntityMatcher:
    """Test suite for EntityMatcher."""

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        pass

    def test_match_single_query(self):
        """Test matching a single query."""
        pass

    def test_batch_match(self):
        """Test batch matching."""
        pass

    @pytest.mark.asyncio
    async def test_async_fit(self):
        """Test async fitting."""
        pass

    @pytest.mark.asyncio
    async def test_async_match(self):
        """Test async matching."""
        pass
```

## Test Markers

### Custom Markers
```python
# Integration tests (require network/external services)
@pytest.mark.integration
def test_external_api():
    """Test integration with external API."""
    pass

# Slow tests (long execution time)
@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset."""
    pass

# Hugging Face model tests
@pytest.mark.hf
def test_sentence_transformer():
    """Test sentence transformer models."""
    pass
```

### Running Marked Tests
```bash
# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only Hugging Face tests
pytest -m hf

# Combine markers
pytest -m "integration and not slow"
```

## Fixtures

### Common Fixtures
```python
@pytest.fixture
def sample_entities():
    """Provide sample entities for testing."""
    return [
        {"name": "Python Developer", "category": "engineering"},
        {"name": "Data Scientist", "category": "data"},
        {"name": "ML Engineer", "category": "engineering"},
    ]

@pytest.fixture
def sample_embeddings():
    """Provide pre-computed embeddings."""
    return np.random.rand(3, 384)

@pytest.fixture
def trained_matcher(sample_entities):
    """Provide a trained matcher instance."""
    matcher = EntityMatcher()
    matcher.fit(sample_entities)
    return matcher

@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for file tests."""
    return tmp_path
```

### Async Fixtures
```python
@pytest.fixture
async def async_trained_matcher(sample_entities):
    """Provide an async-trained matcher instance."""
    matcher = EntityMatcher()
    await matcher.async_fit(sample_entities)
    return matcher
```

## Test Patterns

### Unit Testing
```python
def test_similarity_computation():
    """Test similarity score computation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    score = cosine_similarity(vec1, vec2)
    assert score == pytest.approx(0.0, abs=1e-6)

def test_threshold_filtering():
    """Test threshold-based filtering."""
    results = [
        Match(entity="A", score=0.9),
        Match(entity="B", score=0.6),
        Match(entity="C", score=0.3),
    ]
    filtered = filter_by_threshold(results, threshold=0.5)
    assert len(filtered) == 2
```

### Integration Testing
```python
@pytest.mark.integration
def test_end_to_end_matching():
    """Test complete matching pipeline."""
    # Load entities
    entities = load_test_data()

    # Train matcher
    matcher = EntityMatcher()
    matcher.fit(entities)

    # Perform matching
    results = matcher.match("python engineer")

    # Verify results
    assert len(results) > 0
    assert results[0].score > 0.7
```

### Async Testing
```python
@pytest.mark.asyncio
async def test_async_batch_match():
    """Test async batch matching."""
    matcher = EntityMatcher()
    await matcher.async_fit(entities)

    queries = ["query 1", "query 2", "query 3"]
    results = await matcher.async_batch_match(queries)

    assert len(results) == len(queries)
    assert all(len(r) > 0 for r in results)
```

### Parametrized Testing
```python
@pytest.mark.parametrize("model_name,expected_dim", [
    ("miniLM", 384),
    ("mpnet", 768),
    ("multilingual", 384),
])
def test_model_dimensions(model_name, expected_dim):
    """Test that models produce correct dimensions."""
    model = load_model(model_name)
    embeddings = model.encode(["test"])
    assert embeddings.shape[1] == expected_dim
```

### Exception Testing
```python
def test_invalid_entity_validation():
    """Test that invalid entities raise errors."""
    with pytest.raises(ValidationError) as exc_info:
        validate_entity({"invalid": "data"})

    assert "name" in str(exc_info.value)

def test_untrained_matcher():
    """Test that untrained matcher raises error."""
    matcher = EntityMatcher()
    with pytest.raises(TrainingError):
        matcher.match("query")
```

## Mocking and Patching

### External API Mocking
```python
from unittest.mock import patch, MagicMock

@patch("semanticmatcher.backends.litellm.litellm_embedding")
def test_litellm_backend(mock_embedding):
    """Test LiteLLM backend with mocked API."""
    mock_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

    backend = LiteLLMBackend(model="text-embedding-ada-002")
    embeddings = backend.encode(["test"])

    assert embeddings.shape == (1, 3)
    mock_embedding.assert_called_once()
```

### Model Mocking
```python
@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = MagicMock()
    model.encode.return_value = np.random.rand(10, 384)
    return model

def test_classifier_with_mock(mock_model):
    """Test classifier with mocked model."""
    classifier = EntityClassifier(model=mock_model)
    results = classifier.classify(["test"])
    assert len(results) > 0
```

## Performance Testing

### Benchmark Testing
```python
def test_matching_performance(benchmark):
    """Benchmark matching performance."""
    matcher = EntityMatcher()
    matcher.fit(large_dataset)

    result = benchmark(matcher.match, "test query")
    assert len(result) > 0

@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with large dataset."""
    import time

    start = time.time()
    matcher = EntityMatcher()
    matcher.fit(large_dataset)
    fit_time = time.time() - start

    assert fit_time < 60  # Should complete in under 60 seconds
```

### Memory Testing
```python
@pytest.mark.slow
def test_memory_usage():
    """Test memory usage with large dataset."""
    import tracemalloc

    tracemalloc.start()
    matcher = EntityMatcher()
    matcher.fit(large_dataset)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory should be under 1GB
    assert peak < 1024 * 1024 * 1024
```

## Coverage

### Coverage Configuration
```bash
# Run with coverage
pytest --cov=semanticmatcher --cov-report=html

# Generate terminal report
pytest --cov=semanticmatcher --cov-report=term-missing

# Generate XML report (for CI)
pytest --cov=semanticmatcher --cov-report=xml
```

### Coverage Goals
- **Overall**: >80% coverage
- **Core modules**: >90% coverage
- **Critical paths**: 100% coverage

## Continuous Integration

### GitHub Actions Matrix
```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.9", "3.10", "3.11", "3.12"]
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest -m "not integration" --cov=semanticmatcher
```

## Best Practices

### Test Organization
1. **One test class per module** - Group related tests
2. **Descriptive test names** - `test_<function>_<condition>`
3. **Independent tests** - No shared state between tests
4. **Fast feedback** - Keep tests fast, use markers for slow tests

### Test Data
1. **Fixtures for data** - Reuse test data via fixtures
2. **Minimal datasets** - Use smallest dataset that tests the feature
3. **Realistic data** - Use data that resembles production
4. **Edge cases** - Include boundary conditions and error cases

### Assertion Guidelines
1. **Specific assertions** - Test exact behavior, not just "it works"
2. **Error messages** - Provide helpful failure messages
3. **Approximate comparisons** - Use `pytest.approx()` for floats
4. **Exception testing** - Test both error type and message content

### Async Testing
1. **Use pytest-asyncio** - Automatic async test handling
2. **Mock async operations** - Patch external async calls
3. **Test concurrency** - Verify thread safety where applicable
4. **Clean up resources** - Ensure proper cleanup in async tests
