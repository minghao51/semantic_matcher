# Testing Guidelines

## Testing Framework

### Core Stack
- **Framework**: pytest (>= 8.4.2)
- **Configuration**: `pyproject.toml` [tool.pytest.ini_options]
- **Test Discovery**: Auto-discovery in `tests/` directory
- **Markers**: Custom markers for test categorization

### Test Structure
```
tests/
├── conftest.py                    # Shared fixtures
├── test_config.py                 # Configuration tests
├── test_packaging.py              # Package metadata tests
├── test_core/
│   ├── test_matcher.py           # Matcher tests
│   ├── test_classifier.py        # Classifier tests
│   └── test_normalizer.py        # Normalizer tests
├── test_utils/
│   ├── test_validation.py        # Validation tests
│   ├── test_embeddings.py        # Embedding tests
│   └── test_preprocessing.py     # Preprocessing tests
└── test_backends/
    ├── test_huggingface.py       # HF backend tests
    ├── test_litellm.py           # LiteLLM backend tests
    └── test_reranker_contracts.py # Backend contract tests
```

## Test Organization

### Test Class Structure
```python
class TestEntityMatcher:
    """Tests for EntityMatcher - SetFit-based entity matching."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        ]

    def test_entity_matcher_init(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        assert matcher.entities == sample_entities
```

### Test Naming
- **Test Classes**: `Test{ClassName}` (e.g., `TestEntityMatcher`)
- **Test Methods**: `test_{method}_{scenario}` (e.g., `test_match_below_threshold`)
- **Fixtures**: Descriptive names (e.g., `sample_entities`, `training_data_full`)

## Fixtures

### Shared Fixtures (conftest.py)
```python
@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the global model cache before each test."""
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()
```

### Per-Test Fixtures
```python
@pytest.fixture
def sample_entities(self):
    """Reusable sample entities for testing."""
    return [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
    ]

@pytest.fixture
def training_data(self):
    """Training data for classifier tests."""
    return [
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "France", "label": "FR"},
    ]
```

### Fixture Scope
- **Default**: Function scope (run for each test)
- **Autouse**: Automatically apply to all tests
- **Session**: Expensive resources (models, datasets)

## Test Categories

### Markers
```python
[tool.pytest.ini_options]
markers = [
    "integration: tests that depend on external services or network access",
    "slow: tests that are expensive to run in default CI",
    "hf: Hugging Face model-backed tests",
]
```

### Test Marking
```python
@pytest.mark.integration
def test_external_api_call():
    """Test that requires network access."""
    response = requests.get("https://api.example.com")
    assert response.status_code == 200

@pytest.mark.slow
def test_full_training():
    """Test that takes > 1 minute to run."""
    matcher = Matcher(entities=large_entity_set)
    matcher.fit(large_training_dataset, num_epochs=10)

@pytest.mark.hf
def test_huggingface_model():
    """Test that downloads HF model."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

## CI/CD Testing Strategy

### GitHub Actions Workflows

#### Fast Tests (PR & Branch Push)
```yaml
# .github/workflows/test.yml - test-fast-pr job
- name: Run fast tests (exclude network/model-heavy)
  run: uv run python -m pytest -q -m "not integration and not slow"
```

#### Matrix Testing (Main Branch)
```yaml
# Test across Python 3.9, 3.10, 3.11, 3.12
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]
```

#### Heavy Tests (Main & Workflow Dispatch)
```yaml
# Run integration and slow tests
- name: Run heavy integration tests
  run: uv run python -m pytest -q -m "integration or slow"
```

### Concurrency Control
```yaml
concurrency:
  group: test-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

## Mocking Patterns

### Monkeypatch (pytest)
```python
def test_embedding_matcher_top_k_deduplicates(self, sample_entities, monkeypatch):
    """Test with mocked model for deterministic behavior."""
    vectors = {
        "Germany": [1.0, 0.0],
        "France": [0.0, 1.0],
    }

    class FakeModel:
        def encode(self, texts):
            return np.array([vectors.get(t, [1.0, 0.0]) for t in texts])

    monkeypatch.setattr(
        "semanticmatcher.core.matcher.SentenceTransformer", FakeModel
    )

    matcher = EmbeddingMatcher(entities=sample_entities)
    matcher.build_index()
    assert matcher.match("Germany")["id"] == "DE"
```

### Fake Classes
```python
class FakeClassifier:
    """Minimal fake for testing without model loading."""
    labels = ["DE", "FR", "US"]

    def predict_proba(self, text):
        return np.array([0.82, 0.75, 0.41], dtype=float)

# Use in tests
entity_matcher = EntityMatcher(entities=sample_entities)
entity_matcher.classifier = FakeClassifier()
entity_matcher.is_trained = True
```

### Backend Contract Testing
```python
def test_hf_reranker_score_returns_numeric_list(monkeypatch):
    """Test backend contract without real model."""
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.9, 0.2, -0.1]

    monkeypatch.setattr(st_backend_module, "CrossEncoder", FakeCrossEncoder)

    backend = st_backend_module.HFReranker("fake-model")
    scores = backend.score("query", ["doc1", "doc2", "doc3"])

    assert scores == [0.9, 0.2, -0.1]
    assert all(isinstance(score, float) for score in scores)
```

## Assertion Patterns

### Exception Testing
```python
def test_validate_entity_missing_id(self):
    entity = {"name": "Germany"}
    with pytest.raises(ValueError, match="must have 'id'"):
        validate_entity(entity)

def test_validate_entity_raises_validation_error(self):
    """Test specific exception type."""
    with pytest.raises(ValidationError):
        validate_entity({"name": "Germany"})
```

### Value Assertions
```python
# Simple equality
assert matcher.threshold == 0.7

# Type checking
assert all(isinstance(score, float) for score in scores)

# Collection assertions
assert [r["id"] for r in results] == ["DE", "FR", "US"]

# Numeric comparisons
assert result["score"] > 0.8
assert len(results) == 3
```

### State Assertions
```python
def test_entity_matcher_train(self, sample_entities, training_data):
    matcher = EntityMatcher(entities=sample_entities)
    matcher.train(training_data, num_epochs=1)
    assert matcher.is_trained
```

## Parameterized Testing

### pytest.mark.parametrize
```python
@pytest.mark.parametrize("threshold,expected", [
    (0.5, True),
    (0.0, True),
    (1.0, True),
])
def test_validate_threshold_valid(self, threshold, expected):
    assert validate_threshold(threshold) == threshold

@pytest.mark.parametrize("invalid_threshold", [
    1.5,
    -0.1,
    "invalid",
])
def test_validate_threshold_invalid(self, invalid_threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_threshold(invalid_threshold)
```

## Coverage Goals

### Target Areas
- **Core Logic**: 90%+ coverage (matcher, classifier, hierarchy)
- **Validation**: 100% coverage (all error paths)
- **Utilities**: 80%+ coverage (embeddings, preprocessing)
- **Backends**: Contract tests only (no model loading)

### Coverage Commands
```bash
# Run with coverage report
uv run pytest --cov=semanticmatcher --cov-report=term-missing

# Generate HTML report
uv run pytest --cov=semanticmatcher --cov-report=html

# Check minimum coverage
uv run pytest --cov=semanticmatcher --cov-fail-under=80
```

## Test Data Management

### Fixtures for Data
```python
@pytest.fixture
def sample_entities(self):
    """Small, fast-loading test data."""
    return [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
    ]

@pytest.fixture
def large_entity_set(self):
    """Larger dataset for performance testing."""
    return [{"id": f"entity_{i}", "name": f"Entity {i}"} for i in range(1000)]
```

### Test Data Files
- Place in `tests/fixtures/` directory
- Use small, representative samples
- Version control test data (not auto-generated)

## Performance Testing

### Latency Benchmarks
```python
def test_match_latency_benchmark(self, sample_entities):
    """Benchmark matching performance."""
    matcher = Matcher(entities=sample_entities)
    matcher.fit()

    start = time.time()
    for _ in range(100):
        matcher.match("Germany")
    elapsed = time.time() - start

    assert elapsed < 1.0  # 100 matches in < 1 second
```

### Memory Testing
```python
def test_model_cache_memory_limit(self):
    """Test cache respects memory limits."""
    cache = ModelCache(max_memory_gb=0.1)
    # Load models until limit reached
    # Verify eviction occurs
```

## Integration Testing

### External Service Tests
```python
@pytest.mark.integration
def test_huggingface_model_download():
    """Test actual HF model download and loading."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    assert model is not None
    embeddings = model.encode(["test"])
    assert embeddings.shape == (1, 384)
```

### Network Dependency Tests
```python
@pytest.mark.integration
def test_litellm_api_call():
    """Test LiteLLM backend with real API."""
    backend = LiteLLMEmbedding(model="openai/embedding-3-small")
    embeddings = backend.encode(["test"])
    assert len(embeddings) == 1
```

## Best Practices

### DO
- Use descriptive test names that explain what is being tested
- Keep tests independent (no shared state between tests)
- Use fixtures for common test data
- Mock external dependencies (APIs, models, databases)
- Test both success and failure paths
- Use `pytest.raises` context manager for exception testing
- Add markers for slow/integration tests
- Clear caches between tests (autouse fixture)

### DON'T
- Don't hardcode absolute paths
- Don't sleep in tests (use mocks or proper synchronization)
- Don't test external libraries (test your usage of them)
- Don't skip tests without markers explaining why
- Don't use print statements (use assertions)
- Don't write tests that depend on execution order
- Don't ignore deprecation warnings in tests

## Running Tests

### Run All Tests
```bash
uv run pytest
```

### Run Specific Test File
```bash
uv run pytest tests/test_core/test_matcher.py
```

### Run Specific Test
```bash
uv run pytest tests/test_core/test_matcher.py::TestEntityMatcher::test_entity_matcher_init
```

### Run with Markers
```bash
# Run only fast tests
uv run pytest -m "not integration and not slow"

# Run only integration tests
uv run pytest -m integration

# Run only slow tests
uv run pytest -m slow
```

### Verbose Output
```bash
uv run pytest -v  # Verbose
uv run pytest -vv  # Very verbose
uv run pytest -s  # Show print output
```

### Stop on First Failure
```bash
uv run pytest -x
```

## Debugging Tests

### Drop into Debugger
```python
def test_complex_logic(self):
    result = complex_function()
    import pdb; pdb.set_trace()  # Breakpoint
    assert result == expected
```

### Show Output
```bash
uv run pytest -s -v tests/test_core/test_matcher.py::test_function
```

### Run Last Failed Tests
```bash
uv run pytest --lf
```

## Test Documentation

### Docstrings in Tests
```python
def test_matcher_auto_detect_zero_shot(self, sample_entities):
    """Test auto-detection selects zero-shot without training data."""
    matcher = Matcher(entities=sample_entities)
    assert matcher._detect_training_mode(None) == "zero-shot"
```

### Comments for Complex Tests
```python
def test_matcher_top_k_with_threshold(self, sample_entities):
    """
    Test that top_k results are filtered by threshold.

    Scenario:
    - Matcher with threshold=0.8
    - 3 candidates with scores: 0.9, 0.75, 0.6
    - Should return only first result despite top_k=3
    """
    matcher = Matcher(entities=sample_entities, threshold=0.8)
    # ... test implementation
```
