# SemanticMatcher Testing

## Test Framework

### Core Tools
- **Framework**: pytest (≥8.4.2)
- **Assertions**: pytest built-in assertions
- **Fixtures**: pytest.fixture decorator
- **Exceptions**: pytest.raises()

### Test Execution
```bash
# Run all tests
uv run pytest

# Run specific file
uv run pytest tests/test_core/test_matcher.py

# Run with coverage
uv run pytest --cov=semanticmatcher

# Verbose output
uv run pytest -v
```

## Test Structure

### Directory Layout
```
tests/
├── test_core/
│   ├── test_matcher.py        # EntityMatcher, EmbeddingMatcher
│   └── test_classifier.py     # SetFitClassifier
├── test_utils/
│   ├── test_validation.py     # Input validation
│   └── test_preprocessing.py  # Text preprocessing
└── test_backends/
    └── test_backends.py       # Backend implementations
```

### Test Organization
```python
# Class-based organization
class TestEntityMatcher:
    def test_train(self):
        pass

    def test_match(self):
        pass

# Function-based for simple tests
def test_normalization():
    pass
```

## Test Patterns

### Fixtures
```python
@pytest.fixture
def sample_entities():
    """Sample entity data for testing."""
    return [
        {
            "id": "us",
            "examples": ["USA", "United States", "America"]
        },
        {
            "id": "uk",
            "examples": ["UK", "United Kingdom", "Britain"]
        }
    ]

@pytest.fixture
def trained_matcher(sample_entities):
    """Return a trained matcher instance."""
    matcher = EntityMatcher()
    matcher.train(sample_entities)
    return matcher
```

### Exception Testing
```python
def test_train_invalid_data():
    matcher = EntityMatcher()
    with pytest.raises(ValueError):
        matcher.train([{"id": "test"}])  # Missing examples

def test_match_before_training():
    matcher = EntityMatcher()
    with pytest.raises(RuntimeError):
        matcher.match("test text")
```

### Parametrized Tests
```python
@pytest.mark.parametrize("text,expected", [
    ("USA", "us"),
    ("United Kingdom", "uk"),
])
def test_country_matching(trained_matcher, text, expected):
    results = trained_matcher.match(text)
    assert results[0]["id"] == expected
```

### Float Comparison
```python
def test_similarity_score():
    # Use pytest.approx for float comparison
    assert score == pytest.approx(0.85, rel=1e-2)
```

## Test Coverage

### Coverage Areas
1. **Core Functionality**: ~71% test-to-code ratio
2. **Matcher Classes**: Comprehensive tests
3. **Validation**: Input validation tests
4. **Backends**: Basic backend tests

### Coverage Commands
```bash
# Generate coverage report
uv run pytest --cov=semanticmatcher --cov-report=html

# Terminal coverage
uv run pytest --cov=semanticmatcher
```

## Mocking

### Current State
- **Limited Mocking**: Not extensively used
- **Real Models**: Tests use actual small models
- **Pattern**: Tests use fast/default models

### Recommended Mocking
```python
# Mock expensive operations
from unittest.mock import patch, MagicMock

@patch('semanticmatcher.core.matcher.SentenceTransformer')
def test_with_mocked_model(mock_transformer):
    mock_transformer.return_value = MagicMock()
    # Test logic without loading actual model
```

## Test Data

### Sample Data Location
- **In-code**: Fixtures within test files
- **Examples**: Real examples from `examples/`
- **Size**: Small datasets for fast tests

### Test Data Patterns
```python
# Minimal working examples
MINIMAL_ENTITIES = [
    {"id": "a", "examples": ["example1", "example2"]}
]

# Realistic examples
COUNTRY_ENTITIES = [
    {"id": "us", "examples": ["USA", "United States"]}
]
```

## Test Performance

### Speed Considerations
- **Model Loading**: Tests use small/fast models
- **Training**: Minimal epochs for test speed
- **Fixture Scope**: Use `scope="module"` for expensive setup

### Performance Testing
```python
# Mark slow tests
@pytest.mark.slow
def test_with_large_dataset():
    pass

# Skip slow tests by default
uv run pytest -m "not slow"
```

## CI/CD Integration

### Current State
- **GitHub Actions**: Minimal configuration
- **Workflows**: Located in `.github/workflows/`
- **Status**: Basic setup, can be expanded

### Recommended CI Config
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - run: uv run pytest --cov=semanticmatcher
```

## Best Practices

### Do's
- ✅ Use fixtures for common test data
- ✅ Test exception cases with `pytest.raises()`
- ✅ Use descriptive test names
- ✅ Keep tests independent
- ✅ Use `pytest.approx()` for floats

### Don'ts
- ❌ Don't share state between tests
- ❌ Don't use hardcoded paths
- ❌ Don't skip tests without marking them
- ❌ Don't use print statements (use assertions)
