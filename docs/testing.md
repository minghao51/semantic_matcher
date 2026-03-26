# Testing Guide

## Test Categories

Tests are marked with pytest markers to enable selective test execution:

| Marker | Description | Run by Default |
|--------|-------------|----------------|
| `llm` | Actual LLM API calls (require API key, slow) | No |
| `llm_mocked` | LLM-related logic with mocked responses | Yes |
| `e2e` | End-to-end / feature tests (multi-component) | Yes |
| `unit` | Fast, isolated unit tests | Yes |

## Running Tests

```bash
# Run all tests except llm (actual API calls)
pytest -m "not llm"

# Run only llm_mocked tests
pytest -m llm_mocked

# Run e2e tests
pytest -m e2e

# Run everything including actual LLM calls (requires API key)
pytest -m llm

# Run unit tests only
pytest -m "unit and not llm_mocked"
```

## Test Organization

### LLM Tests (`llm`)

Tests that make **actual** LLM API calls. These are skipped by default.

| File | Test | Notes |
|------|------|-------|
| `tests/test_integration.py` | `TestLLMIntegration::test_llm_proposal_generation` | Skipped by default, requires manual enable |

### LLM Mocked Tests (`llm_mocked`)

Tests that involve LLM logic but use mocks instead of real API calls.

| File | Tests |
|------|-------|
| `tests/test_llm_proposer.py` | All 20+ tests - parsing, fallback, config, litellm mocking |
| `tests/test_discovery_pipeline.py` | `test_discovery_pipeline_creates_review_records` - mocks `llm_proposer.propose_from_clusters` |

### E2E / Feature Tests (`e2e`)

Complex multi-component tests exercising full pipelines.

| File | Description |
|------|-------------|
| `tests/test_integration.py` | `TestNovelClassDetectionIntegration` - full discovery pipeline, async paths, file persistence, batch |
| `tests/test_discovery_pipeline.py` | Pipeline + review lifecycle tests |
| `tests/test_integration_extended.py` | `TestAsyncAPIIntegration`, `TestErrorHandlingIntegration` |
| `tests/test_novel_entity_matcher.py` | High-level matcher integration |

### Unit Tests

Fast, isolated tests - all other test files.

| File | Description |
|------|-------------|
| `tests/test_signal_combiner.py` | Logic only, no I/O |
| `tests/test_persistence.py` | File I/O but isolated |
| `tests/test_pipeline_orchestrator.py` | Orchestration logic |
| `tests/test_review_manager.py` | State transitions |
| `tests/test_schema_validation.py` | Schema validation |
| `tests/test_novelty_detector.py` | Detector logic |
| `tests/test_novelty_detector_lifecycle.py` | Lifecycle logic |
| `tests/test_prototypical_strategy.py` | Prototypical strategy |
| `tests/test_oneclass_strategy.py` | One-class strategy |
| `tests/test_pattern_strategy.py` | Pattern strategy |
| `tests/test_setfit_novelty.py` | SetFit novelty |
| `tests/test_ann_index.py` | ANN index operations |
| `tests/test_config.py` | Config parsing |
| `tests/test_backends/*.py` | Backend contracts and implementations |
| `tests/test_core/*.py` | Core utilities |
| `tests/test_utils/*.py` | Utility functions |
| `tests/test_ingestion/*.py` | Ingestion CLI and timezone handling |
| `tests/test_packaging.py` | Packaging tests |

## Marker Configuration

Markers are defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: tests that depend on external services or network access",
    "slow: tests that are expensive to run in default CI",
    "hf: Hugging Face model-backed tests",
    "llm: tests that make actual LLM API calls (require API key, slow)",
    "llm_mocked: tests that involve LLM logic but use mocks instead of real API calls",
    "e2e: end-to-end / feature tests that exercise multiple components",
]
```

## Best Practices

- Mark new tests appropriately when adding them
- `llm` tests should be skipped by default (`@pytest.mark.skip` or don't mark, only mark `llm_mocked`)
- E2E tests should exercise real component integration without mocking across component boundaries
- Unit tests should be fast and isolated

## Comprehensive Test Run

Run all tests with full reporting (coverage, HTML report):

```bash
uv run python -m pytest \
  -m "not slow and not hf" \
  --cov=novelentitymatcher \
  --cov-report=html \
  --cov-report=term-missing \
  --html=htmlcov/test_report.html \
  -v
```

**Prerequisites:**
```bash
uv pip install pytest-cov pytest-html
```

**Outputs:**
- `htmlcov/index.html` - HTML coverage report (also includes test report via `--html`)

**Full run including HuggingFace tests (slower):**
```bash
uv run python -m pytest -m "not slow" --cov=novelentitymatcher --cov-report=html --html=htmlcov/test_report.html -v
```

**With actual LLM API calls (requires API key):**
```bash
OPENAI_API_KEY=sk-... uv run python -m pytest -m "not slow and not hf" --cov=novelentitymatcher --cov-report=html --html=htmlcov/test_report.html -v
```