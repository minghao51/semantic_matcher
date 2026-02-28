# SemanticMatcher

Map messy text to canonical entities using semantic matching:

- `EmbeddingMatcher` for embedding similarity matching (no training)
- `EntityMatcher` for few-shot SetFit training
- `HybridMatcher` for three-stage waterfall pipelines
- `CrossEncoderReranker` for high-precision reranking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/semantic-matcher)](https://pypi.org/project/semantic-matcher/)
[![Python Version](https://img.shields.io/pypi/pyversions/semantic-matcher)](https://pypi.org/project/semantic-matcher/)

## What It Solves

- Normalize messy entity strings (typos, aliases, alternate names)
- Map text to canonical IDs (for example, country code matching)
- Run locally with Sentence Transformers + SetFit

Example: `"Deutchland"` → `DE`

## Installation

```bash
pip install semantic-matcher
```

## Quick Start

### EmbeddingMatcher (No Training)

Fastest way to get started. No training required.

```python
from semanticmatcher import EmbeddingMatcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()

print(matcher.match("America"))  # {"id": "US", "score": 1.0}
```

### EntityMatcher (Few-Shot Training)

For higher accuracy with labeled examples.

```python
from semanticmatcher import EntityMatcher

training_data = [
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "USA", "label": "US"},
]

matcher = EntityMatcher(entities=entities)
matcher.train(training_data)

print(matcher.predict("Deutschland"))  # "DE"
```

## Feature Comparison

| Matcher | Training | Speed | Best For |
|---|---|---|---|---|
| `EmbeddingMatcher` | No | Fast (~50 q/s) | Prototyping, simple matching |
| `EntityMatcher` | Yes (3-5 examples/entity) | Medium (~30 q/s) | Production, complex variations |
| `HybridMatcher` | No | Medium (3-stage) | Large datasets (>10k entities) |

**Choosing a matcher**:
- Need results immediately? → `EmbeddingMatcher`
- Have labeled examples? → `EntityMatcher`
- Very large dataset? → `HybridMatcher`

## Documentation

- [Quick Start Guide](docs/quickstart.md) - Complete getting started guide
- [Examples Catalog](docs/examples.md) - All examples with difficulty ratings
- [Troubleshooting](docs/troubleshooting.md) - Common issues and fixes
- [Architecture](docs/architecture.md) - Module layout and design

## Where To Start

1. **New Users**: [Quick Start Guide](docs/quickstart.md)
2. **Working Examples**: [examples/embedding_matcher_demo.py](examples/embedding_matcher_demo.py)
3. **Advanced**: [docs/examples.md](docs/examples.md)

## Project Layout

```text
semantic_matcher/              # Repository root
├── src/semanticmatcher/       # Python package
├── examples/                  # Runnable examples (wrapper API)
├── experiments/               # Exploratory scripts
├── tests/                     # Automated tests
├── docs/                      # Documentation
└── pyproject.toml             # Packaging config
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidelines.
