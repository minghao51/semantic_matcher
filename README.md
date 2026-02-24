# SemanticMatcher

Map messy text to canonical entities using:

- `EntityMatcher` for few-shot SetFit training
- `EmbeddingMatcher` for embedding similarity matching (no training)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/semantic-matcher)](https://pypi.org/project/semantic-matcher/)
[![Python Version](https://img.shields.io/pypi/pyversions/semantic-matcher)](https://pypi.org/project/semantic-matcher/)

## What It Solves

- Normalize messy entity strings (typos, aliases, alternate names)
- Map text to canonical IDs (for example, country code matching)
- Run locally with Sentence Transformers

Example: `"Deutchland"` -> `DE`

## Installation

```bash
pip install semantic-matcher
```

## Minimal Example

```python
from semanticmatcher import EmbeddingMatcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()

print(matcher.match("America"))  # {"id": "US", "score": ...}
```

## Choose a Matcher

| Matcher | Best For | Tradeoff |
|---|---|---|
| `EmbeddingMatcher` | Prototyping, no training setup | Usually lower accuracy on harder cases |
| `EntityMatcher` | Production few-shot matching | Requires training data + training time |

## Documentation

- [Docs Index](docs/index.md)
- [Quick Start Guide](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Country Classifier Scripts](docs/country-classifier-scripts.md)

## Development

- Package code: `semanticmatcher/`
- Tests: `tests/`
- Example notebooks/scripts: `notebook/`, `notebooks/`

Run tests (environment permitting):

```bash
uv run python -m pytest
```
