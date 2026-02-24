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
- Run locally with Sentence Transformers + SetFit

Example: `"Deutchland"` -> `DE`

## Installation

```bash
pip install semantic-matcher
```

Optional tools you may also need for experiments/notebooks:

- `jupyter` for `.ipynb` notebooks
- `geograpy` for `notebooks/geograpy.ipynb`

## Minimal Example (Official Package Wrapper API)

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

## Choose Your Path

| Path | Best For | Start Here |
|---|---|---|
| `EmbeddingMatcher` (no training) | Fast prototypes, simple setup | [`docs/quickstart.md`](docs/quickstart.md) |
| `EntityMatcher` (few-shot training) | Better accuracy with labeled examples | [`docs/quickstart.md`](docs/quickstart.md) |
| `notebooks/` experiments (scripts + Jupyter) | Reproducing experiments and explorations | [`docs/notebooks.md`](docs/notebooks.md) |
| Advanced/raw examples (`examples/`) | Lower-level SetFit / sentence-transformers workflows | [`docs/examples.md`](docs/examples.md) |

## Official vs Advanced Examples

- Official beginner path: `semanticmatcher` package wrappers (`EmbeddingMatcher`, `EntityMatcher`)
- Advanced/raw path: direct `setfit` / `sentence-transformers` usage in `examples/` and some notebooks

## Documentation

- [Docs Index](docs/index.md)
- [Quick Start Guide](docs/quickstart.md)
- [Notebook & Experiment Index](docs/notebooks.md)
- [Advanced Examples Guide](docs/examples.md)
- [Country Classifier Scripts](docs/country-classifier-scripts.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)

## Development

- Package code: `semanticmatcher/`
- Tests: `tests/`
- Experiments (scripts + Jupyter): `notebooks/`

Run tests (environment permitting):

```bash
uv run python -m pytest
```

## First-Run Expectations

- First run may download models from Hugging Face (network required).
- CPU works for small examples; training can be much slower than GPU.
- Some experiments/notebooks use extra libraries not required for core package usage.
