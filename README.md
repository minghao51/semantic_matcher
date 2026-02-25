# SemanticMatcher

Map messy text to canonical entities using:

- `EmbeddingMatcher` for embedding similarity matching (no training)
- `EntityMatcher` for few-shot SetFit training
- `CrossEncoderReranker` for high-precision reranking
- `HybridMatcher` for three-stage waterfall pipelines

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

## Project Layout

```text
semantic_matcher/              # Repository root
├── src/semanticmatcher/       # Python package (import path: `semanticmatcher`)
├── tests/                     # Automated tests
├── examples/                  # Runnable user-facing examples
├── experiments/               # Exploratory Python scripts (non-product workflows)
├── notebooks/                 # Jupyter notebooks only (reserved for `.ipynb`)
├── data/                      # Sample/raw/processed datasets for demos/ingestion
├── docs/                      # Hand-written docs (source); generated docs output is ignored
└── pyproject.toml             # Packaging, dependencies, and tool config
```

### Repo Conventions

- Project/package distribution name (PyPI): `semantic-matcher`
- Python import path: `semanticmatcher`
- Repository folder name: `semantic_matcher`

These names differ because Python packaging conventions (distribution names vs import/module names) commonly use hyphens for PyPI and underscores/no hyphens for imports/filesystem paths.

## Where To Start

- Users: start here in `README.md`, then [`docs/quickstart.md`](docs/quickstart.md), then [`examples/basic_usage.py`](examples/basic_usage.py)
- Contributors: start with [`docs/architecture.md`](docs/architecture.md), then `src/semanticmatcher/`, then `tests/`

## Minimal Example (Official Package Wrapper API)

### Basic Embedding Matching

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

### Hybrid Matching Pipeline (New!)

For maximum accuracy with large datasets, use the three-stage pipeline:

```python
from semanticmatcher import HybridMatcher, BM25Blocking

matcher = HybridMatcher(
    entities=products,
    blocking_strategy=BM25Blocking(),  # Fast lexical filtering
    retriever_model="bge-base",        # Semantic search
    reranker_model="bge-m3",            # Precise reranking
)

results = matcher.match(
    "iPhone 15 case",
    blocking_top_k=1000,    # Candidates after blocking
    retrieval_top_k=50,     # Candidates after retrieval
    final_top_k=5           # Final results
)
```

### Cross-Encoder Reranking (New!)

Rerank top candidates for higher precision:

```python
from semanticmatcher import EmbeddingMatcher, CrossEncoderReranker

# Initial retrieval
retriever = EmbeddingMatcher(entities)
retriever.build_index()
candidates = retriever.match(query, top_k=50)

# Rerank with cross-encoder
reranker = CrossEncoderReranker(model="bge-m3")
final_results = reranker.rerank(query, candidates, top_k=5)
```

## Choose Your Path

| Path | Best For | Start Here |
|---|---|---|
| `EmbeddingMatcher` (no training) | Fast prototypes, simple setup | [`docs/quickstart.md`](docs/quickstart.md) |
| `EntityMatcher` (few-shot training) | Better accuracy with labeled examples | [`docs/quickstart.md`](docs/quickstart.md) |
| `HybridMatcher` (three-stage pipeline) | Large datasets, maximum accuracy | [`examples/hybrid_matching_demo.py`](examples/hybrid_matching_demo.py) |
| `CrossEncoderReranker` | Rerank candidates for precision | See Hybrid example above |
| `experiments/` + `notebooks/` | Reproducing exploratory scripts and Jupyter work | [`docs/notebooks.md`](docs/notebooks.md) |
| Advanced/raw examples (`examples/`) | Lower-level SetFit / sentence-transformers workflows | [`docs/examples.md`](docs/examples.md) |

## Official vs Advanced Examples

- Official beginner path: `semanticmatcher` package wrappers (`EmbeddingMatcher`, `EntityMatcher`)
- Advanced/raw path: direct `setfit` / `sentence-transformers` usage in `examples/`, `experiments/`, and optional notebooks

## Documentation

- [Docs Index](docs/index.md)
- [Quick Start Guide](docs/quickstart.md)
- [Notebook & Experiment Index](docs/notebooks.md)
- [Advanced Examples Guide](docs/examples.md)
- [Country Classifier Scripts](docs/country-classifier-scripts.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)

## Development

- Package code: `src/semanticmatcher/`
- Tests: `tests/`
- Script experiments: `experiments/`
- Jupyter notebooks: `notebooks/` (reserved for `.ipynb`)
- CLI ingestion entrypoint: `semanticmatcher-ingest` -> `src/semanticmatcher/ingestion/cli.py`
- Contributor guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

Set up the dev environment (preferred):

```bash
uv sync --group dev
```

Run tests (environment permitting):

```bash
uv run python -m pytest
```

## First-Run Expectations

- First run may download models from Hugging Face (network required).
- CPU works for small examples; training can be much slower than GPU.
- Some experiments/notebooks use extra libraries not required for core package usage.
