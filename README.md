# Novel Entity Matcher

Map messy text to canonical entities with automatic novel entity detection and classification.

**New:** Unified `Matcher` class with smart auto-selection - no need to choose between different matchers!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/novel-entity-matcher)](https://pypi.org/project/novel-entity-matcher/)
[![Python Version](https://img.shields.io/pypi/pyversions/novel-entity-matcher)](https://pypi.org/project/novel-entity-matcher/)

## What It Solves

- Normalize messy entity strings (typos, aliases, alternate names)
- Map text to canonical IDs (for example, country code matching)
- Run locally with Sentence Transformers + SetFit

Example: `"Deutchland"` → `DE`

## Installation

```bash
uv add novel-entity-matcher
```

Optional extras:

```bash
# Novel class detection and ANN-backed discovery
uv add "novel-entity-matcher[novelty]"

# LiteLLM-powered embeddings, reranking, and class proposal features
uv add "novel-entity-matcher[llm]"

# Benchmark visualization scripts
uv add "novel-entity-matcher[viz]"

# Everything
uv add "novel-entity-matcher[all]"
```

If you are not using `uv`, the equivalent `pip` commands still work:

```bash
pip install novel-entity-matcher
pip install "novel-entity-matcher[novelty]"
pip install "novel-entity-matcher[llm]"
pip install "novel-entity-matcher[viz]"
pip install "novel-entity-matcher[all]"
```

## Quick Start

### The New Unified API (Recommended Async Default)

**Single `Matcher` class that auto-detects the best approach, with async shown first:**

```python
import asyncio
from novelentitymatcher import Matcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

async def main():
    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()
        print(await matcher.match_async("America"))  # {"id": "US", "score": 0.95}

        training_data = [
            {"text": "Germany", "label": "DE"},
            {"text": "Deutschland", "label": "DE"},
            {"text": "USA", "label": "US"},
        ]
        await matcher.fit_async(training_data)  # Auto: head-only for <3 examples, full for ≥3
        print(await matcher.match_async("Deutschland"))  # {"id": "DE", "score": 1.0}

asyncio.run(main())
```

Prefer the async API for new integrations, especially in web services, batch jobs, or concurrent workloads. The sync API remains fully supported for scripts and simple one-off usage.

**How it works:**
- No training data → zero-shot (embedding similarity)
- < 3 examples/entity → head-only training (~30s)
- ≥ 3 examples/entity → full training (~3min)

### Sync Alternative

```python
from novelentitymatcher import Matcher

matcher = Matcher(entities=entities)
matcher.fit()
print(matcher.match("America"))
```

### Alternative: Explicit Mode Selection

```python
# Force zero-shot mode
matcher = Matcher(entities=entities, mode="zero-shot")

# Force full training mode
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)

# Force hybrid mode (blocking + retrieval + reranking)
matcher = Matcher(entities=entities, mode="hybrid")
matcher.fit()
results = matcher.match("America", top_k=3)
```

## Feature Comparison

| Mode | Training | Speed | Best For |
|---|---|---|---|---|
| `zero-shot` | No | Fast (~50 q/s) | Prototyping, simple matching |
| `head-only` | Yes (~30s) | Medium (~30 q/s) | Quick accuracy boost |
| `full` | Yes (~3min) | Medium (~30 q/s) | Production, complex variations |
| `hybrid` | No | Slower, highest precision | Large candidate sets, reranking |

**The new `Matcher` class auto-selects the best mode** based on your training data.

## Async API

The async API is the recommended default for new code. It provides non-blocking operations with progress tracking and cancellation support:

```python
import asyncio
from novelentitymatcher import Matcher

async def main():
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "US", "name": "United States", "aliases": ["USA"]},
    ]

    # Use async context manager for automatic cleanup
    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        # Batch processing with progress tracking
        async def show_progress(completed, total):
            print(f"Progress: {completed}/{total}")

        results = await matcher.match_batch_async(
            queries=["USA", "Germany"] * 1000,
            batch_size=100,
            on_progress=show_progress
        )

        # Or use concurrent matchers
        async def match_category(category_entities):
            async with Matcher(entities=category_entities) as m:
                await m.fit_async()
                return await m.match_async("query")

        results = await asyncio.gather(
            match_category(entities_1),
            match_category(entities_2),
            match_category(entities_3),
        )

asyncio.run(main())
```

**Key features:**
- `fit_async()`, `match_async()`, `match_batch_async()` - Async versions of core methods
- Progress tracking via `on_progress` callback
- Cancellation support for long-running operations
- Thread-safe concurrent matching
- Sync and async APIs are both supported

See [Async API Guide](docs/async-guide.md) for comprehensive documentation.

## Embedding Models

### Default: Static Embeddings

Novel Entity Matcher uses **static embeddings by default for retrieval**:

```python
# Retrieval default
matcher = Matcher(mode="zero-shot")  # Uses "potion-8m" by default
```

**Benefits:**
- 10-100x faster than dynamic embeddings
- Lower memory usage
- Sufficient accuracy for most use cases

### Available Models

| Alias | Model | Type | Backend | Best For |
|-------|-------|------|---------|----------|
| `potion-8m` | minishlab/potion-base-8M | Static | model2vec | English general use (retrieval default) |
| `potion-32m` | minishlab/potion-base-32M | Static | model2vec | Fast English with better quality |
| `mrl-en` | RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en | Static (MRL) | StaticEmbedding | English with MRL support* |
| `mrl-multi` | sentence-transformers/static-similarity-mrl-multilingual-v1 | Static (MRL) | StaticEmbedding | Multilingual (NEW!) |
| `bge-m3` | BAAI/bge-m3 | Dynamic | SentenceTransformer | Multilingual (opt-in dynamic) |
| `bge-base` | BAAI/bge-base-en-v1.5 | Dynamic | SentenceTransformer | High accuracy (English) |

*Note: The RikkaBotan MRL model may require MPS fallback on Apple Silicon (set `PYTORCH_ENABLE_MPS_FALLBACK=1`).

### Static Embedding Backends

Novel Entity Matcher supports two static embedding approaches:

1. **StaticEmbedding** (sentence-transformers): For models like RikkaBotan MRL
2. **model2vec**: For minishlab potion models and custom distillations

Both are automatically detected and used based on the model name.

### Training-Safe Default

`Matcher.fit(...)` automatically switches to a training-compatible backbone for
`head-only` and `full` modes. The current training default is `mpnet`, so
`Matcher(model="default")` stays fast for zero-shot retrieval without breaking
SetFit-based training.

### Opting into Dynamic Embeddings

For scenarios requiring contextual understanding:

```python
matcher = Matcher(mode="zero-shot", model="bge-base")
```

### Multilingual Support

```python
matcher = Matcher(mode="zero-shot", model="bge-m3")
```

### Benchmarking

Run the comprehensive benchmark suite with:

```bash
uv run python scripts/benchmark_embeddings.py --track all --output artifacts/benchmarks/benchmark-results.json
```

Run a lightweight async-vs-sync comparison with:

```bash
uv run python scripts/benchmark_async.py --multiplier 20 --concurrency 8
```

## Documentation

- [Documentation Index](docs/index.md) - Organized entry point for guides, experiments, and archive material
- [Quick Start Guide](docs/quickstart.md) - Complete getting started guide
- [Examples Catalog](docs/examples.md) - Maintained runnable examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and fixes
- [Architecture](docs/architecture.md) - Module layout and design

## Where To Start

1. **New Users**: [Quick Start Guide](docs/quickstart.md)
2. **Working Examples**: [examples/current/basic_matcher.py](examples/current/basic_matcher.py)
3. **Advanced**: [docs/examples.md](docs/examples.md)

## Project Layout

```text
novel_entity_matcher/              # Repository root
├── src/novelentitymatcher/       # Python package
├── examples/                  # Maintained runnable examples
├── experiments/               # Exploratory scripts
├── artifacts/                 # Local generated benchmark outputs
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
