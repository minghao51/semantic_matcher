# Benchmarking Guide

Related docs: [`../index.md`](../index.md) | [`../models.md`](../models.md) | [`../static-embeddings.md`](../static-embeddings.md) | [`benchmark-results.md`](./benchmark-results.md) (accuracy/size results) | [`speed-benchmark-results.md`](./speed-benchmark-results.md) (route speed results)

## Overview

Novel Entity Matcher includes a comprehensive benchmarking suite for comparing model performance across accuracy, latency, and throughput.

There are three benchmark entrypoints:

- `scripts/benchmark_embeddings.py` for model-level retrieval/training comparisons
- `scripts/benchmark_async.py` for route-level sync vs async API speed comparisons
- `scripts/render_benchmark_report.py` for rendering JSON artifacts as markdown tables

The current published accuracy benchmark uses query splits instead of entity
splits. That matters for retrieval: the full catalog stays indexed, while the
query text is evaluated across exact-text and synthetic-holdout buckets. The
named perturbation metrics such as `typo_accuracy` and
`remove_parenthetical_accuracy` are more interpretable than the legacy
positional `val` and `test` buckets.

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
uv run python scripts/benchmark_embeddings.py

# Run only embedding benchmarks
uv run python scripts/benchmark_embeddings.py --track embeddings

# Run only training benchmarks
uv run python scripts/benchmark_embeddings.py --track trained
```

### With Custom Options

```bash
# Benchmark specific models
uv run python scripts/benchmark_embeddings.py \
  --embedding-models potion-8m minilm bge-base

# Benchmark specific sections
uv run python scripts/benchmark_embeddings.py \
  --sections languages/languages universities/universities

# Save results
uv run python scripts/benchmark_embeddings.py \
  --output artifacts/benchmarks/benchmark-results.json

# Limit data size (faster testing)
uv run python scripts/benchmark_embeddings.py \
  --max-entities-per-section 30 \
  --max-queries-per-section 10
```

### Route Speed Benchmark

```bash
uv run python scripts/benchmark_async.py \
  --section languages/languages \
  --model default \
  --modes zero-shot head-only full \
  --max-entities 50 \
  --max-queries 25 \
  --multiplier 20 \
  --concurrency 8 \
  --output artifacts/benchmarks/speed-routes-languages.json
```

### Render a Markdown Report

```bash
uv run python scripts/render_benchmark_report.py \
  artifacts/benchmarks/benchmark-results.json

uv run python scripts/render_benchmark_report.py \
  artifacts/benchmarks/speed-routes-languages.json
```

## Understanding the Output

### Console Output

```
BENCHMARK RESULTS

[embedding]
<section: languages/languages>
           model backend  status  throughput_qps  accuracy_split  base_accuracy  val_accuracy  test_accuracy
potion-8m    static       ok        4032.12       val             0.9500         0.7250        1.0000
minilm       dynamic      ok         102.45       val             0.9500         0.7750        0.9474
bge-base     dynamic      ok          41.23       val             0.9600         0.7900        0.9500
```

**Key metrics:**
- `throughput_qps` - Queries per second (higher is better)
- `accuracy` - Top-1 accuracy on the preferred populated split
- `accuracy_split` - Which split that top-line accuracy came from
- `base_accuracy`, `train_accuracy`, `val_accuracy`, `test_accuracy` - split-level accuracy
- perturbation metrics such as `typo_accuracy` - robustness by transformation type
- `speedup_vs_minilm` - Relative speed vs minilm baseline
- `status` - "ok" or "skipped" (with skip_reason)

### JSON Output

```json
[
  {
    "track": "embedding",
    "section": "languages/languages",
    "model": "potion-8m",
    "resolved_model": "minishlab/potion-base-8M",
    "backend": "static",
    "status": "ok",
    "throughput_qps": 4032.12,
    "accuracy": 0.725,
    "accuracy_split": "val",
    "base_accuracy": 0.95,
    "typo_accuracy": 0.68,
    "avg_latency": 0.000248,
    "p95_latency": 0.000312,
    "build_time": 0.125
  }
]
```

## Benchmark Metrics Explained

### Throughput (QPS)

**Queries per second** - How many matches the system can process.

- **Higher is better** - Means faster processing
- **potion-8m**: ~4000 QPS (39x faster than minilm)
- **minilm**: ~100 QPS (baseline)
- **bge-base**: ~40 QPS (2.5x slower than minilm)

**When to care:**
- High-traffic APIs (>100 requests/second)
- Batch processing of large datasets
- Real-time applications with low latency requirements

### Accuracy

**Top-1 match accuracy** - Percentage of queries that match the correct entity.

- **Higher is better** - Means more correct matches
- Typical range: 0.80-0.95 (80-95%)
- Depends on dataset difficulty
- Use perturbation-specific metrics when you want to understand robustness.
  `val` and `test` are legacy positional holdout buckets, not guaranteed
  difficulty levels.

**When to care:**
- All applications - accuracy is always important
- Tradeoff with speed: static models trade slight accuracy for huge speed gains

### Latency

**Time per query** - How long a single match takes.

- **Lower is better** - Means faster response time
- **avg_latency** - Average query time
- **p95_latency** - 95th percentile (worst 5% of queries)
- **p99_latency** - 99th percentile (worst 1% of queries)

**When to care:**
- User-facing applications (need <100ms response time)
- Real-time systems (need <10ms response time)
- SLA requirements (e.g., 99% of requests <50ms)

### Build Time

**Time to initialize** - Model loading and index building.

- **Lower is better** - Means faster cold start
- Static models: ~0.1-0.5 seconds
- Dynamic models: ~0.5-2 seconds

**When to care:**
- Serverless functions (cold start matters)
- Frequent restarts
- Development iteration

### Route Timing Breakdown

The route speed benchmark now reports:

- `construct_seconds` - Matcher object construction
- `fit_seconds` - Zero-shot setup or supervised training time
- `cold_query_seconds` - First query after fit
- `match_seconds` - Steady-state route time for the benchmarked workload
- `end_to_end_seconds` - Combined construct + fit + cold + steady-state time

## Benchmark Sections

### What Are Sections?

Sections are processed datasets in `data/processed/*/*.csv`:

```
data/processed/
├── languages/
│   └── languages.csv
├── universities/
│   └── universities.csv
└── currencies/
    └── currencies.csv
```

Each CSV becomes a benchmark section:
- `languages/languages`
- `universities/universities`
- `currencies/currencies`

### Section Format

CSV columns:
- `id` - Entity ID
- `name` - Primary name
- `aliases` - Pipe-separated aliases (optional)
- `type` - Entity type (optional)

### Custom Sections

Add your own datasets for benchmarking:

```bash
mkdir -p data/processed/mydomain
cat > data/processed/mydomain/entities.csv << EOF
id,name,aliases
DE,Germany,Deutschland|DE
FR,France,frankreich|FR
EOF

# Benchmark your section
uv run python scripts/benchmark_embeddings.py \
  --sections mydomain/entities
```

## Interpreting Results for Model Selection

### Speed-Critical Applications

**Best choice:** `potion-8m`

**Why:**
- 39x faster than minilm
- Minimal accuracy tradeoff (92% vs 93%)
- Lowest latency (~0.25ms per query)

**Use when:**
- High-traffic APIs (>1000 req/s)
- Tight latency budgets (<10ms)
- Resource-constrained environments

### Accuracy-Critical Applications

**Best choice:** `bge-base`

**Why:**
- Highest accuracy (~94-95%)
- Better contextual understanding
- Still reasonable speed

**Use when:**
- Accuracy is paramount
- Lower traffic volumes
- Can tolerate higher latency

### Balanced Approach

**Best choice:** `minilm`

**Why:**
- Good accuracy (~93%)
- Reasonable speed (~100 QPS)
- Proven reliability

**Use when:**
- Uncertain about requirements
- Want a safe default
- Moderate traffic levels

### Multilingual Applications

**Best choices:**
- `mrl-multi` (static, fast)
- `bge-m3` (dynamic, accurate)

**Use when:**
- Supporting multiple languages
- Need language flexibility
- Trading off speed vs coverage

## Advanced Benchmarking

### Benchmarking Custom Models

```python
from novelentitymatcher.utils.benchmarks import benchmark_embedding_models

# Define your custom section
custom_section = {
    "section": "my-data",
    "entities": [...],
    "queries": [...],
    "accuracy_pairs": [...]
}

# Benchmark specific models
results = benchmark_embedding_models(
    model_names=["potion-8m", "my-custom-model"],
    sections_data=[custom_section],
    iterations=3
)

print(results)
```

### Benchmarking Training Modes

```python
from novelentitymatcher.utils.benchmarks import benchmark_trained_modes

results = benchmark_trained_modes(
    model_names=["mpnet", "bge-base"],
    modes=["head-only", "full"],
    num_epochs=1,
    sections_data=[custom_section]
)

print(results)
```

### Custom Metrics

```python
import pandas as pd

results = pd.read_json("artifacts/benchmarks/benchmark-results.json")

# Your custom analysis
avg_accuracy = results.groupby("model")["typo_accuracy"].mean()
speedup = results.set_index("model")["throughput_qps"] / results.loc[results["model"] == "minilm", "throughput_qps"].values[0]

print(f"Average typo accuracy by model:\n{avg_accuracy}")
print(f"\nSpeedup vs minilm:\n{speedup}")
```

## Reproducing Published Results

To reproduce the results in [`benchmark-results.md`](./benchmark-results.md):

```bash
uv run python scripts/benchmark_embeddings.py \
  --track embeddings \
  --sections languages/languages universities/universities \
  --embedding-models potion-8m potion-32m mrl-en mrl-multi minilm \
  --max-entities-per-section 30 \
  --max-queries-per-section 10 \
  --output my-benchmark.json
```

Compare your results with [`benchmark-results.md`](./benchmark-results.md):

```bash
# View published results
cat docs/experiments/benchmark-results.md

# Compare with your results
echo "Your results:"
cat my-benchmark.json | jq '.[] | {model, throughput_qps, accuracy}'
```

## Troubleshooting

### "No benchmark sections found"

**Cause:** No processed data in `data/processed/`.

**Solution:**
```bash
# Check for processed data
ls data/processed/*/*.csv

# Or specify sections explicitly if they exist
uv run python scripts/benchmark_embeddings.py --sections languages/languages
```

### Model loading errors

**Cause:** Model not downloaded or incompatible.

**Solution:**
```python
# Test model loading first
from novelentitymatcher import Matcher
test = Matcher(model="your-model", entities=[{"id": "1", "name": "test"}])
test.fit()
```

### Out of memory errors

**Cause:** Too many models or sections loaded simultaneously.

**Solution:**
```bash
# Benchmark one model at a time
uv run python scripts/benchmark_embeddings.py --embedding-models potion-8m

# Reduce data size
uv run python scripts/benchmark_embeddings.py \
  --max-entities-per-section 10 \
  --max-queries-per-section 5
```

### Slow benchmark execution

**Cause:** Large datasets or too many iterations.

**Solution:**
```bash
# Reduce iterations
uv run python scripts/benchmark_embeddings.py \
  --sections languages/languages \
  --max-entities-per-section 20

# Use fewer models
uv run python scripts/benchmark_embeddings.py \
  --embedding-models potion-8m minilm
```

## Performance Tips

### For Fastest Benchmarks

```bash
# Small dataset, few models, one section
uv run python scripts/benchmark_embeddings.py \
  --embedding-models potion-8m \
  --sections languages/languages \
  --max-entities-per-section 10 \
  --max-queries-per-section 5
```

### For Most Representative Results

```bash
# Real-world dataset sizes, multiple iterations
uv run python scripts/benchmark_embeddings.py \
  --sections languages/languages universities/universities \
  --max-entities-per-section 100 \
  --max-queries-per-section 50
```

### For Production Validation

```bash
# Use your actual data
cp /path/to/your/data.csv data/processed/yourdomain/entities.csv

# Benchmark against your data
uv run python scripts/benchmark_embeddings.py \
  --sections yourdomain/entities \
  --embedding-models potion-8m minilm bge-base
```

## Next Steps

- See [`benchmark-results.md`](./benchmark-results.md) for latest published results
- See [`models.md`](./models.md) for model selection guidance
- See [`static-embeddings.md`](./static-embeddings.md) for static embedding details
- See [`matcher-modes.md`](./matcher-modes.md) for mode selection
