# Benchmark Results

This document was refreshed on March 14, 2026 using the harder
`products/products_mcc` benchmark split instead of the old exact-alias lookup
benchmark.

## What Changed

- Shared aliases are removed from benchmark entities before evaluation.
- Accuracy is now reported by split and perturbation type, not as one overloaded
  number:
  - `base`: first verbatim indexed text sanity check
  - `train`: additional exact query texts, excluding the `base` exact text
  - `val`: first surviving synthetic holdout perturbation
  - `test`: second surviving synthetic holdout perturbation
  - perturbation buckets: `typo`, `remove_parenthetical`,
    `ampersand_expanded`, `first_clause`, `normalized_verbatim`
- This is a query-split benchmark, not an entity-split benchmark.
  The full entity catalog remains indexed because retrieval must search the real
  candidate set; what changes across splits is query difficulty.

## Command

```bash
uv run python scripts/benchmark_embeddings.py \
  --track embeddings \
  --sections products/products_mcc \
  --embedding-models potion-8m potion-32m minilm mpnet \
  --max-entities-per-section 120 \
  --max-queries-per-section 40 \
  --output artifacts/benchmarks/products-mcc-split-benchmark.json
```

## Split Semantics

| Split | Meaning | Pair Count |
|---|---|---:|
| `base` | Verbatim indexed text sanity check | `40` |
| `train` | Exact generated training query texts | `40` |
| `val` | First surviving holdout perturbation | `40` |
| `test` | Second surviving holdout perturbation | `19` |

The smaller `test` count is expected because not every MCC label can produce two
distinct non-verbatim holdout queries.

## Results

| Model | Size | Throughput (qps) | Avg Latency (ms/query) | Base | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| `potion-8m` | `8M` | `23621.66` | `0.1699` | `0.95` | `1.00` | `0.725` | `1.00` |
| `potion-32m` | `32M` | `19472.21` | `0.2070` | `0.95` | `1.00` | `0.75` | `1.00` |
| `minilm` | `22.7M` | `671.52` | `6.1636` | `0.95` | `1.00` | `0.775` | `0.9474` |
| `mpnet` | `109M` | `332.91` | `10.3011` | `0.95` | `1.00` | `0.75` | `0.9474` |

## Perturbation Breakdown

| Model | Typo | Remove Parenthetical | Ampersand Expanded | First Clause | Normalized Verbatim |
|---|---:|---:|---:|---:|---:|
| `potion-8m` | `0.725` | `1.00` | `1.00` | `1.00` | `n/a` |
| `potion-32m` | `0.75` | `1.00` | `1.00` | `1.00` | `n/a` |
| `minilm` | `0.775` | `1.00` | `1.00` | `0.8571` | `n/a` |
| `mpnet` | `0.75` | `0.9167` | `1.00` | `1.00` | `n/a` |

Pair counts by perturbation:

| Bucket | Pair Count |
|---|---:|
| `typo` | `40` |
| `remove_parenthetical` | `12` |
| `ampersand_expanded` | `1` |
| `first_clause` | `7` |
| `normalized_verbatim` | `0` |

## Interpretation

- This benchmark is now materially more honest than the old report because the
  holdout splits are not exact indexed aliases.
- `base` and `train` are still exact-text checks, so they are expected to stay
  high and should not be over-interpreted as generalization metrics.
- `val` and `test` are positional holdout buckets, not calibrated difficulty
  levels. The perturbation-specific metrics are the more trustworthy way to
  compare robustness.
- In this run, typo robustness is the clearest weak spot across all models.
  Structural rewrites like removing parentheticals were much easier than typos.
- `test` is still small and should be treated as directional, not conclusive.
- The static models remain dramatically faster than the dynamic models on this
  machine, and they remain competitive on the harder MCC holdout splits.
- The next quality step, if we want a stronger benchmark still, is a curated
  human-written query set rather than deterministic perturbations.
