# Documentation Index

This folder is organized by audience and task.

## Start Here (New Users)

- [`quickstart.md`](./quickstart.md): unified `Matcher` class with smart auto-selection
- [`troubleshooting.md`](./troubleshooting.md): common install and first-run errors

## Experiments & Notebooks

- [`notebooks.md`](./notebooks.md): canonical index for `experiments/` Python scripts and `notebooks/` Jupyter notebooks
- [`country-classifier-scripts.md`](./country-classifier-scripts.md): deep dive on the country classifier experiment scripts in `experiments/country_classifier/`

## Advanced / Raw Examples

- [`examples.md`](./examples.md): guide to `examples/` (direct `setfit` / `sentence-transformers` workflows)

## Models & Performance

- [`models.md`](./models.md): model registry, aliases, and selection guidance
- [`static-embeddings.md`](./static-embeddings.md): static embedding backend (model2vec, MRL)
- [`matcher-modes.md`](./matcher-modes.md): matcher mode system (zero-shot, head-only, full, hybrid)
- [`benchmarking.md`](./benchmarking.md): how to run and interpret benchmarks
- [`benchmark.md`](./benchmark.md): latest benchmark results

## Configuration & Internals

- [`configuration.md`](./configuration.md): configuration system and model registries
- [`migration-guide.md`](./migration-guide.md): migrating from deprecated classes

## Research & Planning

- [`20260225-alternative-methods-roadmap.md`](./20260225-alternative-methods-roadmap.md): comprehensive analysis of alternative semantic matching methods (traditional NLP, cross-encoder rerankers, contrastive learning, graph neural networks, hybrid pipelines) with implementation guidance and future outlook for 2026-2027

## Contributors / Maintainers

- [`architecture.md`](./architecture.md): module layout and internals

## Reading Paths

### I want to use the library (recommended)

1. Read [`quickstart.md`](./quickstart.md)
2. Choose a mode ([`matcher-modes.md`](./matcher-modes.md)) or use auto-selection
3. Select a model ([`models.md`](./models.md))
4. Use [`troubleshooting.md`](./troubleshooting.md) if setup/runtime issues appear
5. Explore [`notebooks.md`](./notebooks.md) for experiments after first success

### I want to reproduce experiments

1. Read [`notebooks.md`](./notebooks.md) for the experiment inventory and prerequisites
2. If you want country matching benchmarks, read [`country-classifier-scripts.md`](./country-classifier-scripts.md)
3. Run the corresponding file from `experiments/` (scripts) or open a notebook from `notebooks/`

### I want lower-level control

1. Read [`examples.md`](./examples.md)
2. Use raw `setfit` / `sentence-transformers` examples as a base
3. Refer to [`architecture.md`](./architecture.md) for project internals

### I want to optimize performance

1. Read [`static-embeddings.md`](./static-embeddings.md) for fast retrieval
2. Read [`benchmarking.md`](./benchmarking.md) to run your own benchmarks
3. Review [`benchmark.md`](./benchmark.md) for performance comparisons
4. Configure with [`configuration.md`](./configuration.md)

## Notes

- The package code lives in `src/semanticmatcher/` (src-layout).
- Script experiments live in `experiments/`; `notebooks/` is reserved for Jupyter notebooks.
- Some backend integrations are documented as future/planned capabilities and may not be fully wired.
