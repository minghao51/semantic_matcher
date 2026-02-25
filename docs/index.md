# Documentation Index

This folder is organized by audience and task.

## Start Here (New Users)

- [`quickstart.md`](./quickstart.md): official package wrapper API (`EmbeddingMatcher`, `EntityMatcher`)
- [`troubleshooting.md`](./troubleshooting.md): common install and first-run errors

## Experiments (`notebooks/`)

- [`notebooks.md`](./notebooks.md): canonical index for all experiment assets in `notebooks/` (`.py` scripts + `.ipynb`)
- [`country-classifier-scripts.md`](./country-classifier-scripts.md): deep dive on the country classifier experiment scripts now located in `notebooks/`

## Advanced / Raw Examples

- [`examples.md`](./examples.md): guide to `examples/` (direct `setfit` / `sentence-transformers` workflows)

## Research & Planning

- [`20260225-alternative-methods-roadmap.md`](./20260225-alternative-methods-roadmap.md): comprehensive analysis of alternative semantic matching methods (traditional NLP, cross-encoder rerankers, contrastive learning, graph neural networks, hybrid pipelines) with implementation guidance and future outlook for 2026-2027

## Contributors / Maintainers

- [`architecture.md`](./architecture.md): module layout and internals

## Reading Paths

### I want to use the library (recommended)

1. Read [`quickstart.md`](./quickstart.md)
2. Pick a matcher strategy (`EmbeddingMatcher` or `EntityMatcher`)
3. Use [`troubleshooting.md`](./troubleshooting.md) if setup/runtime issues appear
4. Explore [`notebooks.md`](./notebooks.md) for experiments after first success

### I want to reproduce experiments

1. Read [`notebooks.md`](./notebooks.md) for the experiment inventory and prerequisites
2. If you want country matching benchmarks, read [`country-classifier-scripts.md`](./country-classifier-scripts.md)
3. Run the corresponding file from `notebooks/`

### I want lower-level control

1. Read [`examples.md`](./examples.md)
2. Use raw `setfit` / `sentence-transformers` examples as a base
3. Refer to [`architecture.md`](./architecture.md) for project internals

## Notes

- The package code lives in `semanticmatcher/` (not `src/`).
- Previous experiment path `notebook/` was merged into `notebooks/`.
- Some backend integrations are documented as future/planned capabilities and may not be fully wired.
