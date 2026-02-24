# Advanced / Raw Examples (`examples/`)

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`notebooks.md`](./notebooks.md)

This page explains the examples in `examples/` and how they differ from the recommended beginner path.

## What These Examples Are

Most files in `examples/` demonstrate direct use of lower-level libraries such as:

- `setfit`
- `datasets`
- `sentence-transformers`

They are useful for advanced customization and experimentation, but they are not the fastest way to get started with `semanticmatcher`.

## Recommended vs Advanced Paths

- Recommended first path: `semanticmatcher` wrapper APIs in [`quickstart.md`](./quickstart.md)
- Advanced/raw path: direct `setfit` and `sentence-transformers` workflows in `examples/`

## Current Example Inventory

| File | Category | What it demonstrates | When to use it |
|---|---|---|---|
| `examples/basic_usage.py` | Raw SetFit training | Minimal few-shot entity matching with direct SetFit trainer/model usage | Learn/control SetFit internals |
| `examples/country_matching.py` | Raw SetFit training | Country-code matching with expanded labels/training data | Build a larger SetFit baseline outside wrappers |
| `examples/custom_backend.py` | Model/backend exploration | Multilingual/small/large model tradeoffs via direct SetFit usage | Compare embedding backbone choices |
| `examples/zero_shot_classification.py` | Generic SetFit classification | Sentiment/intent examples using SetFit for text classification | Non-entity use cases / SetFit learning |

## Notes for New Users

- These files may bypass `semanticmatcher.EntityMatcher` / `EmbeddingMatcher`.
- They may use APIs/options that differ from the wrapper defaults.
- Read [`quickstart.md`](./quickstart.md) first if your goal is entity matching with the project API.

## Suggested Workflow

1. Start with [`quickstart.md`](./quickstart.md)
2. Use [`notebooks.md`](./notebooks.md) for project experiments
3. Use `examples/` when you need lower-level control or customization
