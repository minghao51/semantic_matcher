# Examples (`examples/`)

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`notebooks.md`](./notebooks.md)

This page catalogs all examples in `examples/` and helps you choose where to start.

## Two Learning Paths

### Path 1: Wrapper API (Recommended)
Start here for production use. These examples use the official `semanticmatcher` wrapper APIs.

### Path 2: Raw Library Examples (Advanced)
These examples demonstrate direct use of lower-level libraries like `setfit` and `sentence-transformers`. They're useful for advanced customization and understanding internals, but aren't the fastest way to get started.

---

## Wrapper API Examples (Recommended Path)

Start with these examples to learn the official API:

| Example | Difficulty | Runtime | API Features Demonstrated |
|---|---|---|---|
| [`embedding_matcher_demo.py`](../examples/embedding_matcher_demo.py) | Beginner | 30s | EmbeddingMatcher, build_index(), match(), threshold, top_k, TextNormalizer |
| [`entity_matcher_demo.py`](../examples/entity_matcher_demo.py) | Beginner | 2-3 min | EntityMatcher, train(), predict(), predict_proba(), threshold, model selection |
| [`model_persistence.py`](../examples/model_persistence.py) | Intermediate | 3-4 min | SetFitClassifier.save/load, model versioning, production deployment |
| [`batch_processing.py`](../examples/batch_processing.py) | Intermediate | 1-2 min | match_bulk(), parallel processing, performance benchmarking |
| [`matcher_comparison.py`](../examples/matcher_comparison.py) | Intermediate | 4-5 min | EntityMatcher vs EmbeddingMatcher comparison, decision matrix |
| [`threshold_tuning.py`](../examples/threshold_tuning.py) | Intermediate | 2 min | Threshold parameter impact, validation, precision/recall tradeoffs |
| [`hybrid_matching_demo.py`](../examples/hybrid_matching_demo.py) | Intermediate | 30s | HybridMatcher, three-stage pipeline, blocking strategies, BM25Blocking |

### Learning Sequence

**Absolute Beginner**:
1. Start with [`embedding_matcher_demo.py`](../examples/embedding_matcher_demo.py) - No training required, immediate results
2. Then [`entity_matcher_demo.py`](../examples/entity_matcher_demo.py) - Learn training workflow
3. Then [`matcher_comparison.py`](../examples/matcher_comparison.py) - Understand when to use each approach

**Production Readiness**:
4. [`model_persistence.py`](../examples/model_persistence.py) - Save/load models for deployment
5. [`batch_processing.py`](../examples/batch_processing.py) - Handle bulk queries efficiently
6. [`threshold_tuning.py`](../examples/threshold_tuning.py) - Optimize accuracy for your use case

**Large Scale / Advanced**:
7. [`hybrid_matching_demo.py`](../examples/hybrid_matching_demo.py) - Three-stage pipeline for big datasets

---

## Raw Library Examples (Advanced Path)

These examples use `setfit`, `sentence-transformers`, or `datasets` directly. They're useful for:
- Understanding the underlying libraries
- Advanced customization beyond what wrappers provide
- Experimental workflows

| Example | Category | What it demonstrates | When to use it |
|---|---|---|---|
| [`basic_usage.py`](../examples/basic_usage.py) | Raw SetFit training | Minimal few-shot entity matching with direct SetFit trainer/model usage | Learn/control SetFit internals |
| [`country_matching.py`](../examples/country_matching.py) | Raw SetFit training | Country-code matching with expanded labels/training data | Build a larger SetFit baseline outside wrappers |
| [`custom_backend.py`](../examples/custom_backend.py) | Model/backend exploration | Multilingual/small/large model tradeoffs via direct SetFit usage | Compare embedding backbone choices |
| [`zero_shot_classification.py`](../examples/zero_shot_classification.py) | Generic SetFit classification | Sentiment/intent examples using SetFit for text classification | Non-entity use cases / SetFit learning |

### Notes for Raw Examples

- These files may bypass `semanticmatcher.EntityMatcher` / `EmbeddingMatcher`
- They may use APIs/options that differ from the wrapper defaults
- Read [`quickstart.md`](./quickstart.md) first if your goal is entity matching with the project API
- The examples in this category are provided for learning and experimentation

---

## Suggested Workflow

1. **Start here**: Read [`quickstart.md`](./quickstart.md) for the wrapper API introduction
2. **Learn by doing**: Run the wrapper API examples in order (Beginner → Intermediate → Advanced)
3. **Experiment**: Use [`notebooks.md`](./notebooks.md) for project experiments and exploratory work
4. **Advanced**: Explore raw examples when you need lower-level control or customization

---

## Quick Reference by Use Case

**I want to...**
- ...match entities without training → [`embedding_matcher_demo.py`](../examples/embedding_matcher_demo.py)
- ...train a model with labeled data → [`entity_matcher_demo.py`](../examples/entity_matcher_demo.py)
- ...choose between matchers → [`matcher_comparison.py`](../examples/matcher_comparison.py)
- ...process many queries efficiently → [`batch_processing.py`](../examples/batch_processing.py)
- ...save/load models → [`model_persistence.py`](../examples/model_persistence.py)
- ...improve accuracy → [`threshold_tuning.py`](../examples/threshold_tuning.py)
- ...handle large datasets (10k+ entities) → [`hybrid_matching_demo.py`](../examples/hybrid_matching_demo.py)
- ...understand SetFit internals → [`basic_usage.py`](../examples/basic_usage.py)
- ...compare different models → [`custom_backend.py`](../examples/custom_backend.py)
