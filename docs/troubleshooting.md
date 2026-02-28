# Troubleshooting

Related docs: [`quickstart.md`](./quickstart.md) | [`notebooks.md`](./notebooks.md) | [`index.md`](./index.md)

This page covers common setup and first-run issues for the package plus exploratory scripts/notebooks.

## Import Errors (`setfit`, `datasets`, `torch`, `sentence-transformers`)

Symptoms:

- `ImportError: setfit is required...`
- `ModuleNotFoundError` for `datasets`, `torch`, or `sentence_transformers`

What to check:

1. Install the project dependencies in the active environment.
2. Make sure your Jupyter kernel matches the environment where the package is installed.
3. Re-run from the repo root if using `PYTHONPATH=.`

## Slow First Run / Model Downloads

First run often downloads model weights from Hugging Face. This can take time depending on model size and network speed.

What to expect:

- Small examples: may still pause while downloading
- Some advanced experiments/notebooks may require larger model downloads and slower startup
- Typical model sizes: 100-500MB

## CPU vs GPU Expectations

- CPU works for basic testing and small examples.
- SetFit training and large embedding models can be significantly slower on CPU.
- GPU is optional but helpful for the country classifier experiments and larger embedding experiments.

## `EmbeddingMatcher` Error: Index Not Built

Symptom:

- `RuntimeError("Index not built. Call build_index() first.")`

Fix:

```python
matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()
result = matcher.match("query")
```

## `EntityMatcher` Error: Model Not Trained

Symptom:

- `RuntimeError("Model not trained. Call train() first.")`

Fix:

```python
matcher = EntityMatcher(entities=entities)
matcher.train(training_data)
result = matcher.predict("query")
```

## Low-Confidence Matches (Returns `None`)

Symptoms:

- `match()` or `predict()` returns `None` for valid queries
- Fewer matches than expected

Causes:

- **Threshold too high**: Confidence score below threshold
- **Insufficient training data**: Model hasn't learned variations
- **Wrong matcher type**: Using EmbeddingMatcher for complex variations

Solutions:

1. **Lower the threshold**:
```python
# Default threshold is 0.7, try lowering
matcher = EmbeddingMatcher(entities, threshold=0.5)
```

2. **Check confidence scores**:
```python
# For EntityMatcher, see probability distribution
probs = matcher.classifier.predict_proba("query")
print({label: prob for label, prob in zip(matcher.classifier.labels, probs)})
```

3. **Use EntityMatcher instead**:
```python
# EmbeddingMatcher struggles with variations
# EntityMatcher handles them better with training
entity_matcher = EntityMatcher(entities)
entity_matcher.train(training_data)  # Include variations in training
```

## Threshold Tuning Guidance

If you're getting **too many matches** (low precision):

- Raise threshold: `0.7` → `0.8` or `0.9`
- Use validation data to find optimal threshold

If you're getting **too few matches** (low recall):

- Lower threshold: `0.7` → `0.6` or `0.5`
- Check if queries have typos or extreme variations
- Consider adding more training examples (EntityMatcher)

**Recommended thresholds by use case**:

- **High precision** (0.8-0.9): Database lookups, exact matching
- **Balanced** (0.7, default): General purpose
- **High recall** (0.5-0.6): Fuzzy search, data cleaning

See [`examples/threshold_tuning.py`](../examples/threshold_tuning.py) for a complete guide.

## Model Selection Issues

**Problem**: Model not working well for your language/domain

**Solutions**:

1. **Try multilingual models**:
```python
# For non-English text
matcher = EntityMatcher(entities, model_name="sentence-transformers/LaBSE")
# or
matcher = EmbeddingMatcher(entities, model_name="paraphrase-multilingual-mpnet-base-v2")
```

2. **Use domain-specific models**:
```python
# For high-accuracy English
matcher = EmbeddingMatcher(entities, model_name="bge-base")

# For speed
matcher = EmbeddingMatcher(entities, model_name="minilm")
```

## Notebook Dependency Issues (`jupyter`, `geograpy`)

### Jupyter

- Install Jupyter in the same environment as `semanticmatcher`
- Launch from repo root to avoid path confusion

### `geograpy`

- If you add a local `geograpy` notebook experiment, expect extra installs and dependency troubleshooting beyond the core project

## Path Migration Note (`notebook/` -> `experiments/` / `notebooks/`)

The old experiment script path `notebook/...` is now split by artifact type:

- script experiments -> `experiments/...`
- Jupyter notebooks -> `notebooks/...`

Updated examples:

- `experiments/country_classifier/country_classifier.py`
- `experiments/country_classifier/country_classifier_quick.py`
- `experiments/country_classifier/country_classifier_advanced.py`

## Performance Issues

**Problem**: Matching is too slow

**Solutions**:

1. **Use faster model**:
```python
# Use minilm for speed (3-5x faster)
matcher = EmbeddingMatcher(entities, model_name="minilm")
```

2. **Use batch processing**:
```python
# For multiple queries
results = matcher.match_bulk(queries)  # HybridMatcher
# Or loop for EmbeddingMatcher (still fast)
results = [matcher.match(q) for q in queries]
```

3. **Reduce embedding dimension** (Matryoshka):
```python
# Use smaller embeddings for faster matching
matcher = EmbeddingMatcher(entities, embedding_dim=256)
```

## Common Errors by Matcher

### EmbeddingMatcher

| Error | Cause | Fix |
|---|---|---|
| `Index not built` | Didn't call `build_index()` | Call `matcher.build_index()` |
| Returns `None` | Threshold too high | Lower `threshold` |
| Slow matching | Large model or dataset | Use `minilm` model |

### EntityMatcher

| Error | Cause | Fix |
|---|---|---|
| `Model not trained` | Didn't call `train()` | Call `matcher.train(data)` |
| Returns `None` | Threshold too high or insufficient training | Lower threshold or add training examples |
| Training slow | Large model or many epochs | Use `minilm` and fewer epochs |

### HybridMatcher

| Error | Cause | Fix |
|---|---|---|
| Empty results | Blocking stage filters everything | Try `NoOpBlocking()` or increase `blocking_top_k` |
| Slow | Small dataset | Use `EmbeddingMatcher` instead |

## Getting More Help

If you're still stuck:

1. Check the examples: [`docs/examples.md`](./examples.md)
2. Review the quickstart: [`docs/quickstart.md`](./quickstart.md)
3. Search issues: [GitHub Issues](https://github.com/your-repo/semantic-matcher/issues)
4. Create an issue with your code, data sample, and error message
