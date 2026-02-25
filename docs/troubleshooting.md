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

## `EmbeddingMatcher` Threshold Confusion

`EmbeddingMatcher` uses the matcher's configured threshold (`threshold` passed to `__init__`) to decide whether to return a match or `None`.

If you are getting too many `None` results:

- lower the threshold (for example `0.7` -> `0.5`)

If you are getting low-confidence matches:

- raise the threshold

Tune thresholds using a validation sample from your domain.
