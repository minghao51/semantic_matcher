# Experiments & Notebook Index

Related docs: [`index.md`](./index.md) | [`country-classifier-scripts.md`](./country-classifier-scripts.md) | [`examples.md`](./examples.md) | [`troubleshooting.md`](./troubleshooting.md)

This page documents exploratory assets and the repo convention split:

- `experiments/` for script-style experiments (`.py`)
- `notebooks/` for Jupyter notebooks (`.ipynb`) only

## Current Status

- Tracked script experiments: `experiments/country_classifier/`
- Tracked Jupyter notebooks: none currently (the folder is reserved for future `.ipynb` work)

## How to Run

### Python script experiments (`experiments/`)

Run from the repo root:

```bash
PYTHONPATH=. uv run python experiments/<path_to_script>.py
```

### Jupyter notebooks (`notebooks/`)

If/when notebook files are added:

```bash
uv run python -m pip install jupyter
uv run jupyter notebook
```

Open files from `notebooks/` while keeping the repo root as the working directory.

## Script Experiment Inventory

| File | Purpose | Runtime | Notes |
|---|---|---|---|
| `experiments/country_classifier/country_classifier.py` | Baseline country-code classifier comparison (A/B/C scenarios) | Minutes (CPU/GPU dependent) | Best first experiment |
| `experiments/country_classifier/country_classifier_quick.py` | Fast optimization checks beyond baseline | Several minutes | Quick iteration path |
| `experiments/country_classifier/country_classifier_advanced.py` | Broader optimization search across models/heads | Longer (~10-15+ min) | Advanced tuning |

## Country Classifier Scripts

For a deeper explanation of the three country classifier experiment scripts (goals, scenarios, and usage), see [`country-classifier-scripts.md`](./country-classifier-scripts.md).
