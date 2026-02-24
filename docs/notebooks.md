# Notebook & Experiment Index (`notebooks/`)

Related docs: [`index.md`](./index.md) | [`country-classifier-scripts.md`](./country-classifier-scripts.md) | [`examples.md`](./examples.md) | [`troubleshooting.md`](./troubleshooting.md)

This page documents every experiment asset in the merged `notebooks/` folder, including Python scripts and Jupyter notebooks.

Migration note: previous path `notebook/...` was merged into `notebooks/...`.

## How to Run Experiments

### Python scripts (`.py`)

Run from the repo root:

```bash
PYTHONPATH=. uv run python notebooks/<script_name>.py
```

### Jupyter notebooks (`.ipynb`)

Install Jupyter if needed, then run from the repo root:

```bash
uv run python -m pip install jupyter
uv run jupyter notebook
```

Open files from `notebooks/`. Using the repo root as the working directory avoids path issues.

## Inventory

| File | Type | Category | Purpose | Primary dependency stack | Runtime expectations | Output / result | Recommended user level |
|---|---|---|---|---|---|---|---|
| `notebooks/country_classifier.py` | `.py` script | Core project experiment | Baseline country-code classifier comparison (A/B/C scenarios) | `semanticmatcher`, `setfit`, `sentence-transformers`, `pandas`, `sklearn` | Minutes on CPU/GPU depending on training | Accuracy + reports for scenarios | Intermediate |
| `notebooks/country_classifier_quick.py` | `.py` script | Core project experiment | Fast optimization checks beyond baseline | `setfit`, `sentence-transformers`, `pandas`, `sklearn` | Several minutes | Accuracy comparisons for quick variants | Intermediate |
| `notebooks/country_classifier_advanced.py` | `.py` script | Core project experiment | Broader optimization search (epochs/models/heads) | `setfit`, `sentence-transformers`, `pandas`, `sklearn` | Longer run (~10-15+ min depending on hardware) | Accuracy/report outputs across many configs | Advanced |
| `notebooks/dev101.ipynb` | `.ipynb` | Core/adjacent config exploration | Inspect config loading behavior via `semanticmatcher.config.Config` | `semanticmatcher`, `pyyaml` | Seconds | Config values / config exploration | Beginner |
| `notebooks/qwen_embedding.ipynb` | `.ipynb` | Adjacent raw embedding exploration | Explore embedding and reranking using raw `sentence-transformers` + `CrossEncoder` | `sentence-transformers`, `torch` | Model download + interactive runtime | Similarity tensors and ranking scores | Intermediate |
| `notebooks/geograpy.ipynb` | `.ipynb` | External exploration | Try `geograpy` place extraction on URL/text inputs | `geograpy` and its dependencies | Seconds to minutes; may fail on setup/network | Extracted place names | Experimental / Advanced |

## Method Documentation Template (Used Below)

Each entry is documented with:

1. What this notebook/script does
2. When to use it
3. How it relates to `semanticmatcher`
4. Dependencies
5. How to run it
6. Method walkthrough
7. Expected outputs
8. Common failure modes
9. Next steps

## `notebooks/country_classifier.py`

### 1. What this script does

Runs a baseline comparison of three country-code classification approaches:
- zero-shot similarity
- head-only SetFit training
- full SetFit training

### 2. When to use it

Use this first if you want to understand the country-matching experiment and benchmark shape before tuning.

### 3. How it relates to `semanticmatcher`

This is a core project experiment. It uses `semanticmatcher.core.classifier.SetFitClassifier` and project training data.

### 4. Dependencies

- Core project dependencies in `pyproject.toml`
- Model downloads from Hugging Face on first run

### 5. How to run it

```bash
PYTHONPATH=. uv run python notebooks/country_classifier.py
```

### 6. Method walkthrough

1. Loads `data/country_training_data.csv`
2. Splits data into train/test (stratified by label)
3. Evaluates zero-shot embedding similarity baseline
4. Trains SetFit classifier variants
5. Prints accuracy and classification outputs

### 7. Expected outputs

- Train/test counts
- Scenario-level metrics
- Accuracy comparisons
- Possibly classification report/confusion matrix text

### 8. Common failure modes

- Missing `setfit` / `datasets` / `torch`
- Slow first run due to model downloads
- CPU runtime is much slower than GPU

### 9. Next steps

- Read [`country-classifier-scripts.md`](./country-classifier-scripts.md) for scenario details
- Try `notebooks/country_classifier_quick.py` next

## `notebooks/country_classifier_quick.py`

### 1. What this script does

Runs a smaller set of optimization experiments (epochs/model/head variants) intended for faster iteration.

### 2. When to use it

Use this after the baseline script when you want quick comparisons without the full advanced sweep.

### 3. How it relates to `semanticmatcher`

Core project experiment for country matching benchmarking. It is closer to raw experimentation than the beginner wrapper API.

### 4. Dependencies

- `setfit`, `datasets`, `sentence-transformers`, `pandas`, `scikit-learn`

### 5. How to run it

```bash
PYTHONPATH=. uv run python notebooks/country_classifier_quick.py
```

### 6. Method walkthrough

1. Loads and splits the country CSV
2. Trains a few targeted variants (epochs/model/head)
3. Evaluates each variant on the test split
4. Prints comparative accuracy results

### 7. Expected outputs

- Baseline and variant accuracy values
- Runtime per experiment (if printed)
- Best/relative configuration notes

### 8. Common failure modes

- Missing model dependencies
- `LinearSVC`/SetFit compatibility issues if package versions drift
- Long runtime on CPU despite the name "quick"

### 9. Next steps

- Try `notebooks/country_classifier_advanced.py` for broader tuning
- Use `EntityMatcher` in [`quickstart.md`](./quickstart.md) for production-oriented wrapper usage

## `notebooks/country_classifier_advanced.py`

### 1. What this script does

Explores a larger optimization space for country-code classification (more epochs, embedding models, classifier heads, and ensemble-like comparisons).

### 2. When to use it

Use this when you are actively tuning accuracy and can tolerate longer iteration times.

### 3. How it relates to `semanticmatcher`

Core project experiment with advanced benchmarking. It is not the recommended first API example for beginners.

### 4. Dependencies

- `setfit`, `datasets`, `sentence-transformers`, `pandas`, `scikit-learn`, `numpy`

### 5. How to run it

```bash
PYTHONPATH=. uv run python notebooks/country_classifier_advanced.py
```

### 6. Method walkthrough

1. Loads and splits country training data
2. Trains multiple SetFit configurations
3. Tests alternative classifier heads / embeddings
4. Compares and reports results

### 7. Expected outputs

- Accuracy metrics for many scenarios
- Classification reports for selected runs
- Comparative performance summary

### 8. Common failure modes

- Long runtime / memory pressure
- Missing dependencies
- Version mismatch in `setfit`/`TrainingArguments` behavior

### 9. Next steps

- Consolidate findings into a production config
- Re-run smaller subsets in `country_classifier_quick.py` for iteration

## `notebooks/dev101.ipynb`

### 1. What this notebook does

Demonstrates basic project config loading with `semanticmatcher.config.Config` and inspects config values.

### 2. When to use it

Use this when learning how configuration is loaded (repo `config.yaml`, packaged defaults, cwd overrides).

### 3. How it relates to `semanticmatcher`

Core/adjacent project exploration. It touches package config utilities directly.

### 4. Dependencies

- `semanticmatcher`
- `pyyaml` (already included in project dependency tree if installed)

### 5. How to run it

Launch Jupyter from repo root and open `notebooks/dev101.ipynb`.

### 6. Method walkthrough

1. Imports `Config`
2. Instantiates `Config()` (default resolution behavior)
3. Reads a config key (`embedding`)

### 7. Expected outputs

- Printed config section or dictionary for the requested key

### 8. Common failure modes

- Import failure if package is not installed in the active kernel environment
- Running from the wrong directory can change config discovery behavior

### 9. Next steps

- Read `semanticmatcher/config.py` for resolution order details
- Use `Config(custom_path=...)` in your own scripts if needed

## `notebooks/qwen_embedding.ipynb`

### 1. What this notebook does

Explores embedding generation and ranking using a Qwen embedding model and a cross-encoder reranker via raw `sentence-transformers` APIs.

### 2. When to use it

Use this for embedding/reranking experimentation outside the `semanticmatcher` wrapper API.

### 3. How it relates to `semanticmatcher`

Adjacent exploration. It does not directly demonstrate `EmbeddingMatcher`; it demonstrates lower-level embedding and reranking building blocks.

### 4. Dependencies

- `sentence-transformers`
- `torch`
- Model downloads from Hugging Face
- Optional GPU acceleration for practical speed

### 5. How to run it

Launch Jupyter from repo root and open `notebooks/qwen_embedding.ipynb`.

### 6. Method walkthrough

1. Loads a Qwen embedding model
2. Encodes queries/documents
3. Computes similarity
4. Demonstrates ranking/search utilities
5. Loads a cross-encoder for passage ranking
6. Ranks passages for a sample query

### 7. Expected outputs

- Similarity tensor/matrix output
- Ranked passages with scores

### 8. Common failure modes

- Large model download time / network issues
- GPU memory limits
- Incompatible `torch` / `transformers` stack versions

### 9. Next steps

- Recreate a smaller version using `EmbeddingMatcher` for entity matching
- Compare raw similarity workflows with project wrapper behavior

## `notebooks/geograpy.ipynb`

### 1. What this notebook does

Tests `geograpy` entity extraction on a URL and a plain text input to extract place names.

### 2. When to use it

Use this only if you want to explore geographic entity extraction with a third-party library.

### 3. How it relates to `semanticmatcher`

External exploration. This notebook is not a core `semanticmatcher` API example.

### 4. Dependencies

- `geograpy`
- Its NLP dependencies (which may require additional setup)
- Network access for URL-based example

### 5. How to run it

Launch Jupyter from repo root and open `notebooks/geograpy.ipynb`.

### 6. Method walkthrough

1. Imports `geograpy.extraction`
2. Extracts geo entities from a URL
3. Prints extracted places
4. Repeats extraction for a text input

### 7. Expected outputs

- Lists of detected place names (`e.places`)

### 8. Common failure modes

- `ModuleNotFoundError: geograpy`
- Dependency install friction for NLP packages
- Network failures for URL extraction

### 9. Next steps

- Use extracted places as upstream inputs to `semanticmatcher` workflows
- Keep this notebook labeled as external/experimental in project docs
