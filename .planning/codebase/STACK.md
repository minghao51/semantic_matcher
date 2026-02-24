# SemanticMatcher Tech Stack

## Languages & Runtime
- **Primary Language**: Python 3.13
- **Minimum Version**: Python 3.9+
- **Version File**: `.python-version`

## Core Package
- **Name**: semanticmatcher
- **Version**: 0.1.0
- **Package Manager**: uv (with pip fallback)
- **Lock File**: `uv.lock`

## AI/ML Dependencies

### Core ML Framework
- **SetFit** (≥1.0.0): Few-shot learning for entity classification
- **Sentence Transformers** (≥3.0.0): Pre-trained sentence embeddings
- **PyTorch** (≥2.0.0): Deep learning backend
- **Hugging Face Hub**: Model repository integration

### Supported Models
- `sentence-transformers/paraphrase-mpnet-base-v2` (default)
- `BAAI/bge-m3` (multilingual)
- Any sentence-transformers compatible model
- LiteLLM-supported models (optional)

### Data Processing
- **NumPy** (≥2.0.0): Numerical operations
- **Pandas** (≥2.0.0): Data manipulation
- **scikit-learn** (≥1.3.0): Cosine similarity computations
- **NLTK** (≥3.9.2): Text normalization (optional)
- **datasets** (≥2.14.0): Hugging Face datasets

## Configuration
- **PyYAML**: Configuration file parsing
- **Config File**: `config.yaml`

## Development Tools
- **pytest** (≥8.4.2): Testing framework
- **black** (≥23.0.0): Code formatting
- **ruff** (≥0.1.0): Linting

## Installation Methods
```bash
# Using uv (recommended)
uv pip install semantic-matcher

# Using pip
pip install semantic-matcher

# From source
uv sync
```

## Version Pinning
All major dependencies pinned in `pyproject.toml` with minimum versions.
