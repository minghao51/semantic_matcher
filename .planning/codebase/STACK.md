# Tech Stack

## Language & Runtime

- **Python**: 3.9, 3.10, 3.11, 3.12 (minimum 3.9)
- **Package Manager**: `uv` (preferred for development), `pip` for production
- **Build System**: Hatchling (via `pyproject.toml`)

## Core Frameworks

### Machine Learning
- **sentence-transformers** (≥3.0.0): Primary embedding framework
- **SetFit** (≥1.0.0): Few-shot learning framework
- **PyTorch** (≥2.0.0): ML backend for sentence transformers

### NLP & Text Processing
- **NLTK** (≥3.9.2): Text normalization and preprocessing
- **RapidFuzz** (≥3.0.0): Fuzzy string matching for blocking

### Data & Math
- **NumPy** (≥2.0.0): Numerical operations
- **pandas** (≥2.0.0): Data manipulation
- **scikit-learn** (≥1.3.0): Cosine similarity, TF-IDF, metrics

### Search & Retrieval
- **rank-bm25** (≥0.2.2): BM25 blocking algorithm

## Key Dependencies by Layer

**Blocking Layer:**
- `rank-bm25` - BM25 lexical filtering
- `scikit-learn` - TF-IDF vectorization
- `rapidfuzz` - Fuzzy string matching

**Retrieval Layer:**
- `sentence-transformers` - Bi-encoder embeddings
- `setfit` - Few-shot classification
- `torch` - Model backend

**Reranking Layer:**
- `sentence-transformers` - Cross-encoder models

**Utilities:**
- `PyYAML` - Configuration management
- `requests` - HTTP for data ingestion
- `tqdm` - Progress bars

## Development Tools

**Testing:**
- `pytest` (≥8.4.2) - Test framework with markers
- `patchright` (≥1.58.0) - Browser automation (optional)

**Code Quality:**
- `ruff` (≥0.1.0) - Fast linter
- `black` (≥23.0.0) - Code formatter

**Packaging:**
- `build` (≥1.2.2) - Package building
- `twine` (≥5.1.1) - PyPI uploads

**Optional Dev Tools:**
- `beautifulsoup4` (≥4.14.3) - HTML parsing for ingestion
- `html-to-markdown` (≥1.8.0) - Content conversion

## Entry Points

**CLI:**
- `semanticmatcher-ingest` → `semanticmatcher.ingestion.cli:main`

**Python Package:**
- Import path: `semanticmatcher`
- Distribution name: `semantic-matcher`

## Model Registry

Built-in model aliases (configured in `src/semanticmatcher/config.py`):

| Alias | Full Model Name |
|-------|----------------|
| `bge-base` | BAAI/bge-base-en-v1.5 |
| `bge-m3` | BAAI/bge-m3 |
| `nomic` | nomic-ai/nomic-embed-text-v1 |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 |
| `default` | sentence-transformers/all-mpnet-base-v2 |

## Configuration

- **Config File**: `pyproject.toml` (Hatchling build, pytest markers, dev dependencies)
- **Linting**: Ruff (configured in pyproject.toml)
- **Formatting**: Black (used via ruff or standalone)
- **Testing**: pytest with 3 markers (integration, slow, hf)
