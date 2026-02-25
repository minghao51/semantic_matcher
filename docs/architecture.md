# Architecture

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md)

## Overview

SemanticMatcher is a text-to-entity matching library built on SetFit few-shot learning with Sentence Transformers.
Use this page for internals and module boundaries rather than first-run usage.

## Module Structure

```text
src/semanticmatcher/
├── __init__.py              # Public exports / lazy import surface
├── config.py                # Config loading and defaults
├── core/                    # Matching pipelines and domain logic
│   ├── matcher.py           # EntityMatcher / EmbeddingMatcher
│   ├── classifier.py        # SetFitClassifier wrapper
│   ├── normalizer.py        # Text normalization
│   ├── blocking.py          # Candidate blocking strategies
│   ├── reranker.py          # Cross-encoder reranking
│   ├── hybrid.py            # Multi-stage matching pipeline
│   └── monitoring.py        # Metrics/monitoring helpers
├── backends/                # Provider integrations (embeddings/reranking)
│   ├── base.py              # Backend interfaces / shared abstractions
│   ├── sentencetransformer.py
│   ├── reranker_st.py
│   ├── litellm.py           # Planned/in-progress cloud backend support
│   └── ...
├── ingestion/               # Dataset ingestion and normalization CLI/pipelines
│   ├── cli.py               # `semanticmatcher-ingest` entrypoint target
│   └── *.py                 # Source-specific ingestors (countries/products/etc.)
├── utils/                   # Cross-cutting helpers (non-domain specific)
└── data/                    # Packaged static data files / defaults
```

## Package Boundaries

- `core/`: orchestration and domain logic for matching, retrieval/reranking pipelines, and normalization.
- `backends/`: provider-specific integrations for embeddings and rerankers (Hugging Face, LiteLLM, etc.).
- `ingestion/`: data acquisition/transformation utilities and the ingestion CLI.
- `utils/`: shared helpers used across modules that are not themselves product/domain entrypoints.
- `data/`: packaged JSON/static assets required at runtime.

## Module Placement Rules

- Add a new matcher or pipeline stage to `core/` unless it is provider-specific.
- Add a new model/provider integration to `backends/`.
- Add dataset import/transformation logic or CLI wiring to `ingestion/`.
- Put generic helpers in `utils/`; avoid moving domain logic there just to “reuse” it.
- Keep the public import surface curated through `src/semanticmatcher/__init__.py` (avoid exposing internal modules unintentionally).

## Core Components

### EntityMatcher

SetFit-based entity matching with optional text normalization.

**Workflow:**
1. Initialize with entities list (id, name, aliases)
2. Train with labeled examples
3. Predict entity ID for new inputs

```python
matcher = EntityMatcher(entities=[
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]}
])
matcher.train(training_data)
result = matcher.predict("Deutschland")  # → "DE"
```

### EmbeddingMatcher

Similarity-based matching without training. Uses cosine similarity between embeddings.

**Workflow:**
1. Initialize with entities
2. Build index (encodes all names/aliases)
3. Match queries against index

```python
matcher = EmbeddingMatcher(entities=[...])
matcher.build_index()
result = matcher.match("Deutschland")  # → {"id": "DE", "score": 0.92}
```

### SetFitClassifier

Low-level wrapper around SetFit for training and prediction.

### TextNormalizer

Text preprocessing with options for:
- Lowercase conversion
- Accent removal
- Punctuation removal

## Data Flow

```
Input Text
    ↓
TextNormalizer (optional)
    ↓
Embedding Model (SentenceTransformer)
    ↓
Similarity Computation / Classification
    ↓
Result (entity ID or score)
```

## Backends

### HuggingFace (SentenceTransformers)

- `HFEmbedding` - Generate embeddings
- `HFReranker` - Cross-encoder reranking

### LiteLLM (future)

- Cloud LLM embedding support (planned/in progress; confirm implementation status before relying on it)

### Ollama (future)

- Local LLM embeddings (planned; may not be fully wired in current release)

## Design Decisions

1. **Optional normalization** - Users can disable if input is already clean
2. **Lazy model loading** - SentenceTransformer loaded on first use
3. **Flexible input** - Single string or list of strings for batch prediction
4. **Threshold-based matching** - Configurable confidence threshold for EmbeddingMatcher
