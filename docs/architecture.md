# Architecture

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md)

## Overview

SemanticMatcher is a text-to-entity matching library built on SetFit few-shot learning with Sentence Transformers.
Use this page for internals and module boundaries rather than first-run usage.

## Module Structure

```
semanticmatcher/
├── core/                    # Core matching functionality
│   ├── matcher.py          # EntityMatcher & EmbeddingMatcher
│   ├── classifier.py       # SetFitClassifier wrapper
│   └── normalizer.py       # Text normalization
├── utils/                   # Utility functions
│   ├── embeddings.py       # Embedding computation
│   ├── preprocessing.py    # Text preprocessing
│   └── validation.py       # Input validation
└── backends/               # Embedding backends
    ├── base.py            # Abstract base classes
    ├── sentencetransformer.py  # HuggingFace backend
    ├── sentencetranformer.py   # Backward-compat shim (legacy typo)
    └── ...
```

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

- Cloud LLM embedding support

### Ollama (future)

- Local LLM embeddings

## Design Decisions

1. **Optional normalization** - Users can disable if input is already clean
2. **Lazy model loading** - SentenceTransformer loaded on first use
3. **Flexible input** - Single string or list of strings for batch prediction
4. **Threshold-based matching** - Configurable confidence threshold for EmbeddingMatcher
