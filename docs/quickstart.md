# Quick Start Guide

Related docs: [`index.md`](./index.md) | [`notebooks.md`](./notebooks.md) | [`examples.md`](./examples.md) | [`architecture.md`](./architecture.md)

## Installation

```bash
pip install semantic-matcher
```

Use this page for the official package wrapper API. If you want exploratory scripts or Jupyter notebooks, see [`notebooks.md`](./notebooks.md). If you want lower-level raw `setfit` examples, see [`examples.md`](./examples.md).

## Choose a Matcher

| Matcher | Best For | Training Required | Speed | Accuracy |
|---|---|---|---|---|---|
| `EmbeddingMatcher` | Prototyping, quick results, simple matching | No | Fast (~50 q/s) | Good on exact matches |
| `EntityMatcher` | Production use, complex variations | Yes (3-5 examples/entity) | Medium (~30 q/s) | High on variations |
| `HybridMatcher` | Large datasets (10k+ entities) | No | Medium (3-stage) | High + precise |

**Decision Guide**:
- **No training data?** → Use `EmbeddingMatcher`
- **Have labeled examples?** → Use `EntityMatcher`
- **Very large dataset?** → Use `HybridMatcher`

---

## Path 1: Embedding Similarity (No Training)

Use cosine similarity without training for quick prototypes. Best when:
- You need results immediately
- Text variations are minimal
- Accuracy requirements are moderate

```python
from semanticmatcher import EmbeddingMatcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
    {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

matcher = EmbeddingMatcher(entities=entities, threshold=0.7)
matcher.build_index()

print(matcher.match("Deutschland"))  # {"id": "DE", "score": ...}
print(matcher.match("UnknownPlace"))  # None (below threshold)
```

**Parameters**:
- `entities` (required): List of entity dicts with `id`, `name`, optional `aliases`
- `model_name` (default: `paraphrase-mpnet-base-v2`): Sentence transformer model
- `threshold` (default: `0.7`): Minimum similarity score (0.0-1.0). Lower = more matches, higher = fewer matches
- `normalize` (default: `True`): Apply text normalization (lowercase, remove accents/punctuation)
- `embedding_dim` (optional): Truncate embeddings to this dimension (Matryoshka embeddings)
- `cache` (optional): Model cache instance (defaults to global cache)

**Methods**:
- `build_index(batch_size=None)`: Build embedding index from entities. Call once before matching.
- `match(texts, top_k=1, batch_size=None)`: Match query/queries. Returns best match or list of matches.
- `match_bulk(queries, n_jobs=-1)`: Batch processing with parallel workers (HybridMatcher only)

See [`examples/embedding_matcher_demo.py`](../examples/embedding_matcher_demo.py) for a complete working example.

---

## Path 2: Few-Shot Training with `EntityMatcher`

Train a SetFit-backed matcher when you have labeled examples. Best when:
- You have 3-5 labeled examples per entity
- Text has significant variations (typos, translations, abbreviations)
- You need higher accuracy on complex cases
- You can afford 1-3 minutes of training time

```python
from semanticmatcher import EntityMatcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
    {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

training_data = [
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "Deutchland", "label": "DE"},
    {"text": "France", "label": "FR"},
    {"text": "Frankreich", "label": "FR"},
    {"text": "USA", "label": "US"},
    {"text": "America", "label": "US"},
]

matcher = EntityMatcher(entities=entities)
matcher.train(training_data, num_epochs=4)

print(matcher.predict("Deutchland"))  # "DE"
print(matcher.predict(["Deutchland", "America", "France"]))  # ["DE", "US", "FR"]
```

**Parameters**:
- `entities` (required): List of entity dicts with `id`, `name`, optional `aliases`
- `model_name` (default: `paraphrase-mpnet-base-v2`): Sentence transformer model
- `threshold` (default: `0.7`): Minimum confidence for predictions (0.0-1.0)
- `normalize` (default: `True`): Apply text normalization

**Methods**:
- `train(training_data, num_epochs=4, batch_size=16)`: Train the model
- `predict(texts)`: Predict entity for query/queries. Returns `None` if below threshold
- Access `classifier.predict_proba(text)` to see confidence scores per entity

See [`examples/entity_matcher_demo.py`](../examples/entity_matcher_demo.py) for a complete working example.

---

## Model Aliases

Common models have short aliases for convenience:

| Alias | Full Model Name | Language | Speed | Use Case |
|---|---|---|---|---|
| `mpnet` | `sentence-transformers/paraphrase-mpnet-base-v2` | English | Medium | Default choice, balanced |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | English | Fast | Prototyping, speed-critical |
| `bge-base` | `BAAI/bge-base-en-v1.5` | English | Medium | High accuracy English |
| `bge-m3` | `BAAI/bge-m3` | Multilingual | Slow | Multilingual, high accuracy |

**Usage**:
```python
# Using alias
matcher = EmbeddingMatcher(entities, model_name="minilm")

# Using full model name
matcher = EntityMatcher(entities, model_name="sentence-transformers/LaBSE")
```

---

## Text Normalization

Both matchers support normalization by default. Use `TextNormalizer` directly for preprocessing.

```python
from semanticmatcher import TextNormalizer

normalizer = TextNormalizer(
    lowercase=True,         # Convert to lowercase
    remove_accents=True,    # Remove accents (é → e)
    remove_punctuation=True,  # Remove punctuation
)

print(normalizer.normalize("HELLO, World!"))  # "hello world"
```

**When to enable/disable normalization**:
- Enable (`normalize=True`, default): For user input, messy data, multilingual text
- Disable (`normalize=False`): For clean, preprocessed data or case-sensitive matching

---

## Path 3: Hybrid Matching Pipeline (For Large Datasets)

For maximum accuracy with large datasets (>10,000 entities), use the three-stage pipeline:

```python
from semanticmatcher import HybridMatcher, BM25Blocking

matcher = HybridMatcher(
    entities=products,
    blocking_strategy=BM25Blocking(),  # Fast lexical filtering
    retriever_model="bge-base",        # Semantic search
    reranker_model="bge-m3",            # Precise reranking
)

results = matcher.match(
    "iPhone 15 case",
    blocking_top_k=1000,    # Candidates after blocking
    retrieval_top_k=50,     # Candidates after retrieval
    final_top_k=5           # Final results
)
```

**Pipeline Stages**:
1. **Blocking** (BM25/TF-IDF/Fuzzy): Fast lexical filtering to ~1000 candidates
2. **Retrieval** (Bi-Encoder): Semantic similarity to ~50 candidates
3. **Reranking** (Cross-Encoder): Precise cross-attention scoring to ~5 final results

See [`examples/hybrid_matching_demo.py`](../examples/hybrid_matching_demo.py) for a complete working example.

---

## Blocking Strategies

When using `HybridMatcher`, choose a blocking strategy:

| Strategy | Best For | Speed | Notes |
|---|---|---|---|---|
| `BM25Blocking` | Keyword-heavy queries, proper nouns | Fast | Default choice |
| `TFIDFBlocking` | Document-level similarity | Fast | Good for longer texts |
| `FuzzyBlocking` | Typos and variations | Slow | Uses RapidFuzz |
| `NoOpBlocking` | Small datasets (<1000 entities) | N/A | Pass-through, no filtering |

**Example**:
```python
from semanticmatcher import HybridMatcher, FuzzyBlocking

matcher = HybridMatcher(
    entities=products,
    blocking_strategy=FuzzyBlocking(score_cutoff=70),
    # ...
)
```

---

## Path 4: Cross-Encoder Reranking (For Higher Precision)

Rerank top candidates for higher precision:

```python
from semanticmatcher import EmbeddingMatcher, CrossEncoderReranker

# Initial retrieval
retriever = EmbeddingMatcher(entities)
retriever.build_index()
candidates = retriever.match(query, top_k=50)

# Rerank with cross-encoder
reranker = CrossEncoderReranker(model="bge-m3")
final_results = reranker.rerank(query, candidates, top_k=5)
```

---

## Model Persistence

Save and load trained models for production:

```python
from semanticmatcher import SetFitClassifier, EntityMatcher

# Train and save
classifier = SetFitClassifier(labels=["DE", "FR", "US"])
classifier.train(training_data)
classifier.save("/path/to/model")

# Load later
loaded = SetFitClassifier.load("/path/to/model")

# Use with EntityMatcher
matcher = EntityMatcher(entities=entities)
matcher.classifier = loaded
matcher.is_trained = True
```

See [`examples/model_persistence.py`](../examples/model_persistence.py) for a complete guide.

---

## Common First-Run Issues

**EmbeddingMatcher**:
- Call `build_index()` before `match()`
- First run downloads model (~100-500MB, requires network)

**EntityMatcher**:
- Call `train()` before `predict()`
- First run downloads model and trains (requires network + 1-3 minutes)

**General**:
- Low matches? Try lowering `threshold` (0.7 → 0.6)
- Too many matches? Try raising `threshold` (0.7 → 0.8)
- `predict()` returns `None`? Query confidence below threshold (see `predict_proba()`)

See [`troubleshooting.md`](./troubleshooting.md) for more fixes.

---

## Performance Expectations

Approximate performance on M1 MacBook Pro (8GB RAM):

| Matcher | Setup Time | Query Speed | Memory |
|---|---|---|---|---|
| EmbeddingMatcher (minilm) | 3s | 50 q/s | 200MB |
| EmbeddingMatcher (mpnet) | 4s | 40 q/s | 400MB |
| EntityMatcher (after training) | 0s (load) | 30 q/s | 400MB |
| HybridMatcher (3-stage) | 5s | 3 q/s | 600MB |

**Tips for performance**:
- Use `minilm` for prototyping, `mpnet` for production
- Use `match_bulk()` for batch processing
- Lower `threshold` reduces precision but increases recall
- Enable `normalize=True` for better matching on messy input

---

## Where to Go Next

- **Working examples**: [`examples.md`](./examples.md) - Complete catalog with difficulty ratings
- **Experiments**: [`notebooks.md`](./notebooks.md) - Interactive exploration
- **Internals**: [`architecture.md`](./architecture.md) - Module layout and design
- **Troubleshooting**: [`troubleshooting.md`](./troubleshooting.md) - Common issues and fixes
