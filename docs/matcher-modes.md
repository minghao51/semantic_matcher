# Matcher Modes

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`migration-guide.md`](./migration-guide.md) | [`architecture.md`](./architecture.md)

## Overview

The unified `Matcher` class supports multiple matching strategies through **modes**. Modes automatically route to the optimal implementation (EmbeddingMatcher, EntityMatcher, or HybridMatcher).

## Available Modes

| Mode | Description | Training Time | Use Case |
|---|---|---|---|
| `zero-shot` | Embedding similarity only | None | No training data available |
| `head-only` | Train classifier head only | ~30s | Minimal training data (1-2 examples/entity) |
| `full` | Full SetFit training | ~3min | Sufficient training data (3+ examples/entity) |
| `hybrid` | Multi-stage pipeline | None | Large datasets (10k+ entities) |
| `auto` | Smart auto-detection | Variable | Let the library choose |

## Mode Comparison

### zero-shot

**What:** Pure embedding similarity using cosine similarity.

**When to use:**
- No labeled training data available
- Need immediate results
- Prototyping or exploration

**Pros:**
- No training required
- Instant setup
- Works out of the box

**Cons:**
- Lower accuracy than trained modes
- Can't learn from your data

**Example:**
```python
from semanticmatcher import Matcher

matcher = Matcher(entities=entities, mode="zero-shot")
matcher.fit()
result = matcher.match("query")
```

**Implementation:** Routes to `EmbeddingMatcher`

---

### head-only

**What:** Lightweight SetFit training that only trains the classification head.

**When to use:**
- Limited training data (1-2 examples per entity)
- Need fast training
- Quick iteration on model

**Pros:**
- Fast training (~30 seconds)
- Better than zero-shot with minimal data
- Good for quick experiments

**Cons:**
- Lower accuracy than full training
- May not capture complex patterns

**Example:**
```python
matcher = Matcher(entities=entities, mode="head-only")
matcher.fit(training_data, num_epochs=1)
result = matcher.match("query")
```

**Implementation:** Routes to `EntityMatcher` with head-only training

---

### full

**What:** Full SetFit training with contrastive learning.

**When to use:**
- Sufficient training data (3+ examples per entity)
- Need best accuracy
- Production deployment

**Pros:**
- Best accuracy
- Learns from your data
- Robust to variations

**Cons:**
- Slower training (~3 minutes)
- Requires more training data

**Example:**
```python
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data, num_epochs=4)
result = matcher.match("query")
```

**Implementation:** Routes to `EntityMatcher` with full training

---

### hybrid

**What:** Three-stage pipeline: blocking → retrieval → reranking.

**Stages:**
1. **Blocking:** Fast candidate filtering (BM25/TF-IDF)
2. **Retrieval:** Embedding similarity on filtered candidates
3. **Reranking:** Cross-encoder scoring for precision

**When to use:**
- Large datasets (10k+ entities)
- Need both speed and accuracy
- Can tolerate some complexity

**Pros:**
- Scales to very large datasets
- High accuracy with reranking
- Efficient candidate pruning

**Cons:**
- More complex setup
- Multiple models to load
- Higher memory usage

**Example:**
```python
matcher = Matcher(entities=entities, mode="hybrid")
matcher.fit()
result = matcher.match("query")
```

**Implementation:** Routes to `HybridMatcher`

**Pipeline Parameters:**
```python
result = matcher.match(
    "query",
    blocking_top_k=1000,     # Candidates after blocking
    retrieval_top_k=50,      # Candidates after retrieval
    final_top_k=5,           # Final results after reranking
)
```

---

### auto

**What:** Smart mode selection based on training data.

**Decision Logic:**
```
No training data → zero-shot
< 3 examples/entity → head-only
≥ 3 examples/entity → full
```

**When to use:**
- Unsure which mode to pick
- Want the library to choose optimally
- Starting a new project

**Example:**
```python
matcher = Matcher(entities=entities, mode="auto")
matcher.fit(training_data)  # Auto-selects based on data
```

**How it works:**
1. Analyzes training data volume per entity
2. Selects appropriate mode automatically
3. Stores detected mode for transparency

**Check detected mode:**
```python
info = matcher.get_training_info()
print(info["detected_mode"])  # "zero-shot", "head-only", or "full"
```

## Mode Selection Decision Tree

```
Do you have training data?
│
├─ No → zero-shot
│
└─ Yes → How many examples per entity?
          │
          ├─ < 3 → head-only (fast, ~30s)
          │
          └─ ≥ 3 → full (accurate, ~3min)
```

**Special case:** Large datasets (10k+ entities)
```
Dataset size > 10k entities?
│
└─ Yes → Consider hybrid mode
```

## Explicit Mode Selection

Override auto-detection when you know what you want:

```python
# Force zero-shot even with training data
matcher = Matcher(entities=entities, mode="zero-shot")
matcher.fit(training_data)  # Training data ignored
```

```python
# Force full training even with minimal data
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)  # Will train but may overfit
```

## Performance & Accuracy Tradeoffs

### Speed Comparison

Training time (100 entities, 50 examples):

| Mode | Training Time | Query Speed |
|---|---|---|
| zero-shot | None | Fast (static: ~1ms, dynamic: ~10ms) |
| head-only | ~30s | Fast (~10ms) |
| full | ~3min | Fast (~10ms) |
| hybrid | None | Medium (~50-100ms with reranking) |

### Accuracy Comparison

Accuracy on typical dataset (higher is better):

| Mode | Accuracy | Notes |
|---|---|---|
| zero-shot | 70-80% | Good baseline |
| head-only | 80-85% | Better with minimal data |
| full | 85-95% | Best with sufficient data |
| hybrid | 90-95% | Best for large datasets |

*Actual results vary by dataset quality and size.*

## Hybrid Mode Deep Dive

### Pipeline Stages

```python
# Stage 1: Blocking (fast candidate filtering)
# BM25, TF-IDF, or fuzzy matching
# Reduces 10k entities → 1000 candidates

# Stage 2: Retrieval (embedding similarity)
# Static or dynamic embeddings
# Reduces 1000 candidates → 50 candidates

# Stage 3: Reranking (cross-encoder scoring)
# Precise but slow
# Reduces 50 candidates → 5 final results
```

### Blocking Strategies

```python
from semanticmatcher.core.blocking import BM25Blocking

matcher = Matcher(
    entities=entities,
    mode="hybrid",
    blocking_strategy=BM25Blocking()
)
```

**Available strategies:**
- `BM25Blocking` - Keyword-based (default)
- `TFIDFBlocking` - Document similarity
- `FuzzyBlocking` - Typos and variations
- `NoOpBlocking` - No filtering (for small datasets)

### Reranker Models

```python
matcher = Matcher(
    entities=entities,
    mode="hybrid",
    reranker_model="bge-m3"  # Default reranker
)
```

**Available rerankers:**
- `bge-m3` - Multilingual, high quality (default)
- `bge-large` - Higher accuracy, slower
- `ms-marco` - Lightweight alternative

## Candidate Filtering (Trained Modes)

When using `head-only` or `full` modes, restrict matching to known candidates:

```python
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)

# Only match against specific candidates
candidates = [
    {"id": "DE", "name": "Germany"},
    {"id": "FR", "name": "France"},
]

result = matcher.match("query", candidates=candidates)
# Only returns DE or FR, not other entities
```

**Use cases:**
- Geographic filtering (e.g., only European countries)
- Category filtering (e.g., only technology companies)
- User permissions (e.g., only entities user can access)

## Mode-Specific Features

### zero-shot Features

```python
# Static embeddings (fastest)
matcher = Matcher(mode="zero-shot", model="potion-8m")

# Dynamic embeddings (better accuracy)
matcher = Matcher(mode="zero-shot", model="bge-base")

# Dimension reduction (MRL models)
matcher = Matcher(
    mode="zero-shot",
    model="mrl-en",
    embedding_dim=256
)
```

### head-only / full Features

```python
# Training parameters
matcher.fit(
    training_data,
    num_epochs=4,      # Training epochs
    batch_size=16,     # Batch size
    show_progress=True # Show progress bar
)

# Candidate filtering
result = matcher.match("query", candidates=candidates)
```

### hybrid Features

```python
# Pipeline tuning
result = matcher.match(
    "query",
    blocking_top_k=1000,
    retrieval_top_k=50,
    final_top_k=5
)

# Batch processing
results = matcher.match(
    ["query1", "query2", ...],
    n_jobs=-1,        # Parallel processing
    chunk_size=100    # Batch size
)
```

## Migration from Deprecated Classes

### Old Way

```python
from semanticmatcher import EmbeddingMatcher, EntityMatcher

# Zero-shot
matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()

# Training
matcher = EntityMatcher(entities=entities)
matcher.train(training_data)
```

### New Way

```python
from semanticmatcher import Matcher

# Zero-shot
matcher = Matcher(entities=entities, mode="zero-shot")
matcher.fit()

# Training
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)
```

**See:** [`migration-guide.md`](./migration-guide.md) for complete migration guide.

## Choosing the Right Mode

### Quick Decision Guide

**I have no training data** → `zero-shot`

**I have some training data (1-2 examples/entity)** → `head-only`

**I have good training data (3+ examples/entity)** → `full`

**I have 10k+ entities** → `hybrid`

**I'm not sure** → `auto` (let the library choose)

### Scenario Examples

**API endpoint with no training data:**
```python
matcher = Matcher(entities=entities, mode="zero-shot", model="potion-8m")
# Fastest, no training needed
```

**Internal tool with a few labeled examples:**
```python
matcher = Matcher(entities=entities, mode="head-only")
matcher.fit(training_data)
# Fast training, better than zero-shot
```

**Production system with good training data:**
```python
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)
# Best accuracy
```

**Enterprise directory (50k employees):**
```python
matcher = Matcher(entities=entities, mode="hybrid")
matcher.fit()
# Scales to large datasets
```

## Diagnostic Tools

### Check Current Mode

```python
info = matcher.get_training_info()
print(f"Mode: {info['mode']}")
print(f"Detected: {info['detected_mode']}")
print(f"Active: {info['active_matcher']}")
```

### Explain Match Results

```python
explanation = matcher.explain_match("query", top_k=5)
print(explanation["matched"])      # True/False
print(explanation["best_match"])   # Top result
print(explanation["top_k"])        # All candidates
```

### Debug Issues

```python
diagnosis = matcher.diagnose("query")
print(diagnosis["issue"])       # What's wrong
print(diagnosis["suggestion"])  # How to fix it
```

## Next Steps

- See [`quickstart.md`](./quickstart.md) for basic usage
- See [`models.md`](./models.md) for model selection
- See [`static-embeddings.md`](./static-embeddings.md) for static embeddings
- See [`migration-guide.md`](./migration-guide.md) for migrating from deprecated classes
