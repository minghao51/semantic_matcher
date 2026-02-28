# Architecture

## Design Philosophy

**Three-Stage Waterfall Pipeline**: Efficient semantic matching through progressive refinement
1. **Blocking** - Fast lexical filtering to reduce candidate set
2. **Retrieval** - Semantic similarity search with bi-encoders
3. **Reranking** - Precise cross-encoder scoring for top candidates

This pattern balances accuracy (cross-encoder) with speed (blocking + bi-encoder).

## Core Patterns

### 1. Strategy Pattern

**Used in**: Blocking strategies

**Implementation**:
- Abstract base: `BlockingStrategy` in `src/semanticmatcher/core/blocking.py`
- Concrete implementations: `BM25Blocking`, `TFIDFBlocking`, `FuzzyBlocking`, `NoOpBlocking`

**Benefits**:
- Interchangeable blocking algorithms
- Easy to add new strategies
- Consistent interface via `block(query, entities, top_k)`

### 2. Pipeline Pattern

**Used in**: `HybridMatcher`

**Implementation**: `src/semanticmatcher/core/hybrid.py`

**Flow**:
```
Input Query
    ↓
[Blocker] → filtered candidates (blocking_top_k)
    ↓
[Retriever] → semantic matches (retrieval_top_k)
    ↓
[Reranker] → final results (final_top_k)
    ↓
Output Results
```

**Benefits**:
- Each stage is independently testable
- Configurable stage boundaries
- Parallel processing support (ThreadPoolExecutor)

### 3. Backend Abstraction

**Used in**: Model loading and inference

**Implementation**: `src/semanticmatcher/backends/`

**Base Class**: `BaseBackend` (abstract)
- `SentenceTransformerBackend` - Standard sentence transformers
- `STReranker` - Cross-encoder rerankers
- `LiteLLMBackend` - OpenAI-compatible APIs (stub, not active)

**Benefits**:
- Swappable model backends
- Consistent embedding API
- Future-proof for new model providers

### 4. Lazy Import Pattern

**Used in**: Package exports

**Implementation**: `src/semanticmatcher/__init__.py`

**Mechanism**:
```python
_EXPORTS = {
    "EntityMatcher": (".core.matcher", "EntityMatcher"),
    # ...
}

def __getattr__(name):
    # Import on first access
```

**Benefits**:
- Faster import times
- Reduced memory footprint
- Optional dependencies not loaded unless used

## Layered Architecture

```
┌─────────────────────────────────────────┐
│         User-Facing APIs                │
│  (EntityMatcher, EmbeddingMatcher,     │
│   HybridMatcher, CrossEncoderReranker)  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         Core Matching Layer             │
│  - TextNormalizer                       │
│  - SetFitClassifier                     │
│  - BlockingStrategy implementations     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         Backend Abstraction             │
│  - BaseBackend                          │
│  - SentenceTransformerBackend           │
│  - STReranker                           │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         ML Frameworks                   │
│  - sentence-transformers                │
│  - SetFit                               │
│  - PyTorch                              │
└─────────────────────────────────────────┘
```

## Data Flow

### Basic Matching (EmbeddingMatcher)

```
Entities List
    ↓
[Normalize Text]
    ↓
[Generate Embeddings] → Cached in ModelCache
    ↓
[Build Index] → Store embeddings
    ↓
Query Text → [Normalize] → [Generate Embedding]
    ↓
[Cosine Similarity] → Compare with index
    ↓
[Filter by Threshold] → Top matches
    ↓
Results List
```

### Training Flow (EntityMatcher)

```
Training Data (text, label pairs)
    ↓
[SetFitClassifier.train()]
    ├── Generate contrastive pairs
    ├── Train sentence transformer
    └── Save model weights
    ↓
[Train] → Fine-tune model
    ↓
[Predict] → Use fine-tuned model for inference
```

### Hybrid Flow (HybridMatcher)

```
Query + Entities
    ↓
[Blocking Stage]
    ├── BM25Blocking (lexical)
    ├── TFIDFBlocking (vector space)
    ├── FuzzyBlocking (approximate match)
    └── NoOpBlocking (pass-through)
    ↓ (blocking_top_k candidates)
[Retrieval Stage]
    ├── Bi-encoder semantic search
    └── Cosine similarity ranking
    ↓ (retrieval_top_k candidates)
[Reranking Stage]
    ├── Cross-encoder scoring
    └── Cross-attention re-ranking
    ↓ (final_top_k results)
Final Matches
```

## Key Abstractions

### TextNormalizer

**Purpose**: Standardize text before matching

**Features**:
- Lowercase conversion
- Accent removal (Unicode normalization)
- Punctuation stripping
- Whitespace normalization

**Location**: `src/semanticmatcher/core/normalizer.py`

### ModelCache

**Purpose**: Cache loaded models to avoid redundant downloads

**Features**:
- Thread-safe (locks)
- TTL-based expiration
- Memory limits
- Shared across instances

**Location**: `src/semanticmatcher/utils/embeddings.py`

### SetFitClassifier

**Purpose**: Few-shot classification wrapper around SetFit

**Features**:
- Train/predict API
- Model persistence (save/load)
- Configurable number of epochs

**Location**: `src/semanticmatcher/core/classifier.py`

## Concurrency

**Thread Safety**:
- ModelCache uses `threading.Lock()` for concurrent access
- HybridMatcher supports parallel bulk matching via `ThreadPoolExecutor`

**Async**:
- No async/await patterns (synchronous only)
- Parallel processing via thread pools

## Error Handling Strategy

**Validation**:
- Input validation in `src/semanticmatcher/utils/validation.py`
- Type hints enforced with `Optional`, `List`, `Dict`

**Exception Propagation**:
- Exceptions bubble up from ML frameworks
- No custom exception hierarchy
- User handles sentence-transformers/SetFit errors

## Performance Optimizations

1. **Blocking** - Reduce search space before expensive operations
2. **Model Caching** - Reuse loaded models across instances
3. **Parallel Bulk Matching** - ThreadPoolExecutor for batch queries
4. **Embedding Index** - Pre-compute embeddings for entities

## Extensibility Points

1. **Custom Blocking Strategies** - Inherit from `BlockingStrategy`
2. **Custom Backends** - Inherit from `BaseBackend`
3. **Model Registry** - Add aliases to `config.py`
4. **Ingestion Modules** - Follow pattern in `ingestion/base.py`
