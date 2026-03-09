# Semantic Matcher Architecture

## Overview

Semantic Matcher is a Python library for text-to-entity matching that uses semantic similarity and few-shot learning. The architecture follows a **layered design** with clear separation of concerns between core matching logic, backend abstractions, data ingestion, and utilities.

## Architectural Pattern

**Pattern**: Layered Architecture with Strategy Pattern

The codebase follows a 4-layer architecture:
1. **API Layer** - Public interfaces and unified entry points
2. **Core Layer** - Business logic and matching algorithms
3. **Backend Layer** - Abstraction over ML/DL models
4. **Utility Layer** - Cross-cutting concerns (validation, preprocessing, embeddings)

## Core Architectural Layers

### 1. API Layer (`__init__.py`)

**Purpose**: Lazy-loaded public API with deprecation warnings

**Key Components**:
- Lazy import mechanism via `__getattr__`
- Export registry mapping names to module locations
- Deprecation warnings for legacy classes
- Unified `Matcher` class as primary entry point

**Design Pattern**: Facade Pattern
- Provides simplified interface to complex subsystem
- Hides internal complexity from users
- Manages backward compatibility

**Data Flow**:
```
User imports → __getattr__ → Module lookup → Lazy import → Return class
```

### 2. Core Layer (`core/`)

**Purpose**: Matching algorithms and business logic

**Components**:

#### Matcher Classes (`matcher.py`)
- **`Matcher`** (Unified API): Smart auto-selection of matching strategy
  - Auto-detects training mode based on data
  - Routes to appropriate specialized matcher
  - Provides consistent interface across all modes

- **`EntityMatcher`**: SetFit-based few-shot classification
  - Requires training data
  - Uses SetFit for efficient fine-tuning
  - Supports candidate filtering

- **`EmbeddingMatcher`**: Zero-shot semantic similarity
  - No training required
  - Uses sentence-transformers embeddings
  - Cosine similarity matching
  - Supports Matryoshka embeddings (dimensionality reduction)

#### Specialized Matchers

- **`HybridMatcher`** (`hybrid.py`): Three-stage pipeline
  - Stage 1: Blocking (lexical filtering)
  - Stage 2: Bi-encoder retrieval (semantic search)
  - Stage 3: Cross-encoder reranking (precise scoring)

- **`HierarchicalMatcher`** (`hierarchy.py`): Multi-parent hierarchy matching
  - Graph-based hierarchy representation (NetworkX)
  - Depth-aware confidence scoring
  - Supports DAG structures with weighted edges

#### Supporting Components

- **`SetFitClassifier`** (`classifier.py`): Wrapper for SetFit training
- **`CrossEncoderReranker`** (`reranker.py`): Cross-encoder reranking
- **`TextNormalizer`** (`normalizer.py`): Text preprocessing
- **`BlockingStrategy`** (`blocking.py`): Candidate filtering strategies
  - `BM25Blocking`: Fast lexical blocking
  - `TFIDFBlocking`: TF-IDF vectorization
  - `FuzzyBlocking`: Approximate string matching
  - `NoOpBlocking`: Pass-through for small datasets

**Design Patterns**:
- **Strategy Pattern**: Interchangeable matching strategies
- **Template Method**: Base classes define algorithms, subclasses implement specifics
- **Facade Pattern**: Unified Matcher hides complexity
- **Lazy Initialization**: Matchers created only when needed

### 3. Backend Layer (`backends/`)

**Purpose**: Abstraction over ML/DL model providers

**Components**:

#### Base Abstractions (`base.py`)
- **`EmbeddingBackend`**: Abstract interface for embedding models
  - `encode(texts)`: Generate embeddings

- **`RerankerBackend`**: Abstract interface for rerankers
  - `score(query, docs)`: Score query-document pairs
  - `rerank(query, candidates, top_k)`: Rerank candidates

#### Concrete Implementations

- **`HFEmbedding`**: HuggingFace sentence-transformers
- **`HFReranker`**: HuggingFace cross-encoders
- **`STReranker`**: SentenceTransformer rerankers

**Design Patterns**:
- **Abstract Factory**: Backend creation via factory functions
- **Strategy Pattern**: Pluggable backends
- **Adapter Pattern**: Adapts external libraries to internal interface

**Data Flow**:
```
Core Layer → Backend Interface → Concrete Implementation → External Library
```

### 4. Utility Layer (`utils/`)

**Purpose**: Cross-cutting concerns and shared utilities

**Components**:

- **`validation.py`**: Input validation with helpful error messages
  - Entity validation (ID, name, uniqueness)
  - Threshold validation (0-1 range)
  - Model name validation

- **`embeddings.py`**: Embedding utilities and caching
  - `ModelCache`: Thread-safe LRU cache for models
  - `compute_embeddings()`: Batch embedding computation
  - `cosine_sim()`: Similarity calculations
  - `batch_encode()`: Batch processing utilities

- **`preprocessing.py`**: Text preprocessing utilities
- **`benchmarks.py`**: Performance benchmarking tools

**Design Patterns**:
- **Singleton**: Global default cache instance
- **Utility Pattern**: Stateless helper functions

### 5. Data Ingestion Layer (`ingestion/`)

**Purpose**: External data source integration

**Components**:

- **`cli.py`**: Command-line interface for data ingestion
- **`base.py`**: Base ingestion classes
- **Domain-specific modules**: `languages`, `currencies`, `industries`, `timezones`, `occupations`, `products`, `universities`

**Design Patterns**:
- **Template Method**: Base classes define ingestion workflow
- **Command Pattern**: CLI commands for ingestion operations

## Data Flow & Execution Flow

### Matching Flow (Unified Matcher)

```
User Input
    ↓
Matcher.fit(training_data?, mode?)
    ↓
Mode Detection (auto or explicit)
    ↓
┌─────────────────┬─────────────────┬──────────────────┐
│   zero-shot     │  head-only/full │     hybrid       │
│   (no training) │  (SetFit)       │  (3-stage)       │
├─────────────────┼─────────────────┼──────────────────┤
│ build_index()   │ train()         │ initialize       │
│ encode entities │ SetFit training │ pipeline         │
└─────────────────┴─────────────────┴──────────────────┘
    ↓
Matcher.match(query, top_k)
    ↓
Route to active matcher
    ↓
Return results (consistent format)
```

### Hybrid Matching Flow

```
Query
    ↓
Stage 1: Blocking (BM25/TF-IDF/Fuzzy)
    → Filter to top_k candidates (e.g., 1000)
    ↓
Stage 2: Bi-Encoder Retrieval
    → Semantic similarity search
    → Filter to top_k candidates (e.g., 50)
    ↓
Stage 3: Cross-Encoder Reranking
    → Precise cross-attention scoring
    → Return final top_k results (e.g., 5)
```

### Training Flow (SetFit)

```
Training Data
    ↓
Validation & Normalization
    ↓
SetFitClassifier.train()
    ↓
Initialize SetFitModel
    ↓
Trainer.train()
    → Body learning rate: 2e-5
    → Head learning rate: 1e-3
    → Num epochs: 4 (default)
    ↓
Model saved in classifier
    ↓
Ready for prediction
```

## Key Abstractions

### 1. Matcher Abstraction

**Interface**:
```python
matcher = Matcher(entities, model, threshold, mode, ...)
matcher.fit(training_data?)  # Optional training
results = matcher.match(query, top_k)  # Match queries
```

**Modes**:
- `auto`: Smart selection based on data
- `zero-shot`: Embedding similarity (no training)
- `head-only`: Train classifier head only (fast)
- `full`: Full SetFit training (accurate)
- `hybrid`: Three-stage pipeline

### 2. Backend Abstraction

**Embedding Backend**:
```python
backend.encode(texts) → List[List[float]]
```

**Reranker Backend**:
```python
backend.score(query, docs) → List[float]
backend.rerank(query, candidates, top_k) → List[Dict]
```

### 3. Blocking Strategy Abstraction

```python
strategy.block(query, entities, top_k) → List[entities]
```

## Configuration Management

**Location**: `config.py`

**Components**:
- **Model Registries**: Aliases to full model names
  - `MODEL_REGISTRY`: Embedding models
  - `RERANKER_REGISTRY`: Reranker models
  - `MATCHER_MODE_REGISTRY`: Mode to class mapping

- **`Config` Class**: YAML/JSON configuration loader
  - Search paths: repo root, package default, CWD
  - Deep merge of custom configs
  - Dot-notation access: `config.get("key.subkey")`

**Model Aliases**:
```python
"default" → "sentence-transformers/all-mpnet-base-v2"
"bge-base" → "BAAI/bge-base-en-v1.5"
"bge-m3" → "BAAI/bge-m3"
"nomic" → "nomic-ai/nomic-embed-text-v1"
```

## Exception Hierarchy

**Base**: `SemanticMatcherError`

**Specialized Exceptions**:
- **`ValidationError`**: Input validation failures with context
  - Attributes: `entity`, `field`, `suggestion`
- **`TrainingError`**: Training failures with diagnostics
  - Attributes: `training_mode`, `details`
- **`MatchingError`**: Matching operation failures
- **`ModeError`**: Invalid mode configuration
  - Attributes: `invalid_mode`, `valid_modes`

**Design Pattern**: Exception Chaining with Context
- Rich error messages with suggestions
- Structured diagnostic information
- Helpful recovery hints

## Caching Strategy

**Model Cache** (`utils/embeddings.py`)

**Features**:
- Thread-safe LRU cache
- Memory-based eviction (configurable GB limit)
- Optional TTL (time-to-live)
- Hit/miss statistics

**Implementation**:
```python
cache = ModelCache(max_memory_gb=4.0, ttl_seconds=None)
model = cache.get_or_load("model_name", lambda: load_model())
```

**Global Instance**:
- Singleton pattern via `get_default_cache()`
- Shared across all matchers by default
- Reduces model loading overhead

## Extensibility Points

### 1. Custom Backends

**How**: Inherit from `EmbeddingBackend` or `RerankerBackend`

```python
class CustomBackend(EmbeddingBackend):
    def encode(self, texts: list[str]) -> list[list[float]]:
        # Custom implementation
        pass
```

### 2. Custom Blocking Strategies

**How**: Inherit from `BlockingStrategy`

```python
class CustomBlocking(BlockingStrategy):
    def block(self, query, entities, top_k):
        # Custom filtering logic
        pass
```

### 3. Custom Matchers

**How**: Implement match interface or extend existing classes

```python
class CustomMatcher:
    def __init__(self, entities, ...): ...
    def fit(self, training_data=None): ...
    def match(self, texts, top_k=1): ...
```

## Technology Stack

**Core Dependencies**:
- `sentence-transformers`: Embedding models
- `setfit`: Few-shot learning
- `torch`: Deep learning backend
- `scikit-learn`: ML utilities (cosine similarity, TF-IDF)
- `networkx`: Graph algorithms (hierarchical matching)
- `rank-bm25`: BM25 blocking
- `rapidfuzz`: Fuzzy matching
- `nltk`: Text preprocessing

**Dev Tools**:
- `pytest`: Testing
- `ruff`: Linting
- `black`: Formatting
- `uv`: Package management

## Performance Considerations

### Optimization Strategies

1. **Lazy Initialization**: Matchers created only when needed
2. **Model Caching**: Reduces loading overhead
3. **Batch Processing**: Efficient bulk operations
4. **Candidate Filtering**: Blocking reduces search space
5. **Dimensionality Reduction**: Matryoshka embeddings support
6. **Parallel Processing**: Hybrid matcher supports parallel bulk matching

### Complexity Analysis

- **EmbeddingMatcher**: O(n) encoding, O(n*k) matching (n=entities, k=top_k)
- **EntityMatcher**: O(1) prediction after training
- **HybridMatcher**: O(n) blocking + O(k) retrieval + O(m) reranking
  - n: total entities, k: blocking_top_k, m: retrieval_top_k

## Testing Strategy

**Test Organization** (`tests/`):
- `test_core/`: Core matcher tests
- `test_backends/`: Backend contract tests
- `test_utils/`: Utility function tests
- `test_ingestion/`: Data ingestion tests

**Test Markers**:
- `integration`: External service/network tests
- `slow`: Expensive tests (not run by default)
- `hf`: HuggingFace model-backed tests

## Deployment & Packaging

**Build System**: Hatchling

**Package Structure**:
```
src/semanticmatcher/  # Source package
pyproject.toml         # Build config
README.md              # Documentation
LICENSE                # MIT license
```

**Entry Points**:
- CLI: `semanticmatcher-ingest` command
- Library: `import semanticmatcher`

**Python Versions**: 3.9, 3.10, 3.11, 3.12

## Design Principles

1. **Simplicity**: Clear, minimal interfaces
2. **Flexibility**: Pluggable backends and strategies
3. **Performance**: Lazy loading, caching, batching
4. **User Experience**: Helpful errors, auto-detection, sensible defaults
5. **Extensibility**: Abstract base classes, factory patterns
6. **Backward Compatibility**: Deprecation warnings, migration guides

## Future Architecture Considerations

**Potential Enhancements**:
1. **Async Support**: Async/await for I/O-bound operations
2. **Distributed Matching**: Ray/Dask for large-scale matching
3. **Model Versioning**: Track and compare model versions
4. **Experiment Tracking**: Integration with MLflow/Weights & Biases
5. **Streaming Support**: Process large datasets without loading all into memory
6. **Multi-modal Support**: Match images, audio, etc.
