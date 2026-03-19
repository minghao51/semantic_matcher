# Architecture

## Core Patterns

### 1. Layered Architecture
```
┌─────────────────────────────────────┐
│         API Layer                   │
│  (Matcher, NovelClassDetector)      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Core Services Layer            │
│  (classifier, normalizer, blocking) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Backend Abstraction Layer      │
│  (embeddings, reranking, LLM)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Utilities Layer                │
│  (validation, preprocessing,        │
│   caching, benchmarks)              │
└─────────────────────────────────────┘
```

### 2. Factory Pattern
- **Backend Factories** - Dynamic backend instantiation based on configuration
- **Model Registry** - 13+ pre-configured models with auto-selection
- **CLI Runners** - Command creation and execution
- **Strategy Selection** - Matcher mode selection (entity, embedding, hybrid, hierarchical)

### 3. Strategy Pattern
- **Matcher Modes** - Entity, embedding, hybrid, hierarchical matching
- **Detection Strategies** - HDBSCAN, Isolation Forest, novelty detection
- **Blocking Strategies** - BM25, exact match, no blocking
- **Reranking Strategies** - Cross-encoder, no reranking

### 4. Template Method Pattern
- **Base Classes** - `BaseMatcher`, `BaseBackend`, `BaseDetector`
- **Consistent Interfaces** - All matchers implement `fit()`, `match()`, `batch_match()`
- **Async Support** - Native async/await throughout the stack

### 5. Observer Pattern
- **Logging** - Configurable verbose logging across all components
- **Monitoring** - Performance tracking and benchmarking
- **Event Handling** - Async event propagation

## Key Components

### Unified API
- **Single Entry Point** - `Matcher` class supports all modes
- **Mode Selection** - Automatic based on input type (entity vs embedding)
- **Async Support** - Native async/await for non-blocking operations
- **Backward Compatibility** - Deprecated classes with warnings

### Core Services
- **Classifier** - Entity classification and categorization
- **Normalizer** - Text preprocessing and normalization
- **Blocking** - Efficient candidate selection (BM25, exact)
- **Similarity** - Vector similarity computation

### Backend Abstraction
- **Embedding Backends**
  - Sentence Transformers (dynamic models)
  - Static Embeddings (Model2Vec)
  - LiteLLM (API-based embeddings)
- **Reranking Backends**
  - Cross-encoder models
  - No reranking option
- **LLM Integration**
  - Anthropic Claude
  - OpenAI GPT
  - OpenRouter (multi-provider)

### Novelty Detection
- **Detector API** - `NovelClassDetector` for detecting new classes
- **Strategies**
  - HDBSCAN clustering
  - Isolation Forest
  - Custom novelty detection
- **LLM Proposer** - Automatic class naming for novel categories
- **Schema Validation** - Pydantic schemas for configuration

### Data Ingestion
- **CLI Tools** - `semanticmatcher-ingest` command
- **Dataset Preparation** - Industries, languages, currencies, timezones
- **External Data** - Fetch from APIs and GitHub
- **JSON Storage** - Entity data persistence

## Data Flow

### Matching Pipeline
```
Input Text
    ↓
Normalization (preprocessing)
    ↓
Validation (schema check)
    ↓
Embedding Generation (backend)
    ↓
Blocking (candidate selection)
    ↓
Similarity Computation (vector search)
    ↓
Reranking (cross-encoder)
    ↓
Post-processing (thresholding)
    ↓
Output Results
```

### Novelty Detection Pipeline
```
Matched Results
    ↓
Low-confidence Items
    ↓
Clustering (HDBSCAN/Isolation Forest)
    ↓
Novel Cluster Identification
    ↓
LLM Class Naming (optional)
    ↓
Novel Class Proposal
    ↓
Schema Update
```

### Training Pipeline
```
Training Data (entities + labels)
    ↓
Validation (schema check)
    ↓
Text Normalization
    ↓
Embedding Generation
    ↓
Model Training (classifier/normalizer)
    ↓
Caching (model + embeddings)
    ↓
Ready for Matching
```

## Abstractions

### Matcher Abstraction
- **Base Interface** - `fit()`, `match()`, `batch_match()`
- **Modes** - Entity, embedding, hybrid, hierarchical
- **Async Variants** - `async_fit()`, `async_match()`, `async_batch_match()`
- **Configuration** - Unified config for all modes

### Backend Abstraction
- **Embedding Backend** - `encode()` interface
- **Reranking Backend** - `rerank()` interface
- **LLM Backend** - `generate()` interface
- **Caching** - LRU cache with thread safety

### Model Abstraction
- **Model Registry** - Centralized model configuration
- **Auto-selection** - Use case-based recommendations
- **Versioning** - Model version management
- **Caching** - Hugging Face cache integration

## Error Handling

### Exception Hierarchy
```
SemanticMatcherError (base)
├── ValidationError (entity/field/suggestion)
├── TrainingError (diagnostics)
└── ConfigurationError (model/backend)
```

### Validation Strategy
- **Input Validation** - Pydantic schemas for all inputs
- **Configuration Validation** - Model/backend compatibility checks
- **Runtime Validation** - Real-time error detection
- **Helpful Messages** - Suggestions for common issues

## Performance Optimization

### Caching Strategy
- **Model Cache** - LRU cache for loaded models
- **Embedding Cache** - Pre-computed embeddings
- **Result Cache** - Match result caching
- **Thread Safety** - Safe concurrent access

### Async Operations
- **Non-blocking I/O** - Async LLM calls
- **Batch Processing** - Efficient batch operations
- **Concurrency Control** - Async task management
- **Memory Management** - Stream processing for large datasets

### Vector Search
- **FAISS** - High-performance similarity search
- **HNSWlib** - Approximate nearest neighbors
- **Blocking** - Candidate selection optimization
- **Reranking** - Two-stage retrieval for accuracy

## Security Considerations

### API Key Management
- Environment variable storage
- No hardcoded credentials
- Multi-provider fallback

### Input Validation
- Schema validation for all inputs
- Sanitization of user-provided text
- Type safety with Pydantic

### External Dependencies
- Network request timeouts
- Fallback mechanisms
- Error recovery strategies
