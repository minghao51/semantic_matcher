# SemanticMatcher Architecture

## Architectural Pattern

### Layered Architecture
```
┌─────────────────────────────────────┐
│         Public API Layer            │
│  (EntityMatcher, EmbeddingMatcher)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│          Core Logic Layer           │
│  (classifier, matcher, normalizer)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Backend Layer               │
│  (SentenceTransformer, LiteLLM)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Utilities Layer             │
│  (embeddings, validation, utils)    │
└─────────────────────────────────────┘
```

## Core Components

### 1. EntityMatcher
- **Purpose**: Training-based semantic matching
- **Algorithm**: SetFit few-shot learning
- **Training**: 8-16 examples per entity
- **File**: `core/matcher.py:23-124`

### 2. EmbeddingMatcher
- **Purpose**: Direct similarity matching (no training)
- **Algorithm**: Cosine similarity on embeddings
- **Trade-off**: Faster but less accurate
- **File**: `core/matcher.py:127-158`

### 3. SetFitClassifier
- **Purpose**: Wraps SetFit training
- **Features**: Model saving/loading, batch inference
- **File**: `core/classifier.py`

### 4. TextNormalizer
- **Purpose**: Text preprocessing pipeline
- **Features**: Lowercase, accent removal, punctuation removal
- **File**: `core/normalizer.py`

## Data Flow

### Training Flow (EntityMatcher)
```
Input Entities (dicts with 'id' and 'examples')
         ↓
Text Normalization (optional)
         ↓
SetFit Training (8-16 examples/entity)
         ↓
Model saved to disk
         ↓
Ready for matching
```

### Matching Flow
```
Query Text(s)
         ↓
Text Normalization (optional)
         ↓
Embedding Generation (backend)
         ↓
Similarity Computation / Classification
         ↓
Results (matched entities + scores)
```

## Backend Abstraction

### Plugin Architecture
- **Base Interface**: `BackendBase` abstract class
- **Implementations**: SentenceTransformer, LiteLLM
- **Extensibility**: Add new backends by implementing base interface
- **File**: `backends/base.py`

### Backend Types
1. **Embedding Backend**: Generate text embeddings
2. **Reranker Backend**: Re-rank results by relevance

## Entry Points

### Main Entry Point
```python
from semanticmatcher import EntityMatcher, EmbeddingMatcher
```

### Example Entry Points
- `hello.py`: Simple demonstration
- `examples/country_matching.py`: Country name matching
- `examples/advanced_matching.py`: Custom backends

## Configuration System

### Config Class
- **Pattern**: Singleton (using `__new__`)
- **File**: `config.py`
- **Sources**: YAML file → Defaults → Environment overrides
- **Caching**: Single instance loaded once

### Default Config
```yaml
default_model: "sentence-transformers/paraphrase-mpnet-base-v2"
training:
  num_epochs: 4
  batch_size: 16
embedding:
  model: "BAAI/bge-m3"
  threshold: 0.7
```

## Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Strategy | Matcher classes | Switch between training/embedding |
| Factory | Backend initialization | Create backend instances |
| Singleton | Config class | Single configuration instance |
| Template | Normalizer pipeline | Configurable preprocessing steps |
