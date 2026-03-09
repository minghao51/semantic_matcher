# Configuration Guide

Related docs: [`index.md`](./index.md) | [`models.md`](./models.md) | [`architecture.md`](./architecture.md)

## Overview

Semantic Matcher provides multiple ways to configure model selection, defaults, and behavior through:
- Model registries (built-in)
- Configuration files (YAML/JSON)
- Environment variables
- Programmatic configuration

## Model Registries

### Built-in Registries

The library includes several registries for easy model selection:

```python
from semanticmatcher.config import (
    MODEL_SPECS,
    STATIC_MODEL_REGISTRY,
    DYNAMIC_MODEL_REGISTRY,
    RERANKER_REGISTRY,
    MATCHER_MODE_REGISTRY
)
```

### MODEL_SPECS

Comprehensive model specifications:

```python
MODEL_SPECS = {
    "potion-8m": {
        "name": "minishlab/potion-base-8M",
        "backend": "static",
        "supports_training": False,
        "language": "en",
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "backend": "sentence-transformers",
        "supports_training": True,
        "language": "en",
    },
    # ... more models
}
```

**Fields:**
- `name` - Full HuggingFace model name
- `backend` - "static" or "sentence-transformers"
- `supports_training` - Can this model be used for SetFit training?
- `language` - "en", "multilingual", etc.

### Adding Custom Models

Extend the registry with your own models:

```python
from semanticmatcher.config import MODEL_SPECS

MODEL_SPECS["my-model"] = {
    "name": "my-org/my-custom-model",
    "backend": "sentence-transformers",
    "supports_training": True,
    "language": "en",
}

# Now you can use the alias
from semanticmatcher import Matcher
matcher = Matcher(entities=entities, model="my-model")
```

### Querying Model Specs

```python
from semanticmatcher.config import get_model_spec

# Get model metadata
spec = get_model_spec("potion-8m")
print(spec["name"])        # "minishlab/potion-base-8M"
print(spec["backend"])     # "static"
print(spec["language"])    # "en"

# Check if model supports training
from semanticmatcher.config import supports_training_model
print(supports_training_model("potion-8m"))  # False
print(supports_training_model("mpnet"))      # True
```

## Configuration Files

### Config File Locations

The `Config` class searches in this order:

1. **Custom path** (if provided)
2. **Repository root** - `config.yaml` in repo root
3. **Package defaults** - `data/default_config.json`
4. **Current working directory** - `config.yaml`

### Config File Format

```yaml
# config.yaml (YAML format)
default_model: potion-8m
training:
  num_epochs: 4
  batch_size: 16
embedding:
  threshold: 0.7
  normalize: true
matcher:
  mode: auto
  verbose: false
```

Or JSON:

```json
{
  "default_model": "potion-8m",
  "training": {
    "num_epochs": 4,
    "batch_size": 16
  },
  "embedding": {
    "threshold": 0.7,
    "normalize": true
  }
}
```

### Using Configuration

```python
from semanticmatcher.config import Config

# Load default config
cfg = Config()
print(cfg.default_model)  # "potion-8m"
print(cfg.training.num_epochs)  # 4

# Load custom config
cfg = Config(custom_path="my-config.yaml")

# Nested access with get()
threshold = cfg.get("embedding.threshold", 0.7)
```

### Per-Project Configuration

Create `config.yaml` in your project root:

```yaml
# my-project/config.yaml
default_model: bge-base
training:
  num_epochs: 8
embedding:
  threshold: 0.8
```

```python
# my-project/script.py
from semanticmatcher.config import Config

cfg = Config()  # Automatically finds project/config.yaml
model = cfg.default_model  # "bge-base"
epochs = cfg.training.num_epochs  # 8
```

## Environment Variables

### Supported Variables

```bash
# Set default embedding model
export SEMANTIC_MATCHER_DEFAULT_MODEL="potion-8m"

# Set training default model
export SEMANTIC_MATCHER_TRAINING_MODEL="mpnet"

# Enable verbose logging
export SEMANTIC_MATCHER_VERBOSE="true"

# Disable text normalization
export SEMANTIC_MATCHER_NORMALIZE="false"

# PyTorch device selection
export CUDA_VISIBLE_DEVICES="0"  # Use GPU 0
export PYTORCH_ENABLE_MPS_FALLBACK="1"  # Apple Silicon fallback
```

### Reading Environment Variables

```python
import os
from semanticmatcher import Matcher

model = os.getenv("SEMANTIC_MATCHER_DEFAULT_MODEL", "default")
verbose = os.getenv("SEMANTIC_MATCHER_VERBOSE", "false").lower() == "true"

matcher = Matcher(
    entities=entities,
    model=model,
    verbose=verbose
)
```

## Programmatic Configuration

### Matcher Configuration

```python
from semanticmatcher import Matcher

matcher = Matcher(
    entities=entities,
    model="potion-8m",           # Model selection
    threshold=0.7,               # Matching threshold
    normalize=True,              # Text normalization
    mode="auto",                 # Mode selection
    verbose=False,               # Logging
    blocking_strategy=None,      # For hybrid mode
    reranker_model="default"     # For hybrid mode
)
```

### Runtime Configuration

```python
# Update threshold after initialization
matcher.set_threshold(0.8)

# Check current configuration
info = matcher.get_training_info()
stats = matcher.get_statistics()

print(f"Mode: {info['mode']}")
print(f"Threshold: {stats['threshold']}")
print(f"Model: {stats['model_name']}")
```

## Model Selection Configuration

### Default Models

```python
from semanticmatcher.config import (
    RETRIEVAL_DEFAULT_MODEL,
    TRAINING_DEFAULT_MODEL
)

print(f"Retrieval: {RETRIEVAL_DEFAULT_MODEL}")  # "potion-8m"
print(f"Training: {TRAINING_DEFAULT_MODEL}")    # "mpnet"
```

### Custom Defaults

Override defaults in config file:

```yaml
# config.yaml
retrieval_default_model: minilm
training_default_model: bge-base
```

Or programmatically:

```python
from semanticmatcher.config import MODEL_REGISTRY

MODEL_REGISTRY["default"] = "minishlab/potion-base-32M"
```

## Mode Configuration

### Mode Registry

```python
from semanticmatcher.config import MATCHER_MODE_REGISTRY

print(MATCHER_MODE_REGISTRY)
# {
#     "zero-shot": "EmbeddingMatcher",
#     "head-only": "EntityMatcher",
#     "full": "EntityMatcher",
#     "hybrid": "HybridMatcher",
#     "auto": "SmartSelection"
# }
```

### Mode Resolution

```python
from semanticmatcher.config import resolve_matcher_mode

mode_class = resolve_matcher_mode("zero-shot")
print(mode_class)  # "EmbeddingMatcher"
```

### Default Mode

```yaml
# config.yaml
matcher:
  default_mode: auto  # or zero-shot, head-only, full, hybrid
```

## Training Configuration

### Default Training Parameters

```yaml
# config.yaml
training:
  num_epochs: 4
  batch_size: 16
  show_progress: true
```

### Applying Training Config

```python
from semanticmatcher.config import Config

cfg = Config()
matcher = Matcher(entities=entities)

# Use config values
matcher.fit(
    training_data,
    num_epochs=cfg.training.num_epochs,
    batch_size=cfg.training.batch_size
)
```

## Embedding Configuration

### Static Embedding Config

```yaml
# config.yaml
static_embeddings:
  default_model: potion-8m
  enable_dimension_reduction: true
  default_dimension: 256
```

```python
from semanticmatcher import Matcher

matcher = Matcher(
    entities=entities,
    model="mrl-en",
    embedding_dim=256  # MRL dimension reduction
)
```

### Normalization Config

```yaml
# config.yaml
normalization:
  enabled: true
  lowercase: true
  remove_accents: true
  remove_punctuation: false
```

## Hybrid Mode Configuration

### Blocking Strategy Config

```yaml
# config.yaml
hybrid:
  blocking_strategy: bm25  # or tfidf, fuzzy, none
  blocking_top_k: 1000
  retrieval_top_k: 50
  final_top_k: 5
```

```python
from semanticmatcher import Matcher
from semanticmatcher.core.blocking import BM25Blocking

matcher = Matcher(
    entities=entities,
    mode="hybrid",
    blocking_strategy=BM25Blocking()
)

result = matcher.match(
    "query",
    blocking_top_k=1000,
    retrieval_top_k=50,
    final_top_k=5
)
```

## Advanced Configuration

### Custom Model Resolution

```python
from semanticmatcher.config import resolve_model_alias

# Resolve alias to full model name
full_name = resolve_model_alias("potion-8m")
print(full_name)  # "minishlab/potion-base-8M"

# Pass through if already full name
full_name = resolve_model_alias("org/custom-model")
print(full_name)  # "org/custom-model"
```

### Training Model Resolution

```python
from semanticmatcher.config import resolve_training_model_alias

# Static models auto-fallback to training-compatible
training_model = resolve_training_model_alias("potion-8m")
print(training_model)  # "sentence-transformers/all-mpnet-base-v2"

# Training-compatible models pass through
training_model = resolve_training_model_alias("bge-base")
print(training_model)  # "BAAI/bge-base-en-v1.5"
```

### Checking Model Capabilities

```python
from semanticmatcher.config import (
    is_static_embedding_model,
    supports_training_model,
    get_model_spec
)

# Check if model is static
print(is_static_embedding_model("potion-8m"))  # True
print(is_static_embedding_model("bge-base"))   # False

# Check if model supports training
print(supports_training_model("potion-8m"))  # False
print(supports_training_model("mpnet"))      # True

# Get full model metadata
spec = get_model_spec("potion-8m")
print(spec)
# {
#     'name': 'minishlab/potion-base-8M',
#     'backend': 'static',
#     'supports_training': False,
#     'language': 'en'
# }
```

## Configuration Best Practices

### For Development

```yaml
# dev-config.yaml
default_model: minilm  # Fast iteration
training:
  num_epochs: 1  # Quick testing
matcher:
  verbose: true  # Debug output
```

### For Production

```yaml
# prod-config.yaml
default_model: potion-8m  # Fast inference
training:
  num_epochs: 4  # Full training
matcher:
  verbose: false  # Clean logs
embedding:
  threshold: 0.8  # Higher precision
```

### For Testing

```yaml
# test-config.yaml
default_model: minilm  # Fast, reliable
training:
  num_epochs: 1
matcher:
  verbose: false
```

## Troubleshooting

### Config Not Loading

**Cause:** Config file not in search path.

**Solution:**
```python
from semanticmatcher.config import Config

# Specify path explicitly
cfg = Config(custom_path="/path/to/config.yaml")

# Check what's being loaded
print(cfg.to_dict())
```

### Model Alias Not Resolved

**Cause:** Model not in registry.

**Solution:**
```python
from semanticmatcher.config import MODEL_SPECS, MODEL_REGISTRY

# Check if alias exists
print("my-model" in MODEL_SPECS)  # False
print("my-model" in MODEL_REGISTRY)  # False

# Add to registry
MODEL_SPECS["my-model"] = {
    "name": "org/model",
    "backend": "sentence-transformers",
    "supports_training": True,
    "language": "en",
}
```

### Wrong Model Used for Training

**Cause:** Static model specified for training.

**Solution:**
```python
# Check training compatibility
from semanticmatcher.config import supports_training_model

print(supports_training_model("potion-8m"))  # False - will fallback
print(supports_training_model("mpnet"))      # True - will work

# Use training-compatible model
matcher = Matcher(model="mpnet")  # Not potion-8m
matcher.fit(training_data, mode="full")
```

## Next Steps

- See [`models.md`](./models.md) for model selection
- See [`matcher-modes.md`](./matcher-modes.md) for mode configuration
- See [`static-embeddings.md`](./static-embeddings.md) for static embedding config
- See [`architecture.md`](./architecture.md) for internal configuration
