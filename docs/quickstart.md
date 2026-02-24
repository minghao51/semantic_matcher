# Quick Start Guide

Related docs: [`index.md`](./index.md) | [`architecture.md`](./architecture.md)

## Installation

```bash
pip install semantic-matcher
```

## Basic Usage

Use this page for first-run examples. For internals and module layout, see [`architecture.md`](./architecture.md).

### Option 1: SetFit Training (Recommended for Production)

Train a few-shot model for better accuracy:

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

result = matcher.predict("Deutchland")
print(result)  # → "DE"

results = matcher.predict(["Deutchland", "America", "France"])
print(results)  # → ["DE", "US", "FR"]
```

### Option 2: Embedding Similarity (No Training)

Use cosine similarity without training - great for prototyping:

```python
from semanticmatcher import EmbeddingMatcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
    {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()

result = matcher.match("Deutschland")
print(result)  # → {"id": "DE", "score": 0.92}

result = matcher.match("UnknownPlace", threshold=0.7)
print(result)  # → None (below threshold)
```

## Text Normalization

Both matchers support text normalization:

```python
from semanticmatcher import EntityMatcher, TextNormalizer

normalizer = TextNormalizer(
    lowercase=True,
    remove_accents=True,
    remove_punctuation=True
)

result = normalizer.normalize("HELLO, World!")  # → "hello world"
```

## Custom Models

```python
from semanticmatcher import EntityMatcher, EmbeddingMatcher

# Use multilingual model
matcher = EntityMatcher(
    entities=entities,
    model_name="sentence-transformers/LaBSE"
)

# Embedding matcher with custom model
matcher = EmbeddingMatcher(
    entities=entities,
    model_name="BAAI/bge-m3",
    threshold=0.8
)
```

## Saving and Loading Models

```python
from semanticmatcher import SetFitClassifier

# Save trained model
classifier.save("/path/to/model")

# Load model
classifier = SetFitClassifier.load("/path/to/model")
```
