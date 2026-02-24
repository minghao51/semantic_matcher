# Quick Start Guide

Related docs: [`index.md`](./index.md) | [`notebooks.md`](./notebooks.md) | [`examples.md`](./examples.md) | [`architecture.md`](./architecture.md)

## Installation

```bash
pip install semantic-matcher
```

Use this page for the official package wrapper API. If you want experiment notebooks/scripts, see [`notebooks.md`](./notebooks.md). If you want lower-level raw `setfit` examples, see [`examples.md`](./examples.md).

## Choose a Matcher

| Matcher | Best For | Tradeoff |
|---|---|---|
| `EmbeddingMatcher` | Prototyping, no training setup | Usually lower accuracy on harder cases |
| `EntityMatcher` | Few-shot production matching | Requires labeled training data + training time |

## Path 1: Embedding Similarity (No Training)

Use cosine similarity without training for quick prototypes.

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

## Path 2: Few-Shot Training with `EntityMatcher`

Train a SetFit-backed matcher when you have labeled examples.

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

## Text Normalization (Optional)

Both matchers support normalization by default. You can also use `TextNormalizer` directly.

```python
from semanticmatcher import TextNormalizer

normalizer = TextNormalizer(
    lowercase=True,
    remove_accents=True,
    remove_punctuation=True,
)

print(normalizer.normalize("HELLO, World!"))  # "hello world"
```

## Custom Model Names

```python
from semanticmatcher import EntityMatcher, EmbeddingMatcher

entity_matcher = EntityMatcher(
    entities=entities,
    model_name="sentence-transformers/LaBSE",
)

embedding_matcher = EmbeddingMatcher(
    entities=entities,
    model_name="BAAI/bge-m3",
    threshold=0.8,
)
```

## Save/Load (Low-Level `SetFitClassifier`)

`EntityMatcher` does not currently expose a top-level save/load API. If you need lower-level model persistence, use `SetFitClassifier` directly.

```python
from semanticmatcher import SetFitClassifier

# Assuming `classifier` is an already-trained SetFitClassifier instance:
classifier.save("/path/to/model")

# Load it later
classifier = SetFitClassifier.load("/path/to/model")
```

## Common First-Run Issues

- `EmbeddingMatcher`: call `build_index()` before `match()`
- `EntityMatcher`: call `train()` before `predict()`
- First run may download models (network required)

See [`troubleshooting.md`](./troubleshooting.md) for fixes.

## Where to Go Next

- Experiments and notebook methods: [`notebooks.md`](./notebooks.md)
- Raw/advanced examples: [`examples.md`](./examples.md)
- Internals and module layout: [`architecture.md`](./architecture.md)
