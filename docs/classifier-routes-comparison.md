# Classifier Route Comparison

Related docs: [`matcher-modes.md`](./matcher-modes.md) | [`bert-classifier.md`](./bert-classifier.md) | [`models.md`](./models.md)

This document compares the main routes exposed by `Matcher`:

- `zero-shot`
- `head-only`
- `full`
- `bert`
- `hybrid`

The goal is not to rank them globally. Each route optimizes for a different mix of setup cost, latency, accuracy, and hardware footprint.

## Quick Summary

| Route | Training data needed | Latency profile | Quality profile | Compute profile | Best fit |
|---|---|---|---|---|---|
| `zero-shot` | None | Fast startup, moderate query latency | Good baseline, weakest task adaptation | CPU-friendly | Cold start, prototypes, no labels |
| `head-only` | 1-2 examples per entity | Fast inference | Better than zero-shot on simple labeled tasks | CPU-friendly, modest RAM | Quick supervised iteration |
| `full` | 3+ examples per entity | Fast inference | Strong default trained classifier route | CPU okay, GPU optional for faster training | Most production classifier use cases |
| `bert` | Best with 100+ total and 8+ per entity | Slowest inference of classifier modes | Highest ceiling for nuanced text classification | GPU recommended, highest memory use | Accuracy-first deployments |
| `hybrid` | No classifier labels required | Higher end-to-end latency, scalable retrieval | Best for large candidate sets and precision-sensitive search | Multiple models, highest operational complexity | Large catalogs and long-tail retrieval |

## Route Details

### `zero-shot`

**What it is:** embedding similarity against entity names and aliases, with no supervised training.

**Pros**
- No labeling or training loop.
- Lowest setup cost.
- Easy to operate in CPU-only environments.
- Good first pass for evaluating entity coverage and alias quality.

**Cons**
- Cannot learn task-specific decision boundaries.
- More sensitive to weak entity names or missing aliases.
- Usually below trained routes on ambiguous or domain-specific language.

**Recommended when**
- You have no labeled data yet.
- You need an immediate baseline.
- The entity list is small to medium and the wording is fairly literal.

**Compute guidance**
- CPU: recommended.
- GPU: not needed.
- RAM: low to moderate, mostly driven by embedding model size and entity index size.
- VRAM: none required.

### `head-only`

**What it is:** supervised SetFit route for very small labeled datasets.

**Pros**
- Fastest trained route.
- Good improvement over zero-shot with very little data.
- Keeps inference relatively cheap.
- Easy to rerun during labeling iterations.

**Cons**
- Lower ceiling than `full` or `bert`.
- Less robust when label boundaries depend on subtle wording.
- Can plateau quickly once the task becomes more semantic than lexical.

**Recommended when**
- You have only 1-2 examples per entity.
- You want a cheap supervised baseline before investing in more labels.
- Training speed matters more than squeezing out maximum quality.

**Compute guidance**
- CPU: good default.
- GPU: optional, mainly for faster experimentation.
- RAM: modest.
- VRAM: not required.

### `full`

**What it is:** the main SetFit training route for classifier-style matching.

**Pros**
- Best general-purpose tradeoff for trained classification.
- Faster inference than `bert`.
- Usually more data-efficient and cheaper to operate than full transformer classifiers.
- Easier to deploy on CPU-only infrastructure than `bert`.

**Cons**
- Still depends on labeled data quality.
- Lower ceiling than `bert` on nuanced or pattern-heavy tasks.
- Less attractive when you need multilingual transformer classification behavior.

**Recommended when**
- You have at least 3 examples per entity.
- You want a production-ready default with balanced quality and speed.
- You need trained behavior but want to avoid transformer-classifier serving cost.

**Compute guidance**
- CPU: viable for training and serving.
- GPU: optional and useful if training repeatedly.
- RAM: modest to moderate.
- VRAM: optional.

### `bert`

**What it is:** fine-tuned transformer classification using a BERT-family backbone such as `distilbert`, `roberta-base`, `deberta-v3`, or `bert-multilingual`.

**Pros**
- Highest accuracy ceiling among the classifier routes.
- Strongest option for subtle phrasing, context-heavy labels, and harder edge cases.
- Better fit for tasks where exact wording patterns matter.
- Model family choice lets you trade size for quality.

**Cons**
- Slowest classifier inference path.
- Highest training and serving cost among classifier routes.
- More memory pressure on both CPU and GPU.
- Longer test and CI runtime if live model training is exercised by default.

**Recommended when**
- Accuracy matters more than throughput.
- You have richer supervision: roughly 100+ total examples and at least 8+ per entity is a sensible threshold.
- You can justify GPU-backed training, and possibly GPU-backed serving for lower latency.
- The task contains nuanced phrasing that SetFit misses.

**Compute guidance**
- CPU: acceptable for experimentation and low-QPS serving, but slower.
- GPU: recommended for training; helpful for serving when latency matters.
- RAM: moderate to high depending on model.
- VRAM:
  - `tinybert`: low, suitable for constrained GPUs.
  - `distilbert`: moderate and the best default balance.
  - `roberta-base` / `deberta-v3`: moderate to high, more likely to need careful batch sizing.
- Disk footprint: larger than SetFit-style classifier artifacts.

**Backbone selection**

| Model | Relative size | Relative speed | Relative quality | Recommended use |
|---|---|---|---|---|
| `tinybert` | Smallest | Fastest | Lowest of BERT options | Tight resource budgets |
| `distilbert` | Small | Fast | High | Default BERT choice |
| `roberta-base` | Medium | Medium | Very high | Accuracy-focused English workloads |
| `deberta-v3` | Larger | Slowest | Highest ceiling | Maximum quality over speed |
| `bert-multilingual` | Large | Slow | High | Multilingual classification |

### `hybrid`

**What it is:** a retrieval pipeline, not a classifier-training route. It combines blocking, embedding retrieval, and cross-encoder reranking.

**Pros**
- Handles much larger entity sets than the classifier routes.
- Candidate pruning makes large-search problems tractable.
- Reranking improves precision on hard retrieval tasks.
- Strong fit when entity matching is closer to search than closed-set classification.

**Cons**
- Highest system complexity.
- Multiple models and stages to tune.
- More latency variance than a single classifier.
- Harder to reason about operationally than `zero-shot`, `full`, or `bert`.

**Recommended when**
- The entity inventory is large, often tens of thousands or more.
- You need high recall first, then precision via reranking.
- Matching resembles document retrieval more than small-label classification.

**Compute guidance**
- CPU: usable for smaller deployments, but reranking can become expensive.
- GPU: helpful for reranker-heavy workloads.
- RAM: moderate to high because multiple indexes/models may be resident.
- VRAM: useful when the cross-encoder is on GPU.

## Recommendation Matrix

| Situation | Recommended route | Why |
|---|---|---|
| No labels yet | `zero-shot` | Cheapest baseline and immediate feedback |
| 1-2 examples per entity | `head-only` | Fastest supervised improvement |
| 3+ examples per entity, typical production API | `full` | Best balance of quality, cost, and latency |
| Accuracy-first classification with enough data | `bert` | Highest ceiling for nuanced decisions |
| Large candidate catalog or retrieval-style matching | `hybrid` | Scales better than closed-set classifiers |

## Practical Selection Guidance

Start with `zero-shot` if you are still validating the taxonomy. Move to `head-only` as soon as you have a handful of trustworthy labels. Use `full` as the default trained route for most production classifier workloads. Move to `bert` only when you have enough supervision and a clear accuracy gap to justify the extra compute. Use `hybrid` when the problem stops looking like small-label classification and starts looking like large-scale search plus reranking.

If the choice is between `full` and `bert`, the main question is usually not “which is better?” but “is the incremental quality worth the extra cost and latency?” In many CPU-first deployments, `full` remains the practical default even if `bert` is slightly more accurate.
