# Novel Class Detection & Proposal System

## Overview

The Novel Class Detection & Proposal System enables discovery of new categories in unlabeled text data. It combines multi-strategy novelty detection with LLM-based class naming to help researchers and developers expand their taxonomies systematically.

`NovelEntityMatcher` is the primary orchestration API. `NovelClassDetector` remains available as a compatibility wrapper for older code, but new integrations should use `NovelEntityMatcher`.

## Key Features

- **Multi-Strategy Detection**: Top-k uncertainty scoring, ANN-backed kNN novelty scoring, and validated density-based clustering
- **Efficient ANN Search**: O(log n) similarity search using HNSWlib or FAISS
- **LLM-Powered Proposals**: Intelligent class naming and justification using litellm
- **Structured Output**: Type-safe Pydantic models for all results
- **File-Based Storage**: Automatic saving of discovery sessions in YAML/JSON format
- **Multi-Provider LLM Support**: OpenRouter, Anthropic, OpenAI, and more

## Setup

### 1. Install Dependencies

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Configure API Keys (for LLM Proposals)

The novel class detection system supports LLM-based class naming. To use this feature, configure your API keys:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# For OpenRouter (recommended - supports multiple providers):
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Or for Anthropic Claude:
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or for OpenAI:
OPENAI_API_KEY=sk-your-key-here
```

**API Key Sources:**
- **OpenRouter**: https://openrouter.ai/keys (supports Claude, GPT-4, Gemini, etc.)
- **Anthropic**: https://console.anthropic.com/
- **OpenAI**: https://platform.openai.com/api-keys

### 3. Verify Installation

```bash
# Run tests to verify installation
uv run pytest tests/test_ann_index.py -v

# Try the example
uv run python examples/novel_discovery_example.py
```

## Architecture

```
User Queries
    ↓
Tier 1: Enhanced Matchers (return_metadata=True)
    → MatchResultWithMetadata: embeddings, scores, probabilities
    ↓
Tier 2: NovelEntityMatcher
    1. Collect matcher metadata + reference corpus
    2. Delegate multi-signal scoring to NoveltyDetector
    3. Return operational match results or batch discovery reports
    ↓
Tier 3: NoveltyDetector
    1. Top-k Uncertainty Scoring → uncertainty_scores
    2. ANN kNN Distance Analysis → knn_novelty_scores
    3. Validated HDBSCAN Clustering → cluster_support_scores
    4. Weighted Signal Fusion → novel_sample_indices
    ↓
Tier 4: LLMClassProposer (optional)
    → NovelClassAnalysis: proposed classes with names, descriptions, justification
    ↓
File Storage: proposals/{timestamp}.yaml
```

## Quick Start

### Basic Usage

```python
from semanticmatcher import Matcher, NovelEntityMatcher

# 1. Train a matcher on known classes
matcher = Matcher(entities=entities, model="minilm")
matcher.fit(texts=train_texts, labels=train_labels)

# 2. Create NovelEntityMatcher
novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    auto_save=True,
    output_dir="./proposals",
)

# 3. Discover novel classes
report = await novel_matcher.discover_novel_classes(
    queries=test_queries,
    existing_classes=["class1", "class2", "class3"],
)

# 4. Check results
print(f"Found {len(report.novel_sample_report.novel_samples)} novel samples")
```

### With LLM Proposals

```python
novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    llm_provider="openrouter",  # or "anthropic", "openai"
    llm_model="anthropic/claude-sonnet-4",
)

report = await novel_matcher.discover_novel_classes(
    queries=test_queries,
    existing_classes=existing_classes,
    context="Scientific research domain",  # Optional domain context
    run_llm_proposal=True,
)

# Access proposed classes
for proposal in report.class_proposals.proposed_classes:
    print(f"Class: {proposal.name}")
    print(f"Description: {proposal.description}")
    print(f"Confidence: {proposal.confidence:.2%}")
    print(f"Justification: {proposal.justification}")
```

## Configuration

### Detection Configuration

```python
from semanticmatcher.novelty.schemas import DetectionConfig, DetectionStrategy

config = DetectionConfig(
    strategies=[
        DetectionStrategy.CONFIDENCE,  # Low confidence detection
        DetectionStrategy.KNN_DISTANCE,  # Distance to known ANN neighbors
        DetectionStrategy.CLUSTERING,    # Validated density-based clustering
    ],
    confidence_threshold=0.7,       # Backward-compatible top-1 confidence cutoff
    uncertainty_top_k=5,           # Candidates inspected for uncertainty scoring
    knn_k=5,                       # Neighbors inspected in ANN novelty scoring
    knn_distance_threshold=0.55,   # Threshold for ANN kNN novelty scoring
    min_cluster_size=5,            # Minimum samples for clustering
    ann_backend="hnswlib",          # "hnswlib" or "faiss"
    combine_method="weighted",      # Default weighted fusion; legacy boolean modes still available
)

novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    detection_config=config,
)
```

### LLM Configuration

```python
# Set API keys as environment variables
import os
os.environ["OPENROUTER_API_KEY"] = "sk-..."
# or
os.environ["ANTHROPIC_API_KEY"] = "sk-..."

# Or pass directly
novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    llm_provider="openrouter",
    llm_model="anthropic/claude-sonnet-4",
    llm_api_keys={
        "openrouter": "sk-...",
    },
)
```

## API Reference

### NovelEntityMatcher

Primary orchestration class for novelty-aware matching and novel-class discovery.

#### Parameters

- `matcher`: Fitted `Matcher` instance
- `entities`: Optional entity definitions when constructing the inner matcher directly
- `acceptance_threshold`: Threshold for accepting a known-class match
- `detection_config`: DetectionConfig instance
- `llm_provider`: LLM provider name ("openrouter", "anthropic", "openai")
- `llm_model`: LLM model name
- `llm_api_keys`: Dict of API keys for providers
- `output_dir`: Directory to save reports
- `auto_save`: Automatically save reports to disk

#### Methods

##### `match(text, return_alternatives=False, existing_classes=None)`

Run novelty-aware matching for a single query.

**Returns:** `NovelEntityMatchResult`

##### `match_batch(texts, return_alternatives=False, existing_classes=None)`

Run novelty-aware matching for a batch of queries.

**Returns:** List of `NovelEntityMatchResult`

##### `async discover_novel_classes(queries, existing_classes, context=None, return_metadata=True, run_llm_proposal=True)`

Discover novel classes in query texts.

**Parameters:**
- `queries`: List of text queries to analyze
- `existing_classes`: List of known/expected classes
- `context`: Optional domain context for LLM
- `return_metadata`: Whether matcher should return rich metadata
- `run_llm_proposal`: Whether to run LLM class proposal

**Returns:** `NovelClassDiscoveryReport`

##### `batch_discover(queries_batch, existing_classes, context=None)`

Run discovery on multiple query lists.

**Parameters:**
- `queries_batch`: List of query lists
- `existing_classes`: List of known classes
- `context`: Optional domain context

**Returns:** List of `NovelClassDiscoveryReport`

### NovelClassDetector

Compatibility wrapper over `NovelEntityMatcher`.

Use this only for older code that already imports `semanticmatcher.novelty.detector_api.NovelClassDetector`. New code should prefer `NovelEntityMatcher`.

### NoveltyDetector

Lower-level detector for multi-strategy novelty detection.

```python
from semanticmatcher.novelty.detector import NoveltyDetector

detector = NoveltyDetector(
    config=DetectionConfig(),
    embedding_dim=768,
)

report = detector.detect_novel_samples(
    texts=texts,
    confidences=confidences,
    embeddings=embeddings,
    predicted_classes=predictions,
    candidate_results=raw_match_results,
    known_classes=known_classes,
    reference_embeddings=reference_embeddings,
    reference_labels=reference_labels,
)
```

### LLMClassProposer

LLM-based class proposal system.

```python
from semanticmatcher.novelty.llm_proposer import LLMClassProposer

proposer = LLMClassProposer(
    primary_model="anthropic/claude-sonnet-4",
    fallback_models=["openai/gpt-4o"],
)

analysis = proposer.propose_classes(
    novel_samples=novel_samples,
    existing_classes=existing_classes,
    context="Domain context",
)
```

### Enhanced Matchers

All matcher types support `return_metadata` parameter:

```python
# Standard matching
result = matcher.match(["query1", "query2"])

# Matching with metadata
result = matcher.match(
    ["query1", "query2"],
    return_metadata=True,
)

# Access metadata
result.predictions     # Predicted classes
result.confidences     # Confidence scores
result.embeddings      # Text embeddings
result.metadata        # Additional metadata
```

## Data Models

### MatchResultWithMetadata

```python
@dataclass
class MatchResultWithMetadata:
    predictions: List[str]      # Predicted class IDs
    confidences: np.ndarray     # Confidence scores
    embeddings: np.ndarray      # Text embeddings
    scores: Optional[np.ndarray]  # Raw similarity scores
    metadata: Optional[Dict]    # Additional metadata
```

### NovelSampleMetadata

```python
class NovelSampleMetadata:
    text: str                   # Original text
    index: int                  # Index in query list
    confidence: float           # Classification confidence
    predicted_class: str        # Predicted class
    embedding_distance: Optional[float]  # Distance to centroid
    cluster_id: Optional[int]   # Cluster assignment
    signals: Dict[str, bool]    # Detection signals
```

### NovelClassDiscoveryReport

```python
class NovelClassDiscoveryReport:
    discovery_id: str
    timestamp: datetime
    matcher_config: Dict
    detection_config: Dict
    novel_sample_report: NovelSampleReport
    class_proposals: Optional[NovelClassAnalysis]
    metadata: Dict
    output_file: Optional[str]
```

## Storage

### Saving Reports

Reports are automatically saved when `auto_save=True`:

```python
novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    auto_save=True,
    output_dir="./proposals",
)

report = await novel_matcher.discover_novel_classes(...)
# Automatically saved to ./proposals/discovery_YYYYMMDD-HHMMSS.yaml
```

### Loading Reports

```python
from semanticmatcher.novelty.storage import load_proposals, list_proposals

# List all saved discoveries
proposals = list_proposals("./proposals")
for proposal in proposals:
    print(f"{proposal['filename']}: {proposal['timestamp']}")

# Load a specific report
report = load_proposals("./proposals/discovery_20250317-143000.yaml")
```

### Exporting Summaries

```python
from semanticmatcher.novelty.storage import export_summary

export_summary(
    report=report,
    output_path="./summary.md",
    format="markdown",
)
```

## Detection Strategies

### 1. Confidence Thresholding

Flags samples with classification confidence below threshold.

```python
config = DetectionConfig(
    strategies=[DetectionStrategy.CONFIDENCE],
    confidence_threshold=0.7,
)
```

### 2. ANN kNN Distance Analysis

Flags samples whose nearest known neighbors still look too far away or weakly supported.

```python
config = DetectionConfig(
    strategies=[DetectionStrategy.KNN_DISTANCE],
    knn_k=5,
    knn_distance_threshold=0.55,
    ann_backend="hnswlib",
)
```

### 3. Density-Based Clustering

Uses HDBSCAN to find dense clusters of novel samples.

```python
config = DetectionConfig(
    strategies=[DetectionStrategy.CLUSTERING],
    min_cluster_size=5,
)
```

### Signal Combination

Combine signals from multiple strategies:

- **`weighted`**: Blend uncertainty, kNN novelty, and validated cluster support (default)
- **`and` / `intersection`**: Flag sample only if ALL strategies flag it
- **`or` / `union`**: Flag sample if ANY strategy flags it (most sensitive)
- **`voting`**: Flag sample if MAJORITY of strategies flag it (balanced)

```python
config = DetectionConfig(
    strategies=[...],
    combine_method="weighted",  # or "and", "or", "voting"
)
```

## Performance Considerations

### ANN Backend Selection

- **HNSWlib**: Best for medium-scale data (< 1M embeddings), faster queries
- **FAISS**: Better for large-scale data (> 1M embeddings), more memory efficient

```python
config = DetectionConfig(
    ann_backend="hnswlib",  # or "faiss"
)
```

### Batch Processing

Process multiple query lists efficiently:

```python
reports = novel_matcher.batch_discover(
    queries_batch=[
        ["query1", "query2"],
        ["query3", "query4"],
    ],
    existing_classes=existing_classes,
)
```

### Memory Usage

ANN indices require memory proportional to number of embeddings:
- HNSWlib: ~ 1GB per 100K embeddings (384-dim)
- FAISS: ~ 500MB per 100K embeddings (384-dim)

## Examples

### Scientific Research Domain

```python
# Known classes
existing_classes = ["physics", "chemistry", "biology", "computer_science"]

# Queries with potential novel classes
queries = [
    "quantum biology applications",      # Novel: Quantum Biology
    "bioinformatics algorithms",         # Novel: Computational Biology
    "machine learning for drug discovery",  # Novel: AI Drug Discovery
]

novel_matcher = NovelEntityMatcher(matcher=matcher)
report = await novel_matcher.discover_novel_classes(
    queries=queries,
    existing_classes=existing_classes,
    context="Biomedical research",
)
```

### E-commerce Product Categorization

```python
existing_classes = ["electronics", "clothing", "home", "sports"]

queries = [
    "smart home fitness equipment",  # Novel: Connected Fitness
    "wearable health monitors",      # Novel: Health Wearables
    "eco-friendly phone cases",      # Novel: Sustainable Accessories
]

report = await novel_matcher.discover_novel_classes(
    queries=queries,
    existing_classes=existing_classes,
)
```

## Troubleshooting

### No Novel Samples Detected

- Lower `confidence_threshold`
- Lower `distance_threshold`
- Reduce `min_cluster_size`
- Try different `combine_method`

### Too Many False Positives

- Raise `confidence_threshold`
- Raise `distance_threshold`
- Use `combine_method="intersection"`
- Increase `min_cluster_size`

### LLM Proposals Failing

- Check API keys are set correctly
- Verify network connectivity
- Try fallback models
- Check litellm installation

### Memory Issues

- Switch to FAISS backend
- Process in smaller batches
- Reduce embedding dimension
- Use incremental index updates

## Best Practices

1. **Start with default settings**: use the default `DetectionConfig` first, then tune uncertainty and kNN thresholds based on false positives vs misses
2. **Use domain context**: Provide context parameter for better LLM proposals
3. **Review proposals**: Always validate LLM suggestions manually
4. **Iterate**: Use discovered classes to retrain the underlying `Matcher` and improve future novelty decisions
5. **Monitor signals**: Check which strategies are flagging samples
6. **Save experiments**: Keep records of different parameter combinations

## Advanced Usage

### Custom ANN Index Configuration

```python
from semanticmatcher.novelty.detector import NoveltyDetector

detector = NoveltyDetector(
    config=DetectionConfig(
        ann_config={
            "backend": "hnswlib",
            "hnswlib": {
                "ef_construction": 200,  # Higher = better quality
                "M": 16,                 # Higher = better quality
            },
        },
    ),
)
```

### Incremental Index Updates

```python
from semanticmatcher.novelty.ann_index import ANNIndex

index = ANNIndex(dim=384, backend="hnswlib")
index.add_vectors(batch1)
index.add_vectors(batch2)  # Incremental update
```

### Custom LLM Prompts

```python
from semanticmatcher.novelty.llm_proposer import LLMClassProposer

proposer = LLMClassProposer(primary_model="custom-model")
prompt = proposer._build_proposal_prompt(
    novel_samples=samples,
    existing_classes=classes,
    context=context,
)
# Modify prompt as needed
```

## References

- [HNSWlib Documentation](https://github.com/nmslib/hnswlib)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [litellm Documentation](https://litellm.ai/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
