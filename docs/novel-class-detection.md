# Novel Class Detection

`NovelEntityMatcher` is the supported orchestration API for novelty-aware matching and class discovery.

## Main Flow

```python
from semanticmatcher import Matcher, NovelEntityMatcher
from semanticmatcher.novelty import DetectionConfig
from semanticmatcher.novelty.config.strategies import ConfidenceConfig, KNNConfig

entities = [
    {"id": "physics", "name": "Physics"},
    {"id": "cs", "name": "Computer Science"},
]

matcher = Matcher(entities=entities, model="minilm", threshold=0.6)
matcher.fit(
    texts=["quantum mechanics", "neural networks"],
    labels=["physics", "cs"],
)

novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    detection_config=DetectionConfig(
        strategies=["confidence", "knn_distance"],
        confidence=ConfidenceConfig(threshold=0.65),
        knn_distance=KNNConfig(distance_threshold=0.45),
    ),
    auto_save=False,
)
```

```python
report = await novel_matcher.discover_novel_classes(
    queries=["quantum biology", "new interdisciplinary topic"],
    existing_classes=["physics", "cs"],
    run_llm_proposal=False,
)
```

## Lower-Level Components

- `semanticmatcher.novelty.core.detector.NoveltyDetector`: modular detector used by `NovelEntityMatcher`
- `semanticmatcher.novelty.proposal.llm.LLMClassProposer`: LLM-backed naming and summarization
- `semanticmatcher.novelty.storage.ANNIndex`: ANN search index used by distance-based strategies
- `semanticmatcher.novelty.storage.save_proposals` / `load_proposals`: persistence helpers for discovery reports

## Reports

Discovery returns `NovelClassDiscoveryReport`, which contains:

- `novel_sample_report.novel_samples`: flagged samples with confidence, novelty score, signals, and per-sample metrics
- `class_proposals`: optional LLM-generated class names and justifications
- `metadata`: counts and output paths for saved artifacts

## Example

Run the maintained end-to-end example with:

```bash
uv run python examples/novel_discovery_example.py
```
