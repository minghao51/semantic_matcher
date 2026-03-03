# Hierarchical Entity Matching Architecture

## Overview

The `HierarchicalMatcher` enables entity matching that considers hierarchical relationships between entities. This allows matching at multiple granularity levels and supports multi-parent hierarchies.

## Architecture

### Components

1. **HierarchyIndex** - Graph-based representation using NetworkX
   - Directed acyclic graph (DAG) structure
   - Supports multi-parent relationships
   - Cached ancestor/descendant queries

2. **HierarchicalScoring** - Depth-aware confidence calculation
   - Combines semantic similarity with hierarchical boost
   - Applies depth penalties for distant relationships
   - Configurable alpha/beta parameters

3. **HierarchicalMatcher** - User-facing API
   - Composes EmbeddingMatcher for semantic similarity
   - Provides flexible granularity matching
   - Hierarchy exploration methods

### Scoring Formula

```
final_score = (
    semantic_similarity * α +
    hierarchical_boost * β
) * depth_penalty

Where:
- semantic_similarity: Cosine similarity (0-1)
- hierarchical_boost: 0.2-0.5 based on relationship type
- depth_penalty: 1.0 (self), 0.9 (parent), 0.75 (grandparent), etc.
- α, β: Tunable weights (default α=0.7, β=0.3)
```

### Data Model

Hierarchical entities include a `hierarchy` key:

```python
{
    "id": "DE",
    "name": "Germany",
    "hierarchy": {
        "parents": ["EU", "Europe"],           # Multi-parent
        "children": ["DE-BY", "DE-BW"],        # Children
        "level": 2,                            # Hierarchy depth
        "weights": {"EU": 1.0, "Europe": 0.8}  # Relationship strength
    }
}
```

## Performance

- **Query latency:** ~50-100ms per query (depends on hierarchy size)
- **Index build time:** ~1-5 seconds for 10K entities
- **Memory overhead:** ~2-3x base embedding storage (graph + cache)

## Use Cases

1. **Geographic hierarchies** - Countries, regions, cities
2. **Product taxonomies** - Categories, subcategories, SKUs
3. **Organizational structures** - Companies, departments, teams
4. **Knowledge graphs** - Concepts, sub-concepts, instances

## Limitations

- Requires hierarchy metadata (not auto-discovered)
- Static hierarchy (no dynamic updates without rebuild)
- Linear scaling with hierarchy depth (O(depth) for ancestor/descendant queries)

## Future Enhancements

- Dynamic hierarchy updates
- Hierarchical HybridMatcher integration
- Graph neural network embeddings
- Matryoshka embeddings for faster search
