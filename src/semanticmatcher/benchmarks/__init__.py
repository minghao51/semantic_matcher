"""HuggingFace benchmark module for semantic_matcher.

This module provides tools for benchmarking semantic_matcher on external
HuggingFace datasets for entity resolution, text classification, and
novelty detection tasks.

Usage:
    from semanticmatcher.benchmarks import BenchmarkRunner, DATASET_REGISTRY

    runner = BenchmarkRunner()
    runner.load_all()

    er_results = runner.run_entity_resolution()
    clf_results = runner.run_classification()
    novelty_results = runner.run_novelty()

    all_results = runner.run_all()
"""

from .registry import (
    DATASET_REGISTRY,
    DatasetConfig,
    CacheConfig,
    get_dataset_config,
    get_datasets_by_task,
    get_default_datasets,
)

from .loader import DatasetLoader

from .runner import BenchmarkRunner

from .entity_resolution import (
    EntityResolutionEvaluator,
    sweep_threshold,
    find_optimal_threshold,
    MatchPair,
)

from .classification import (
    ClassificationEvaluator,
    sweep_num_classes,
    evaluate_by_class_count,
    ClassificationSample,
)

from .novelty import (
    NoveltyEvaluator,
    sweep_knn_params,
    find_optimal_knn_params,
    NoveltySample,
)

from .novel_entity_matcher import (
    NovelEntityMatcher,
    NoveltyMatchResult,
    create_novel_entity_matcher,
)

from .base import BaseEvaluator, EvaluationResult

__all__ = [
    "DATASET_REGISTRY",
    "DatasetConfig",
    "CacheConfig",
    "get_dataset_config",
    "get_datasets_by_task",
    "get_default_datasets",
    "DatasetLoader",
    "BenchmarkRunner",
    "BaseEvaluator",
    "EvaluationResult",
    "EntityResolutionEvaluator",
    "sweep_threshold",
    "find_optimal_threshold",
    "MatchPair",
    "ClassificationEvaluator",
    "sweep_num_classes",
    "evaluate_by_class_count",
    "ClassificationSample",
    "NoveltyEvaluator",
    "sweep_knn_params",
    "find_optimal_knn_params",
    "NoveltySample",
    "NovelEntityMatcher",
    "NoveltyMatchResult",
    "create_novel_entity_matcher",
]
