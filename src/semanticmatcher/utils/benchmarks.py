"""Benchmark utilities for comparing semantic matching performance."""

import time
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

from ..core.matcher import EmbeddingMatcher
from ..config import MODEL_REGISTRY


def compare_models(
    entities: List[Dict[str, Any]],
    queries: List[str],
    model_names: List[str],
    num_iterations: int = 3,
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.

    Args:
        entities: List of entity dictionaries
        queries: List of test queries
        model_names: List of model names/aliases to compare
        num_iterations: Number of times to run each model

    Returns:
        DataFrame with comparison results:
            - model: Model name
            - avg_time: Average time per query (seconds)
            - total_time: Total time for all queries
            - queries_per_second: Throughput metric
    """
    results = []

    for model in tqdm(model_names, desc="Comparing models"):
        # Resolve alias if needed
        full_model_name = MODEL_REGISTRY.get(model, model)

        matcher = EmbeddingMatcher(entities, model_name=full_model_name)
        matcher.build_index()

        times = []
        match_bulk = getattr(matcher, "match_bulk", None)
        for _ in range(num_iterations):
            start = time.time()
            if callable(match_bulk):
                match_bulk(queries)
            else:
                [matcher.match(q) for q in queries]
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        total_time = avg_time
        qps = len(queries) / avg_time if avg_time > 0 else 0

        results.append(
            {
                "model": model,
                "avg_time": avg_time,
                "total_time": total_time,
                "queries_per_second": qps,
            }
        )

    return pd.DataFrame(results)


def benchmark_latency(
    matcher,
    queries: List[str],
    iterations: int = 10,
    warmup_iterations: int = 2,
) -> Dict[str, float]:
    """
    Measure matcher latency over multiple iterations.

    Args:
        matcher: Initialized matcher instance
        queries: List of test queries
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations (not counted)

    Returns:
        Dictionary with latency statistics:
            - avg_time: Average time per query
            - min_time: Minimum time
            - max_time: Maximum time
            - p50_time: 50th percentile
            - p95_time: 95th percentile
            - p99_time: 99th percentile
            - total_time: Total time for all queries
    """
    # Warmup
    for _ in range(warmup_iterations):
        for query in queries:
            matcher.match(query)

    # Benchmark
    timings = []
    for _ in range(iterations):
        start = time.time()
        for query in queries:
            matcher.match(query)
        elapsed = time.time() - start
        timings.append(elapsed)

    timings_sorted = sorted(timings)

    return {
        "avg_time": sum(timings) / len(timings) / len(queries),
        "min_time": min(timings) / len(queries),
        "max_time": max(timings) / len(queries),
        "p50_time": timings_sorted[len(timings) // 2] / len(queries),
        "p95_time": timings_sorted[int(len(timings) * 0.95)] / len(queries),
        "p99_time": timings_sorted[int(len(timings) * 0.99)] / len(queries),
        "total_time": sum(timings),
    }


def benchmark_accuracy(
    matcher,
    test_pairs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Benchmark matching accuracy on labeled test pairs.

    Args:
        matcher: Initialized matcher instance
        test_pairs: List of dicts with 'query', 'expected_id', and optionally 'threshold'

    Returns:
        Dictionary with accuracy metrics:
            - accuracy: Percentage of correct matches
            - avg_score: Average confidence score
            - avg_threshold: Average threshold used
            - total_pairs: Total number of test pairs
    """
    correct = 0
    scores = []

    for pair in test_pairs:
        query = pair["query"]
        expected_id = pair["expected_id"]
        result = matcher.match(query)

        if result and result.get("id") == expected_id:
            correct += 1

        if result:
            scores.append(result.get("score", 0.0))

    return {
        "accuracy": correct / len(test_pairs) if test_pairs else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "total_pairs": len(test_pairs),
    }


def print_benchmark_report(results: pd.DataFrame):
    """Print a formatted benchmark report."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)

    # Find winner (highest queries per second)
    winner_idx = results["queries_per_second"].idxmax()
    print(
        f"\nWinner: {results.loc[winner_idx, 'model']} "
        f"({results.loc[winner_idx, 'queries_per_second']:.2f} qps)"
    )


if __name__ == "__main__":
    # Simple CLI for running benchmarks
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m semanticmatcher.utils.benchmarks <command>")
        print("Commands:")
        print("  compare <num_entities> <num_queries>  - Compare model performance")
        print(
            "  latency <model> <num_queries>         - Benchmark single model latency"
        )
        sys.exit(1)

    command = sys.argv[1]

    if command == "compare":
        num_entities = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        num_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 10

        # Generate sample data
        entities = [
            {
                "id": f"entity_{i}",
                "name": f"Entity {i}",
                "text": f"Description for entity {i}",
            }
            for i in range(num_entities)
        ]
        queries = [f"Entity {i}" for i in range(min(num_queries, num_entities))]

        # Compare models
        models = ["minilm", "mpnet", "bge-base"]
        results = compare_models(entities, queries, models)
        print_benchmark_report(results)

    elif command == "latency":
        model = sys.argv[2] if len(sys.argv) > 2 else "minilm"
        num_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 100

        entities = [
            {
                "id": f"entity_{i}",
                "name": f"Entity {i}",
                "text": f"Description for entity {i}",
            }
            for i in range(1000)
        ]
        queries = [f"Entity {i % 100}" for i in range(num_queries)]

        matcher = EmbeddingMatcher(
            entities, model_name=MODEL_REGISTRY.get(model, model)
        )
        matcher.build_index()

        stats = benchmark_latency(matcher, queries)
        print(f"\nLatency Benchmark for {model}:")
        print(f"  Average: {stats['avg_time'] * 1000:.2f} ms/query")
        print(f"  Min:     {stats['min_time'] * 1000:.2f} ms/query")
        print(f"  Max:     {stats['max_time'] * 1000:.2f} ms/query")
        print(f"  P95:     {stats['p95_time'] * 1000:.2f} ms/query")
        print(f"  P99:     {stats['p99_time'] * 1000:.2f} ms/query")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
