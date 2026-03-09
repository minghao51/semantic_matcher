"""Benchmark utilities for comparing retrieval and trained matching models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..config import (
    get_embedding_model_aliases,
    get_model_spec,
    get_training_model_aliases,
)
from ..core.matcher import EmbeddingMatcher, Matcher

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency

    def tqdm(iterable, **_kwargs):
        return iterable


PROCESSED_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def _parse_aliases(raw_aliases: str) -> List[str]:
    if not raw_aliases:
        return []
    return [alias.strip() for alias in raw_aliases.split("|") if alias.strip()]


def _dataset_section_name(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _row_to_entity(row: Dict[str, str]) -> Dict[str, Any]:
    aliases = _parse_aliases(row.get("aliases", ""))
    entity = {
        "id": row["id"],
        "name": row["name"],
    }
    if aliases:
        entity["aliases"] = aliases
    if row.get("type"):
        entity["type"] = row["type"]
    return entity


def iter_processed_dataset_paths(
    processed_dir: Path = PROCESSED_DATA_DIR,
    sections: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Return benchmarkable processed dataset CSVs."""
    selected = {section for section in sections} if sections else None
    paths = sorted(processed_dir.glob("*/*.csv"))
    if selected is None:
        return paths
    return [path for path in paths if _dataset_section_name(path) in selected]


def load_processed_sections(
    processed_dir: Path = PROCESSED_DATA_DIR,
    sections: Optional[Iterable[str]] = None,
    max_entities_per_section: int = 200,
    max_queries_per_section: int = 50,
) -> List[Dict[str, Any]]:
    """Load processed CSV datasets as benchmark sections."""
    loaded_sections: List[Dict[str, Any]] = []

    for path in iter_processed_dataset_paths(
        processed_dir=processed_dir, sections=sections
    ):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        entities = []
        queries = []
        accuracy_pairs = []
        training_data = []
        evaluation_pairs = []

        for row in rows[:max_entities_per_section]:
            entity = _row_to_entity(row)
            entities.append(entity)

            texts = [entity["name"], *entity.get("aliases", []), entity["id"]]
            unique_texts = []
            for text in texts:
                if text and text not in unique_texts:
                    unique_texts.append(text)

            if len(queries) < max_queries_per_section:
                query = unique_texts[1] if len(unique_texts) > 1 else unique_texts[0]
                queries.append(query)
                accuracy_pairs.append({"query": query, "expected_id": entity["id"]})

            for text in unique_texts[:3]:
                training_data.append({"text": text, "label": entity["id"]})

            evaluation_query = (
                unique_texts[1] if len(unique_texts) > 1 else unique_texts[0]
            )
            evaluation_pairs.append(
                {"query": evaluation_query, "expected_id": entity["id"]}
            )

        if not entities or not queries:
            continue

        loaded_sections.append(
            {
                "section": _dataset_section_name(path),
                "path": path,
                "entities": entities,
                "queries": queries,
                "accuracy_pairs": accuracy_pairs,
                "training_data": training_data[: max_queries_per_section * 3],
                "evaluation_pairs": evaluation_pairs[:max_queries_per_section],
            }
        )

    return loaded_sections


def build_embedding_benchmark_dataset(
    num_entities: int = 200,
    num_queries: int = 50,
) -> Dict[str, Any]:
    """Backward-compatible single-section loader."""
    section = load_processed_sections(
        max_entities_per_section=num_entities,
        max_queries_per_section=num_queries,
    )[0]
    return {
        "entities": section["entities"],
        "queries": section["queries"],
        "accuracy_pairs": section["accuracy_pairs"],
    }


def build_trained_benchmark_dataset() -> Dict[str, Any]:
    """Backward-compatible single-section loader for trained benchmarks."""
    section = load_processed_sections(
        max_entities_per_section=40, max_queries_per_section=20
    )[0]
    return {
        "entities": section["entities"],
        "training_data": section["training_data"],
        "evaluation_pairs": section["evaluation_pairs"],
        "queries": [pair["query"] for pair in section["evaluation_pairs"]],
    }


def _top_level_match_id(result: Any) -> Optional[str]:
    if isinstance(result, dict):
        return result.get("id")
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("id")
    return None


def benchmark_accuracy(
    matcher: Any,
    test_pairs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Benchmark top-1 accuracy on labeled test pairs."""
    correct = 0
    scores = []

    for pair in test_pairs:
        result = matcher.match(pair["query"])
        if _top_level_match_id(result) == pair["expected_id"]:
            correct += 1
        if isinstance(result, dict):
            scores.append(float(result.get("score", 0.0)))

    return {
        "accuracy": correct / len(test_pairs) if test_pairs else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "total_pairs": len(test_pairs),
    }


def benchmark_latency(
    matcher: Any,
    queries: List[str],
    iterations: int = 5,
    warmup_iterations: int = 1,
) -> Dict[str, float]:
    """Measure single-query latency statistics."""
    for _ in range(warmup_iterations):
        for query in queries:
            matcher.match(query)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        for query in queries:
            matcher.match(query)
        elapsed = time.perf_counter() - start
        timings.append(elapsed / len(queries))

    timings_sorted = sorted(timings)
    return {
        "avg_time": mean(timings),
        "min_time": min(timings),
        "max_time": max(timings),
        "p50_time": timings_sorted[len(timings_sorted) // 2],
        "p95_time": timings_sorted[
            min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.95))
        ],
        "p99_time": timings_sorted[
            min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.99))
        ],
        "total_time": sum(timings) * len(queries),
    }


def compare_models(
    entities: List[Dict[str, Any]],
    queries: List[str],
    model_names: List[str],
    num_iterations: int = 3,
) -> pd.DataFrame:
    """Backward-compatible wrapper for retrieval benchmarking."""
    accuracy_pairs = [
        {"query": query, "expected_id": entities[index]["id"]}
        for index, query in enumerate(queries[: len(entities)])
    ]
    return benchmark_embedding_models(
        entities=entities,
        queries=queries,
        accuracy_pairs=accuracy_pairs,
        model_names=model_names,
        iterations=num_iterations,
    )


def benchmark_embedding_models(
    entities: Optional[List[Dict[str, Any]]] = None,
    queries: Optional[List[str]] = None,
    accuracy_pairs: Optional[List[Dict[str, Any]]] = None,
    model_names: Optional[List[str]] = None,
    iterations: int = 3,
    batch_size: Optional[int] = None,
    sections_data: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Benchmark retrieval models across one or more processed-data sections."""
    model_names = model_names or get_embedding_model_aliases()
    if sections_data is None:
        if entities is None or queries is None or accuracy_pairs is None:
            sections_data = load_processed_sections()
        else:
            sections_data = [
                {
                    "section": "custom",
                    "entities": entities,
                    "queries": queries,
                    "accuracy_pairs": accuracy_pairs,
                }
            ]

    records: List[Dict[str, Any]] = []

    for section_data in sections_data:
        section_name = section_data["section"]
        section_entities = section_data["entities"]
        section_queries = section_data["queries"]
        section_accuracy = section_data["accuracy_pairs"]

        for alias in tqdm(model_names, desc=f"Embedding benchmarks [{section_name}]"):
            spec = get_model_spec(alias) or {}
            backend = spec.get("backend", "sentence-transformers")

            try:
                matcher = EmbeddingMatcher(
                    section_entities, model_name=alias, threshold=0.0
                )
                build_start = time.perf_counter()
                matcher.build_index(batch_size=batch_size)
                build_time = time.perf_counter() - build_start

                cold_start = time.perf_counter()
                matcher.match(section_queries[0], batch_size=batch_size)
                cold_query_time = time.perf_counter() - cold_start

                latency = benchmark_latency(
                    matcher, section_queries, iterations=iterations, warmup_iterations=1
                )
                accuracy = benchmark_accuracy(matcher, section_accuracy)

                bulk_times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    matcher.match(section_queries, batch_size=batch_size)
                    bulk_times.append(time.perf_counter() - start)
                avg_bulk_time = mean(bulk_times)

                records.append(
                    {
                        "track": "embedding",
                        "section": section_name,
                        "model": alias,
                        "resolved_model": matcher.model_name,
                        "backend": backend,
                        "status": "ok",
                        "build_time": build_time,
                        "cold_query_time": cold_query_time,
                        "avg_latency": latency["avg_time"],
                        "p95_latency": latency["p95_time"],
                        "throughput_qps": len(section_queries) / avg_bulk_time
                        if avg_bulk_time
                        else 0.0,
                        "bulk_time": avg_bulk_time,
                        "accuracy": accuracy["accuracy"],
                        "avg_score": accuracy["avg_score"],
                        "skip_reason": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised via monkeypatch
                records.append(
                    {
                        "track": "embedding",
                        "section": section_name,
                        "model": alias,
                        "resolved_model": spec.get("name", alias),
                        "backend": backend,
                        "status": "skipped",
                        "build_time": None,
                        "cold_query_time": None,
                        "avg_latency": None,
                        "p95_latency": None,
                        "throughput_qps": None,
                        "bulk_time": None,
                        "accuracy": None,
                        "avg_score": None,
                        "skip_reason": str(exc),
                    }
                )

        section_baseline = next(
            (
                row["throughput_qps"]
                for row in records
                if row["section"] == section_name
                and row["status"] == "ok"
                and row["model"] == "minilm"
            ),
            None,
        )
        for row in records:
            if row["section"] != section_name:
                continue
            if section_baseline and row["status"] == "ok" and row["throughput_qps"]:
                row["speedup_vs_minilm"] = row["throughput_qps"] / section_baseline
            else:
                row["speedup_vs_minilm"] = None

    return pd.DataFrame(records)


def benchmark_trained_modes(
    entities: Optional[List[Dict[str, Any]]] = None,
    training_data: Optional[List[Dict[str, Any]]] = None,
    evaluation_pairs: Optional[List[Dict[str, Any]]] = None,
    queries: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    modes: Optional[Iterable[str]] = None,
    num_epochs: int = 1,
    sections_data: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Benchmark SetFit-backed matcher modes across one or more sections."""
    model_names = model_names or get_training_model_aliases()
    modes = list(modes or ("head-only", "full"))
    if sections_data is None:
        if (
            entities is None
            or training_data is None
            or evaluation_pairs is None
            or queries is None
        ):
            sections_data = load_processed_sections(
                max_entities_per_section=40, max_queries_per_section=20
            )
        else:
            sections_data = [
                {
                    "section": "custom",
                    "entities": entities,
                    "training_data": training_data,
                    "evaluation_pairs": evaluation_pairs,
                    "queries": queries,
                }
            ]

    records: List[Dict[str, Any]] = []

    for section_data in sections_data:
        section_name = section_data["section"]
        section_entities = section_data["entities"]
        section_training = section_data["training_data"]
        section_evaluation = section_data["evaluation_pairs"]
        section_queries = section_data["queries"]

        for alias in tqdm(model_names, desc=f"Training benchmarks [{section_name}]"):
            for mode in modes:
                try:
                    matcher = Matcher(
                        entities=section_entities,
                        model=alias,
                        mode=mode,
                        threshold=0.0,
                    )
                    train_start = time.perf_counter()
                    matcher.fit(
                        section_training,
                        num_epochs=num_epochs,
                        show_progress=False,
                    )
                    train_time = time.perf_counter() - train_start

                    latency = benchmark_latency(
                        matcher, section_queries, iterations=3, warmup_iterations=1
                    )
                    accuracy = benchmark_accuracy(matcher, section_evaluation)

                    bulk_times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        matcher.match(section_queries)
                        bulk_times.append(time.perf_counter() - start)
                    avg_bulk_time = mean(bulk_times)

                    records.append(
                        {
                            "track": "trained",
                            "section": section_name,
                            "mode": mode,
                            "model": alias,
                            "resolved_model": matcher.entity_matcher.model_name,
                            "status": "ok",
                            "training_time": train_time,
                            "avg_latency": latency["avg_time"],
                            "p95_latency": latency["p95_time"],
                            "throughput_qps": len(section_queries) / avg_bulk_time
                            if avg_bulk_time
                            else 0.0,
                            "accuracy": accuracy["accuracy"],
                            "avg_score": accuracy["avg_score"],
                            "skip_reason": "",
                        }
                    )
                except Exception as exc:  # pragma: no cover - exercised via monkeypatch
                    spec = get_model_spec(alias)
                    records.append(
                        {
                            "track": "trained",
                            "section": section_name,
                            "mode": mode,
                            "model": alias,
                            "resolved_model": spec.get("name", alias)
                            if spec
                            else alias,
                            "status": "skipped",
                            "training_time": None,
                            "avg_latency": None,
                            "p95_latency": None,
                            "throughput_qps": None,
                            "accuracy": None,
                            "avg_score": None,
                            "skip_reason": str(exc),
                        }
                    )

    return pd.DataFrame(records)


def save_benchmark_report(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Write benchmark results as JSON or CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(results.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    elif path.suffix.lower() == ".csv":
        results.to_csv(path, index=False)
    else:
        raise ValueError("output_path must end with .json or .csv")

    return path


def format_benchmark_summary(results: pd.DataFrame) -> str:
    """Return a readable benchmark summary grouped by track and section."""
    if results.empty:
        return "No benchmark results collected."

    lines = ["BENCHMARK RESULTS"]
    for track in results["track"].dropna().unique():
        track_subset = results[results["track"] == track]
        lines.append("")
        lines.append(f"[{track}]")
        for section in track_subset["section"].dropna().unique():
            section_subset = track_subset[track_subset["section"] == section]
            lines.append(f"<section: {section}>")
            if (
                "mode" in section_subset.columns
                and section_subset["mode"].notna().any()
            ):
                columns = [
                    col
                    for col in [
                        "mode",
                        "model",
                        "status",
                        "throughput_qps",
                        "accuracy",
                        "skip_reason",
                    ]
                    if col in section_subset.columns
                ]
            else:
                columns = [
                    col
                    for col in [
                        "model",
                        "backend",
                        "status",
                        "throughput_qps",
                        "accuracy",
                        "speedup_vs_minilm",
                        "skip_reason",
                    ]
                    if col in section_subset.columns
                ]
            lines.append(section_subset[columns].to_string(index=False))
            lines.append("")
    return "\n".join(line for line in lines if line is not None).rstrip()


def print_benchmark_report(results: pd.DataFrame):
    """Print a formatted benchmark report."""
    print(format_benchmark_summary(results))


def run_benchmark_suite(
    track: str = "all",
    embedding_models: Optional[List[str]] = None,
    training_models: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
    sections: Optional[List[str]] = None,
    max_entities_per_section: int = 200,
    max_queries_per_section: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Run the configured benchmark suite."""
    suite: Dict[str, pd.DataFrame] = {}
    loaded_sections = load_processed_sections(
        sections=sections,
        max_entities_per_section=max_entities_per_section,
        max_queries_per_section=max_queries_per_section,
    )

    if track in ("embeddings", "all"):
        suite["embeddings"] = benchmark_embedding_models(
            model_names=embedding_models,
            sections_data=loaded_sections,
        )

    if track in ("trained", "all"):
        suite["trained"] = benchmark_trained_modes(
            model_names=training_models,
            sections_data=loaded_sections,
        )

    if output_path:
        combined = (
            pd.concat(suite.values(), ignore_index=True) if suite else pd.DataFrame()
        )
        save_benchmark_report(combined, output_path)

    return suite


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark semantic matcher models")
    parser.add_argument(
        "--track",
        choices=("embeddings", "trained", "all"),
        default="all",
        help="Benchmark embeddings, trained modes, or both",
    )
    parser.add_argument(
        "--output",
        help="Optional output path (.json or .csv) for the combined results",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="*",
        default=None,
        help="Optional subset of embedding aliases to benchmark",
    )
    parser.add_argument(
        "--training-models",
        nargs="*",
        default=None,
        help="Optional subset of training-compatible aliases to benchmark",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Optional subset of processed-data sections such as languages/languages",
    )
    parser.add_argument(
        "--max-entities-per-section",
        type=int,
        default=200,
        help="Maximum entities loaded from each processed dataset section",
    )
    parser.add_argument(
        "--max-queries-per-section",
        type=int,
        default=50,
        help="Maximum benchmark queries generated per processed dataset section",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for benchmark execution."""
    args = _parse_args(argv)
    suite = run_benchmark_suite(
        track=args.track,
        embedding_models=args.embedding_models,
        training_models=args.training_models,
        output_path=args.output,
        sections=args.sections,
        max_entities_per_section=args.max_entities_per_section,
        max_queries_per_section=args.max_queries_per_section,
    )

    if args.track in ("embeddings", "all") and "embeddings" in suite:
        print_benchmark_report(suite["embeddings"])
    if args.track in ("trained", "all") and "trained" in suite:
        print("")
        print_benchmark_report(suite["trained"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
