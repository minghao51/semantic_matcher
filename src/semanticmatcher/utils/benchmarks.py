"""Benchmark utilities for comparing retrieval and trained matching models."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..config import (
    get_embedding_model_aliases,
    get_model_spec,
    get_training_model_aliases,
)
from ..core.matcher import EmbeddingMatcher, Matcher
from .preprocessing import clean_text

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


def _row_to_entity(
    row: Dict[str, str],
    alias_counts: Optional[Counter[str]] = None,
) -> Dict[str, Any]:
    aliases = _parse_aliases(row.get("aliases", ""))
    if alias_counts is not None:
        aliases = [
            alias
            for alias in aliases
            if alias_counts.get(alias, 0) == 1 and alias != row["id"]
        ]
    entity = {
        "id": row["id"],
        "name": row["name"],
    }
    if aliases:
        entity["aliases"] = aliases
    if row.get("type"):
        entity["type"] = row["type"]
    return entity


def _dedupe_texts(entity: Dict[str, Any], include_id: bool = True) -> List[str]:
    texts = [entity["name"], *entity.get("aliases", [])]
    if include_id:
        texts.append(entity["id"])
    unique_texts = []
    for text in texts:
        if text and text not in unique_texts:
            unique_texts.append(text)
    return unique_texts


def _normalize_eval_text(text: str) -> str:
    return clean_text(text, lowercase=True, remove_punct=True)


def _remove_parenthetical(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def _first_clause(text: str) -> str:
    return re.split(r"[,;]", text, maxsplit=1)[0]


def _introduce_typo(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    longest_idx = max(range(len(tokens)), key=lambda idx: len(tokens[idx]))
    token = tokens[longest_idx]
    if len(token) < 5:
        return text

    midpoint = len(token) // 2
    typo = token[:midpoint] + token[midpoint + 1 :]
    if typo == token:
        return text

    tokens[longest_idx] = typo
    return " ".join(tokens)


def _generate_holdout_queries(
    source_text: str,
    excluded_texts: Iterable[str],
) -> List[Tuple[str, str]]:
    """Generate labeled non-verbatim evaluation queries from a source text."""
    excluded = {_normalize_eval_text(text) for text in excluded_texts if text}
    candidates: List[Tuple[str, str]] = []
    raw_candidates = [
        (
            "typo",
            _normalize_eval_text(_introduce_typo(_normalize_eval_text(source_text))),
        ),
        (
            "remove_parenthetical",
            _normalize_eval_text(_remove_parenthetical(source_text)),
        ),
        ("ampersand_expanded", _normalize_eval_text(source_text.replace("&", " and "))),
        ("first_clause", _normalize_eval_text(_first_clause(source_text))),
        ("normalized_verbatim", _normalize_eval_text(source_text)),
    ]

    seen_candidates = set()
    for label, candidate in raw_candidates:
        if not candidate or candidate in excluded:
            continue
        if candidate in seen_candidates:
            continue
        candidates.append((label, candidate))
        seen_candidates.add(candidate)

    return candidates


def _build_split_pairs(entity: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build exact-match and holdout query splits for benchmarking."""
    indexed_texts = _dedupe_texts(entity, include_id=False)
    if not indexed_texts:
        return {
            "base": [],
            "train": [],
            "val": [],
            "test": [],
            "typo": [],
            "remove_parenthetical": [],
            "ampersand_expanded": [],
            "first_clause": [],
            "normalized_verbatim": [],
        }

    base_query = indexed_texts[0]
    base_pairs = [{"query": base_query, "expected_id": entity["id"]}]
    train_pairs = [
        {"query": text, "expected_id": entity["id"]}
        for text in indexed_texts[1:3]
        if text != base_query
    ]

    holdout_source = indexed_texts[-1]
    holdout_queries = _generate_holdout_queries(holdout_source, indexed_texts)
    perturbation_pairs = {
        label: [{"query": query, "expected_id": entity["id"]}]
        for label, query in holdout_queries
    }
    val_pairs = (
        [{"query": holdout_queries[0][1], "expected_id": entity["id"]}]
        if len(holdout_queries) >= 1
        else []
    )
    test_pairs = (
        [{"query": holdout_queries[1][1], "expected_id": entity["id"]}]
        if len(holdout_queries) >= 2
        else []
    )

    return {
        "base": base_pairs,
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
        "typo": perturbation_pairs.get("typo", []),
        "remove_parenthetical": perturbation_pairs.get("remove_parenthetical", []),
        "ampersand_expanded": perturbation_pairs.get("ampersand_expanded", []),
        "first_clause": perturbation_pairs.get("first_clause", []),
        "normalized_verbatim": perturbation_pairs.get("normalized_verbatim", []),
    }


def _split_train_eval_texts(entity: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    """Reserve one unseen text for trained-mode evaluation when possible."""
    unique_texts = _dedupe_texts(entity, include_id=False)
    if len(unique_texts) < 2:
        return unique_texts[:1], None
    return unique_texts[:-1], unique_texts[-1]


def _select_primary_queries(split_pairs: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    for split in ("test", "val", "train", "base"):
        pairs = split_pairs.get(split, [])
        if pairs:
            return [pair["query"] for pair in pairs]
    return []


def _split_accuracy_fields(
    matcher: Any,
    split_pairs: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}

    metric_splits = (
        "base",
        "train",
        "val",
        "test",
        "typo",
        "remove_parenthetical",
        "ampersand_expanded",
        "first_clause",
        "normalized_verbatim",
    )
    for split_name in metric_splits:
        metrics = benchmark_accuracy(matcher, split_pairs.get(split_name, []))
        fields[f"{split_name}_accuracy"] = metrics["accuracy"]
        fields[f"{split_name}_avg_score"] = metrics["avg_score"]
        fields[f"{split_name}_total_pairs"] = metrics["total_pairs"]

    for preferred in ("test", "val", "train", "base"):
        if fields[f"{preferred}_total_pairs"] > 0:
            fields["accuracy"] = fields[f"{preferred}_accuracy"]
            fields["avg_score"] = fields[f"{preferred}_avg_score"]
            fields["accuracy_split"] = preferred
            break
    else:
        fields["accuracy"] = 0.0
        fields["avg_score"] = 0.0
        fields["accuracy_split"] = "none"

    return fields


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

        alias_counts: Counter[str] = Counter()
        for row in rows:
            alias_counts.update(_parse_aliases(row.get("aliases", "")))

        entities = []
        queries = []
        accuracy_pairs = []
        training_data = []
        evaluation_pairs = []
        base_pairs = []
        train_pairs = []
        val_pairs = []
        test_pairs = []
        perturbation_pairs = {
            "typo": [],
            "remove_parenthetical": [],
            "ampersand_expanded": [],
            "first_clause": [],
            "normalized_verbatim": [],
        }

        for row in rows[:max_entities_per_section]:
            entity = _row_to_entity(row, alias_counts=alias_counts)
            entities.append(entity)
            split_pairs = _build_split_pairs(entity)

            if len(base_pairs) < max_queries_per_section:
                base_pairs.extend(split_pairs["base"])
            if len(train_pairs) < max_queries_per_section:
                train_pairs.extend(split_pairs["train"])
            if len(val_pairs) < max_queries_per_section:
                val_pairs.extend(split_pairs["val"])
            if len(test_pairs) < max_queries_per_section:
                test_pairs.extend(split_pairs["test"])
            for label in perturbation_pairs:
                if len(perturbation_pairs[label]) < max_queries_per_section:
                    perturbation_pairs[label].extend(split_pairs[label])

            training_texts = [base_query["query"] for base_query in split_pairs["base"]]
            training_texts.extend(pair["query"] for pair in split_pairs["train"])
            for text in training_texts[:3]:
                training_data.append({"text": text, "label": entity["id"]})

        split_map = {
            "base": base_pairs[:max_queries_per_section],
            "train": train_pairs[:max_queries_per_section],
            "val": val_pairs[:max_queries_per_section],
            "test": test_pairs[:max_queries_per_section],
        }
        queries = _select_primary_queries(split_map)
        accuracy_pairs = split_map["base"]
        evaluation_pairs = split_map["test"] or split_map["val"]

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
                "base_pairs": split_map["base"],
                "train_pairs": split_map["train"],
                "val_pairs": split_map["val"],
                "test_pairs": split_map["test"],
                "typo_pairs": perturbation_pairs["typo"][:max_queries_per_section],
                "remove_parenthetical_pairs": perturbation_pairs[
                    "remove_parenthetical"
                ][:max_queries_per_section],
                "ampersand_expanded_pairs": perturbation_pairs["ampersand_expanded"][
                    :max_queries_per_section
                ],
                "first_clause_pairs": perturbation_pairs["first_clause"][
                    :max_queries_per_section
                ],
                "normalized_verbatim_pairs": perturbation_pairs["normalized_verbatim"][
                    :max_queries_per_section
                ],
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
    sections = load_processed_sections(
        max_entities_per_section=40, max_queries_per_section=20
    )
    section = next(
        (
            item
            for item in sections
            if item["training_data"] and item["evaluation_pairs"]
        ),
        None,
    )
    if section is None:
        raise ValueError(
            "No processed sections contain a non-overlapping train/eval split"
        )
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
        split_pairs = {
            "base": section_data.get(
                "base_pairs", section_data.get("accuracy_pairs", [])
            ),
            "train": section_data.get("train_pairs", []),
            "val": section_data.get("val_pairs", []),
            "test": section_data.get("test_pairs", []),
            "typo": section_data.get("typo_pairs", []),
            "remove_parenthetical": section_data.get("remove_parenthetical_pairs", []),
            "ampersand_expanded": section_data.get("ampersand_expanded_pairs", []),
            "first_clause": section_data.get("first_clause_pairs", []),
            "normalized_verbatim": section_data.get("normalized_verbatim_pairs", []),
        }

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
                accuracy_fields = _split_accuracy_fields(matcher, split_pairs)

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
                        **accuracy_fields,
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
                        "accuracy_split": None,
                        "base_accuracy": None,
                        "train_accuracy": None,
                        "val_accuracy": None,
                        "test_accuracy": None,
                        "base_avg_score": None,
                        "train_avg_score": None,
                        "val_avg_score": None,
                        "test_avg_score": None,
                        "base_total_pairs": None,
                        "train_total_pairs": None,
                        "val_total_pairs": None,
                        "test_total_pairs": None,
                        "typo_accuracy": None,
                        "remove_parenthetical_accuracy": None,
                        "ampersand_expanded_accuracy": None,
                        "first_clause_accuracy": None,
                        "normalized_verbatim_accuracy": None,
                        "typo_avg_score": None,
                        "remove_parenthetical_avg_score": None,
                        "ampersand_expanded_avg_score": None,
                        "first_clause_avg_score": None,
                        "normalized_verbatim_avg_score": None,
                        "typo_total_pairs": None,
                        "remove_parenthetical_total_pairs": None,
                        "ampersand_expanded_total_pairs": None,
                        "first_clause_total_pairs": None,
                        "normalized_verbatim_total_pairs": None,
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
        section_queries = section_data["queries"]
        split_pairs = {
            "base": section_data.get("base_pairs", []),
            "train": section_data.get("train_pairs", []),
            "val": section_data.get("val_pairs", []),
            "test": section_data.get(
                "test_pairs", section_data.get("evaluation_pairs", [])
            ),
            "typo": section_data.get("typo_pairs", []),
            "remove_parenthetical": section_data.get("remove_parenthetical_pairs", []),
            "ampersand_expanded": section_data.get("ampersand_expanded_pairs", []),
            "first_clause": section_data.get("first_clause_pairs", []),
            "normalized_verbatim": section_data.get("normalized_verbatim_pairs", []),
        }

        if (
            not section_training
            or not (split_pairs["val"] or split_pairs["test"])
            or not section_queries
        ):
            continue

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
                    accuracy_fields = _split_accuracy_fields(matcher, split_pairs)

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
                            **accuracy_fields,
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
                            "accuracy_split": None,
                            "base_accuracy": None,
                            "train_accuracy": None,
                            "val_accuracy": None,
                            "test_accuracy": None,
                            "base_avg_score": None,
                            "train_avg_score": None,
                            "val_avg_score": None,
                            "test_avg_score": None,
                            "base_total_pairs": None,
                            "train_total_pairs": None,
                            "val_total_pairs": None,
                            "test_total_pairs": None,
                            "typo_accuracy": None,
                            "remove_parenthetical_accuracy": None,
                            "ampersand_expanded_accuracy": None,
                            "first_clause_accuracy": None,
                            "normalized_verbatim_accuracy": None,
                            "typo_avg_score": None,
                            "remove_parenthetical_avg_score": None,
                            "ampersand_expanded_avg_score": None,
                            "first_clause_avg_score": None,
                            "normalized_verbatim_avg_score": None,
                            "typo_total_pairs": None,
                            "remove_parenthetical_total_pairs": None,
                            "ampersand_expanded_total_pairs": None,
                            "first_clause_total_pairs": None,
                            "normalized_verbatim_total_pairs": None,
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
                        "base_accuracy",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
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
                        "base_accuracy",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
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
