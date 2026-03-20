"""CLI for HuggingFace benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .runner import BenchmarkRunner
from .registry import DATASET_REGISTRY, get_datasets_by_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_run_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("run", help="Run benchmarks")
    parser.add_argument(
        "--task",
        choices=[
            "all",
            "entity_resolution",
            "er",
            "classification",
            "novelty",
            "processed",
        ],
        default="all",
        help="Which benchmark task to run (er=entity_resolution)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to run (default: all available)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["potion-8m"],
        help="Embedding models to benchmark",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zero-shot"],
        help="Matcher modes to test",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Thresholds to sweep for entity resolution",
    )
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.2, 0.3, 0.4, 0.5],
        help="Confidence thresholds for novelty detection",
    )
    parser.add_argument(
        "--class-counts",
        nargs="+",
        type=int,
        default=[4, 10, 28],
        help="Class counts to test for classification scaling",
    )
    parser.add_argument(
        "--ood-ratio",
        type=float,
        default=0.2,
        help="Ratio of classes to hold out as OOD for novelty detection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for results JSON",
    )
    return parser


def add_load_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("load", help="Load/download datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to load (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    return parser


def add_list_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("list", help="List available datasets")
    parser.add_argument(
        "--task",
        choices=["all", "entity_resolution", "classification", "novelty"],
        default="all",
        help="Filter by task type",
    )
    return parser


def add_clear_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("clear", help="Clear cached datasets")
    parser.add_argument(
        "--dataset",
        help="Specific dataset to clear (default: all)",
    )
    return parser


def add_sweep_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("sweep", help="Run parameter sweep")
    parser.add_argument(
        "--task",
        choices=["er", "clf", "novelty"],
        required=True,
        help="Task type for sweep",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to use for sweep",
    )
    parser.add_argument(
        "--param",
        choices=["threshold", "k", "distance"],
        required=True,
        help="Parameter to sweep",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Values to sweep",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novelentitymatcher-bench",
        description="HuggingFace benchmark runner for novel_entity_matcher",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_run_parser(subparsers)
    add_load_parser(subparsers)
    add_list_parser(subparsers)
    add_clear_parser(subparsers)
    add_sweep_parser(subparsers)

    return parser


def list_datasets(task: str = "all") -> None:
    if task == "all":
        for name, config in DATASET_REGISTRY.items():
            print(f"{name}: {config.hf_path} ({config.task_type})")
    else:
        task_map = {
            "entity_resolution": "entity_matching",
            "classification": "classification",
            "novelty": "novelty",
        }
        task_type = task_map.get(task, task)
        datasets = get_datasets_by_task(task_type)
        for name, config in datasets.items():
            print(f"{name}: {config.hf_path}")


def load_datasets(
    datasets: list[str] | None = None,
    force: bool = False,
) -> None:
    runner = BenchmarkRunner()
    results = runner.load_all(datasets=datasets, force_redownload=force)
    for name, data in results.items():
        if "error" in data:
            print(f"Failed to load {name}: {data['error']}")
        else:
            print(
                f"Loaded {name}: {data.get('metadata', {}).get('num_rows', 'unknown')} rows"
            )


def run_benchmarks(
    task: str = "all",
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    modes: list[str] | None = None,
    thresholds: list[float] | None = None,
    class_counts: list[int] | None = None,
    ood_ratio: float = 0.2,
    output: Path | None = None,
    confidence_thresholds: list[float] | None = None,
) -> None:
    runner = BenchmarkRunner()
    results_to_save = None

    if task == "all":
        results = runner.run_all(
            embedding_models=models,
            modes=modes,
            thresholds=thresholds,
            class_counts=class_counts,
            ood_ratio=ood_ratio,
        )
        results_to_save = results
        print("\n=== Entity Resolution Results ===")
        for model_results in results["entity_resolution"]:
            for r in model_results:
                print(f"  {r.get('dataset')}: F1={r.get('f1', 0):.4f}")

        print("\n=== Classification Results ===")
        for model_results in results["classification"]:
            for r in model_results:
                print(f"  {r.get('dataset')}: Accuracy={r.get('accuracy', 0):.4f}")

        print("\n=== Novelty Detection Results ===")
        for model_results in results["novelty"]:
            for r in model_results:
                print(f"  {r.get('dataset')}: AUROC={r.get('auroc', 0):.4f}")

    elif task in ("entity_resolution", "er"):
        df = runner.run_entity_resolution_benchmark(
            datasets=datasets,
            model=models[0] if models else "all-MiniLM-L6-v2",
            thresholds=thresholds,
        )
        results_to_save = {
            "metadata": {"task": task},
            "entity_resolution": df.to_dict(orient="records"),
        }
        print(df.to_string(index=False))

    elif task == "classification":
        df = runner.run_classification(
            datasets=datasets,
            model=models[0] if models else "potion-8m",
            class_counts=class_counts,
        )
        results_to_save = {
            "metadata": {"task": task},
            "classification": df.to_dict(orient="records"),
        }
        print(df.to_string(index=False))

    elif task == "novelty":
        df = runner.run_novelty(
            datasets=datasets,
            model=models[0] if models else "potion-8m",
            ood_ratio=ood_ratio,
        )
        results_to_save = {
            "metadata": {"task": task},
            "novelty": df.to_dict(orient="records"),
        }
        print(df.to_string(index=False))

    elif task == "processed":
        df = runner.run_novelty_on_processed(
            datasets=datasets,
            model=models[0] if models else "potion-8m",
            confidence_thresholds=confidence_thresholds
            or thresholds
            or [0.2, 0.3, 0.4, 0.5],
        )
        results_to_save = {
            "metadata": {"task": task},
            "processed": df.to_dict(orient="records"),
        }
        print(df.to_string(index=False))

    if output:
        runner.save_results(
            results_to_save or {"metadata": {"task": task}}, str(output)
        )
        print(f"\nResults saved to {output}")


def clear_cache(dataset: str | None = None) -> None:
    runner = BenchmarkRunner()
    runner.loader.clear_cache(dataset)
    print("Cache cleared" if dataset else "All caches cleared")


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.command == "run":
            run_benchmarks(
                task=args.task,
                datasets=args.datasets,
                models=args.models,
                modes=args.modes,
                thresholds=args.thresholds,
                class_counts=args.class_counts,
                ood_ratio=args.ood_ratio,
                output=args.output,
                confidence_thresholds=getattr(args, "confidence_thresholds", None),
            )

        elif args.command == "load":
            load_datasets(
                datasets=args.datasets,
                force=args.force,
            )

        elif args.command == "list":
            list_datasets(task=args.task)

        elif args.command == "clear":
            clear_cache(dataset=args.dataset)

        elif args.command == "sweep":
            print(f"Sweep command: {args.task} {args.dataset} {args.param}")
            print("Note: Detailed sweep functionality coming soon")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
