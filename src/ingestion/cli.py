"""CLI for running data ingestion scripts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion import (
    run_languages,
    run_currencies,
    run_industries,
    run_timezones,
    run_occupations,
    run_products,
    run_universities,
)


INGESTORS = {
    "languages": run_languages,
    "currencies": run_currencies,
    "industries": run_industries,
    "timezones": run_timezones,
    "occupations": run_occupations,
    "products": run_products,
    "universities": run_universities,
    "all": None,
}


def main():
    parser = argparse.ArgumentParser(description="Ingest external datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=list(INGESTORS.keys()),
        help="Dataset to ingest (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name in INGESTORS:
            print(f"  - {name}")
        return

    if args.dataset == "all":
        print("Running all ingestions...")
        for name, func in INGESTORS.items():
            if name == "all":
                continue
            print(f"\n{'='*50}")
            print(f"Ingesting {name}...")
            print("=" * 50)
            try:
                func()
            except Exception as e:
                print(f"Error ingesting {name}: {e}")
        print("\nAll ingestions complete!")
    else:
        func = INGESTORS.get(args.dataset)
        if func:
            func()
        else:
            print(f"Unknown dataset: {args.dataset}")
            sys.exit(1)


if __name__ == "__main__":
    main()
