"""Base classes for data ingestion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
import csv

PathLike = Union[str, Path]


class BaseFetcher(ABC):
    """Base class for fetching and processing external datasets."""

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Fetch raw data from source."""

    @abstractmethod
    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process raw data into standardized format."""

    def save_csv(self, data: list[dict[str, Any]], filename: str) -> Path:
        """Save data to CSV file."""
        if not data:
            raise ValueError("No data to save")

        output_path = self.processed_dir / filename
        fieldnames = list(data[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        return output_path

    def run(self, output_filename: str) -> Path:
        """Execute full ingestion pipeline."""
        print(f"Fetching {self.__class__.__name__} data...")
        raw_data = self.fetch()

        print(f"Processing {len(raw_data)} records...")
        processed_data = self.process(raw_data)

        print(f"Saving to {output_filename}...")
        output_path = self.save_csv(processed_data, output_filename)

        print(f"Done! Saved {len(processed_data)} records to {output_path}")
        return output_path


def resolve_output_dirs(
    dataset: str,
    raw_dir: Optional[PathLike] = None,
    processed_dir: Optional[PathLike] = None,
) -> tuple[Path, Path]:
    """Resolve per-dataset output directories without relying on repo layout."""
    raw_base = Path(raw_dir) if raw_dir is not None else Path.cwd() / "data" / "raw"
    processed_base = (
        Path(processed_dir)
        if processed_dir is not None
        else Path.cwd() / "data" / "processed"
    )
    return raw_base / dataset, processed_base / dataset
