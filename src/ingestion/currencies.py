"""Ingestion script for ISO 4217 currency codes."""

import sys
from pathlib import Path
from typing import Any
import csv
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ingestion.base import BaseFetcher


class CurrenciesFetcher(BaseFetcher):
    """Fetch ISO 4217 currency codes with symbols."""

    SOURCE_URL = "https://datahub.io/core/currency-codes/r/codes-all.csv"

    def fetch(self) -> list[dict[str, Any]]:
        """Download currency codes from DataHub."""
        output_path = self.raw_dir / "currencies.csv"

        if not output_path.exists():
            response = requests.get(self.SOURCE_URL, timeout=60)
            response.raise_for_status()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)

        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return data

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format with aliases."""
        entities = []

        seen = set()
        for row in raw_data:
            code = row.get("AlphabeticCode", "").strip()
            if not code or code in seen:
                continue
            seen.add(code)

            entity = row.get("Entity", "").strip()
            currency = row.get("Currency", "").strip()
            numeric = row.get("NumericCode", "").strip()

            name = currency if currency else entity
            if not name:
                continue

            aliases = [code]
            if numeric:
                aliases.append(numeric)

            entities.append({
                "id": code,
                "name": name,
                "aliases": "|".join(aliases),
                "type": "currency"
            })

        return entities


def run():
    """Execute currency data ingestion."""
    base_path = Path(__file__).parent.parent.parent
    raw_dir = base_path / "data" / "raw" / "currencies"
    processed_dir = base_path / "data" / "processed" / "currencies"

    fetcher = CurrenciesFetcher(raw_dir, processed_dir)
    fetcher.run("currencies.csv")


if __name__ == "__main__":
    run()
