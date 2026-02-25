"""Ingestion script for NAICS/SIC industry codes."""

from typing import Any
import csv
import json
import requests

from .base import BaseFetcher, resolve_output_dirs


class IndustriesFetcher(BaseFetcher):
    """Fetch NAICS and SIC industry classification codes."""

    NAICS_URLS = [
        "https://raw.githubusercontent.com/erickogore/country-code-json/refs/heads/master/industry-codes.json",
        "https://raw.githubusercontent.com/datasets/industry-codes/master/data/industry-codes.csv",
    ]

    FALLBACK_NAICS = [
        {"Code": "11", "Title": "Agriculture, Forestry, Fishing and Hunting"},
        {"Code": "21", "Title": "Mining, Quarrying, and Oil and Gas Extraction"},
        {"Code": "22", "Title": "Utilities"},
        {"Code": "23", "Title": "Construction"},
        {"Code": "31-33", "Title": "Manufacturing"},
        {"Code": "42", "Title": "Wholesale Trade"},
        {"Code": "44-45", "Title": "Retail Trade"},
        {"Code": "48-49", "Title": "Transportation and Warehousing"},
        {"Code": "51", "Title": "Information"},
        {"Code": "52", "Title": "Finance and Insurance"},
        {"Code": "53", "Title": "Real Estate and Rental and Leasing"},
        {"Code": "54", "Title": "Professional, Scientific, and Technical Services"},
        {"Code": "55", "Title": "Management of Companies and Enterprises"},
        {"Code": "56", "Title": "Administrative and Support and Waste Management"},
        {"Code": "61", "Title": "Educational Services"},
        {"Code": "62", "Title": "Health Care and Social Assistance"},
        {"Code": "71", "Title": "Arts, Entertainment, and Recreation"},
        {"Code": "72", "Title": "Accommodation and Food Services"},
        {"Code": "81", "Title": "Other Services (except Public Administration)"}
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download industry codes from various sources."""
        output_path = self.raw_dir / "naics_2022.json"

        for url in self.NAICS_URLS:
            try:
                if not output_path.exists():
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(response.text)
                        break
            except Exception as e:
                print(f"Failed to fetch from {url}: {e}")
                continue
        else:
            print("Using fallback NAICS data")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.FALLBACK_NAICS, f)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                return data
            return self.FALLBACK_NAICS

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format with aliases."""
        entities = []

        for item in raw_data:
            code = str(item.get("Code", "")).strip()
            title = item.get("Title", "").strip()

            if not code or not title:
                continue

            aliases = []
            sector = ""
            if len(code) >= 2:
                sector = code[:2]
                aliases.append(sector)
            if len(code) >= 3:
                subsector = code[:3]
                if subsector != sector:
                    aliases.append(subsector)

            entities.append({
                "id": code,
                "name": title,
                "aliases": "|".join(aliases) if aliases else "",
                "type": "industry",
                "system": "NAICS"
            })

        return entities


class SICFetcher(BaseFetcher):
    """Fetch SIC industry codes from BLS."""

    SIC_URL = "https://www.bls.gov/cew/classifications/industry/sic-industry-titles.csv"

    FALLBACK_SIC = [
        {"SIC Code": "01", "SIC Industry Title": "Agricultural Production - Crops"},
        {"SIC Code": "10", "SIC Industry Title": "Metal Mining"},
        {"SIC Code": "15", "SIC Industry Title": "Building Construction - General Contractors"},
        {"SIC Code": "17", "SIC Industry Title": "Construction - Special Trade Contractors"},
        {"SIC Code": "20", "SIC Industry Title": "Food and Kindred Products"},
        {"SIC Code": "25", "SIC Industry Title": "Furniture and Fixtures"},
        {"SIC Code": "27", "SIC Industry Title": "Printing, Publishing and Allied Industries"},
        {"SIC Code": "28", "SIC Industry Title": "Chemicals and Allied Products"},
        {"SIC Code": "30", "SIC Industry Title": "Rubber and Miscellaneous Plastics Products"},
        {"SIC Code": "33", "SIC Industry Title": "Primary Metal Industries"},
        {"SIC Code": "35", "SIC Industry Title": "Industrial and Commercial Machinery"},
        {"SIC Code": "36", "SIC Industry Title": "Electronic and Other Electrical Equipment"},
        {"SIC Code": "37", "SIC Industry Title": "Transportation Equipment"},
        {"SIC Code": "38", "SIC Industry Title": "Instruments and Related Products"},
        {"SIC Code": "40", "SIC Industry Title": "Railroad Transportation"},
        {"SIC Code": "42", "SIC Industry Title": "Trucking and Warehousing"},
        {"SIC Code": "45", "SIC Industry Title": "Transportation by Air"},
        {"SIC Code": "48", "SIC Industry Title": "Communications"},
        {"SIC Code": "49", "SIC Industry Title": "Electric, Gas, and Sanitary Services"},
        {"SIC Code": "50", "SIC Industry Title": "Wholesale Trade - Durable Goods"},
        {"SIC Code": "51", "SIC Industry Title": "Wholesale Trade - Non-Durable Goods"},
        {"SIC Code": "52", "SIC Industry Title": "Building Materials and Garden Supplies"},
        {"SIC Code": "53", "SIC Industry Title": "General Merchandise Stores"},
        {"SIC Code": "54", "SIC Industry Title": "Food Stores"},
        {"SIC Code": "55", "SIC Industry Title": "Automotive Dealers and Gasoline Stations"},
        {"SIC Code": "56", "SIC Industry Title": "Apparel and Accessory Stores"},
        {"SIC Code": "57", "SIC Industry Title": "Furniture and Home Furnishings Stores"},
        {"SIC Code": "58", "SIC Industry Title": "Eating and Drinking Places"},
        {"SIC Code": "60", "SIC Industry Title": "Depository Institutions"},
        {"SIC Code": "61", "SIC Industry Title": "Non-Depository Credit Institutions"},
        {"SIC Code": "62", "SIC Industry Title": "Security and Commodity Brokers"},
        {"SIC Code": "63", "SIC Industry Title": "Insurance Carriers"},
        {"SIC Code": "65", "SIC Industry Title": "Real Estate"},
        {"SIC Code": "70", "SIC Industry Title": "Hotels, Rooming Houses, Camps"},
        {"SIC Code": "72", "SIC Industry Title": "Personal Services"},
        {"SIC Code": "73", "SIC Industry Title": "Business Services"},
        {"SIC Code": "75", "SIC Industry Title": "Automotive Repair and Services"},
        {"SIC Code": "78", "SIC Industry Title": "Motion Pictures"},
        {"SIC Code": "79", "SIC Industry Title": "Amusement and Recreation Services"},
        {"SIC Code": "80", "SIC Industry Title": "Health Services"},
        {"SIC Code": "82", "SIC Industry Title": "Educational Services"},
        {"SIC Code": "83", "SIC Industry Title": "Social Services"},
        {"SIC Code": "86", "SIC Industry Title": "Membership Organizations"},
        {"SIC Code": "87", "SIC Industry Title": "Engineering and Management Services"},
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download SIC codes."""
        output_path = self.raw_dir / "sic_titles.csv"

        if not output_path.exists():
            try:
                response = requests.get(self.SIC_URL, timeout=30)
                response.raise_for_status()
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
            except Exception as e:
                print(f"BLS fetch failed: {e}, using fallback data")
                with open(output_path, "w", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["SIC Code", "SIC Industry Title"])
                    writer.writeheader()
                    writer.writerows(self.FALLBACK_SIC)

        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return data if data else self.FALLBACK_SIC

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        for row in raw_data:
            code = row.get("SIC Code", "").strip()
            title = row.get("SIC Industry Title", "").strip()

            if not code or not title:
                continue

            aliases = []
            if len(code) >= 2:
                aliases.append(code[:2])

            entities.append({
                "id": code,
                "name": title,
                "aliases": "|".join(aliases) if aliases else "",
                "type": "industry",
                "system": "SIC"
            })

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute industry data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("industries", raw_dir, processed_dir)

    fetcher = IndustriesFetcher(raw_dir, processed_dir)
    fetcher.run("industries_naics.csv")

    sic_fetcher = SICFetcher(raw_dir, processed_dir)
    sic_fetcher.run("industries_sic.csv")


if __name__ == "__main__":
    run()
