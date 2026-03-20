"""Data ingestion package for semantic matcher datasets."""

from .languages import run as run_languages
from .currencies import run as run_currencies
from .industries import run as run_industries
from .timezones import run as run_timezones
from .occupations import run as run_occupations
from .products import run as run_products
from .universities import run as run_universities

__all__ = [
    "run_languages",
    "run_currencies",
    "run_industries",
    "run_timezones",
    "run_occupations",
    "run_products",
    "run_universities",
]
