"""Backward-compatible shim for the stable matcher metadata contract."""

from ..pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
    build_match_records,
    build_match_result_with_metadata,
    convert_match_result_to_metadata,
    normalize_candidate_results,
)

__all__ = [
    "MatchRecord",
    "MatchResultWithMetadata",
    "build_match_records",
    "build_match_result_with_metadata",
    "convert_match_result_to_metadata",
    "normalize_candidate_results",
]
