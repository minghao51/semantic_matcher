"""
Match result schemas with metadata support.

Provides enhanced match result classes that include embeddings,
confidence scores, and other metadata for novel class detection.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MatchResultWithMetadata:
    """
    Enhanced match result with metadata for novel class detection.

    Attributes:
        predictions: Predicted class/entity IDs
        confidences: Confidence scores for predictions
        embeddings: Text embeddings
        scores: Raw similarity scores
        metadata: Additional prediction metadata
    """

    predictions: List[str]
    confidences: np.ndarray
    embeddings: np.ndarray
    scores: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.confidences, np.ndarray):
            self.confidences = np.array(self.confidences)
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.array(self.embeddings)
        if self.scores is not None and not isinstance(self.scores, np.ndarray):
            self.scores = np.array(self.scores)

    @property
    def num_samples(self) -> int:
        """Number of samples in the result."""
        return len(self.predictions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "predictions": self.predictions,
            "confidences": self.confidences.tolist(),
            "embeddings": self.embeddings.tolist(),
            "scores": self.scores.tolist() if self.scores is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchResultWithMetadata":
        """Create from dictionary representation."""
        return cls(
            predictions=data["predictions"],
            confidences=np.array(data["confidences"]),
            embeddings=np.array(data["embeddings"]),
            scores=np.array(data["scores"]) if data.get("scores") else None,
            metadata=data.get("metadata"),
        )


def convert_match_result_to_metadata(
    match_result: Any,
    embeddings: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> MatchResultWithMetadata:
    """
    Convert standard match result to metadata-enhanced result.

    Args:
        match_result: Standard match result (dict, list of dicts, or list of strings)
        embeddings: Text embeddings
        confidences: Optional confidence scores

    Returns:
        MatchResultWithMetadata instance
    """
    # Extract predictions
    if isinstance(match_result, dict):
        # Single result
        predictions = [match_result.get("id", "unknown")]
        scores = np.array([match_result.get("score", 0.0)])
    elif isinstance(match_result, list):
        if all(isinstance(r, dict) for r in match_result):
            # List of results
            predictions = [r.get("id", "unknown") for r in match_result]
            scores = np.array([r.get("score", 0.0) for r in match_result])
        else:
            # List of strings (predictions)
            predictions = match_result
            scores = None
    else:
        # Single prediction
        predictions = [str(match_result)]
        scores = None

    # Default confidences if not provided
    if confidences is None:
        confidences = np.ones(len(predictions))

    return MatchResultWithMetadata(
        predictions=predictions,
        confidences=confidences,
        embeddings=embeddings,
        scores=scores,
    )
