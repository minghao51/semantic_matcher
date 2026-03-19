"""
Pydantic schemas for novel class detection and proposal system.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DetectionStrategy(str, Enum):
    """Available detection strategies."""

    CONFIDENCE = "confidence"
    CENTROID = "centroid"
    KNN_DISTANCE = "knn_distance"
    CLUSTERING = "clustering"


class NovelSampleMetadata(BaseModel):
    """Metadata for a single novel sample."""

    text: str = Field(..., description="The original text sample")
    index: int = Field(..., description="Index in the original query list")
    confidence: float = Field(..., description="Classification confidence score")
    predicted_class: str = Field(..., description="The predicted class")
    embedding_distance: Optional[float] = Field(
        default=None,
        description="Compatibility field for geometric distance-based novelty scoring",
    )
    margin_score: Optional[float] = Field(
        default=None, description="Novelty contribution from top-1 vs top-2 margin"
    )
    entropy_score: Optional[float] = Field(
        default=None, description="Novelty contribution from normalized entropy"
    )
    uncertainty_score: Optional[float] = Field(
        default=None, description="Aggregate uncertainty score"
    )
    knn_novelty_score: Optional[float] = Field(
        default=None, description="Aggregate ANN kNN novelty score"
    )
    knn_mean_distance: Optional[float] = Field(
        default=None, description="Mean cosine distance to top-k known neighbors"
    )
    knn_nearest_distance: Optional[float] = Field(
        default=None, description="Cosine distance to nearest known neighbor"
    )
    predicted_class_neighbor_ratio: Optional[float] = Field(
        default=None,
        description="Fraction of top-k neighbors matching the predicted class",
    )
    predicted_class_support: Optional[float] = Field(
        default=None,
        description="Average similarity among top-k neighbors matching the predicted class",
    )
    neighbor_labels: List[str] = Field(
        default_factory=list,
        description="Labels of the nearest known neighbors used for ANN scoring",
    )
    neighbor_distances: List[float] = Field(
        default_factory=list,
        description="Cosine distances to the nearest known neighbors",
    )
    cluster_id: Optional[int] = Field(
        default=None, description="Cluster assignment from HDBSCAN"
    )
    cluster_validation_passed: bool = Field(
        default=False,
        description="Whether this sample belongs to a cluster that passed quality checks",
    )
    cluster_support_score: Optional[float] = Field(
        default=None, description="Support added by validated novel-cluster membership"
    )
    novelty_score: Optional[float] = Field(
        default=None, description="Final aggregate novelty score"
    )
    signals: Dict[str, bool] = Field(
        default_factory=dict,
        description="Detection signals from each strategy",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


class ClassProposal(BaseModel):
    """A proposed new class."""

    name: str = Field(..., description="Proposed class name")
    description: str = Field(..., description="Description of the class")
    confidence: float = Field(
        ..., description="Confidence in this proposal (0-1)", ge=0, le=1
    )
    sample_count: int = Field(
        ..., description="Number of samples supporting this proposal", ge=0
    )
    example_samples: List[str] = Field(
        ..., description="Example samples that belong to this class"
    )
    justification: str = Field(..., description="Justification for this proposal")
    suggested_parent: Optional[str] = Field(
        default=None, description="Suggested parent class if hierarchical"
    )

    @field_validator("example_samples")
    @classmethod
    def validate_examples(cls, v: List[str]) -> List[str]:
        if len(v) == 0:
            raise ValueError("example_samples cannot be empty")
        return v


class NovelClassAnalysis(BaseModel):
    """Complete analysis of novel classes from LLM."""

    proposed_classes: List[ClassProposal] = Field(
        ..., description="List of proposed new classes"
    )
    rejected_as_noise: List[str] = Field(
        default_factory=list,
        description="Samples rejected as noise/outliers",
    )
    analysis_summary: str = Field(..., description="Summary of the analysis process")
    cluster_count: int = Field(..., description="Number of clusters found", ge=0)
    model_used: str = Field(..., description="LLM model used for analysis")


class NovelSampleReport(BaseModel):
    """Report of detected novel samples."""

    novel_samples: List[NovelSampleMetadata] = Field(
        default_factory=list, description="List of detected novel samples"
    )
    detection_strategies: List[DetectionStrategy] = Field(
        ..., description="Strategies used for detection"
    )
    config: Dict[str, Any] = Field(..., description="Configuration used for detection")
    signal_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of samples flagged by each strategy",
    )


class NovelClassDiscoveryReport(BaseModel):
    """Complete report from novel class discovery pipeline."""

    discovery_id: str = Field(
        ..., description="Unique identifier for this discovery session"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the discovery was performed",
    )
    matcher_config: Dict[str, Any] = Field(
        ..., description="Configuration of the matcher used"
    )
    detection_config: Dict[str, Any] = Field(
        ..., description="Configuration used for novelty detection"
    )
    novel_sample_report: NovelSampleReport = Field(
        ..., description="Results from novelty detection"
    )
    class_proposals: Optional[NovelClassAnalysis] = Field(
        default=None, description="Proposed classes from LLM analysis"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the discovery session",
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Path to saved proposal file (if saved)",
    )


class MatcherConfig(BaseModel):
    """Matcher configuration for novel class detection."""

    matcher_type: str = Field(..., description="Type of matcher used")
    model: str = Field(..., description="Model name/alias")
    mode: Optional[str] = Field(default=None, description="Matcher mode")
    threshold: Optional[float] = Field(default=None, description="Similarity threshold")


class DetectionConfig(BaseModel):
    """Configuration for novelty detection."""

    strategies: List[DetectionStrategy] = Field(
        default_factory=lambda: [
            DetectionStrategy.CONFIDENCE,
            DetectionStrategy.KNN_DISTANCE,
            DetectionStrategy.CLUSTERING,
        ],
        description="Detection strategies to use",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for novelty detection",
        ge=0,
        le=1,
    )
    distance_threshold: float = Field(
        default=0.3,
        description="Legacy distance threshold for centroid analysis compatibility",
        ge=0,
        le=1,
    )
    uncertainty_top_k: int = Field(
        default=5,
        description="Number of matcher candidates to inspect for uncertainty scoring",
        gt=1,
    )
    uncertainty_threshold: float = Field(
        default=0.55,
        description="Threshold for aggregate uncertainty signal",
        ge=0,
        le=1,
    )
    strong_uncertainty_threshold: float = Field(
        default=0.85,
        description="Threshold for unusually strong uncertainty",
        ge=0,
        le=1,
    )
    knn_k: int = Field(
        default=5,
        description="Number of nearest known neighbors to inspect in ANN scoring",
        gt=0,
    )
    knn_distance_threshold: float = Field(
        default=0.55,
        description="Threshold for aggregate kNN novelty scoring",
        ge=0,
        le=1,
    )
    strong_knn_novelty_threshold: float = Field(
        default=0.85,
        description="Threshold for very strong point-anomaly kNN novelty",
        ge=0,
        le=1,
    )
    knn_gate_threshold: float = Field(
        default=0.45,
        description="Minimum kNN novelty needed for weighted promotion",
        ge=0,
        le=1,
    )
    min_cluster_size: int = Field(
        default=5,
        description="Minimum cluster size for HDBSCAN",
        gt=0,
    )
    ann_backend: str = Field(
        default="hnswlib",
        description="ANN backend to use ('hnswlib' or 'faiss')",
    )
    combine_method: str = Field(
        default="weighted",
        description=(
            "Method to combine signals "
            "('weighted', 'and'/'intersection', 'or'/'union', or 'voting')"
        ),
    )
    novelty_threshold: float = Field(
        default=0.6,
        description="Threshold for the final weighted novelty score",
        ge=0,
        le=1,
    )
    candidate_score_threshold: float = Field(
        default=0.45,
        description="Minimum subsystem score required for clustering candidacy",
        ge=0,
        le=1,
    )
    uncertainty_weight: float = Field(
        default=0.35,
        description="Weight of uncertainty in the aggregate novelty score",
        ge=0,
        le=1,
    )
    knn_weight: float = Field(
        default=0.45,
        description="Weight of kNN novelty in the aggregate novelty score",
        ge=0,
        le=1,
    )
    cluster_weight: float = Field(
        default=0.2,
        description="Weight of validated cluster support in the aggregate novelty score",
        ge=0,
        le=1,
    )
    cluster_persistence_threshold: float = Field(
        default=0.1,
        description="Minimum HDBSCAN persistence required for a valid novel cluster",
        ge=0,
        le=1,
    )
    cluster_cohesion_threshold: float = Field(
        default=0.45,
        description="Maximum average pairwise cosine distance allowed within a cluster",
        ge=0,
        le=1,
    )
    cluster_separation_threshold: float = Field(
        default=0.35,
        description="Minimum mean kNN distance required for cluster separation from known classes",
        ge=0,
        le=1,
    )
    cluster_known_support_threshold: float = Field(
        default=0.6,
        description="Maximum average predicted-class support allowed for a valid novel cluster",
        ge=0,
        le=1,
    )

    @field_validator("combine_method")
    @classmethod
    def validate_combine_method(cls, value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "weighted": "weighted",
            "and": "intersection",
            "intersection": "intersection",
            "or": "union",
            "union": "union",
            "voting": "voting",
        }
        if normalized not in aliases:
            raise ValueError(
                "combine_method must be one of: weighted, and, intersection, or, union, voting"
            )
        return aliases[normalized]
