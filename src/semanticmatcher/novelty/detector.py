"""
Novelty detection using ANN-backed multi-signal scoring.

Combines uncertainty scoring, ANN kNN distance analysis, and
validated density-based clustering for novel-class detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from .ann_index import ANNIndex
from .schemas import (
    DetectionConfig,
    DetectionStrategy,
    NovelSampleMetadata,
    NovelSampleReport,
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class NoveltyDetector:
    """
    Detect novel samples using multiple complementary signals.

    The detector expects a stable reference corpus of known embeddings and labels.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        embedding_dim: int = 768,
    ):
        self.config = config or DetectionConfig()
        self.embedding_dim = embedding_dim
        self._ann_index: Optional[ANNIndex] = None
        self._reference_labels: List[str] = []
        self._reference_embeddings: Optional[np.ndarray] = None
        self._class_centroids: Dict[str, np.ndarray] = {}

    def detect_novel_samples(
        self,
        texts: List[str],
        confidences: np.ndarray,
        embeddings: np.ndarray,
        predicted_classes: List[str],
        known_classes: Optional[List[str]] = None,
        candidate_results: Optional[Sequence[Any]] = None,
        reference_embeddings: Optional[np.ndarray] = None,
        reference_labels: Optional[List[str]] = None,
    ) -> NovelSampleReport:
        """
        Detect novel samples using configured strategies.

        Args:
            texts: Input text samples
            confidences: Top-1 confidence scores (0-1)
            embeddings: Query embeddings
            predicted_classes: Predicted class per sample
            known_classes: Optional list of known classes
            candidate_results: Top-k matcher results used for uncertainty scoring
            reference_embeddings: Embeddings for known/reference corpus
            reference_labels: Labels for known/reference corpus

        Returns:
            NovelSampleReport with detected novel samples
        """
        n_samples = len(texts)
        known_classes = known_classes or sorted(
            set(reference_labels or predicted_classes)
        )

        logger.info(
            "Detecting novel samples among %s samples using strategies: %s",
            n_samples,
            self.config.strategies,
        )

        if n_samples == 0:
            return NovelSampleReport(
                novel_samples=[],
                detection_strategies=self.config.strategies,
                config=self.config.model_dump(),
                signal_counts={},
            )

        self._build_reference_index(reference_embeddings, reference_labels)

        all_signals: Dict[int, Dict[str, bool]] = {i: {} for i in range(n_samples)}
        signal_counts: Dict[str, int] = {}

        uncertainty_metrics = self._compute_uncertainty_metrics(
            confidences,
            candidate_results,
        )
        uncertainty_flags = {
            idx
            for idx, metrics in uncertainty_metrics.items()
            if metrics["uncertainty_score"] >= self.config.uncertainty_threshold
        }
        if DetectionStrategy.CONFIDENCE in self.config.strategies:
            for idx in uncertainty_flags:
                all_signals[idx][DetectionStrategy.CONFIDENCE] = True
            signal_counts[DetectionStrategy.CONFIDENCE] = len(uncertainty_flags)

        use_knn = any(
            strategy in self.config.strategies
            for strategy in (DetectionStrategy.KNN_DISTANCE, DetectionStrategy.CENTROID)
        )
        knn_metrics: Dict[int, Dict[str, Any]] = {}
        knn_flags: Set[int] = set()
        active_distance_strategy = None
        if use_knn:
            knn_metrics, knn_flags = self._detect_by_knn_distance(
                embeddings,
                predicted_classes,
            )
            active_distance_strategy = (
                DetectionStrategy.KNN_DISTANCE
                if DetectionStrategy.KNN_DISTANCE in self.config.strategies
                else DetectionStrategy.CENTROID
            )
            for idx in knn_flags:
                all_signals[idx][active_distance_strategy] = True
            signal_counts[active_distance_strategy] = len(knn_flags)

        candidate_indices: List[int] = []
        cluster_assignments: Dict[int, int] = {}
        cluster_support_scores: Dict[int, float] = {}
        cluster_validation: Dict[int, Dict[str, Any]] = {}
        if DetectionStrategy.CLUSTERING in self.config.strategies:
            candidate_indices = self._get_candidates(
                all_signals,
                uncertainty_metrics=uncertainty_metrics,
                knn_metrics=knn_metrics,
            )
            if len(candidate_indices) >= self.config.min_cluster_size:
                (
                    cluster_assignments,
                    cluster_support_scores,
                    cluster_validation,
                ) = self._detect_by_clustering_ann(
                    embeddings,
                    candidate_indices,
                    knn_metrics,
                )
                for idx, cluster_id in cluster_assignments.items():
                    if cluster_id >= 0 and cluster_support_scores.get(idx, 0.0) > 0:
                        all_signals[idx][DetectionStrategy.CLUSTERING] = True
                signal_counts[DetectionStrategy.CLUSTERING] = sum(
                    1
                    for idx in cluster_assignments
                    if cluster_support_scores.get(idx, 0.0) > 0
                )
            else:
                signal_counts[DetectionStrategy.CLUSTERING] = 0

        novel_indices, novelty_scores = self._combine_signals(
            all_signals,
            uncertainty_metrics=uncertainty_metrics,
            knn_metrics=knn_metrics,
            cluster_support_scores=cluster_support_scores,
        )

        novel_samples = self._create_sample_metadata(
            texts=texts,
            confidences=confidences,
            predicted_classes=predicted_classes,
            novel_indices=novel_indices,
            all_signals=all_signals,
            uncertainty_metrics=uncertainty_metrics,
            knn_metrics=knn_metrics,
            cluster_assignments=cluster_assignments,
            cluster_support_scores=cluster_support_scores,
            cluster_validation=cluster_validation,
            novelty_scores=novelty_scores,
        )

        logger.info("Detected %s novel samples", len(novel_samples))

        return NovelSampleReport(
            novel_samples=novel_samples,
            detection_strategies=self.config.strategies,
            config=self.config.model_dump(),
            signal_counts={
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in signal_counts.items()
            },
        )

    def _build_reference_index(
        self,
        reference_embeddings: Optional[np.ndarray],
        reference_labels: Optional[List[str]],
    ) -> None:
        """Build an ANN index over known/reference embeddings."""
        if reference_embeddings is None or reference_labels is None:
            raise RuntimeError(
                "Novelty detection requires reference embeddings and labels for known classes."
            )

        if len(reference_labels) == 0:
            raise RuntimeError(
                "Reference corpus is empty; cannot run novelty detection."
            )

        embeddings = np.asarray(reference_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("reference_embeddings must be a 2D array")
        if embeddings.shape[0] != len(reference_labels):
            raise ValueError(
                "reference_embeddings and reference_labels must have the same length"
            )

        self.embedding_dim = embeddings.shape[1]
        self._reference_embeddings = embeddings
        self._reference_labels = list(reference_labels)
        self._ann_index = ANNIndex(
            dim=self.embedding_dim,
            backend=self.config.ann_backend,
            max_elements=max(len(reference_labels), 1),
        )
        self._ann_index.add_vectors(embeddings, self._reference_labels)

    def _build_class_centroids(
        self,
        embeddings: np.ndarray,
        predicted_classes: List[str],
        known_classes: List[str],
    ) -> None:
        """
        Legacy helper retained for compatibility.

        Builds per-class centroids and stores them in the ANN index.
        """
        self._class_centroids = {}
        centroid_vectors = []
        centroid_labels = []
        for class_name in known_classes:
            mask = np.array([c == class_name for c in predicted_classes])
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(axis=0)
                self._class_centroids[class_name] = centroid
                centroid_vectors.append(centroid)
                centroid_labels.append(class_name)
        if centroid_vectors:
            self._build_reference_index(np.vstack(centroid_vectors), centroid_labels)

    def _detect_by_confidence(self, confidences: np.ndarray) -> Set[int]:
        """Legacy confidence-threshold detector retained for compatibility."""
        low_confidence_mask = confidences < self.config.confidence_threshold
        return set(np.where(low_confidence_mask)[0])

    def _detect_by_distance_ann(
        self,
        embeddings: np.ndarray,
        predicted_classes: List[str],
    ) -> Set[int]:
        """Legacy distance-based detector mapped to kNN novelty scoring."""
        _, flags = self._detect_by_knn_distance(embeddings, predicted_classes)
        return flags

    def _compute_uncertainty_metrics(
        self,
        confidences: np.ndarray,
        candidate_results: Optional[Sequence[Any]],
    ) -> Dict[int, Dict[str, float]]:
        """Compute top-k uncertainty features from matcher scores."""
        metrics: Dict[int, Dict[str, float]] = {}
        normalized_results = list(candidate_results or [])

        for idx, confidence in enumerate(np.asarray(confidences, dtype=float)):
            raw_result = (
                normalized_results[idx] if idx < len(normalized_results) else None
            )
            scores = self._extract_scores(raw_result)
            if not scores:
                scores = [float(confidence)]

            top1 = float(scores[0])
            top2 = float(scores[1]) if len(scores) > 1 else 0.0
            margin = float(np.clip(top1 - top2, 0.0, 1.0))
            margin_score = float(np.clip(1.0 - margin, 0.0, 1.0))
            confidence_novelty = float(np.clip(1.0 - top1, 0.0, 1.0))

            if len(scores) > 1:
                stable_scores = np.asarray(scores, dtype=float)
                stable_scores = stable_scores - np.max(stable_scores)
                probs = np.exp(stable_scores)
                probs = probs / probs.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                entropy_score = float(entropy / np.log(len(scores)))
            else:
                entropy_score = 0.0

            uncertainty_score = float(
                np.clip(
                    (0.4 * confidence_novelty)
                    + (0.3 * margin_score)
                    + (0.3 * entropy_score),
                    0.0,
                    1.0,
                )
            )
            metrics[idx] = {
                "margin_score": margin_score,
                "entropy_score": entropy_score,
                "uncertainty_score": uncertainty_score,
            }

        return metrics

    def _detect_by_knn_distance(
        self,
        embeddings: np.ndarray,
        predicted_classes: List[str],
    ) -> Tuple[Dict[int, Dict[str, Any]], Set[int]]:
        """Score novelty against the known ANN reference corpus."""
        if self._ann_index is None:
            raise RuntimeError("ANN reference index is not initialized.")

        query_embeddings = np.asarray(embeddings, dtype=np.float32)
        k = min(self.config.knn_k, self._ann_index.n_elements)
        similarities, neighbor_indices = self._ann_index.knn_query(
            query_embeddings, k=k
        )

        metrics: Dict[int, Dict[str, Any]] = {}
        flags: Set[int] = set()
        labels = self._ann_index.labels

        for idx in range(len(query_embeddings)):
            sims = similarities[idx]
            indices = neighbor_indices[idx]
            neighbor_labels = [
                labels[int(neighbor_idx)]
                for neighbor_idx in indices
                if 0 <= int(neighbor_idx) < len(labels)
            ]
            distances = [
                float(np.clip(1.0 - sim, 0.0, 1.0))
                for sim in sims[: len(neighbor_labels)]
            ]
            mean_distance = float(np.mean(distances)) if distances else 1.0
            nearest_distance = distances[0] if distances else 1.0

            predicted_class = predicted_classes[idx]
            matching_sims = [
                float(sim)
                for sim, label in zip(sims[: len(neighbor_labels)], neighbor_labels)
                if label == predicted_class
            ]
            predicted_ratio = (
                len(matching_sims) / len(neighbor_labels) if neighbor_labels else 0.0
            )
            predicted_support = float(np.mean(matching_sims)) if matching_sims else 0.0

            novelty_score = float(
                np.clip(
                    (0.15 * mean_distance)
                    + (0.45 * nearest_distance)
                    + (0.15 * (1.0 - predicted_ratio))
                    + (0.25 * (1.0 - predicted_support)),
                    0.0,
                    1.0,
                )
            )

            metrics[idx] = {
                "neighbor_labels": neighbor_labels,
                "neighbor_distances": distances,
                "knn_mean_distance": mean_distance,
                "knn_nearest_distance": nearest_distance,
                "predicted_class_neighbor_ratio": predicted_ratio,
                "predicted_class_support": predicted_support,
                "knn_novelty_score": novelty_score,
            }

            if novelty_score >= self.config.knn_distance_threshold:
                flags.add(idx)

        return metrics, flags

    def _detect_by_clustering_ann(
        self,
        embeddings: np.ndarray,
        candidate_indices: List[int],
        knn_metrics: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, Dict[str, Any]]]:
        """Run HDBSCAN and validate clusters before adding support."""
        try:
            import hdbscan
        except ImportError:
            logger.warning(
                "hdbscan not available, skipping clustering strategy. Install with: pip install hdbscan"
            )
            return {}, {}, {}

        candidate_embeddings = np.asarray(
            embeddings[candidate_indices], dtype=np.float32
        )
        distance_matrix = cosine_distances(candidate_embeddings).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            metric="precomputed",
            prediction_data=True,
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
        probabilities = getattr(
            clusterer, "probabilities_", np.ones(len(candidate_indices))
        )
        persistences = getattr(clusterer, "cluster_persistence_", [])

        assignments: Dict[int, int] = {}
        support_scores: Dict[int, float] = {}
        validation: Dict[int, Dict[str, Any]] = {}

        for original_idx, cluster_id in zip(candidate_indices, cluster_labels):
            assignments[original_idx] = int(cluster_id)
            support_scores[original_idx] = 0.0

        unique_clusters = sorted(
            {int(label) for label in cluster_labels if int(label) >= 0}
        )
        if (
            not unique_clusters
            and len(candidate_indices) >= self.config.min_cluster_size
        ):
            unique_clusters = [0]
            cluster_labels = np.zeros(len(candidate_indices), dtype=int)
            probabilities = np.ones(len(candidate_indices), dtype=float)
            for original_idx in candidate_indices:
                assignments[original_idx] = 0

        for cluster_id in unique_clusters:
            local_member_positions = np.where(cluster_labels == cluster_id)[0]
            member_indices = [candidate_indices[pos] for pos in local_member_positions]
            submatrix = distance_matrix[
                np.ix_(local_member_positions, local_member_positions)
            ]

            if len(local_member_positions) > 1:
                tri = submatrix[np.triu_indices_from(submatrix, k=1)]
                cohesion = float(np.mean(tri)) if tri.size else 0.0
            else:
                cohesion = 0.0

            persistence = (
                float(persistences[cluster_id])
                if len(persistences) > cluster_id
                else 1.0
            )
            separation = float(
                np.mean(
                    [
                        knn_metrics.get(idx, {}).get("knn_mean_distance", 0.0)
                        for idx in member_indices
                    ]
                )
            )
            known_support = float(
                np.mean(
                    [
                        knn_metrics.get(idx, {}).get("predicted_class_support", 1.0)
                        for idx in member_indices
                    ]
                )
            )
            validation_passed = (
                len(member_indices) >= self.config.min_cluster_size
                and persistence >= self.config.cluster_persistence_threshold
                and cohesion <= self.config.cluster_cohesion_threshold
                and separation >= self.config.cluster_separation_threshold
                and known_support <= self.config.cluster_known_support_threshold
            )

            support_base = 0.0
            if validation_passed:
                cohesion_component = np.clip(
                    1.0
                    - (cohesion / max(self.config.cluster_cohesion_threshold, 1e-6)),
                    0.0,
                    1.0,
                )
                support_base = float(
                    np.clip(
                        np.mean(
                            [
                                persistence,
                                separation,
                                1.0 - known_support,
                                cohesion_component,
                            ]
                        ),
                        0.0,
                        1.0,
                    )
                )

            validation[cluster_id] = {
                "size": len(member_indices),
                "persistence": persistence,
                "cohesion": cohesion,
                "separation": separation,
                "known_support": known_support,
                "validation_passed": validation_passed,
                "cluster_support_score": support_base,
            }

            for local_pos, original_idx in zip(local_member_positions, member_indices):
                if validation_passed:
                    membership = (
                        float(probabilities[local_pos])
                        if len(probabilities) > local_pos
                        else 1.0
                    )
                    support_scores[original_idx] = float(
                        np.clip(support_base * membership, 0.0, 1.0)
                    )

        return assignments, support_scores, validation

    def _get_candidates(
        self,
        all_signals: Dict[int, Dict[str, bool]],
        uncertainty_metrics: Optional[Dict[int, Dict[str, float]]] = None,
        knn_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        """Select clustering candidates using the configured combine policy."""
        if self.config.combine_method == "weighted":
            candidates = []
            for idx in all_signals:
                uncertainty_score = (
                    (uncertainty_metrics or {})
                    .get(idx, {})
                    .get(
                        "uncertainty_score",
                        0.0,
                    )
                )
                knn_score = (
                    (knn_metrics or {}).get(idx, {}).get("knn_novelty_score", 0.0)
                )
                if (
                    max(uncertainty_score, knn_score)
                    >= self.config.candidate_score_threshold
                ):
                    candidates.append(idx)
            return candidates

        if self.config.combine_method == "union":
            return [
                idx for idx, signals in all_signals.items() if any(signals.values())
            ]
        if self.config.combine_method == "intersection":
            return [
                idx
                for idx, signals in all_signals.items()
                if signals and all(signals.values())
            ]

        threshold = max(1, int(np.ceil(len(self.config.strategies) / 2)))
        return [
            idx
            for idx, signals in all_signals.items()
            if sum(bool(v) for v in signals.values()) >= threshold
        ]

    def _combine_signals(
        self,
        all_signals: Dict[int, Dict[str, bool]],
        uncertainty_metrics: Optional[Dict[int, Dict[str, float]]] = None,
        knn_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        cluster_support_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[Set[int], Dict[int, float]]:
        """Combine subsystem scores using weighted fusion or legacy boolean policies."""
        if self.config.combine_method != "weighted":
            novel_indices = self._combine_signals_legacy(all_signals)
            denominator = max(len(self.config.strategies), 1)
            novelty_scores = {
                idx: float(sum(bool(v) for v in signals.values()) / denominator)
                for idx, signals in all_signals.items()
            }
            return novel_indices, novelty_scores

        novelty_scores: Dict[int, float] = {}
        novel_indices: Set[int] = set()

        for idx in all_signals:
            uncertainty_score = (
                (uncertainty_metrics or {})
                .get(idx, {})
                .get(
                    "uncertainty_score",
                    0.0,
                )
            )
            knn_score = (knn_metrics or {}).get(idx, {}).get("knn_novelty_score", 0.0)
            cluster_score = (cluster_support_scores or {}).get(idx, 0.0)

            novelty_score = float(
                np.clip(
                    (self.config.uncertainty_weight * uncertainty_score)
                    + (self.config.knn_weight * knn_score)
                    + (self.config.cluster_weight * cluster_score),
                    0.0,
                    1.0,
                )
            )
            novelty_scores[idx] = novelty_score

            strong_knn = knn_score >= self.config.strong_knn_novelty_threshold
            strong_uncertainty = (
                uncertainty_score >= self.config.strong_uncertainty_threshold
                and knn_score >= self.config.knn_gate_threshold
            )
            weighted_positive = novelty_score >= self.config.novelty_threshold and (
                knn_score >= self.config.knn_gate_threshold
                or cluster_score > 0
                or uncertainty_score >= self.config.strong_uncertainty_threshold
            )

            if strong_knn or strong_uncertainty or weighted_positive:
                novel_indices.add(idx)

        return novel_indices, novelty_scores

    def _combine_signals_legacy(
        self, all_signals: Dict[int, Dict[str, bool]]
    ) -> Set[int]:
        """Combine detection signals using the pre-upgrade boolean logic."""
        if self.config.combine_method == "union":
            return {
                idx for idx, signals in all_signals.items() if any(signals.values())
            }
        if self.config.combine_method == "intersection":
            required_strategies = set(self.config.strategies)
            return {
                idx
                for idx, signals in all_signals.items()
                if required_strategies.issubset(signals.keys())
                and all(signals.get(s, False) for s in required_strategies)
            }

        threshold = len(self.config.strategies) / 2
        return {
            idx
            for idx, signals in all_signals.items()
            if sum(bool(v) for v in signals.values()) >= threshold
        }

    def _create_sample_metadata(
        self,
        texts: List[str],
        confidences: np.ndarray,
        predicted_classes: List[str],
        novel_indices: Set[int],
        all_signals: Dict[int, Dict[str, bool]],
        uncertainty_metrics: Dict[int, Dict[str, float]],
        knn_metrics: Dict[int, Dict[str, Any]],
        cluster_assignments: Dict[int, int],
        cluster_support_scores: Dict[int, float],
        cluster_validation: Dict[int, Dict[str, Any]],
        novelty_scores: Dict[int, float],
    ) -> List[NovelSampleMetadata]:
        """Create structured metadata for each novel sample."""
        metadata_list: List[NovelSampleMetadata] = []

        for idx in sorted(novel_indices):
            uncertainty = uncertainty_metrics.get(idx, {})
            knn = knn_metrics.get(idx, {})
            cluster_id = cluster_assignments.get(idx)
            cluster_info = (
                cluster_validation.get(cluster_id, {})
                if cluster_id is not None and cluster_id >= 0
                else {}
            )
            metadata = NovelSampleMetadata(
                text=texts[idx],
                index=int(idx),
                confidence=float(confidences[idx]),
                predicted_class=predicted_classes[idx],
                embedding_distance=knn.get("knn_nearest_distance"),
                margin_score=uncertainty.get("margin_score"),
                entropy_score=uncertainty.get("entropy_score"),
                uncertainty_score=uncertainty.get("uncertainty_score"),
                knn_novelty_score=knn.get("knn_novelty_score"),
                knn_mean_distance=knn.get("knn_mean_distance"),
                knn_nearest_distance=knn.get("knn_nearest_distance"),
                predicted_class_neighbor_ratio=knn.get(
                    "predicted_class_neighbor_ratio"
                ),
                predicted_class_support=knn.get("predicted_class_support"),
                neighbor_labels=knn.get("neighbor_labels", []),
                neighbor_distances=knn.get("neighbor_distances", []),
                cluster_id=cluster_id,
                cluster_validation_passed=bool(
                    cluster_info.get("validation_passed", False)
                ),
                cluster_support_score=cluster_support_scores.get(idx),
                novelty_score=novelty_scores.get(idx),
                signals=all_signals.get(idx, {}),
            )
            metadata_list.append(metadata)

        return metadata_list

    @staticmethod
    def _extract_scores(raw_result: Any) -> List[float]:
        """Extract ordered matcher scores from a raw top-k result payload."""
        if raw_result is None:
            return []
        if isinstance(raw_result, dict):
            return [float(raw_result.get("score", 0.0))]
        if isinstance(raw_result, list):
            scores = [
                float(item.get("score", 0.0))
                for item in raw_result
                if isinstance(item, dict)
            ]
            return scores
        return []
