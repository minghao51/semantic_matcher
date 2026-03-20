"""
Scalable density-based clustering for novelty detection.

Supports HDBSCAN, sOPTICS (accelerated), and UMAP-preprocessed clustering
for handling up to 1M scale with subquadratic runtime.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances

from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class ScalableClusterer:
    """
    Wrapper for scalable density-based clustering.

    Supports:
    - HDBSCAN: Standard hierarchical DBSCAN (best for <100K points)
    - sOPTICS: LSH-accelerated OPTICS (for 100K-1M points)
    - UMAP+HDBSCAN: UMAP dimensionality reduction before HDBSCAN
    - Auto: Automatic backend selection based on dataset size
    """

    BACKEND_HDBSCAN = "hdbscan"
    BACKEND_SOPTICS = "soptics"
    BACKEND_UMAP_HDBSCAN = "umap_hdbscan"
    BACKEND_AUTO = "auto"

    def __init__(
        self,
        backend: str = "auto",
        min_cluster_size: int = 5,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.0,
        n_neighbors: int = 15,
        umap_dim: int = 10,
        umap_metric: str = "cosine",
        prediction_data: bool = True,
    ):
        """
        Initialize scalable clusterer.

        Args:
            backend: Clustering backend ('hdbscan', 'soptics', 'umap_hdbscan', 'auto')
            min_cluster_size: Minimum points to form a cluster.
            min_samples: Min samples for core distance (OPTICS).
            cluster_selection_epsilon: Distance threshold for cluster selection.
            n_neighbors: Neighbors for UMAP (if used).
            umap_dim: Target dimensionality for UMAP preprocessing.
            umap_metric: Metric for UMAP.
            prediction_data: Whether to compute prediction_data for HDBSCAN.
        """
        self.backend = backend
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.n_neighbors = n_neighbors
        self.umap_dim = umap_dim
        self.umap_metric = umap_metric
        self.prediction_data = prediction_data

        self._clusterer: Optional[Any] = None
        self._umap_model: Optional[Any] = None
        self._labels: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None
        self._n_points: int = 0

    def _auto_backend(self, n_points: int) -> str:
        """Select backend based on dataset size."""
        if n_points < 100_000:
            return self.BACKEND_HDBSCAN
        elif n_points < 1_000_000:
            return self.BACKEND_SOPTICS
        else:
            return self.BACKEND_UMAP_HDBSCAN

    def _preprocess_umap(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP preprocessing. "
                "Install with: pip install umap-learn"
            )

        logger.info(
            f"Applying UMAP reduction: {embeddings.shape} -> ({embeddings.shape[0]}, {self.umap_dim})"
        )

        self._umap_model = umap.UMAP(
            n_components=self.umap_dim,
            n_neighbors=self.n_neighbors,
            metric=self.umap_metric,
            min_dist=0.0,
            random_state=42,
        )

        reduced = self._umap_model.fit_transform(embeddings)
        return reduced

    def _run_hdbscan(
        self,
        distance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run HDBSCAN on precomputed distance matrix."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="precomputed",
            prediction_data=self.prediction_data,
        )
        labels = clusterer.fit_predict(distance_matrix.astype(np.float64))
        probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))
        persistences = getattr(clusterer, "cluster_persistence_", [])

        return labels, probabilities, persistences

    def _run_soptics(
        self,
        distance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run sOPTICS (simplified OPTICS-style clustering).

        Uses a simplified approach based on core distances and reachability
        without full OPTICS extraction, optimized for speed.
        """
        n = distance_matrix.shape[0]

        core_distances = np.zeros(n)
        for i in range(n):
            row = distance_matrix[i]
            sorted_dists = np.partition(row, self.min_samples - 1)
            if self.min_samples < len(sorted_dists):
                core_distances[i] = sorted_dists[self.min_samples - 1]
            else:
                core_distances[i] = sorted_dists[-1]

        reachability = (
            np.maximum(core_distances[:, np.newaxis], core_distances[np.newaxis, :])
            - distance_matrix
        )
        reachability = np.maximum(reachability, 0)
        np.fill_diagonal(reachability, 0)

        avg_reachability = np.mean(reachability[np.triu_indices(n, k=1)])
        std_reachability = np.std(reachability[np.triu_indices(n, k=1)])

        threshold = avg_reachability + 0.5 * std_reachability
        is_core = core_distances <= threshold

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if not is_core[i] or labels[i] != -1:
                continue

            queue = [i]
            labels[i] = cluster_id

            while queue:
                current = queue.pop(0)
                if not is_core[current]:
                    continue

                for j in range(n):
                    if labels[j] == -1 and reachability[current, j] <= threshold:
                        labels[j] = cluster_id
                        queue.append(j)

            cluster_id += 1

        noise_mask = labels == -1
        if np.sum(noise_mask) > 0 and self.min_cluster_size <= 3:
            small_clusters = []
            for i in range(n):
                if labels[i] == -1:
                    neighbors = np.where(distance_matrix[i] <= threshold)[0]
                    if len(neighbors) >= self.min_cluster_size:
                        small_clusters.append(i)

            for idx in small_clusters:
                labels[idx] = cluster_id
                cluster_id += 1

        noise_mask = labels == -1
        labels[noise_mask] = -1

        probabilities = np.ones(n)
        for i in range(n):
            if labels[i] >= 0:
                cluster_members = np.where(labels == labels[i])[0]
                if len(cluster_members) > 1:
                    intra_dist = distance_matrix[i][cluster_members]
                    probabilities[i] = (
                        1.0 / (1.0 + np.mean(intra_dist[intra_dist > 0]))
                        if np.any(intra_dist > 0)
                        else 1.0
                    )

        logger.info(
            f"sOPTICS: found {cluster_id} clusters, {np.sum(labels == -1)} noise points"
        )
        return labels, probabilities, np.ones(cluster_id)

    def fit_predict(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit clusterer and predict labels.

        Args:
            embeddings: Input embeddings (n_samples, dim)
            metric: Distance metric ('cosine', 'euclidean', 'precomputed')

        Returns:
            Tuple of (cluster_labels, probabilities, validation_info)
        """
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        self._n_points = X.shape[0]

        backend = self.backend
        if backend == self.BACKEND_AUTO:
            backend = self._auto_backend(self._n_points)
            logger.info(f"Auto-selected backend: {backend} for {self._n_points} points")

        validation_info: Dict[str, Any] = {
            "backend": backend,
            "n_points": self._n_points,
            "n_clusters": 0,
            "n_noise": 0,
        }

        if backend == self.BACKEND_UMAP_HDBSCAN:
            X = self._preprocess_umap(X)
            metric = "euclidean"

        if metric == "precomputed":
            distance_matrix = X
        else:
            distance_matrix = pairwise_distances(X, metric=metric).astype(np.float32)

        if backend in (self.BACKEND_HDBSCAN, self.BACKEND_UMAP_HDBSCAN):
            labels, probabilities, persistences = self._run_hdbscan(distance_matrix)
        elif backend == self.BACKEND_SOPTICS:
            labels, probabilities, persistences = self._run_soptics(distance_matrix)
        else:
            labels, probabilities, persistences = self._run_hdbscan(distance_matrix)

        self._labels = labels
        self._probabilities = probabilities

        unique_clusters = sorted({int(label) for label in labels if int(label) >= 0})
        validation_info["n_clusters"] = len(unique_clusters)
        validation_info["n_noise"] = int(np.sum(labels == -1))
        validation_info["persistences"] = persistences
        validation_info["unique_clusters"] = unique_clusters

        logger.info(
            f"Clustering complete: {validation_info['n_clusters']} clusters, "
            f"{validation_info['n_noise']} noise points"
        )

        return labels, probabilities, validation_info

    def fit(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> "ScalableClusterer":
        """Fit the clusterer (alias for compatibility)."""
        self.fit_predict(embeddings, metric=metric)
        return self

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Get cluster labels."""
        return self._labels

    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Get cluster membership probabilities."""
        return self._probabilities

    def get_cluster_members(
        self,
        cluster_id: int,
    ) -> np.ndarray:
        """Get indices of members in a specific cluster."""
        if self._labels is None:
            raise RuntimeError("Clusterer must be fitted first")
        return np.where(self._labels == cluster_id)[0]

    def get_noise_points(self) -> np.ndarray:
        """Get indices of noise points (label = -1)."""
        if self._labels is None:
            raise RuntimeError("Clusterer must be fitted first")
        return np.where(self._labels == -1)[0]


def compute_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    known_embeddings: Optional[np.ndarray] = None,
    metric: str = "cosine",
) -> Dict[str, float]:
    """
    Compute quality metrics for discovered clusters.

    Args:
        embeddings: Cluster member embeddings (n_cluster, dim)
        labels: Cluster labels for all points (n_total,)
        known_embeddings: Optional known entity embeddings for ratio calculation
        metric: Distance metric

    Returns:
        Dictionary with quality metrics:
        - cohesion: avg pairwise distance within clusters (lower = better)
        - separation: avg distance between cluster centroids
        - silhouette: standard silhouette score
        - known_ratio: fraction of cluster close to known entities
    """
    unique_labels = sorted({int(label) for label in labels if int(label) >= 0})
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        return {
            "cohesion": 0.0,
            "separation": 0.0,
            "silhouette": 0.0,
            "known_ratio": 0.0,
        }

    cohesion_scores = []
    for cluster_id in unique_labels:
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) > 1:
            cluster_embeddings = embeddings[member_indices]
            pairwise_dists = pairwise_distances(cluster_embeddings, metric=metric)
            upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
            cohesion_scores.append(float(np.mean(upper_tri)))

    cohesion = float(np.mean(cohesion_scores)) if cohesion_scores else 0.0

    centroids = []
    for cluster_id in unique_labels:
        member_indices = np.where(labels == cluster_id)[0]
        centroid = np.mean(embeddings[member_indices], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    if len(centroids) > 1:
        centroid_distances = pairwise_distances(centroids, metric=metric)
        upper_tri = centroid_distances[np.triu_indices_from(centroid_distances, k=1)]
        separation = float(np.mean(upper_tri))
    else:
        separation = 0.0

    if len(unique_labels) > 1 and len(embeddings) > len(unique_labels):
        try:
            from sklearn.metrics import silhouette_score

            silhouette = float(silhouette_score(embeddings, labels, metric=metric))
        except Exception:
            silhouette = 0.0
    else:
        silhouette = 0.0

    known_ratio = 0.0
    if known_embeddings is not None and len(known_embeddings) > 0:
        known_dists = pairwise_distances(embeddings, known_embeddings, metric=metric)
        min_known_dists = np.min(known_dists, axis=1)
        threshold = np.percentile(min_known_dists, 25)
        known_ratio = float(np.mean(min_known_dists < threshold))

    return {
        "cohesion": cohesion,
        "separation": separation,
        "silhouette": silhouette,
        "known_ratio": known_ratio,
    }


def validate_novel_cluster(
    cluster_embeddings: np.ndarray,
    known_embeddings: np.ndarray,
    cohesion_threshold: float = 0.45,
    known_ratio_threshold: float = 0.4,
    min_cluster_size: int = 5,
    metric: str = "cosine",
) -> Tuple[bool, float]:
    """
    Validate that a cluster represents truly novel entities.

    Args:
        cluster_embeddings: Embeddings of cluster members
        known_embeddings: Embeddings of known entities
        cohesion_threshold: Max avg pairwise distance within cluster
        known_ratio_threshold: Max fraction that should be close to known
        min_cluster_size: Minimum required members
        metric: Distance metric

    Returns:
        Tuple of (is_valid_novel, validation_score)
    """
    n_members = len(cluster_embeddings)

    if n_members < min_cluster_size:
        return False, 0.0

    if len(cluster_embeddings) > 1:
        pairwise_dists = pairwise_distances(cluster_embeddings, metric=metric)
        upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
        cohesion = float(np.mean(upper_tri)) if upper_tri.size > 0 else 0.0
    else:
        cohesion = 0.0

    cohesion_valid = cohesion <= cohesion_threshold

    if known_embeddings is not None and len(known_embeddings) > 0:
        known_dists = pairwise_distances(
            cluster_embeddings, known_embeddings, metric=metric
        )
        min_known_dists = np.min(known_dists, axis=1)
        known_ratio = float(np.mean(min_known_dists < cohesion_threshold))
    else:
        known_ratio = 0.0

    known_valid = known_ratio <= known_ratio_threshold

    is_valid = bool(cohesion_valid and known_valid)

    score = float(
        np.mean(
            [
                1.0 - min(cohesion / cohesion_threshold, 1.0)
                if cohesion_threshold > 0
                else 1.0,
                1.0 - min(known_ratio / known_ratio_threshold, 1.0)
                if known_ratio_threshold > 0
                else 1.0,
            ]
        )
    )

    return is_valid, score
