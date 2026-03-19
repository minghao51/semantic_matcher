"""
Prototype Network Detector for few-shot novel entity discovery.

Implements CEPTNER-style contrastive prototypical learning for
entity-level novelty detection. Builds class prototypes from support
embeddings and scores queries based on distance to prototypes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances

from semanticmatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class PrototypeNetwork:
    """
    Contrastive Prototype Network for entity-level classification.

    Builds class prototypes from support set embeddings and uses
    distance-based classification with contrastive loss.
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        temperature: float = 0.1,
        contrastive_margin: float = 0.5,
        prototype_type: str = "mean",  # "mean", "weighted", "multi"
        num_prototypes: int = 1,
    ):
        """
        Initialize prototype network.

        Args:
            distance_metric: Distance metric for similarity ("cosine", "euclidean")
            temperature: Temperature for softmax-based distance scaling
            contrastive_margin: Margin for contrastive loss
            prototype_type: How to build prototypes ("mean", "weighted", "multi")
            num_prototypes: Number of prototypes per class (for multi-prototype)
        """
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.contrastive_margin = contrastive_margin
        self.prototype_type = prototype_type
        self.num_prototypes = num_prototypes

        self.class_prototypes: Dict[str, np.ndarray] = {}
        self.class_embeddings: Dict[str, np.ndarray] = {}
        self._is_fitted = False
        self._embedding_dim: Optional[int] = None

    def fit(
        self,
        support_embeddings: np.ndarray,
        support_labels: List[str],
        known_classes: Optional[List[str]] = None,
    ) -> "PrototypeNetwork":
        """
        Build class prototypes from support set.

        Args:
            support_embeddings: Embeddings of support examples (n, dim)
            support_labels: Labels for support examples
            known_classes: Optional list of known classes (uses all seen if None)

        Returns:
            self
        """
        X = np.asarray(support_embeddings, dtype=np.float32)
        labels = list(support_labels)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got {X.ndim}D")
        self._embedding_dim = X.shape[1]

        if len(X) != len(labels):
            raise ValueError(f"Mismatch: {len(X)} embeddings but {len(labels)} labels")

        classes = known_classes or sorted(set(labels))

        self.class_prototypes = {}
        self.class_embeddings = {}

        for cls in classes:
            mask = np.array([label == cls for label in labels])
            cls_embeddings = X[mask]

            if len(cls_embeddings) == 0:
                continue

            self.class_embeddings[cls] = cls_embeddings

            if self.prototype_type == "mean":
                prototype = np.mean(cls_embeddings, axis=0)
                self.class_prototypes[cls] = prototype

            elif self.prototype_type == "weighted":
                distances = pairwise_distances(
                    cls_embeddings, metric=self.distance_metric
                )
                avg_dist = np.mean(distances, axis=1)
                weights = 1.0 / (avg_dist + 1e-8)
                weights = weights / weights.sum()
                prototype = np.average(cls_embeddings, axis=0, weights=weights)
                self.class_prototypes[cls] = prototype

            elif self.prototype_type == "multi":
                prototypes = []
                n_proto = min(self.num_prototypes, len(cls_embeddings))
                if n_proto == 1:
                    prototypes.append(np.mean(cls_embeddings, axis=0))
                else:
                    from sklearn.cluster import KMeans

                    kmeans = KMeans(n_clusters=n_proto, random_state=42, n_init=10)
                    kmeans.fit(cls_embeddings)
                    for centroid in kmeans.cluster_centers_:
                        prototypes.append(centroid)
                self.class_prototypes[cls] = np.array(prototypes)

        self._is_fitted = True
        logger.info(
            f"Built {len(self.class_prototypes)} class prototypes "
            f"(type={self.prototype_type})"
        )
        return self

    def compute_distances_to_prototypes(
        self,
        query_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distances from queries to all class prototypes.

        Args:
            query_embeddings: Query embeddings (n, dim)

        Returns:
            Tuple of (min_distances, prototype_labels)
            - min_distances: (n,) distance to nearest prototype per class
            - prototype_labels: (n,) class of nearest prototype
        """
        if not self._is_fitted:
            raise RuntimeError("PrototypeNetwork must be fitted before inference")

        X = np.asarray(query_embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        all_distances = []
        all_labels = []

        for cls, prototypes in self.class_prototypes.items():
            if self.prototype_type == "multi":
                prototypes = np.array(prototypes)
                dists = pairwise_distances(X, prototypes, metric=self.distance_metric)
                min_dists = np.min(dists, axis=1)
            else:
                dists = pairwise_distances(
                    X, prototypes.reshape(1, -1), metric=self.distance_metric
                )
                min_dists = dists.flatten()

            all_distances.append(min_dists)
            all_labels.append(cls)

        all_distances = np.array(all_distances).T
        all_labels = np.array(all_labels)

        min_distances = np.min(all_distances, axis=1)
        nearest_class_idx = np.argmin(all_distances, axis=1)
        nearest_labels = all_labels[nearest_class_idx]

        return min_distances, nearest_labels

    def compute_novelty_scores(
        self,
        query_embeddings: np.ndarray,
        known_classes: Optional[List[str]] = None,
        return_distances: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute novelty scores based on distance to known class prototypes.

        Higher score = more novel (far from all known classes).

        Args:
            query_embeddings: Query embeddings
            known_classes: Optional subset of classes considered "known"
            return_distances: If True, also return per-class distances

        Returns:
            Tuple of (novelty_scores, per_class_distances or None)
        """
        if not self._is_fitted:
            raise RuntimeError("PrototypeNetwork must be fitted before inference")

        X = np.asarray(query_embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        classes_to_use = known_classes or list(self.class_prototypes.keys())

        target_prototypes = []
        for cls in classes_to_use:
            if cls in self.class_prototypes:
                target_prototypes.append((cls, self.class_prototypes[cls]))

        if not target_prototypes:
            novelty = np.ones(len(X))
            return novelty, None

        all_class_distances = []
        for cls, prototype in target_prototypes:
            if self.prototype_type == "multi":
                prototype = np.array(prototype)
                dists = pairwise_distances(X, prototype, metric=self.distance_metric)
                min_dists = np.min(dists, axis=1)
            else:
                dists = pairwise_distances(
                    X, prototype.reshape(1, -1), metric=self.distance_metric
                )
                min_dists = dists.flatten()
            all_class_distances.append(min_dists)

        all_class_distances = np.array(all_class_distances).T

        if self.distance_metric == "cosine":
            all_class_distances = np.clip(all_class_distances, 0, 2)

        min_distances = np.min(all_class_distances, axis=1)

        novelty_scores = np.clip(min_distances / (self.temperature + 1e-8), 0, 1)

        if return_distances:
            return novelty_scores, all_class_distances
        return novelty_scores, None

    def classify(
        self,
        query_embeddings: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Classify queries to nearest known class.

        Args:
            query_embeddings: Query embeddings

        Returns:
            Tuple of (predicted_labels, distances)
        """
        min_distances, nearest_labels = self.compute_distances_to_prototypes(
            query_embeddings
        )
        return list(nearest_labels), min_distances


class PrototypeDetector:
    """
    Novelty detector using prototype networks.

    Identifies novel entities by measuring distance to known class
    prototypes - novel entities are far from all known classes.
    """

    def __init__(
        self,
        prototype_threshold: float = 0.5,
        temperature: float = 0.1,
        use_contrastive: bool = True,
        min_cluster_size: int = 3,
    ):
        """
        Initialize prototype detector.

        Args:
            prototype_threshold: Distance threshold for novelty (higher = more novel)
            temperature: Temperature for softmax distance scaling
            use_contrastive: Use contrastive learning for prototype refinement
            min_cluster_size: Minimum samples for prototype construction
        """
        self.prototype_threshold = prototype_threshold
        self.temperature = temperature
        self.use_contrastive = use_contrastive
        self.min_cluster_size = min_cluster_size

        self.prototype_network: Optional[PrototypeNetwork] = None
        self._is_fitted = False
        self._known_classes: List[str] = []

    def fit(
        self,
        known_embeddings: np.ndarray,
        known_labels: List[str],
        known_classes: Optional[List[str]] = None,
    ) -> "PrototypeDetector":
        """
        Build prototypes from known entity embeddings.

        Args:
            known_embeddings: Embeddings of known entities (n, dim)
            known_labels: Labels for known entities
            known_classes: Optional explicit list of known classes

        Returns:
            self
        """
        self._known_classes = known_classes or sorted(set(known_labels))

        valid_mask = np.array([label in self._known_classes for label in known_labels])
        valid_embeddings = np.asarray(known_embeddings, dtype=np.float32)[valid_mask]
        valid_labels = [label for label, m in zip(known_labels, valid_mask) if m]

        if len(valid_embeddings) < self.min_cluster_size:
            logger.warning(
                f"Only {len(valid_embeddings)} valid samples for "
                f"{len(self._known_classes)} classes"
            )

        self.prototype_network = PrototypeNetwork(
            temperature=self.temperature,
            prototype_type="mean",
        )
        self.prototype_network.fit(
            support_embeddings=valid_embeddings,
            support_labels=valid_labels,
            known_classes=self._known_classes,
        )
        self._is_fitted = True
        return self

    def detect_novel_samples(
        self,
        embeddings: np.ndarray,
        threshold: Optional[float] = None,
        return_scores: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect novel samples based on prototype distances.

        Args:
            embeddings: Query embeddings (n, dim)
            threshold: Override for prototype_threshold
            return_scores: If True, return (novelty_scores, is_novel).
                          If False, return (is_novel,) only.

        Returns:
            Tuple of (novelty_scores, is_novel_mask)
        """
        if not self._is_fitted:
            raise RuntimeError("PrototypeDetector must be fitted before detection")

        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        novelty_scores, _ = self.prototype_network.compute_novelty_scores(
            X,
            known_classes=self._known_classes,
        )

        thresh = threshold if threshold is not None else self.prototype_threshold
        is_novel = novelty_scores >= thresh

        if return_scores:
            return novelty_scores, is_novel
        return is_novel

    def compute_prototype_distances(
        self,
        embeddings: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute distances to each known class prototype.

        Args:
            embeddings: Query embeddings

        Returns:
            Dictionary mapping class name to distance array
        """
        if not self._is_fitted:
            raise RuntimeError("PrototypeDetector must be fitted before detection")

        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        _, per_class_dists = self.prototype_network.compute_novelty_scores(
            X,
            known_classes=self._known_classes,
            return_distances=True,
        )

        result = {}
        for i, cls in enumerate(self._known_classes):
            if per_class_dists is not None:
                result[cls] = per_class_dists[:, i]

        return result

    @property
    def known_classes(self) -> List[str]:
        """Get list of known classes."""
        return self._known_classes
