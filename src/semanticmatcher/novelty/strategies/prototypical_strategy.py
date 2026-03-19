"""
Prototypical network novelty detection strategy.

Computes prototype (mean embedding) for each class and detects
novelty by distance to nearest prototype.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


class PrototypicalNoveltyDetector:
    """
    Detect novel entities using prototypical network approach.

    This strategy computes a prototype (mean embedding) for each class
    and identifies novel entities as those far from all known prototypes.
    Training is very fast - just computing means.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_threshold: float = 0.5,
        distance_metric: str = "cosine",
    ):
        """
        Initialize the prototypical detector.

        Args:
            model_name: Sentence transformer model name
            distance_threshold: Maximum distance to be considered known (0-1)
            distance_metric: Distance metric ('cosine', 'euclidean', 'mahalanobis')
        """
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric

        self.model: Optional[SentenceTransformer] = None
        self.prototypes: Dict[str, np.ndarray] = {}
        self.class_covariances: Dict[str, np.ndarray] = {}
        self.is_trained = False

    def train(
        self,
        training_data: List[Dict[str, str]],
        show_progress: bool = False,
    ) -> None:
        """
        Compute prototypes for each class.

        Args:
            training_data: List of {"text": str, "label": str} dictionaries
            show_progress: Whether to show progress messages
        """
        if not training_data:
            raise ValueError("training_data cannot be empty")

        # Validate input format
        for item in training_data:
            if "text" not in item or "label" not in item:
                raise ValueError(
                    "Each item in training_data must have 'text' and 'label' keys"
                )

        if show_progress:
            print(f"Loading sentence transformer model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        # Group by class
        class_texts: Dict[str, List[str]] = {}
        for item in training_data:
            label = item["label"]
            text = item["text"]
            if label not in class_texts:
                class_texts[label] = []
            class_texts[label].append(text)

        if show_progress:
            print(f"Computing prototypes for {len(class_texts)} classes...")

        # Compute prototype for each class
        for label, texts in class_texts.items():
            if show_progress:
                print(f"  Encoding {len(texts)} examples for class '{label}'...")

            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Prototype is mean embedding
            prototype = np.mean(embeddings, axis=0)
            self.prototypes[label] = prototype

            # For Mahalanobis distance, compute covariance matrix
            if self.distance_metric == "mahalanobis":
                centered = embeddings - prototype
                cov = np.cov(centered.T)
                # Add small regularization for numerical stability
                cov += np.eye(cov.shape[0]) * 1e-6
                self.class_covariances[label] = cov

        self.is_trained = True

        if show_progress:
            print(f"Training complete! Computed {len(self.prototypes)} prototypes.")

    def is_novel(
        self,
        text: str,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check if a text is novel.

        Args:
            text: The text to check

        Returns:
            Tuple of (is_novel: bool, distance: float, nearest_label: Optional[str])
            - is_novel: True if the text is considered novel
            - distance: Distance to nearest prototype
            - nearest_label: Label of nearest prototype (None if no prototypes)
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling is_novel()")

        if self.model is None or not self.prototypes:
            raise RuntimeError("Model or prototypes not initialized")

        embedding = self.model.encode([text], convert_to_numpy=True)[0]

        # Find nearest prototype
        nearest_label = None
        min_distance = float("inf")

        for label, prototype in self.prototypes.items():
            distance = self._compute_distance(embedding, prototype, label)

            if distance < min_distance:
                min_distance = distance
                nearest_label = label

        is_novel = min_distance > self.distance_threshold

        return is_novel, float(min_distance), nearest_label

    def score_batch(
        self,
        texts: List[str],
    ) -> List[Tuple[bool, float, Optional[str]]]:
        """
        Score novelty for a batch of texts.

        Args:
            texts: List of texts to score

        Returns:
            List of (is_novel, distance, nearest_label) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling score_batch()")

        if self.model is None or not self.prototypes:
            raise RuntimeError("Model or prototypes not initialized")

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        results = []
        for embedding in embeddings:
            nearest_label = None
            min_distance = float("inf")

            for label, prototype in self.prototypes.items():
                distance = self._compute_distance(embedding, prototype, label)

                if distance < min_distance:
                    min_distance = distance
                    nearest_label = label

            is_novel = min_distance > self.distance_threshold
            results.append((is_novel, float(min_distance), nearest_label))

        return results

    def _compute_distance(
        self,
        embedding: np.ndarray,
        prototype: np.ndarray,
        label: str,
    ) -> float:
        """Compute distance between embedding and prototype."""
        emb_reshaped = embedding.reshape(1, -1)
        proto_reshaped = prototype.reshape(1, -1)

        if self.distance_metric == "cosine":
            distances = cosine_distances(emb_reshaped, proto_reshaped)
            return float(distances[0, 0])

        elif self.distance_metric == "euclidean":
            distances = euclidean_distances(emb_reshaped, proto_reshaped)
            return float(distances[0, 0])

        elif self.distance_metric == "mahalanobis":
            from scipy.spatial.distance import mahalanobis

            cov = self.class_covariances.get(label)
            if cov is None:
                # Fallback to euclidean if covariance not available
                distances = euclidean_distances(emb_reshaped, proto_reshaped)
                return float(distances[0, 0])

            try:
                inv_cov = np.linalg.inv(cov)
                distance = mahalanobis(embedding, prototype, inv_cov)
                return float(distance)
            except np.linalg.LinAlgError:
                # Fallback to euclidean if matrix is singular
                distances = euclidean_distances(emb_reshaped, proto_reshaped)
                return float(distances[0, 0])

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def get_prototype_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all prototypes.

        Returns:
            Dictionary mapping class names to prototype information
        """
        info = {}

        for label, prototype in self.prototypes.items():
            info[label] = {
                "prototype_norm": float(np.linalg.norm(prototype)),
                "prototype_mean": float(np.mean(prototype)),
                "prototype_std": float(np.std(prototype)),
                "has_covariance": label in self.class_covariances,
            }

        return info

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Directory path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained detector")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save prototypes
        np.save(path / "prototypes.npy", self.prototypes)

        # Save covariances if using mahalanobis
        if self.class_covariances:
            np.save(path / "covariances.npy", self.class_covariances)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "distance_threshold": self.distance_threshold,
            "distance_metric": self.distance_metric,
            "is_trained": self.is_trained,
            "num_classes": len(self.prototypes),
            "class_names": list(self.prototypes.keys()),
        }

        import json

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PrototypicalNoveltyDetector":
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing the saved model

        Returns:
            Loaded PrototypicalNoveltyDetector instance
        """
        path = Path(path)

        # Load metadata
        import json

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance with saved parameters
        detector = cls(
            model_name=metadata["model_name"],
            distance_threshold=metadata["distance_threshold"],
            distance_metric=metadata["distance_metric"],
        )

        # Load prototypes
        detector.prototypes = np.load(path / "prototypes.npy", allow_pickle=True).item()

        # Load covariances if available
        cov_path = path / "covariances.npy"
        if cov_path.exists():
            detector.class_covariances = np.load(cov_path, allow_pickle=True).item()

        detector.is_trained = metadata["is_trained"]

        # Load the sentence transformer model
        detector.model = SentenceTransformer(metadata["model_name"])

        return detector
