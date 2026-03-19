"""
One-Class SVM novelty detection strategy.

Uses sklearn's OneClassSVM with sentence transformer embeddings
to detect novel entities without requiring negative examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.svm import OneClassSVM
from sentence_transformers import SentenceTransformer


class OneClassNoveltyDetector:
    """
    Detect novel entities using One-Class SVM classification.

    This strategy trains a One-Class SVM on known entity embeddings,
    learning the boundary of known entities. Anything outside this
    boundary is considered novel.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
    ):
        """
        Initialize the One-Class SVM detector.

        Args:
            model_name: Sentence transformer model name
            nu: Expected outlier fraction (0-1). Lower = stricter.
            kernel: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
        """
        self.model_name = model_name
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.model: Optional[SentenceTransformer] = None
        self.oc_svm: Optional[OneClassSVM] = None
        self.known_embeddings: Optional[np.ndarray] = None
        self.is_trained = False

    def train(
        self,
        known_entities: List[str],
        show_progress: bool = False,
    ) -> None:
        """
        Train the One-Class SVM on known entities.

        Args:
            known_entities: List of known entity strings
            show_progress: Whether to show progress (not used for sklearn, but for consistency)
        """
        if not known_entities:
            raise ValueError("known_entities cannot be empty")

        if show_progress:
            print(f"Loading sentence transformer model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        if show_progress:
            print(f"Encoding {len(known_entities)} known entities...")

        self.known_embeddings = self.model.encode(
            known_entities,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        if show_progress:
            print(f"Training One-Class SVM (nu={self.nu}, kernel={self.kernel})...")

        self.oc_svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.oc_svm.fit(self.known_embeddings)

        self.is_trained = True

        if show_progress:
            print("Training complete!")

    def is_novel(self, text: str) -> Tuple[bool, float]:
        """
        Check if a text is novel.

        Args:
            text: The text to check

        Returns:
            Tuple of (is_novel: bool, confidence_score: float)
            - is_novel: True if the text is considered novel
            - confidence_score: Distance from decision boundary (higher = more novel)
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling is_novel()")

        if self.model is None or self.oc_svm is None:
            raise RuntimeError("Model or SVM not initialized")

        embedding = self.model.encode([text], convert_to_numpy=True)

        # predict returns 1 for inliers (known), -1 for outliers (novel)
        prediction = self.oc_svm.predict(embedding)[0]
        is_novel = prediction == -1

        # decision_function returns signed distance to decision boundary
        # Negative values indicate outliers (novel), positive indicate inliers (known)
        distance = self.oc_svm.decision_function(embedding)[0]

        # Convert to confidence score (0-1, where 1 = definitely novel)
        # We normalize by assuming most distances will be in range [-1, 1]
        confidence = float(np.clip(-distance, 0.0, 1.0))

        return is_novel, confidence

    def score_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Score novelty for a batch of texts.

        Args:
            texts: List of texts to score

        Returns:
            List of (is_novel, confidence) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling score_batch()")

        if self.model is None or self.oc_svm is None:
            raise RuntimeError("Model or SVM not initialized")

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        predictions = self.oc_svm.predict(embeddings)
        distances = self.oc_svm.decision_function(embeddings)

        results = []
        for pred, dist in zip(predictions, distances):
            is_novel = pred == -1
            confidence = float(np.clip(-dist, 0.0, 1.0))
            results.append((is_novel, confidence))

        return results

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

        # Save SVM model using sklearn's joblib
        import joblib

        joblib.dump(self.oc_svm, path / "oc_svm_model.pkl")

        # Save embeddings and metadata
        np.save(path / "known_embeddings.npy", self.known_embeddings)

        metadata = {
            "model_name": self.model_name,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "is_trained": self.is_trained,
            "num_known_entities": len(self.known_embeddings)
            if self.known_embeddings is not None
            else 0,
        }

        import json

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OneClassNoveltyDetector":
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing the saved model

        Returns:
            Loaded OneClassNoveltyDetector instance
        """
        path = Path(path)

        # Load metadata
        import json

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance with saved parameters
        detector = cls(
            model_name=metadata["model_name"],
            nu=metadata["nu"],
            kernel=metadata["kernel"],
            gamma=metadata["gamma"],
        )

        # Load SVM model
        import joblib

        detector.oc_svm = joblib.load(path / "oc_svm_model.pkl")

        # Load embeddings
        detector.known_embeddings = np.load(path / "known_embeddings.npy")
        detector.is_trained = metadata["is_trained"]

        # Load the sentence transformer model
        detector.model = SentenceTransformer(metadata["model_name"])

        return detector

    def get_support_vectors_info(self) -> Dict[str, Any]:
        """
        Get information about support vectors.

        Returns:
            Dictionary with support vector statistics
        """
        if not self.is_trained or self.oc_svm is None:
            return {}

        n_support = (
            self.oc_svm.n_support_[0] if hasattr(self.oc_svm, "n_support_") else 0
        )

        return {
            "n_support_vectors": int(n_support),
            "dual_coef_mean": float(np.mean(np.abs(self.oc_svm.dual_coef_)))
            if hasattr(self.oc_svm, "dual_coef_")
            else 0.0,
            "intercept": float(self.oc_svm.intercept_[0])
            if hasattr(self.oc_svm, "intercept_")
            else 0.0,
        }
