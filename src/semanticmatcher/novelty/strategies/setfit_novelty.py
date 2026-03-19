"""
SetFit contrastive novelty detection strategy.

Uses SetFit with contrastive loss for novelty detection with
few-shot learning capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


class SetFitNoveltyDetector:
    """
    Detect novel entities using SetFit contrastive learning.

    This strategy uses SetFit's contrastive learning approach to
    learn an embedding space where known entities cluster together
    and novel entities are pushed away.
    """

    def __init__(
        self,
        known_entities: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_epochs: int = 4,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        margin: float = 0.5,
    ):
        """
        Initialize the SetFit novelty detector.

        Args:
            known_entities: List of known entity strings
            model_name: Sentence transformer model name
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            margin: Contrastive loss margin
        """
        self.known_entities = known_entities
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin

        self.model: Optional[Any] = None
        self.novelty_threshold: Optional[float] = None
        self.known_embeddings: Optional[np.ndarray] = None
        self.is_trained = False

    def train(
        self,
        synthetic_novels: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> None:
        """
        Train with contrastive loss for novelty detection.

        Args:
            synthetic_novels: Optional synthetic novel examples for training
            show_progress: Whether to show progress messages
        """
        try:
            from setfit import SetFitModel, SetFitTrainer
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "SetFit is not installed. Install with: pip install setfit"
            )

        if not self.known_entities:
            raise ValueError("known_entities cannot be empty")

        if show_progress:
            print(f"Loading SetFit model: {self.model_name}")

        # Initialize SetFit model
        self.model = SetFitModel.from_pretrained(self.model_name)

        # Prepare training data
        # We create pairs: (anchor, positive) for same entity with augmentations
        # and (anchor, negative) for different entities or synthetic novels

        train_texts = []
        train_labels = []

        # Add known entities with label 0
        for entity in self.known_entities:
            train_texts.append(entity)
            train_labels.append(0)

        # Add synthetic novels with label 1 (if provided)
        if synthetic_novels:
            for novel in synthetic_novels:
                train_texts.append(novel)
                train_labels.append(1)

        if show_progress:
            print(
                f"Training with {len(train_texts)} examples ({self.known_entities.__len__()} known, {len(synthetic_novels) if synthetic_novels else 0} novel)..."
            )

        # Create dataset
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})

        # Train SetFit model
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_dataset,
            num_iterations=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        trainer.train()

        # Encode known entities for threshold calibration
        if show_progress:
            print("Calibrating novelty threshold...")

        self.known_embeddings = self.model.model.encode(
            self.known_entities,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Calibrate threshold based on known entity similarities
        # Use 95th percentile of pairwise distances as threshold
        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(self.known_embeddings)
        # Get upper triangle (excluding diagonal)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]

        if len(upper_tri) > 0:
            self.novelty_threshold = float(np.percentile(upper_tri, 95))
        else:
            self.novelty_threshold = 0.5  # Default fallback

        self.is_trained = True

        if show_progress:
            print(f"Training complete! Threshold: {self.novelty_threshold:.4f}")

    def is_novel(self, text: str) -> Tuple[bool, float]:
        """
        Check if a text is novel.

        Args:
            text: The text to check

        Returns:
            Tuple of (is_novel: bool, confidence_score: float)
            - is_novel: True if the text is considered novel
            - confidence_score: Distance to nearest known entity (0-1)
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling is_novel()")

        if (
            self.model is None
            or self.known_embeddings is None
            or self.novelty_threshold is None
        ):
            raise RuntimeError("Model, embeddings, or threshold not initialized")

        # Encode the text
        embedding = self.model.model.encode([text], convert_to_numpy=True)[0]

        # Compute distance to nearest known entity
        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances([embedding], self.known_embeddings)[0]
        min_distance = float(np.min(distances))

        is_novel = min_distance > self.novelty_threshold

        # Normalize confidence score to 0-1 range
        # Use a sigmoid-like transformation
        confidence = 1.0 / (
            1.0 + np.exp(-5.0 * (min_distance - self.novelty_threshold))
        )

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

        if (
            self.model is None
            or self.known_embeddings is None
            or self.novelty_threshold is None
        ):
            raise RuntimeError("Model, embeddings, or threshold not initialized")

        # Encode all texts
        embeddings = self.model.model.encode(texts, convert_to_numpy=True)

        # Compute distances to known entities
        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(embeddings, self.known_embeddings)
        min_distances = np.min(distances, axis=1)

        results = []
        for min_distance in min_distances:
            is_novel = min_distance > self.novelty_threshold
            confidence = 1.0 / (
                1.0 + np.exp(-5.0 * (float(min_distance) - self.novelty_threshold))
            )
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

        # Save SetFit model
        if self.model is not None:
            self.model.save_model(str(path / "setfit_model"))

        # Save embeddings and threshold
        if self.known_embeddings is not None:
            np.save(path / "known_embeddings.npy", self.known_embeddings)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "margin": self.margin,
            "novelty_threshold": self.novelty_threshold,
            "is_trained": self.is_trained,
            "num_known_entities": len(self.known_entities),
        }

        import json

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SetFitNoveltyDetector":
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing the saved model

        Returns:
            Loaded SetFitNoveltyDetector instance
        """
        path = Path(path)

        # Load metadata
        import json

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance (known_entities will be loaded from embeddings)
        detector = cls(
            known_entities=[],  # Will be populated from embeddings
            model_name=metadata["model_name"],
            num_epochs=metadata["num_epochs"],
            batch_size=metadata["batch_size"],
            learning_rate=metadata["learning_rate"],
            margin=metadata["margin"],
        )

        # Load SetFit model
        try:
            from setfit import SetFitModel

            detector.model = SetFitModel.from_pretrained(str(path / "setfit_model"))
        except ImportError:
            raise ImportError(
                "SetFit is not installed. Install with: pip install setfit"
            )

        # Load embeddings and threshold
        detector.known_embeddings = np.load(path / "known_embeddings.npy")
        detector.novelty_threshold = metadata["novelty_threshold"]
        detector.is_trained = metadata["is_trained"]

        # Reconstruct known_entities count
        detector.known_entities = [""] * (metadata.get("num_known_entities", 0))

        return detector

    def generate_synthetic_novels(
        self,
        num_samples: int = 100,
        augmentation_methods: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate synthetic novel examples for training.

        Args:
            num_samples: Number of synthetic examples to generate
            augmentation_methods: List of augmentation methods to use

        Returns:
            List of synthetic novel strings
        """
        if augmentation_methods is None:
            augmentation_methods = ["typos", "case_change", "spacing", "substring"]

        synthetic = []

        import random

        for _ in range(num_samples):
            # Select random known entity
            base = random.choice(self.known_entities)
            method = random.choice(augmentation_methods)

            if method == "typos":
                synthetic.append(self._add_typos(base))
            elif method == "case_change":
                synthetic.append(self._change_case(base))
            elif method == "spacing":
                synthetic.append(self._modify_spacing(base))
            elif method == "substring":
                synthetic.append(self._create_substring_variant(base))
            else:
                synthetic.append(base)

        return synthetic

    def _add_typos(self, text: str, num_typos: int = 1) -> str:
        """Add random typos to text."""
        import random

        chars = list(text)
        for _ in range(num_typos):
            if chars:
                idx = random.randint(0, len(chars) - 1)
                # Common keyboard adjacent pairs
                replacements = {
                    "a": "s",
                    "s": "a",
                    "d": "s",
                    "e": "r",
                    "r": "e",
                    "t": "y",
                    "y": "t",
                    "i": "o",
                    "o": "i",
                    "n": "m",
                }
                if chars[idx] in replacements:
                    chars[idx] = replacements[chars[idx]]
        return "".join(chars)

    def _change_case(self, text: str) -> str:
        """Randomly change case."""
        import random

        methods = [str.upper, str.lower, str.title]
        return random.choice(methods)(text)

    def _modify_spacing(self, text: str) -> str:
        """Modify spacing in text."""
        import random

        if " " in text:
            # Randomly remove or duplicate spaces
            parts = text.split(" ")
            if random.random() < 0.5 and len(parts) > 1:
                # Remove space
                idx = random.randint(0, len(parts) - 2)
                parts[idx] = parts[idx] + parts[idx + 1]
                parts.pop(idx + 1)
            else:
                # Add space
                idx = random.randint(0, len(parts) - 1)
                if len(parts[idx]) > 1:
                    split_idx = random.randint(1, len(parts[idx]) - 1)
                    parts.insert(idx + 1, parts[idx][split_idx:])
                    parts[idx] = parts[idx][:split_idx]

            return " ".join(parts)

        return text

    def _create_substring_variant(self, text: str) -> str:
        """Create a substring variant."""
        import random

        if len(text) > 3:
            start = random.randint(0, len(text) // 2)
            end = random.randint(len(text) // 2 + 1, len(text))
            return text[start:end]

        return text
