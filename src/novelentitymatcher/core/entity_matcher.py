from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .bert_classifier import BERTClassifier
from .classifier import SetFitClassifier
from .matcher_shared import (
    TextInput,
    coerce_texts,
    normalize_texts,
    normalize_training_data,
    resolve_threshold,
    unwrap_single,
)
from .normalizer import TextNormalizer
from ..utils.validation import (
    validate_entities,
    validate_model_name,
    validate_threshold,
)


class EntityMatcher:
    """SetFit-based or BERT-based entity matching with optional text normalization."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
        classifier_type: str = "setfit",
    ):
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize
        self.classifier_type = classifier_type

        self.normalizer = TextNormalizer() if normalize else None
        self.classifier: Optional[Union[SetFitClassifier, BERTClassifier]] = None
        self.is_trained = False
        self._async_executor = None
        self._reference_texts: List[str] = []
        self._reference_labels: List[str] = []
        self._reference_embeddings: Optional[np.ndarray] = None

    def _ensure_async_executor(self):
        if self._async_executor is None:
            from .async_utils import AsyncExecutor

            self._async_executor = AsyncExecutor()
        return self._async_executor

    def _get_training_data(self, training_data: List[dict]) -> List[dict]:
        return normalize_training_data(training_data, self.normalizer, self.normalize)

    def train(
        self,
        training_data: List[dict],
        num_epochs: int = 4,
        batch_size: int = 16,
        show_progress: bool = True,
    ):
        normalized_data = self._get_training_data(training_data)
        labels = list(dict.fromkeys(item["label"] for item in normalized_data))

        if self.classifier_type == "bert":
            self.classifier = BERTClassifier(
                labels=labels,
                model_name=self.model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
        else:
            self.classifier = SetFitClassifier(
                labels=labels,
                model_name=self.model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )

        self.classifier.train(
            normalized_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        self.is_trained = True
        self._reference_texts = [item["text"] for item in normalized_data]
        self._reference_labels = [item["label"] for item in normalized_data]
        self._reference_embeddings = None

    def predict(self, texts: TextInput) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")

        matches = self.match(texts, top_k=1)
        if isinstance(matches, list):
            return [match["id"] if match else None for match in matches]
        return matches["id"] if matches else None

    def match(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
    ) -> Any:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        texts, single_input = coerce_texts(texts)
        texts = normalize_texts(texts, self.normalizer, self.normalize)
        entity_lookup = {entity["id"]: entity for entity in self.entities}
        candidate_ids = None
        effective_threshold = resolve_threshold(threshold_override, self.threshold)
        if candidates is not None:
            candidate_ids = {candidate["id"] for candidate in candidates}

        results = []
        for text in texts:
            try:
                proba = self.classifier.predict_proba(text)
                ranked_matches = sorted(
                    zip(self.classifier.labels, proba),
                    key=lambda item: item[1],
                    reverse=True,
                )
                matches = []
                for label, score in ranked_matches:
                    score = float(score)
                    if score < effective_threshold:
                        continue
                    if candidate_ids is not None and label not in candidate_ids:
                        continue
                    entity = entity_lookup.get(label, {})
                    matches.append(
                        {
                            "id": label,
                            "score": score,
                            "text": entity.get("name", ""),
                        }
                    )
                    if len(matches) >= top_k:
                        break
                if top_k == 1:
                    results.append(matches[0] if matches else None)
                else:
                    results.append(matches)
            except ValueError:
                results.append(None if top_k == 1 else [])

        return unwrap_single(results, single_input)

    async def train_async(
        self,
        training_data: List[dict],
        num_epochs: int = 4,
        batch_size: int = 16,
        show_progress: bool = True,
    ):
        await self._ensure_async_executor().run_in_thread(
            self.train,
            training_data,
            num_epochs,
            batch_size,
            show_progress,
        )

    async def match_async(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
    ) -> Any:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError(
                "Model not trained. Call train() or train_async() first."
            )

        return await self._ensure_async_executor().run_in_thread(
            self.match,
            texts,
            candidates,
            top_k,
            threshold_override,
        )

    async def predict_async(
        self, texts: TextInput
    ) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError(
                "Model not trained. Call train() or train_async() first."
            )

        matches = await self.match_async(texts, top_k=1)
        if isinstance(matches, list):
            return [match["id"] if match else None for match in matches]
        return matches["id"] if matches else None

    def get_reference_corpus(self, encoder: Optional[Any] = None) -> Dict[str, Any]:
        if not self._reference_texts or not self._reference_labels:
            raise RuntimeError(
                "No reference corpus available. Train the matcher before novelty detection."
            )

        if self._reference_embeddings is None:
            encode_model = encoder
            if encode_model is None and self.classifier is not None:
                encode_model = getattr(self.classifier, "model", None)
            if encode_model is None or not hasattr(encode_model, "encode"):
                raise RuntimeError(
                    "Could not derive reference embeddings from the trained matcher. "
                    "Provide an embedding-capable encoder or use a matcher mode with "
                    "reference embeddings."
                )
            embeddings = encode_model.encode(self._reference_texts)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            self._reference_embeddings = embeddings

        return {
            "texts": list(self._reference_texts),
            "labels": list(self._reference_labels),
            "embeddings": self._reference_embeddings,
            "source": "training_examples",
        }
