from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .classifier import SetFitClassifier
from .normalizer import TextNormalizer
from ..utils.validation import (
    validate_entities,
    validate_model_name,
    validate_threshold,
)


TextInput = Union[str, List[str]]


def _coerce_texts(texts: TextInput) -> Tuple[List[str], bool]:
    if isinstance(texts, str):
        return [texts], True
    return texts, False


def _unwrap_single(results: List[Any], single_input: bool) -> Any:
    if single_input:
        return results[0]
    return results


def _normalize_texts(
    texts: List[str],
    normalizer: Optional[TextNormalizer],
    normalize: bool,
) -> List[str]:
    if not (normalizer and normalize):
        return texts
    return [normalizer.normalize(text) for text in texts]


def _normalize_training_data(
    training_data: List[dict],
    normalizer: Optional[TextNormalizer],
    normalize: bool,
) -> List[dict]:
    if not (normalizer and normalize):
        return training_data
    return [
        {"text": normalizer.normalize(item["text"]), "label": item["label"]}
        for item in training_data
    ]


def _flatten_entity_texts(entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    entity_texts: List[str] = []
    entity_ids: List[str] = []
    for entity in entities:
        entity_texts.append(entity["name"])
        entity_ids.append(entity["id"])
        for alias in entity.get("aliases", []):
            entity_texts.append(alias)
            entity_ids.append(entity["id"])
    return entity_texts, entity_ids


class EntityMatcher:
    """SetFit-based entity matching with optional text normalization."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
    ):
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize

        self.normalizer = TextNormalizer() if normalize else None
        self.classifier: Optional[SetFitClassifier] = None
        self.is_trained = False

    def _get_training_data(self, training_data: List[dict]) -> List[dict]:
        return _normalize_training_data(training_data, self.normalizer, self.normalize)

    def train(
        self,
        training_data: List[dict],
        num_epochs: int = 4,
        batch_size: int = 16,
    ):
        normalized_data = self._get_training_data(training_data)
        labels = list(set(item["label"] for item in normalized_data))

        self.classifier = SetFitClassifier(
            labels=labels,
            model_name=self.model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        self.classifier.train(normalized_data, num_epochs=num_epochs, batch_size=batch_size)
        self.is_trained = True

    def predict(
        self,
        texts: TextInput
    ) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)

        predictions = []
        for text in texts:
            try:
                pred = self.classifier.predict(text)
                predictions.append(pred)
            except ValueError:
                predictions.append(None)

        return _unwrap_single(predictions, single_input)


class EmbeddingMatcher:
    """Embedding-based similarity matching without training."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
    ):
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize

        self.normalizer = TextNormalizer() if normalize else None
        self.model: Optional[SentenceTransformer] = None
        self.entity_texts: List[str] = []
        self.entity_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self):
        self.model = SentenceTransformer(self.model_name)

        self.entity_texts, self.entity_ids = _flatten_entity_texts(self.entities)
        self.entity_texts = _normalize_texts(
            self.entity_texts, self.normalizer, self.normalize
        )

        self.embeddings = self.model.encode(self.entity_texts)

    def match(
        self,
        texts: TextInput
    ) -> Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
        if self.embeddings is None or self.model is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)

        query_embeddings = self.model.encode(texts)
        similarities = cosine_similarity(query_embeddings, self.embeddings)

        results = []
        for i, sim_row in enumerate(similarities):
            best_idx = np.argmax(sim_row)
            best_score = sim_row[best_idx]

            if best_score < self.threshold:
                results.append(None)
            else:
                results.append({
                    "id": self.entity_ids[best_idx],
                    "score": float(best_score),
                })

        return _unwrap_single(results, single_input)
