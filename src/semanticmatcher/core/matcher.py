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
from ..utils.embeddings import ModelCache, get_default_cache

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


def _flatten_entity_texts(
    entities: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
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
        self.classifier.train(
            normalized_data, num_epochs=num_epochs, batch_size=batch_size
        )
        self.is_trained = True

    def predict(self, texts: TextInput) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)

        predictions = []
        for text in texts:
            try:
                proba = self.classifier.predict_proba(text)
                if float(np.max(proba)) < self.threshold:
                    predictions.append(None)
                    continue
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
        embedding_dim: Optional[int] = None,
        cache: Optional[ModelCache] = None,
    ):
        """
        Initialize the embedding matcher.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            model_name: Name of the sentence-transformer model
            threshold: Minimum similarity score threshold (0-1)
            normalize: Whether to normalize text
            embedding_dim: Optional dimension for Matryoshka embeddings
            cache: Optional ModelCache instance. If None, uses global default cache.
        """
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize
        self.embedding_dim = embedding_dim  # Matryoshka support

        self.normalizer = TextNormalizer() if normalize else None
        self.cache = cache if cache is not None else get_default_cache()
        self.model: Optional[SentenceTransformer] = None
        self.entity_texts: List[str] = []
        self.entity_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, batch_size: Optional[int] = None):
        """
        Build the embedding index from entities.

        Args:
            batch_size: Batch size for encoding. None = use model's default.
        """
        # Use cache to get or load the model
        self.model = self.cache.get_or_load(
            self.model_name, lambda: SentenceTransformer(self.model_name)
        )

        # Validate embedding_dim if provided
        if self.embedding_dim is not None:
            # Get actual model embedding dimension
            actual_dim = self.model.get_sentence_embedding_dimension()

            # Validate against model's actual dimension
            if actual_dim is not None and self.embedding_dim > actual_dim:
                raise ValueError(
                    f"embedding_dim ({self.embedding_dim}) cannot exceed "
                    f"model embedding dimension ({actual_dim})"
                )

            # Validate positive value
            if self.embedding_dim <= 0:
                raise ValueError(
                    f"embedding_dim must be positive, got {self.embedding_dim}"
                )

        self.entity_texts, self.entity_ids = _flatten_entity_texts(self.entities)
        self.entity_texts = _normalize_texts(
            self.entity_texts, self.normalizer, self.normalize
        )

        if batch_size is not None:
            self.embeddings = self.model.encode(
                self.entity_texts, batch_size=batch_size
            )
        else:
            self.embeddings = self.model.encode(self.entity_texts)

        # Matryoshka embedding support: truncate to specified dimension
        if (
            self.embedding_dim is not None
            and self.embeddings.shape[1] > self.embedding_dim
        ):
            self.embeddings = self.embeddings[:, : self.embedding_dim]

    def match(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        batch_size: Optional[int] = None,
    ) -> Any:
        """
        Match texts against indexed entities.

        Args:
            texts: Query text(s) to match
            candidates: Optional list of candidate entities to restrict search
            top_k: Number of top results to return
            batch_size: Batch size for encoding queries. None = use model's default.

        Returns:
            Matched entity/ies with scores
        """
        if self.embeddings is None or self.model is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)
        entity_lookup = {entity["id"]: entity for entity in self.entities}

        # Use provided candidates or all entities
        if candidates is not None:
            candidate_ids = {c["id"] for c in candidates}
            candidate_indices = [
                i for i, eid in enumerate(self.entity_ids) if eid in candidate_ids
            ]
        else:
            candidate_indices = list(range(len(self.entity_ids)))

        if not candidate_indices:
            empty = None if top_k == 1 else []
            return empty if single_input else [empty for _ in texts]

        candidate_embeddings = self.embeddings[candidate_indices]
        candidate_ids_list = [self.entity_ids[i] for i in candidate_indices]

        if batch_size is not None:
            query_embeddings = self.model.encode(texts, batch_size=batch_size)
        else:
            query_embeddings = self.model.encode(texts)

        # Ensure both query and candidate embeddings use same dimension
        # Use the smaller of: model's output dim or configured embedding_dim
        effective_dim = (
            self.embedding_dim
            if self.embedding_dim is not None
            else query_embeddings.shape[1]
        )

        # Truncate query embeddings if needed
        if query_embeddings.shape[1] > effective_dim:
            query_embeddings = query_embeddings[:, :effective_dim]

        # Ensure candidate embeddings match (may have been truncated in build_index)
        if candidate_embeddings.shape[1] > effective_dim:
            candidate_embeddings = candidate_embeddings[:, :effective_dim]

        similarities = cosine_similarity(query_embeddings, candidate_embeddings)

        results = []
        for sim_row in similarities:
            sorted_indices = np.argsort(sim_row)[::-1]
            matches = []
            seen_ids = set()
            for idx in sorted_indices:
                score = sim_row[idx]
                if score < self.threshold:
                    continue
                entity_id = candidate_ids_list[idx]
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                entity = entity_lookup.get(entity_id, {})
                matches.append(
                    {
                        "id": entity_id,
                        "score": float(score),
                        "text": entity.get(
                            "text", self.entity_texts[candidate_indices[idx]]
                        ),
                    }
                )
                if len(matches) >= top_k:
                    break

            if top_k == 1:
                results.append(matches[0] if matches else None)
            else:
                results.append(matches)

        return _unwrap_single(results, single_input)
