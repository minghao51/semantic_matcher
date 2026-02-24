import pytest
from semanticmatcher.core.matcher import EntityMatcher, EmbeddingMatcher


class TestEntityMatcher:
    """Tests for EntityMatcher - SetFit-based entity matching."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
            {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        ]

    @pytest.fixture
    def training_data(self):
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "Deutschland", "label": "DE"},
            {"text": "Deutchland", "label": "DE"},
            {"text": "France", "label": "FR"},
            {"text": "Frankreich", "label": "FR"},
            {"text": "USA", "label": "US"},
            {"text": "America", "label": "US"},
        ]

    def test_entity_matcher_init(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        assert matcher.entities == sample_entities

    def test_entity_matcher_with_model(self, sample_entities):
        matcher = EntityMatcher(
            entities=sample_entities,
            model_name="sentence-transformers/paraphrase-mpnet-base-v2"
        )
        assert matcher.model_name == "sentence-transformers/paraphrase-mpnet-base-v2"

    def test_entity_matcher_default_threshold(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        assert matcher.threshold == 0.7

    def test_entity_matcher_custom_threshold(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities, threshold=0.5)
        assert matcher.threshold == 0.5

    def test_entity_matcher_without_training_raises(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        with pytest.raises(RuntimeError, match="not trained"):
            matcher.predict("Germany")

    def test_entity_matcher_train(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities)
        matcher.train(training_data, num_epochs=1)
        assert matcher.is_trained

    def test_entity_matcher_predict_single(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities)
        matcher.train(training_data, num_epochs=1)
        result = matcher.predict("Deutchland")
        assert result == "DE"

    def test_entity_matcher_predict_multiple(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities)
        matcher.train(training_data, num_epochs=1)
        results = matcher.predict(["Deutchland", "America", "France"])
        assert results == ["DE", "US", "FR"]

    def test_entity_matcher_predict_below_threshold(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities, threshold=0.99)
        matcher.train(training_data, num_epochs=1)
        result = matcher.predict("UnknownCountry123")
        assert result is None


class TestEmbeddingMatcher:
    """Tests for EmbeddingMatcher - similarity-based matching."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
            {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        ]

    def test_embedding_matcher_init(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        assert matcher.entities == sample_entities

    def test_embedding_matcher_default_model(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        assert matcher.model_name == "sentence-transformers/paraphrase-mpnet-base-v2"

    def test_embedding_matcher_custom_model(self, sample_entities):
        matcher = EmbeddingMatcher(
            entities=sample_entities,
            model_name="BAAI/bge-m3"
        )
        assert matcher.model_name == "BAAI/bge-m3"

    def test_embedding_matcher_build_index(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        assert matcher.embeddings is not None
        assert len(matcher.embeddings) > 0

    def test_embedding_matcher_match(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        result = matcher.match("Deutschland")
        assert result is not None
        assert result["id"] == "DE"
        assert "score" in result

    def test_embedding_matcher_match_below_threshold(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities, threshold=0.99)
        matcher.build_index()
        result = matcher.match("xyzunknown123")
        assert result is None

    def test_embedding_matcher_match_multiple(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        results = matcher.match(["Deutschland", "America"])
        assert len(results) == 2
        assert results[0]["id"] == "DE"
        assert results[1]["id"] == "US"

    def test_embedding_matcher_with_aliases(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        result = matcher.match("Deutchland")
        assert result["id"] == "DE"
