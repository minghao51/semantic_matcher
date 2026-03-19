"""Tests for ANN-backed NoveltyDetector."""

import importlib.util

import numpy as np
import pytest

from semanticmatcher.novelty.detector import NoveltyDetector
from semanticmatcher.novelty.schemas import DetectionConfig, DetectionStrategy

HAS_HDBSCAN = importlib.util.find_spec("hdbscan") is not None


class TestNoveltyDetector:
    """Test suite for NoveltyDetector."""

    @pytest.fixture
    def sample_texts(self):
        return [
            "quantum physics research",
            "machine learning algorithms",
            "neural network architecture",
            "quantum entanglement experiments",
            "biology research methods",
            "gene editing techniques",
            "novel interdisciplinary sample",
            "another emerging topic",
        ]

    @pytest.fixture
    def reference_embeddings(self):
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.95, 0.05, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.05, 0.95, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.95, 0.05],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def reference_labels(self):
        return ["physics", "physics", "cs", "cs", "biology", "biology"]

    @pytest.fixture
    def sample_embeddings(self):
        return np.array(
            [
                [0.98, 0.02, 0.0, 0.0],
                [0.0, 0.99, 0.01, 0.0],
                [0.02, 0.97, 0.01, 0.0],
                [0.96, 0.04, 0.0, 0.0],
                [0.0, 0.0, 0.99, 0.01],
                [0.0, 0.02, 0.97, 0.01],
                [0.55, 0.55, 0.0, 0.62],
                [0.56, 0.54, 0.02, 0.61],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def sample_confidences(self):
        return np.array([0.97, 0.96, 0.95, 0.97, 0.94, 0.93, 0.58, 0.56], dtype=float)

    @pytest.fixture
    def candidate_results(self):
        return [
            [{"id": "physics", "score": 0.97}, {"id": "cs", "score": 0.12}],
            [{"id": "cs", "score": 0.96}, {"id": "biology", "score": 0.08}],
            [{"id": "cs", "score": 0.95}, {"id": "physics", "score": 0.11}],
            [{"id": "physics", "score": 0.97}, {"id": "cs", "score": 0.1}],
            [{"id": "biology", "score": 0.94}, {"id": "cs", "score": 0.07}],
            [{"id": "biology", "score": 0.93}, {"id": "cs", "score": 0.1}],
            [{"id": "cs", "score": 0.58}, {"id": "physics", "score": 0.54}],
            [{"id": "cs", "score": 0.56}, {"id": "physics", "score": 0.53}],
        ]

    @pytest.fixture
    def sample_predictions(self):
        return ["physics", "cs", "cs", "physics", "biology", "biology", "cs", "cs"]

    @pytest.fixture
    def detector(self):
        config = DetectionConfig(
            strategies=[
                DetectionStrategy.CONFIDENCE,
                DetectionStrategy.KNN_DISTANCE,
                DetectionStrategy.CLUSTERING,
            ],
            uncertainty_threshold=0.4,
            knn_distance_threshold=0.25,
            strong_knn_novelty_threshold=0.6,
            candidate_score_threshold=0.3,
            novelty_threshold=0.35,
            min_cluster_size=2,
        )
        return NoveltyDetector(config=config, embedding_dim=4)

    def test_initialization(self, detector):
        assert detector.config is not None
        assert detector.embedding_dim == 4
        assert detector._ann_index is None

    def test_default_initialization(self):
        detector = NoveltyDetector(embedding_dim=4)
        assert detector.config.combine_method == "weighted"

    def test_detect_by_confidence_legacy(self, detector, sample_confidences):
        low_confidence_indices = detector._detect_by_confidence(sample_confidences)
        assert 6 in low_confidence_indices
        assert 7 in low_confidence_indices

    def test_knn_distance_detection(
        self,
        detector,
        sample_embeddings,
        sample_predictions,
        reference_embeddings,
        reference_labels,
    ):
        detector._build_reference_index(reference_embeddings, reference_labels)
        knn_metrics, knn_flags = detector._detect_by_knn_distance(
            sample_embeddings, sample_predictions
        )

        assert isinstance(knn_metrics, dict)
        assert 6 in knn_flags
        assert 7 in knn_flags
        assert knn_metrics[0]["knn_novelty_score"] < knn_metrics[6]["knn_novelty_score"]
        assert knn_metrics[6]["neighbor_labels"]

    def test_missing_reference_embeddings_raises(
        self,
        detector,
        sample_texts,
        sample_confidences,
        sample_embeddings,
        sample_predictions,
    ):
        with pytest.raises(RuntimeError, match="reference embeddings"):
            detector.detect_novel_samples(
                texts=sample_texts,
                confidences=sample_confidences,
                embeddings=sample_embeddings,
                predicted_classes=sample_predictions,
            )

    def test_full_detection_pipeline(
        self,
        detector,
        sample_texts,
        sample_confidences,
        sample_embeddings,
        sample_predictions,
        candidate_results,
        reference_embeddings,
        reference_labels,
    ):
        report = detector.detect_novel_samples(
            texts=sample_texts,
            confidences=sample_confidences,
            embeddings=sample_embeddings,
            predicted_classes=sample_predictions,
            candidate_results=candidate_results,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            known_classes=["physics", "cs", "biology"],
        )

        assert len(report.novel_samples) >= 2
        novel_indices = {sample.index for sample in report.novel_samples}
        assert {6, 7}.issubset(novel_indices)

        sample = next(item for item in report.novel_samples if item.index == 6)
        assert sample.uncertainty_score is not None
        assert sample.knn_novelty_score is not None
        assert sample.novelty_score is not None
        assert sample.neighbor_labels

    def test_legacy_combine_modes_still_work(self):
        config = DetectionConfig(
            strategies=[
                DetectionStrategy.CONFIDENCE,
                DetectionStrategy.KNN_DISTANCE,
            ],
            combine_method="intersection",
        )
        detector = NoveltyDetector(config=config, embedding_dim=4)
        all_signals = {
            0: {DetectionStrategy.CONFIDENCE: True},
            1: {DetectionStrategy.KNN_DISTANCE: True},
            2: {
                DetectionStrategy.CONFIDENCE: True,
                DetectionStrategy.KNN_DISTANCE: True,
            },
        }

        combined = detector._combine_signals_legacy(all_signals)
        assert combined == {2}

    def test_empty_samples(self, detector):
        report = detector.detect_novel_samples(
            texts=[],
            confidences=np.array([]),
            embeddings=np.array([]).reshape(0, 4),
            predicted_classes=[],
        )

        assert len(report.novel_samples) == 0


@pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not available")
class TestNoveltyDetectorWithClustering:
    """Cluster-validation tests."""

    def test_validated_cluster_adds_support(self):
        config = DetectionConfig(
            strategies=[
                DetectionStrategy.CONFIDENCE,
                DetectionStrategy.KNN_DISTANCE,
                DetectionStrategy.CLUSTERING,
            ],
            uncertainty_threshold=0.4,
            knn_distance_threshold=0.2,
            novelty_threshold=0.3,
            candidate_score_threshold=0.2,
            min_cluster_size=2,
        )
        detector = NoveltyDetector(config=config, embedding_dim=4)
        reference_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        reference_labels = ["physics", "cs", "biology"]
        query_embeddings = np.array(
            [
                [0.55, 0.55, 0.0, 0.62],
                [0.56, 0.54, 0.02, 0.61],
            ],
            dtype=np.float32,
        )
        report = detector.detect_novel_samples(
            texts=["novel 1", "novel 2"],
            confidences=np.array([0.55, 0.54]),
            embeddings=query_embeddings,
            predicted_classes=["cs", "cs"],
            candidate_results=[
                [{"id": "cs", "score": 0.55}, {"id": "physics", "score": 0.54}],
                [{"id": "cs", "score": 0.54}, {"id": "physics", "score": 0.53}],
            ],
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
        )

        assert len(report.novel_samples) == 2
        assert any(sample.cluster_support_score for sample in report.novel_samples)
