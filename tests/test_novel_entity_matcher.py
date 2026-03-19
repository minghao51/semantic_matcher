"""Tests for the promoted NovelEntityMatcher orchestration API."""

import pytest

from semanticmatcher import Matcher, NovelEntityMatcher
from semanticmatcher.novelty.schemas import DetectionConfig, DetectionStrategy


@pytest.fixture
def sample_entities():
    return [
        {"id": "physics", "name": "Quantum Physics"},
        {"id": "cs", "name": "Computer Science"},
        {"id": "biology", "name": "Molecular Biology"},
    ]


@pytest.fixture
def trained_matcher(sample_entities):
    matcher = Matcher(entities=sample_entities, model="minilm", threshold=0.6)
    matcher.fit(
        texts=[
            "quantum mechanics",
            "wave function",
            "machine learning",
            "data structures",
            "gene expression",
            "DNA replication",
        ],
        labels=["physics", "physics", "cs", "cs", "biology", "biology"],
    )
    return matcher


class TestNovelEntityMatcher:
    def test_root_export_is_available(self):
        assert NovelEntityMatcher is not None

    def test_match_returns_operational_result(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=[DetectionStrategy.CONFIDENCE],
                confidence_threshold=0.7,
            ),
            auto_save=False,
        )

        result = novelty_matcher.match("quantum superposition", return_alternatives=True)

        assert result.predicted_id is not None
        assert isinstance(result.is_novel, bool)
        assert isinstance(result.alternatives, list)

    @pytest.mark.asyncio
    async def test_discover_novel_classes_matches_legacy_shape(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=[DetectionStrategy.CONFIDENCE],
            ),
            auto_save=False,
        )

        report = await novelty_matcher.discover_novel_classes(
            queries=["quantum superposition", "random unrelated text here"],
            existing_classes=["physics", "cs", "biology"],
            run_llm_proposal=False,
        )

        assert report.discovery_id
        assert report.novel_sample_report is not None

    def test_threshold_updates_delegate_to_inner_matcher(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(matcher=trained_matcher, auto_save=False)

        novelty_matcher.adjust_threshold(0.42)

        assert novelty_matcher.acceptance_threshold == 0.42
        assert trained_matcher.threshold == 0.42
