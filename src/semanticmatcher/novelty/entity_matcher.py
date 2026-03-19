"""
Primary orchestration API for classification plus novel-class detection.

This module promotes NovelEntityMatcher to the main public entry point for
novelty-aware matching. It wraps a fitted ``Matcher`` together with the
multi-signal ``NoveltyDetector`` and optional ``LLMClassProposer``.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .detector import NoveltyDetector
from .llm_proposer import LLMClassProposer
from .schemas import DetectionConfig, DetectionStrategy, NovelClassDiscoveryReport
from .storage import export_summary, save_proposals
from ..core.matcher import Matcher
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NovelEntityMatchResult:
    """Operational result for a single novelty-aware match decision."""

    id: str | None
    score: float
    is_match: bool
    is_novel: bool
    novel_score: float | None = None
    match_method: str = "accepted_known"
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, bool] = field(default_factory=dict)
    predicted_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Backwards-compatible alias used by benchmark imports.
NoveltyMatchResult = NovelEntityMatchResult


class NovelEntityMatcher:
    """
    Primary public API for novelty-aware matching.

    The class orchestrates three stages:
    1. Retrieve rich matcher metadata with top-k candidates and embeddings.
    2. Score novelty using ANN-backed multi-signal detection.
    3. Optionally propose new class names for novel batches.
    """

    def __init__(
        self,
        entities: Optional[list[dict[str, Any]]] = None,
        *,
        matcher: Optional[Matcher] = None,
        model: str = "potion-8m",
        mode: str = "zero-shot",
        acceptance_threshold: Optional[float] = None,
        detection_config: Optional[Union[DetectionConfig, Dict[str, Any]]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_keys: Optional[Dict[str, str]] = None,
        output_dir: str = "./proposals",
        auto_save: bool = True,
        match_threshold: Optional[float] = None,
        novelty_strategy: str = "confidence",
        confidence_threshold: float = 0.3,
        knn_k: int = 5,
        knn_distance_threshold: float = 0.6,
        min_cluster_size: int = 5,
        use_novelty_detector: bool = True,
    ):
        if matcher is None:
            if entities is None:
                raise ValueError("entities is required when matcher is not provided")
            threshold = (
                acceptance_threshold
                if acceptance_threshold is not None
                else (match_threshold if match_threshold is not None else 0.5)
            )
            matcher = Matcher(
                entities=entities,
                model=model,
                mode=mode,
                threshold=threshold,
            )

        self.matcher = matcher
        self.entities = (
            entities
            if entities is not None
            else list(getattr(matcher, "entities", []))
        )
        self.acceptance_threshold = (
            acceptance_threshold
            if acceptance_threshold is not None
            else (
                match_threshold
                if match_threshold is not None
                else getattr(self.matcher, "threshold", 0.5)
            )
        )
        self.output_dir = output_dir
        self.auto_save = auto_save
        self.use_novelty_detector = use_novelty_detector

        self.detection_config = self._coerce_detection_config(
            detection_config=detection_config,
            novelty_strategy=novelty_strategy,
            confidence_threshold=confidence_threshold,
            knn_k=knn_k,
            knn_distance_threshold=knn_distance_threshold,
            min_cluster_size=min_cluster_size,
        )
        self.detector = NoveltyDetector(config=self.detection_config)
        self.llm_proposer = LLMClassProposer(
            primary_model=llm_model,
            provider=llm_provider,
            api_keys=llm_api_keys,
        )

    @staticmethod
    def _coerce_detection_config(
        detection_config: Optional[Union[DetectionConfig, Dict[str, Any]]],
        novelty_strategy: str,
        confidence_threshold: float,
        knn_k: int,
        knn_distance_threshold: float,
        min_cluster_size: int,
    ) -> DetectionConfig:
        if isinstance(detection_config, DetectionConfig):
            return detection_config
        if isinstance(detection_config, dict):
            return DetectionConfig(**detection_config)

        strategies: list[DetectionStrategy]
        strategy = novelty_strategy.lower()
        if strategy == "confidence":
            strategies = [DetectionStrategy.CONFIDENCE]
        elif strategy in {"knn", "knn_distance", "distance"}:
            strategies = [DetectionStrategy.CONFIDENCE, DetectionStrategy.KNN_DISTANCE]
        elif strategy in {"cluster", "clustering"}:
            strategies = [
                DetectionStrategy.CONFIDENCE,
                DetectionStrategy.KNN_DISTANCE,
                DetectionStrategy.CLUSTERING,
            ]
        else:
            strategies = [
                DetectionStrategy.CONFIDENCE,
                DetectionStrategy.KNN_DISTANCE,
                DetectionStrategy.CLUSTERING,
            ]

        return DetectionConfig(
            strategies=strategies,
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=confidence_threshold,
            knn_k=knn_k,
            knn_distance_threshold=knn_distance_threshold,
            min_cluster_size=min_cluster_size,
        )

    def fit(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "NovelEntityMatcher":
        self.matcher.fit(
            training_data=training_data,
            mode=mode,
            show_progress=show_progress,
            **kwargs,
        )
        return self

    async def fit_async(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "NovelEntityMatcher":
        await self.matcher.fit_async(
            training_data=training_data,
            mode=mode,
            show_progress=show_progress,
            **kwargs,
        )
        return self

    def set_threshold(self, threshold: float) -> "NovelEntityMatcher":
        self.acceptance_threshold = threshold
        self.matcher.set_threshold(threshold)
        return self

    def adjust_threshold(self, new_threshold: float) -> None:
        self.set_threshold(new_threshold)

    def get_reference_corpus(self) -> Dict[str, Any]:
        return self.matcher.get_reference_corpus()

    def set_novelty_detector(self, detector: NoveltyDetector | None) -> None:
        if detector is None:
            self.use_novelty_detector = False
            self.detector = NoveltyDetector(config=self.detection_config)
            return
        self.use_novelty_detector = True
        self.detector = detector

    def get_stats(self) -> dict[str, Any]:
        return {
            "num_entities": len(self.entities),
            "model": getattr(self.matcher, "model_name", None),
            "mode": getattr(self.matcher, "_training_mode", None),
            "acceptance_threshold": self.acceptance_threshold,
            "use_novelty_detector": self.use_novelty_detector,
            "detection_config": self.detection_config.model_dump(),
        }

    def _derive_existing_classes(
        self, existing_classes: Optional[List[str]] = None
    ) -> List[str]:
        if existing_classes:
            return list(existing_classes)
        if self.entities:
            return [str(entity["id"]) for entity in self.entities if "id" in entity]
        reference = self.get_reference_corpus()
        return list(reference.get("labels", []))

    @staticmethod
    def _normalize_candidate_results(
        raw_match_results: Any,
        num_queries: int,
    ) -> List[Any]:
        if raw_match_results is None:
            return [None] * num_queries
        if num_queries == 1:
            if isinstance(raw_match_results, list):
                if raw_match_results and all(isinstance(item, dict) for item in raw_match_results):
                    return [raw_match_results]
                if len(raw_match_results) == 1:
                    return [raw_match_results[0]]
            return [raw_match_results]
        if isinstance(raw_match_results, list):
            return raw_match_results
        return [raw_match_results] * num_queries

    async def _collect_metadata_async(self, queries: List[str]) -> Dict[str, Any]:
        match_async = getattr(self.matcher, "match_async", None)
        if callable(match_async):
            result = await match_async(
                queries,
                return_metadata=True,
                top_k=self.detection_config.uncertainty_top_k,
            )
        else:
            result = await asyncio.to_thread(
                self.matcher.match,
                queries,
                return_metadata=True,
                top_k=self.detection_config.uncertainty_top_k,
            )

        reference = self.get_reference_corpus()
        raw_match_results = (result.metadata or {}).get("raw_match_results")
        return {
            "predicted_classes": result.predictions,
            "confidences": result.confidences,
            "embeddings": result.embeddings,
            "candidate_results": self._normalize_candidate_results(
                raw_match_results,
                num_queries=len(queries),
            ),
            "reference_embeddings": reference["embeddings"],
            "reference_labels": reference["labels"],
        }

    def _collect_metadata_sync(self, queries: List[str]) -> Dict[str, Any]:
        result = self.matcher.match(
            queries,
            return_metadata=True,
            top_k=self.detection_config.uncertainty_top_k,
        )
        reference = self.get_reference_corpus()
        raw_match_results = (result.metadata or {}).get("raw_match_results")
        return {
            "predicted_classes": result.predictions,
            "confidences": result.confidences,
            "embeddings": result.embeddings,
            "candidate_results": self._normalize_candidate_results(
                raw_match_results,
                num_queries=len(queries),
            ),
            "reference_embeddings": reference["embeddings"],
            "reference_labels": reference["labels"],
        }

    def _build_match_result(
        self,
        query: str,
        metadata: Dict[str, Any],
        existing_classes: Optional[List[str]] = None,
        return_alternatives: bool = False,
    ) -> NovelEntityMatchResult:
        predicted_id = (
            metadata["predicted_classes"][0]
            if metadata["predicted_classes"]
            else None
        )
        score = (
            float(metadata["confidences"][0])
            if len(metadata["confidences"]) > 0
            else 0.0
        )
        alternatives = metadata["candidate_results"][0]
        if not isinstance(alternatives, list):
            alternatives = [alternatives] if alternatives else []

        if self.use_novelty_detector:
            report = self.detector.detect_novel_samples(
                texts=[query],
                confidences=np.asarray(metadata["confidences"], dtype=float),
                embeddings=np.asarray(metadata["embeddings"]),
                predicted_classes=list(metadata["predicted_classes"]),
                known_classes=self._derive_existing_classes(existing_classes),
                candidate_results=metadata["candidate_results"],
                reference_embeddings=metadata["reference_embeddings"],
                reference_labels=metadata["reference_labels"],
            )
            sample = report.novel_samples[0] if report.novel_samples else None
        else:
            report = None
            sample = None

        is_novel = False
        novel_score = max(0.0, 1.0 - score)
        signals: dict[str, bool] = {}
        if sample is not None:
            is_novel = True
            novel_score = float(sample.novelty_score or 0.0)
            signals = dict(sample.signals)

        accepted_known = (
            predicted_id not in (None, "unknown")
            and score >= self.acceptance_threshold
            and not is_novel
        )

        if accepted_known:
            match_method = "accepted_known"
        elif is_novel:
            match_method = "novelty_detector"
        elif predicted_id in (None, "unknown"):
            match_method = "no_match"
        else:
            match_method = "below_acceptance_threshold"

        return NovelEntityMatchResult(
            id=predicted_id if accepted_known else None,
            score=score,
            is_match=accepted_known,
            is_novel=is_novel or not accepted_known,
            novel_score=novel_score,
            match_method=match_method,
            alternatives=alternatives if return_alternatives else [],
            signals=signals,
            predicted_id=predicted_id,
            metadata={
                "query": query,
                "acceptance_threshold": self.acceptance_threshold,
            },
        )

    def match(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: Optional[List[str]] = None,
    ) -> NovelEntityMatchResult:
        metadata = self._collect_metadata_sync([text])
        return self._build_match_result(
            text,
            metadata,
            existing_classes=existing_classes,
            return_alternatives=return_alternatives,
        )

    async def match_async(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: Optional[List[str]] = None,
    ) -> NovelEntityMatchResult:
        metadata = await self._collect_metadata_async([text])
        return self._build_match_result(
            text,
            metadata,
            existing_classes=existing_classes,
            return_alternatives=return_alternatives,
        )

    def match_batch(
        self,
        texts: list[str],
        return_alternatives: bool = False,
        existing_classes: Optional[List[str]] = None,
    ) -> list[NovelEntityMatchResult]:
        metadata = self._collect_metadata_sync(texts)
        return [
            self._build_match_result(
                text,
                {
                    "predicted_classes": [metadata["predicted_classes"][idx]],
                    "confidences": np.asarray([metadata["confidences"][idx]], dtype=float),
                    "embeddings": np.asarray([metadata["embeddings"][idx]]),
                    "candidate_results": [metadata["candidate_results"][idx]],
                    "reference_embeddings": metadata["reference_embeddings"],
                    "reference_labels": metadata["reference_labels"],
                },
                existing_classes=existing_classes,
                return_alternatives=return_alternatives,
            )
            for idx, text in enumerate(texts)
        ]

    async def discover_novel_classes(
        self,
        queries: List[str],
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
        return_metadata: bool = True,
        run_llm_proposal: bool = True,
    ) -> NovelClassDiscoveryReport:
        discovery_id = str(uuid.uuid4())[:8]
        logger.info(
            "[%s] Starting novel class discovery for %s queries",
            discovery_id,
            len(queries),
        )

        if return_metadata:
            results = await self._collect_metadata_async(queries)
        else:
            results = self._collect_metadata_sync(queries)

        known_classes = self._derive_existing_classes(existing_classes)
        novel_sample_report = self.detector.detect_novel_samples(
            texts=queries,
            confidences=results["confidences"],
            embeddings=results["embeddings"],
            predicted_classes=results["predicted_classes"],
            candidate_results=results.get("candidate_results"),
            known_classes=known_classes,
            reference_embeddings=results["reference_embeddings"],
            reference_labels=results["reference_labels"],
        )

        class_proposals = None
        if run_llm_proposal and novel_sample_report.novel_samples:
            try:
                class_proposals = self.llm_proposer.propose_classes(
                    novel_samples=novel_sample_report.novel_samples,
                    existing_classes=known_classes,
                    context=context,
                )
            except Exception as exc:
                logger.error("[%s] LLM proposal failed: %s", discovery_id, exc)

        report = NovelClassDiscoveryReport(
            discovery_id=discovery_id,
            timestamp=datetime.now(),
            matcher_config=self._get_matcher_config(),
            detection_config=self.detection_config.model_dump(),
            novel_sample_report=novel_sample_report,
            class_proposals=class_proposals,
            metadata={
                "num_queries": len(queries),
                "num_existing_classes": len(known_classes),
                "num_novel_samples": len(novel_sample_report.novel_samples),
                "context": context,
            },
        )

        if self.auto_save:
            output_file = save_proposals(report, output_dir=self.output_dir)
            report.output_file = output_file
            summary_path = output_file.replace(
                f".{output_file.split('.')[-1]}",
                "_summary.md",
            )
            export_summary(report, summary_path)
            report.metadata["summary_file"] = summary_path

        return report

    def batch_discover(
        self,
        queries_batch: List[List[str]],
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[NovelClassDiscoveryReport]:
        async def run_all():
            tasks = [
                self.discover_novel_classes(
                    queries=queries,
                    existing_classes=existing_classes,
                    context=context,
                )
                for queries in queries_batch
            ]
            return await asyncio.gather(*tasks)

        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_all())
                return future.result()
        except RuntimeError:
            return asyncio.run(run_all())

    def _get_matcher_config(self) -> Dict[str, Any]:
        config = {
            "matcher_type": self.matcher.__class__.__name__,
        }
        if hasattr(self.matcher, "model_name"):
            config["model"] = str(self.matcher.model_name)
        if hasattr(self.matcher, "threshold"):
            config["threshold"] = self.matcher.threshold
        if hasattr(self.matcher, "_training_mode"):
            config["mode"] = getattr(self.matcher, "_training_mode")
        return config


def create_novel_entity_matcher(
    entities: list[dict[str, Any]],
    model: str = "potion-8m",
    mode: str = "zero-shot",
    threshold: float = 0.5,
    enable_novelty_detection: bool = True,
    **kwargs,
) -> NovelEntityMatcher:
    return NovelEntityMatcher(
        entities=entities,
        model=model,
        mode=mode,
        acceptance_threshold=threshold,
        use_novelty_detector=enable_novelty_detection,
        **kwargs,
    )
