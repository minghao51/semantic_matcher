"""Adapters that route existing matcher and discovery capabilities through stages."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

from ..novelty.schemas import ClusterEvidence, DiscoveryCluster, NovelSampleReport
from .contracts import PipelineStage, StageContext, StageResult

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class MatcherMetadataStage(PipelineStage):
    """Collect rich matcher metadata and reference corpus for downstream stages."""

    name = "match"

    def __init__(
        self,
        collect_sync: Callable[[List[str]], Any],
        collect_async: Callable[[List[str]], Any],
    ):
        self._collect_sync = collect_sync
        self._collect_async = collect_async

    def run(self, context: StageContext) -> StageResult:
        match_result, reference = self._collect_sync(context.inputs)
        return StageResult(
            stage_name=self.name,
            artifacts={
                "match_result": match_result,
                "reference_corpus": reference,
            },
            metadata={
                "num_queries": len(context.inputs),
                "candidate_top_k": (match_result.metadata or {}).get("top_k"),
            },
        )

    async def run_async(self, context: StageContext) -> StageResult:
        match_result, reference = await self._collect_async(context.inputs)
        return StageResult(
            stage_name=self.name,
            artifacts={
                "match_result": match_result,
                "reference_corpus": reference,
            },
            metadata={
                "num_queries": len(context.inputs),
                "candidate_top_k": (match_result.metadata or {}).get("top_k"),
            },
        )


class OODDetectionStage(PipelineStage):
    """Run novelty detection against the stable matcher metadata contract."""

    name = "ood"

    def __init__(self, detector: Any, enabled: bool = True):
        self.detector = detector
        self.enabled = enabled

    def run(self, context: StageContext) -> StageResult:
        if not self.enabled:
            report = NovelSampleReport()
        else:
            match_result = context.artifacts["match_result"]
            reference = context.artifacts["reference_corpus"]
            report = self.detector.detect_novel_samples(
                texts=context.inputs,
                confidences=match_result.confidences,
                embeddings=match_result.embeddings,
                predicted_classes=match_result.predictions,
                candidate_results=match_result.candidate_results,
                reference_embeddings=reference["embeddings"],
                reference_labels=reference["labels"],
            )

        return StageResult(
            stage_name=self.name,
            artifacts={"novel_sample_report": report},
            metadata={
                "num_novel_samples": len(report.novel_samples),
                "strategies": list(getattr(report, "detection_strategies", [])),
            },
        )


class CommunityDetectionStage(PipelineStage):
    """Cluster likely novel samples into discovery communities."""

    name = "cluster"

    def __init__(
        self,
        clusterer: Any | None = None,
        *,
        enabled: bool = True,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 2,
    ):
        self.clusterer = clusterer
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def run(self, context: StageContext) -> StageResult:
        report = context.artifacts["novel_sample_report"]
        match_result = context.artifacts["match_result"]
        novel_samples = list(report.novel_samples)

        if not self.enabled or not novel_samples:
            return StageResult(
                stage_name=self.name,
                artifacts={"discovery_clusters": []},
                metadata={"num_clusters": 0, "backend": "disabled"},
            )

        sample_indices = [sample.index for sample in novel_samples]
        embeddings = np.asarray(
            [match_result.embeddings[idx] for idx in sample_indices]
        )

        cluster_labels, backend = self._cluster_embeddings(embeddings)
        index_to_cluster: dict[int, int] = {}
        groups: dict[int, list[Any]] = defaultdict(list)
        for sample, cluster_id in zip(novel_samples, cluster_labels):
            sample.cluster_id = cluster_id if cluster_id >= 0 else None
            if cluster_id >= 0:
                groups[int(cluster_id)].append(sample)
                index_to_cluster[sample.index] = int(cluster_id)

        clusters = [
            self._build_cluster(cluster_id, samples)
            for cluster_id, samples in sorted(groups.items())
            if len(samples) >= self.min_cluster_size
        ]
        if not clusters and novel_samples:
            for sample in novel_samples:
                sample.cluster_id = 0
                index_to_cluster[sample.index] = 0
            clusters = [self._build_cluster(0, novel_samples)]

        return StageResult(
            stage_name=self.name,
            artifacts={
                "discovery_clusters": clusters,
                "cluster_assignments": index_to_cluster,
            },
            metadata={
                "num_clusters": len(clusters),
                "backend": backend,
                "cluster_sizes": [cluster.sample_count for cluster in clusters],
            },
        )

    def _cluster_embeddings(self, embeddings: np.ndarray) -> tuple[list[int], str]:
        if len(embeddings) == 1:
            return [0], "singleton"

        if self.clusterer is not None:
            try:
                labels, _, info = self.clusterer.fit_predict(embeddings)
                return [int(label) for label in labels], str(
                    info.get("backend", "custom")
                )
            except Exception:
                pass

        labels = self._fallback_connected_components(embeddings)
        return labels, "fallback_connected_components"

    def _fallback_connected_components(self, embeddings: np.ndarray) -> list[int]:
        normalized = embeddings / np.clip(
            np.linalg.norm(embeddings, axis=1, keepdims=True),
            a_min=1e-12,
            a_max=None,
        )
        similarity = normalized @ normalized.T
        labels = [-1] * len(embeddings)
        cluster_id = 0

        for start in range(len(embeddings)):
            if labels[start] != -1:
                continue
            stack = [start]
            component: list[int] = []
            labels[start] = cluster_id
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = np.where(similarity[current] >= self.similarity_threshold)[
                    0
                ]
                for neighbor in neighbors:
                    if labels[int(neighbor)] == -1:
                        labels[int(neighbor)] = cluster_id
                        stack.append(int(neighbor))

            if len(component) < self.min_cluster_size:
                for member in component:
                    labels[member] = -1
            else:
                cluster_id += 1

        return labels

    def _build_cluster(
        self, cluster_id: int, samples: Iterable[Any]
    ) -> DiscoveryCluster:
        sample_list = list(samples)
        novelty_scores = [
            float(sample.novelty_score or 0.0)
            for sample in sample_list
            if sample.novelty_score is not None
        ]
        confidences = [float(sample.confidence) for sample in sample_list]
        return DiscoveryCluster(
            cluster_id=cluster_id,
            sample_indices=[sample.index for sample in sample_list],
            sample_count=len(sample_list),
            example_texts=[sample.text for sample in sample_list[:5]],
            mean_novelty_score=sum(novelty_scores) / len(novelty_scores)
            if novelty_scores
            else None,
            mean_confidence=sum(confidences) / len(confidences)
            if confidences
            else None,
            metadata={
                "predicted_classes": [sample.predicted_class for sample in sample_list],
            },
        )


class ClusterEvidenceStage(PipelineStage):
    """Extract compact evidence from clusters before proposal generation."""

    name = "evidence"

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_keywords: int = 8,
        max_examples: int = 4,
        token_budget: int = 256,
    ):
        self.enabled = enabled
        self.max_keywords = max_keywords
        self.max_examples = max_examples
        self.token_budget = token_budget

    def run(self, context: StageContext) -> StageResult:
        clusters: list[DiscoveryCluster] = list(
            context.artifacts.get("discovery_clusters", [])
        )
        report = context.artifacts["novel_sample_report"]
        samples_by_index = {sample.index: sample for sample in report.novel_samples}

        if not self.enabled or not clusters:
            return StageResult(
                stage_name=self.name,
                artifacts={"discovery_clusters": clusters},
                metadata={"num_clusters_with_evidence": 0},
            )

        enriched: list[DiscoveryCluster] = []
        for cluster in clusters:
            samples = [
                samples_by_index[index]
                for index in cluster.sample_indices
                if index in samples_by_index
            ]
            evidence = self._build_evidence(samples)
            cluster.keywords = list(evidence.keywords)
            cluster.evidence = evidence
            enriched.append(cluster)

        return StageResult(
            stage_name=self.name,
            artifacts={"discovery_clusters": enriched},
            metadata={
                "num_clusters_with_evidence": len(enriched),
                "token_budget": self.token_budget,
            },
        )

    def _build_evidence(self, samples: list[Any]) -> ClusterEvidence:
        texts = [sample.text for sample in samples]
        keyword_counts: Counter[str] = Counter()
        for text in texts:
            keyword_counts.update(self._tokenize(text))

        predicted_classes = [str(sample.predicted_class) for sample in samples]
        confidences = [float(sample.confidence) for sample in samples]
        novelty_scores = [
            float(sample.novelty_score or 0.0)
            for sample in samples
            if sample.novelty_score is not None
        ]

        return ClusterEvidence(
            keywords=[
                token for token, _ in keyword_counts.most_common(self.max_keywords)
            ],
            representative_examples=self._truncate_examples(texts),
            sample_indices=[sample.index for sample in samples],
            predicted_classes=sorted(set(predicted_classes)),
            confidence_summary={
                "mean_confidence": sum(confidences) / len(confidences)
                if confidences
                else 0.0,
                "mean_novelty_score": (
                    sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
                ),
            },
            token_budget=self.token_budget,
            metadata={"sample_count": len(samples)},
        )

    def _truncate_examples(self, texts: list[str]) -> list[str]:
        examples: list[str] = []
        tokens_used = 0
        for text in texts[: self.max_examples]:
            token_estimate = max(1, math.ceil(len(text.split()) * 1.2))
            if tokens_used + token_estimate > self.token_budget and examples:
                break
            examples.append(text)
            tokens_used += token_estimate
        return examples

    def _tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) > 2 and token not in _STOPWORDS
        ]


class ProposalStage(PipelineStage):
    """Optionally generate class proposals from a novelty report."""

    name = "proposal"

    def __init__(
        self,
        proposer: Any,
        existing_classes_resolver: Callable[[], List[str]],
        enabled: bool = True,
        context_text: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.proposer = proposer
        self._existing_classes_resolver = existing_classes_resolver
        self.enabled = enabled
        self.context_text = context_text
        self.max_retries = max_retries

    def run(self, context: StageContext) -> StageResult:
        report = context.artifacts["novel_sample_report"]
        discovery_clusters = list(context.artifacts.get("discovery_clusters", []))
        class_proposals = None
        existing_classes = self._existing_classes_resolver()
        error = None

        if self.enabled and report.novel_samples:
            try:
                if discovery_clusters and hasattr(
                    self.proposer, "propose_from_clusters"
                ):
                    class_proposals = self.proposer.propose_from_clusters(
                        discovery_clusters=discovery_clusters,
                        existing_classes=existing_classes,
                        context=self.context_text,
                        max_retries=self.max_retries,
                    )
                else:
                    class_proposals = self.proposer.propose_classes(
                        novel_samples=report.novel_samples,
                        existing_classes=existing_classes,
                        context=self.context_text,
                    )
            except Exception as exc:  # pragma: no cover - defensive wrapper
                error = str(exc)

        return StageResult(
            stage_name=self.name,
            artifacts={"class_proposals": class_proposals},
            metadata={
                "enabled": self.enabled,
                "num_existing_classes": len(existing_classes),
                "generated": class_proposals is not None,
                "num_clusters": len(discovery_clusters),
                "error": error,
            },
        )
