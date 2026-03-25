"""Public pipeline-first discovery API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..core.matcher import Matcher
from ..novelty.entity_matcher import NovelEntityMatchResult, NovelEntityMatcher
from ..novelty.schemas import NovelClassDiscoveryReport, ProposalReviewRecord
from ..novelty.storage.review import ProposalReviewManager


class DiscoveryPipeline:
    """Pipeline-first public entry point for discovery and promotion workflows."""

    def __init__(
        self,
        entities: Optional[list[dict[str, Any]]] = None,
        *,
        matcher: Optional[Matcher] = None,
        review_storage_path: str | Path = "./proposals/review_records.json",
        **kwargs: Any,
    ):
        self.novel_entity_matcher = NovelEntityMatcher(
            entities=entities,
            matcher=matcher,
            review_storage_path=str(review_storage_path),
            **kwargs,
        )
        self.review_manager = ProposalReviewManager(review_storage_path)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        entities: Optional[list[dict[str, Any]]] = None,
        matcher: Optional[Matcher] = None,
        **overrides: Any,
    ) -> "DiscoveryPipeline":
        config = Config(config_path)
        matcher_kwargs: Dict[str, Any] = {
            "model": config.get("embedding.model"),
            "acceptance_threshold": config.get("embedding.threshold"),
        }
        matcher_kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(entities=entities, matcher=matcher, **matcher_kwargs)

    def fit(self, *args: Any, **kwargs: Any) -> "DiscoveryPipeline":
        self.novel_entity_matcher.fit(*args, **kwargs)
        return self

    async def fit_async(self, *args: Any, **kwargs: Any) -> "DiscoveryPipeline":
        await self.novel_entity_matcher.fit_async(*args, **kwargs)
        return self

    def match(self, text: str, **kwargs: Any) -> NovelEntityMatchResult:
        return self.novel_entity_matcher.match(text, **kwargs)

    async def discover(
        self,
        queries: List[str],
        **kwargs: Any,
    ) -> NovelClassDiscoveryReport:
        return await self.novel_entity_matcher.discover_novel_classes(queries, **kwargs)

    def list_review_records(
        self, discovery_id: str | None = None
    ) -> list[ProposalReviewRecord]:
        return self.review_manager.list_records(discovery_id)

    def approve_proposal(
        self, review_id: str, *, notes: str | None = None
    ) -> ProposalReviewRecord:
        return self.review_manager.update_state(review_id, "approved", notes=notes)

    def reject_proposal(
        self, review_id: str, *, notes: str | None = None
    ) -> ProposalReviewRecord:
        return self.review_manager.update_state(review_id, "rejected", notes=notes)

    def promote_proposal(
        self,
        review_id: str,
        *,
        promoter: Any | None = None,
    ) -> ProposalReviewRecord:
        return self.review_manager.promote(review_id, promoter=promoter)
