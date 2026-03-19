"""Compatibility wrapper for the old NovelClassDetector entry point."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .entity_matcher import NovelEntityMatcher
from .schemas import DetectionConfig


class NovelClassDetector(NovelEntityMatcher):
    """
    Backwards-compatible alias for the older batch detector API.

    NovelEntityMatcher is now the primary orchestration class. This wrapper keeps
    the old import path stable while delegating all behavior to the new API.
    """

    def __init__(
        self,
        matcher,
        detection_config: Optional[Union[DetectionConfig, Dict[str, Any]]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_keys: Optional[Dict[str, str]] = None,
        output_dir: str = "./proposals",
        auto_save: bool = True,
    ):
        super().__init__(
            entities=list(getattr(matcher, "entities", [])),
            matcher=matcher,
            detection_config=detection_config,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_keys=llm_api_keys,
            output_dir=output_dir,
            auto_save=auto_save,
            acceptance_threshold=getattr(matcher, "threshold", 0.5),
            use_novelty_detector=True,
        )
