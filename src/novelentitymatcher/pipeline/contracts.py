"""
Internal staged discovery pipeline contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StageContext:
    """Mutable context passed between internal pipeline stages."""

    inputs: List[str]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result returned by a single pipeline stage."""

    stage_name: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRunResult:
    """Terminal result for an internal pipeline run."""

    context: StageContext
    stage_results: List[StageResult] = field(default_factory=list)


class PipelineStage(ABC):
    """Base contract for internal discovery stages."""

    name: str

    @abstractmethod
    def run(self, context: StageContext) -> StageResult:
        """Execute the stage synchronously."""

    async def run_async(self, context: StageContext) -> StageResult:
        """Async entrypoint; stages can override when they have real async work."""
        return self.run(context)
