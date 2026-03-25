"""
Internal pipeline orchestrator.
"""

from __future__ import annotations

from typing import Iterable, List

from .contracts import PipelineRunResult, PipelineStage, StageContext


class PipelineOrchestrator:
    """Runs an ordered list of internal stages against a shared context."""

    def __init__(self, stages: Iterable[PipelineStage]):
        self.stages: List[PipelineStage] = list(stages)

    def run(self, context: StageContext) -> PipelineRunResult:
        stage_results = []
        for stage in self.stages:
            result = stage.run(context)
            context.artifacts.update(result.artifacts)
            context.metadata[stage.name] = result.metadata
            stage_results.append(result)
        return PipelineRunResult(context=context, stage_results=stage_results)

    async def run_async(self, context: StageContext) -> PipelineRunResult:
        stage_results = []
        for stage in self.stages:
            result = await stage.run_async(context)
            context.artifacts.update(result.artifacts)
            context.metadata[stage.name] = result.metadata
            stage_results.append(result)
        return PipelineRunResult(context=context, stage_results=stage_results)
