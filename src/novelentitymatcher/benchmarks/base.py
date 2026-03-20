"""Base evaluator abstract class for benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import pandas as pd

T = TypeVar("T")


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    dataframe: pd.DataFrame | None = None


class BaseEvaluator(ABC, Generic[T]):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        data: T,
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate on the given data."""
        raise NotImplementedError

    @abstractmethod
    def get_default_metrics(self) -> list[str]:
        """Return list of default metric names this evaluator computes."""
        raise NotImplementedError

    def format_results(
        self,
        result: EvaluationResult,
        include_details: bool = False,
    ) -> str:
        lines = [f"=== {self.name} ==="]
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
            else:
                lines.append(f"  {metric}: {value}")
        return "\n".join(lines)
