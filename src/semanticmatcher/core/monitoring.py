"""Performance monitoring utilities for semantic matchers."""

import functools
import time
from typing import Callable, Dict, List, Optional


def track_performance(func: Callable) -> Callable:
    """
    Decorator to track method performance metrics.

    Tracks:
        - Number of calls
        - Total time
        - Average time per call
        - Last call duration

    Usage:
        @track_performance
        def match(self, query, top_k=5):
            ...

        # Access metrics
        matcher._metrics  # {'calls': 10, 'total_time': 1.5, 'avg_time': 0.15}
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start

        if not hasattr(self, "_metrics"):
            self._metrics = {
                "calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_time": 0.0,
            }

        self._metrics["calls"] += 1
        self._metrics["total_time"] += elapsed
        self._metrics["avg_time"] = self._metrics["total_time"] / self._metrics["calls"]
        self._metrics["last_time"] = elapsed

        return result

    return wrapper


class PerformanceMonitor:
    """
    Simple performance tracking for matchers and other components.

    Provides detailed metrics for different operations.

    Example:
        monitor = PerformanceMonitor()

        with monitor.track("match_operation"):
            result = matcher.match(query)

        print(monitor.summary())
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record(self, operation: str, duration: float):
        """Record a timing for an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)

    def track(self, operation: str):
        """Context manager for tracking operation timing."""

        class _Timer:
            def __init__(self, monitor: PerformanceMonitor, op: str):
                self.monitor = monitor
                self.op = op
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                duration = time.time() - self.start
                self.monitor.record(self.op, duration)

        return _Timer(self, operation)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return summary statistics for all tracked operations.

        Returns:
            Dict mapping operation names to statistics:
                - count: Number of recordings
                - total: Total time
                - mean: Average time
                - min: Minimum time
                - max: Maximum time
        """
        summary = {}
        for op, timings in self.metrics.items():
            if timings:
                summary[op] = {
                    "count": len(timings),
                    "total": sum(timings),
                    "mean": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                }
        return summary

    def reset(self):
        """Clear all recorded metrics."""
        self.metrics.clear()

    def get_operation_metrics(self, operation: str) -> Optional[Dict[str, float]]:
        """Get metrics for a specific operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        timings = self.metrics[operation]
        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def to_dict(self) -> Dict[str, List[float]]:
        """Return raw metrics as dictionary."""
        return dict(self.metrics)
