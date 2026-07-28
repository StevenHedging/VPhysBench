"""Benchmark-owned, scene-aware evaluation."""

from .protocols import load_evaluation_protocol
from .task_evaluator import aggregate_task_results, evaluate_task

__all__ = [
    "aggregate_task_results",
    "evaluate_task",
    "load_evaluation_protocol",
]
