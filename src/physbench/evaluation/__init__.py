"""Benchmark-owned, scene-aware evaluation."""

from .protocols import load_evaluation_protocol
from .task_evaluator import evaluate_task

__all__ = ["evaluate_task", "load_evaluation_protocol"]
