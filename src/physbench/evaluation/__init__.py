"""Benchmark-owned, scene-aware evaluation."""

from .protocols import load_evaluation_protocol

__all__ = [
    "aggregate_task_results",
    "evaluate_task",
    "load_evaluation_protocol",
]


def __getattr__(name: str):
    if name not in {"aggregate_task_results", "evaluate_task"}:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    try:
        from . import task_evaluator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Scene evaluation requires optional dependencies; install them "
            "with `pip install '.[scene-evaluation]'`."
        ) from exc
    return getattr(task_evaluator, name)
