"""Benchmark-owned, scene-aware evaluation."""

from importlib import import_module

from .protocols import load_evaluation_protocol


_SCENE_EVALUATION_DEPENDENCY_ROOTS = frozenset({
    "cv2",
    "matplotlib",
    "numpy",
    "sam2",
    "scipy",
    "torch",
})


class SceneEvaluationDependencyError(RuntimeError):
    """The optional scene-evaluation runtime dependencies are unavailable."""

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
        task_evaluator = import_module(".task_evaluator", __name__)
    except ModuleNotFoundError as exc:
        dependency_root = (exc.name or "").split(".", 1)[0]
        if dependency_root not in _SCENE_EVALUATION_DEPENDENCY_ROOTS:
            raise
        raise SceneEvaluationDependencyError(
            "Scene evaluation requires optional dependencies; install them "
            "with `pip install '.[scene-evaluation]'`."
        ) from exc
    return getattr(task_evaluator, name)
