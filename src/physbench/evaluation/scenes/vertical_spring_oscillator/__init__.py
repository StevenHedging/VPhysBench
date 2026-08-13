"""Lazy public exports for the vertical spring scene evaluator."""

from __future__ import annotations

from importlib import import_module


_SCORING_EXPORTS = frozenset(
    {
        "SpringTrace",
        "SpringTraceError",
        "extract_spring_trace",
        "score_spring_traces",
        "theoretical_period_s",
    }
)
_EVALUATOR_EXPORTS = frozenset({"VerticalSpringOscillatorCaseEvaluator"})

__all__ = [
    "SpringTrace",
    "SpringTraceError",
    "VerticalSpringOscillatorCaseEvaluator",
    "extract_spring_trace",
    "score_spring_traces",
    "theoretical_period_s",
]


def __getattr__(name: str):
    if name in _SCORING_EXPORTS:
        module = import_module(f"{__name__}.scoring")
    elif name in _EVALUATOR_EXPORTS:
        module = import_module(f"{__name__}.evaluator")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
