"""Pure trace extraction and physics scoring for vertical spring oscillators."""

from .scoring import (
    SpringTrace,
    SpringTraceError,
    extract_spring_trace,
    score_spring_traces,
    theoretical_period_s,
)
from .evaluator import VerticalSpringOscillatorCaseEvaluator

__all__ = [
    "SpringTrace",
    "SpringTraceError",
    "extract_spring_trace",
    "score_spring_traces",
    "theoretical_period_s",
    "VerticalSpringOscillatorCaseEvaluator",
]
