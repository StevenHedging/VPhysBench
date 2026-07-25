from __future__ import annotations

import math


def scaled_delta(
    reference: float,
    prediction: float,
    *,
    scale: float,
) -> float:
    """Absolute reference/prediction difference expressed in protocol units."""
    return abs(float(prediction) - float(reference)) / max(float(scale), 1e-12)


def relative_delta(
    reference: float,
    prediction: float,
    *,
    minimum_scale: float,
) -> float:
    """Reference-relative difference with a stable lower-bound scale."""
    denominator = max(abs(float(reference)), float(minimum_scale), 1e-12)
    return abs(float(prediction) - float(reference)) / denominator


def exponential_delta_similarity(
    reference: float,
    prediction: float,
    *,
    scale: float,
) -> float:
    """Reflexive similarity: identical finite values always produce exactly 1."""
    return float(math.exp(-scaled_delta(reference, prediction, scale=scale)))


def bounded_ratio_similarity(reference: float, prediction: float) -> float:
    """Similarity for quantities already bounded to [0, 1]."""
    difference = min(abs(float(prediction) - float(reference)), 1.0)
    return float(1.0 - difference)
