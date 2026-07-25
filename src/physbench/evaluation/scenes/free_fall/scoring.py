from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...common.errors import SceneAnalysisError
from ...common.fitting import exponential_similarity, fit_polynomial, normalized_rmse
from ...common.similarity import bounded_ratio_similarity, scaled_delta
from ...common.tracking import CentroidTrace


@dataclass(frozen=True)
class FreeFallTrace:
    times_s: np.ndarray
    xy: np.ndarray
    valid: np.ndarray
    valid_ratio: float
    vertical_displacement_px: np.ndarray
    normalized_vertical_displacement: np.ndarray
    vertical_span_px: float
    acceleration_px_s2: float
    normalized_acceleration_s2: float
    quadratic_rmse_ratio: float
    impact_time_s: float
    horizontal_drift_ratio: float
    downward_progress_ratio: float


def extract_free_fall_trace(
    centroid: CentroidTrace,
    times_s: list[float],
    *,
    minimum_vertical_span_px: float,
) -> FreeFallTrace:
    times = np.asarray(times_s, dtype=np.float64)
    xy = centroid.xy
    vertical = xy[:, 1] - xy[0, 1]
    span = float(np.max(np.abs(vertical)))
    if span < minimum_vertical_span_px:
        raise SceneAnalysisError(
            "insufficient_vertical_motion",
            f"falling subject spans only {span:.3f}px",
        )
    normalized = vertical / span
    coefficients, fitted, rmse = fit_polynomial(times, vertical, degree=2)
    acceleration = float(2.0 * coefficients[0])
    normalized_acceleration = acceleration / span
    target = 0.95 * float(np.max(vertical))
    reached = np.flatnonzero(vertical >= target)
    impact_time = float(times[reached[0]]) if len(reached) else float(times[-1])
    horizontal_drift = float(np.ptp(xy[:, 0]) / span)
    downward_steps = np.diff(vertical) >= -0.02 * span
    return FreeFallTrace(
        times_s=times,
        xy=xy,
        valid=centroid.valid,
        valid_ratio=centroid.valid_ratio,
        vertical_displacement_px=vertical,
        normalized_vertical_displacement=normalized,
        vertical_span_px=span,
        acceleration_px_s2=acceleration,
        normalized_acceleration_s2=normalized_acceleration,
        quadratic_rmse_ratio=float(rmse / span),
        impact_time_s=impact_time,
        horizontal_drift_ratio=horizontal_drift,
        downward_progress_ratio=float(np.mean(downward_steps)),
    )


def score_free_fall(
    reference: FreeFallTrace,
    prediction: FreeFallTrace,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    trajectory_rmse, trajectory_error = normalized_rmse(
        reference.normalized_vertical_displacement,
        prediction.normalized_vertical_displacement,
        minimum_scale=1.0,
    )
    trajectory_score = exponential_similarity(
        trajectory_error, scale=float(config["trajectory_error_scale"])
    )
    acceleration_error = abs(
        prediction.normalized_acceleration_s2
        - reference.normalized_acceleration_s2
    ) / max(abs(reference.normalized_acceleration_s2), 1e-9)
    acceleration_score = exponential_similarity(
        acceleration_error, scale=float(config["acceleration_error_scale"])
    )
    duration = max(float(reference.times_s[-1]), 1e-9)
    impact_error = abs(
        prediction.impact_time_s - reference.impact_time_s
    ) / duration
    impact_score = exponential_similarity(
        impact_error, scale=float(config["impact_time_error_scale"])
    )
    drift_error = scaled_delta(
        reference.horizontal_drift_ratio,
        prediction.horizontal_drift_ratio,
        scale=float(config["horizontal_drift_scale"]),
    )
    drift_score = exponential_similarity(drift_error)
    direction_score = bounded_ratio_similarity(
        reference.downward_progress_ratio,
        prediction.downward_progress_ratio,
    )
    constraint_score = 0.5 * (drift_score + direction_score)
    components = {
        "vertical_trajectory": trajectory_score,
        "normalized_acceleration": acceleration_score,
        "impact_time": impact_score,
        "motion_constraints": constraint_score,
    }
    weights = {key: float(value) for key, value in config["weights"].items()}
    denominator = sum(weights[name] for name in components)
    score = sum(weights[name] * value for name, value in components.items())
    return {
        "score": float(score / denominator),
        "components": components,
        "vertical_trajectory_rmse": trajectory_rmse,
        "vertical_trajectory_normalized_error": trajectory_error,
        "reference_normalized_acceleration_s2": (
            reference.normalized_acceleration_s2
        ),
        "prediction_normalized_acceleration_s2": (
            prediction.normalized_acceleration_s2
        ),
        "normalized_acceleration_relative_error": acceleration_error,
        "reference_impact_time_s": reference.impact_time_s,
        "prediction_impact_time_s": prediction.impact_time_s,
        "impact_time_normalized_error": impact_error,
        "reference_horizontal_drift_ratio": reference.horizontal_drift_ratio,
        "prediction_horizontal_drift_ratio": prediction.horizontal_drift_ratio,
        "horizontal_drift_scaled_error": drift_error,
        "reference_downward_progress_ratio": reference.downward_progress_ratio,
        "prediction_downward_progress_ratio": prediction.downward_progress_ratio,
        "downward_progress_absolute_error": abs(
            prediction.downward_progress_ratio
            - reference.downward_progress_ratio
        ),
        "reference_quadratic_rmse_ratio": reference.quadratic_rmse_ratio,
        "prediction_quadratic_rmse_ratio": prediction.quadratic_rmse_ratio,
        "weights_used": {
            name: weights[name] / denominator for name in components
        },
    }
