from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...common.errors import SceneAnalysisError
from ...common.fitting import exponential_similarity, fit_polynomial, normalized_rmse
from ...common.geometry import AxisModel, fit_axis
from ...common.similarity import bounded_ratio_similarity, scaled_delta
from ...common.tracking import CentroidTrace, interpolate_trace


@dataclass(frozen=True)
class InclineTrace:
    times_s: np.ndarray
    xy: np.ndarray
    valid: np.ndarray
    valid_ratio: float
    axis: AxisModel
    along_displacement_px: np.ndarray
    normalized_along_displacement: np.ndarray
    cross_displacement_px: np.ndarray
    span_px: float
    acceleration_px_s2: float
    normalized_acceleration_s2: float
    quadratic_rmse_ratio: float
    descent_time_s: float
    monotonic_progress_ratio: float
    cross_track_std_ratio: float
    orientation_std_deg: float


def _mask_orientations(
    masks: list[np.ndarray], valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.full(len(masks), np.nan, dtype=np.float64)
    angle_valid = valid.copy()
    for index, mask in enumerate(masks):
        points = cv2.findNonZero((mask > 0).astype(np.uint8))
        if points is None or len(points) < 5 or not valid[index]:
            angle_valid[index] = False
            continue
        (_, _), (width, height), angle = cv2.minAreaRect(points)
        if width < height:
            angle += 90.0
        angles[index] = angle
    if int(angle_valid.sum()) < 2:
        return np.zeros(len(masks), dtype=np.float64), angle_valid
    doubled = np.unwrap(np.deg2rad(2.0 * angles[angle_valid]))
    unwrapped = np.rad2deg(doubled) / 2.0
    angles[angle_valid] = unwrapped
    return interpolate_trace(angles, angle_valid), angle_valid


def extract_incline_trace(
    centroid: CentroidTrace,
    masks: list[np.ndarray],
    times_s: list[float],
    *,
    minimum_span_px: float,
) -> InclineTrace:
    times = np.asarray(times_s, dtype=np.float64)
    axis = fit_axis(centroid.xy)
    along, cross = axis.project(centroid.xy)
    along = along - along[0]
    cross = cross - np.median(cross)
    span = float(np.max(along) - np.min(along))
    if span < minimum_span_px:
        raise SceneAnalysisError(
            "insufficient_incline_motion",
            f"sliding subject spans only {span:.3f}px",
        )
    normalized = along / span
    fit_end_candidates = np.flatnonzero(along >= 0.9 * float(np.max(along)))
    fit_end = int(fit_end_candidates[0]) + 1 if len(fit_end_candidates) else len(along)
    fit_end = max(5, min(fit_end, len(along)))
    coefficients, _, rmse = fit_polynomial(
        times[:fit_end], along[:fit_end], degree=2
    )
    acceleration = float(2.0 * coefficients[0])
    reached = np.flatnonzero(along >= 0.95 * float(np.max(along)))
    descent_time = float(times[reached[0]]) if len(reached) else float(times[-1])
    tolerance = 0.015 * span
    monotonic = float(np.mean(np.diff(along) >= -tolerance))
    orientations, orientation_valid = _mask_orientations(masks, centroid.valid)
    orientation_std = (
        float(np.std(orientations[orientation_valid]))
        if int(orientation_valid.sum()) >= 2
        else 180.0
    )
    return InclineTrace(
        times_s=times,
        xy=centroid.xy,
        valid=centroid.valid,
        valid_ratio=centroid.valid_ratio,
        axis=axis,
        along_displacement_px=along,
        normalized_along_displacement=normalized,
        cross_displacement_px=cross,
        span_px=span,
        acceleration_px_s2=acceleration,
        normalized_acceleration_s2=acceleration / span,
        quadratic_rmse_ratio=float(rmse / span),
        descent_time_s=descent_time,
        monotonic_progress_ratio=monotonic,
        cross_track_std_ratio=float(np.std(cross) / span),
        orientation_std_deg=orientation_std,
    )


def score_incline(
    reference: InclineTrace,
    prediction: InclineTrace,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    trajectory_rmse, trajectory_error = normalized_rmse(
        reference.normalized_along_displacement,
        prediction.normalized_along_displacement,
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
    time_error = abs(
        prediction.descent_time_s - reference.descent_time_s
    ) / duration
    time_score = exponential_similarity(
        time_error, scale=float(config["descent_time_error_scale"])
    )
    cross_track_error = scaled_delta(
        reference.cross_track_std_ratio,
        prediction.cross_track_std_ratio,
        scale=float(config["cross_track_scale"]),
    )
    orientation_error = scaled_delta(
        reference.orientation_std_deg,
        prediction.orientation_std_deg,
        scale=float(config["orientation_std_scale_deg"]),
    )
    contact_score = exponential_similarity(cross_track_error)
    orientation_score = exponential_similarity(orientation_error)
    monotonic_score = bounded_ratio_similarity(
        reference.monotonic_progress_ratio,
        prediction.monotonic_progress_ratio,
    )
    constraint_score = (
        0.45 * contact_score + 0.35 * monotonic_score + 0.2 * orientation_score
    )
    components = {
        "along_plane_trajectory": trajectory_score,
        "normalized_acceleration": acceleration_score,
        "descent_time": time_score,
        "contact_and_pose_constraints": constraint_score,
    }
    weights = {key: float(value) for key, value in config["weights"].items()}
    denominator = sum(weights[name] for name in components)
    score = sum(weights[name] * value for name, value in components.items())
    return {
        "score": float(score / denominator),
        "components": components,
        "along_trajectory_rmse": trajectory_rmse,
        "along_trajectory_normalized_error": trajectory_error,
        "reference_normalized_acceleration_s2": (
            reference.normalized_acceleration_s2
        ),
        "prediction_normalized_acceleration_s2": (
            prediction.normalized_acceleration_s2
        ),
        "normalized_acceleration_relative_error": acceleration_error,
        "reference_descent_time_s": reference.descent_time_s,
        "prediction_descent_time_s": prediction.descent_time_s,
        "descent_time_normalized_error": time_error,
        "reference_axis_explained_ratio": reference.axis.explained_ratio,
        "prediction_axis_explained_ratio": prediction.axis.explained_ratio,
        "reference_cross_track_std_ratio": reference.cross_track_std_ratio,
        "prediction_cross_track_std_ratio": prediction.cross_track_std_ratio,
        "cross_track_scaled_error": cross_track_error,
        "reference_orientation_std_deg": reference.orientation_std_deg,
        "prediction_orientation_std_deg": prediction.orientation_std_deg,
        "orientation_std_scaled_error": orientation_error,
        "reference_monotonic_progress_ratio": (
            reference.monotonic_progress_ratio
        ),
        "prediction_monotonic_progress_ratio": (
            prediction.monotonic_progress_ratio
        ),
        "monotonic_progress_absolute_error": abs(
            prediction.monotonic_progress_ratio
            - reference.monotonic_progress_ratio
        ),
        "reference_quadratic_rmse_ratio": reference.quadratic_rmse_ratio,
        "prediction_quadratic_rmse_ratio": prediction.quadratic_rmse_ratio,
        "weights_used": {
            name: weights[name] / denominator for name in components
        },
    }
