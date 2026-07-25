from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...common.errors import SceneAnalysisError
from ...common.fitting import exponential_similarity, normalized_rmse
from ...common.geometry import AxisModel, fit_axis


@dataclass(frozen=True)
class CollisionTrace:
    times_s: np.ndarray
    xy: np.ndarray
    valid: np.ndarray
    valid_ratio: np.ndarray
    axis: AxisModel
    scalar_position: np.ndarray
    normalized_position: np.ndarray
    scene_span_px: float
    event_frame: int
    event_time_s: float
    pre_velocity_normalized_s: np.ndarray
    post_velocity_normalized_s: np.ndarray
    cross_track_std_ratio: float
    momentum_residual_ratio: float
    effective_restitution: float | None


def _window_velocity(
    times: np.ndarray, positions: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            np.polyfit(times[indices], positions[indices, object_index], 1)[0]
            for object_index in range(positions.shape[1])
        ],
        dtype=np.float64,
    )


def extract_collision_trace(
    xy: np.ndarray,
    valid: np.ndarray,
    times_s: list[float],
    *,
    masses_kg: np.ndarray,
    minimum_span_px: float,
    velocity_window_fraction: float,
) -> CollisionTrace:
    times = np.asarray(times_s, dtype=np.float64)
    points = np.asarray(xy, dtype=np.float64)
    axis = fit_axis(points.reshape(-1, 2), orient_with_time=False)
    striker_motion = points[min(3, len(points) - 1), 0] - points[0, 0]
    if np.dot(striker_motion, axis.direction_xy) < 0:
        axis = AxisModel(
            axis.origin_xy,
            -axis.direction_xy,
            -axis.normal_xy,
            axis.explained_ratio,
        )
    flattened_s, flattened_d = axis.project(points.reshape(-1, 2))
    scalar = flattened_s.reshape(points.shape[:2])
    cross = flattened_d.reshape(points.shape[:2])
    span = float(np.max(scalar) - np.min(scalar))
    if span < minimum_span_px:
        raise SceneAnalysisError(
            "insufficient_collision_motion",
            f"collision subjects span only {span:.3f}px",
        )
    relative = scalar - scalar[0]
    normalized = relative / span
    striker_target_distance = np.abs(scalar[:, 1] - scalar[:, 0])
    event_frame = int(np.argmin(striker_target_distance))
    minimum_window = 3
    window = max(
        minimum_window, int(round(len(times) * velocity_window_fraction))
    )
    pre_end = max(minimum_window, event_frame - 1)
    pre_indices = np.arange(max(0, pre_end - window), pre_end)
    if len(pre_indices) < minimum_window:
        pre_indices = np.arange(minimum_window)
    post_start = min(len(times) - minimum_window, event_frame + 2)
    post_indices = np.arange(post_start, min(len(times), post_start + window))
    if len(post_indices) < minimum_window:
        post_indices = np.arange(len(times) - minimum_window, len(times))
    pre_velocity = _window_velocity(times, normalized, pre_indices)
    post_velocity = _window_velocity(times, normalized, post_indices)
    momentum_before = float(np.dot(masses_kg, pre_velocity))
    momentum_after = float(np.dot(masses_kg, post_velocity))
    momentum_scale = max(
        float(np.sum(np.abs(masses_kg * pre_velocity))), 1e-9
    )
    momentum_residual = abs(momentum_after - momentum_before) / momentum_scale
    closing_speed = float(pre_velocity[0] - pre_velocity[-1])
    separation_speed = float(post_velocity[-1] - post_velocity[0])
    restitution = (
        separation_speed / closing_speed if abs(closing_speed) > 1e-9 else None
    )
    return CollisionTrace(
        times_s=times,
        xy=points,
        valid=valid,
        valid_ratio=np.mean(valid, axis=0),
        axis=axis,
        scalar_position=scalar,
        normalized_position=normalized,
        scene_span_px=span,
        event_frame=event_frame,
        event_time_s=float(times[event_frame]),
        pre_velocity_normalized_s=pre_velocity,
        post_velocity_normalized_s=post_velocity,
        cross_track_std_ratio=float(np.std(cross, axis=0).mean() / span),
        momentum_residual_ratio=float(momentum_residual),
        effective_restitution=(
            float(restitution) if restitution is not None else None
        ),
    )


def score_collision(
    reference: CollisionTrace,
    prediction: CollisionTrace,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    trajectory_errors = []
    for object_index in range(reference.normalized_position.shape[1]):
        _, error = normalized_rmse(
            reference.normalized_position[:, object_index],
            prediction.normalized_position[:, object_index],
            minimum_scale=1.0,
        )
        trajectory_errors.append(error)
    trajectory_error = float(np.mean(trajectory_errors))
    trajectory_score = exponential_similarity(
        trajectory_error, scale=float(config["trajectory_error_scale"])
    )
    duration = max(float(reference.times_s[-1]), 1e-9)
    event_error = abs(
        prediction.event_time_s - reference.event_time_s
    ) / duration
    event_score = exponential_similarity(
        event_error, scale=float(config["event_time_error_scale"])
    )
    velocity_reference = np.concatenate(
        [
            reference.pre_velocity_normalized_s,
            reference.post_velocity_normalized_s,
        ]
    )
    velocity_prediction = np.concatenate(
        [
            prediction.pre_velocity_normalized_s,
            prediction.post_velocity_normalized_s,
        ]
    )
    velocity_rmse = float(
        np.sqrt(np.mean(np.square(velocity_reference - velocity_prediction)))
    )
    velocity_scale = max(float(np.max(np.abs(velocity_reference))), 1e-9)
    velocity_error = velocity_rmse / velocity_scale
    velocity_score = exponential_similarity(
        velocity_error, scale=float(config["velocity_error_scale"])
    )
    momentum_score = exponential_similarity(
        prediction.momentum_residual_ratio,
        scale=float(config["momentum_residual_scale"]),
    )
    if (
        reference.effective_restitution is None
        or prediction.effective_restitution is None
    ):
        restitution_score = 0.0
        restitution_error = None
    else:
        restitution_error = abs(
            prediction.effective_restitution
            - reference.effective_restitution
        ) / max(abs(reference.effective_restitution), 0.1)
        restitution_score = exponential_similarity(
            restitution_error,
            scale=float(config["restitution_error_scale"]),
        )
    conservation_score = 0.6 * momentum_score + 0.4 * restitution_score
    one_dimensional_score = exponential_similarity(
        prediction.cross_track_std_ratio,
        scale=float(config["cross_track_scale"]),
    )
    components = {
        "instance_trajectories": trajectory_score,
        "contact_event_time": event_score,
        "pre_post_velocities": velocity_score,
        "collision_physics": conservation_score,
        "one_dimensional_constraint": one_dimensional_score,
    }
    weights = {key: float(value) for key, value in config["weights"].items()}
    denominator = sum(weights[name] for name in components)
    score = sum(weights[name] * value for name, value in components.items())
    return {
        "score": float(score / denominator),
        "components": components,
        "per_object_trajectory_normalized_errors": trajectory_errors,
        "mean_trajectory_normalized_error": trajectory_error,
        "reference_event_time_s": reference.event_time_s,
        "prediction_event_time_s": prediction.event_time_s,
        "event_time_normalized_error": event_error,
        "reference_pre_velocity_normalized_s": (
            reference.pre_velocity_normalized_s.tolist()
        ),
        "prediction_pre_velocity_normalized_s": (
            prediction.pre_velocity_normalized_s.tolist()
        ),
        "reference_post_velocity_normalized_s": (
            reference.post_velocity_normalized_s.tolist()
        ),
        "prediction_post_velocity_normalized_s": (
            prediction.post_velocity_normalized_s.tolist()
        ),
        "velocity_normalized_error": velocity_error,
        "prediction_momentum_residual_ratio": (
            prediction.momentum_residual_ratio
        ),
        "reference_effective_restitution": reference.effective_restitution,
        "prediction_effective_restitution": prediction.effective_restitution,
        "effective_restitution_relative_error": restitution_error,
        "prediction_cross_track_std_ratio": (
            prediction.cross_track_std_ratio
        ),
        "weights_used": {
            name: weights[name] / denominator for name in components
        },
    }
