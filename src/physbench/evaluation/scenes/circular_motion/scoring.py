from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...common.fitting import exponential_similarity, normalized_rmse
from ...common.geometry import CircleModel, fit_circle
from ...common.tracking import InstanceTracks


@dataclass(frozen=True)
class OrbitTrace:
    xy: np.ndarray
    valid: np.ndarray
    valid_ratio: float
    circle: CircleModel
    relative_angle_rad: np.ndarray
    angular_velocity_rad_s: float
    angular_fit_rmse_rad: float


def extract_orbit_traces(
    tracks: InstanceTracks, times_s: list[float]
) -> list[OrbitTrace]:
    times = np.asarray(times_s, dtype=np.float64)
    output = []
    for object_index in range(tracks.xy.shape[1]):
        xy = tracks.xy[:, object_index]
        circle = fit_circle(xy)
        raw_angle = np.arctan2(
            xy[:, 1] - circle.center_xy[1],
            xy[:, 0] - circle.center_xy[0],
        )
        relative = np.unwrap(raw_angle)
        relative = relative - relative[0]
        coefficients = np.polyfit(times, relative, 1)
        fitted = np.polyval(coefficients, times)
        output.append(
            OrbitTrace(
                xy=xy,
                valid=tracks.valid[:, object_index],
                valid_ratio=float(tracks.valid_ratio[object_index]),
                circle=circle,
                relative_angle_rad=relative,
                angular_velocity_rad_s=float(coefficients[0]),
                angular_fit_rmse_rad=float(
                    np.sqrt(np.mean(np.square(relative - fitted)))
                ),
            )
        )
    return sorted(output, key=lambda trace: trace.circle.radius_px)


def score_orbits(
    reference: list[OrbitTrace],
    prediction: list[OrbitTrace],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(reference) != len(prediction):
        raise ValueError("reference and prediction object counts differ")
    object_metrics = []
    for reference_orbit, prediction_orbit in zip(reference, prediction):
        angle_rmse, _ = normalized_rmse(
            reference_orbit.relative_angle_rad,
            prediction_orbit.relative_angle_rad,
            minimum_scale=1.0,
        )
        angle_score = exponential_similarity(
            angle_rmse, scale=float(config["angular_trajectory_scale_rad"])
        )
        omega_error = abs(
            prediction_orbit.angular_velocity_rad_s
            - reference_orbit.angular_velocity_rad_s
        ) / max(abs(reference_orbit.angular_velocity_rad_s), 1e-9)
        omega_score = exponential_similarity(
            omega_error, scale=float(config["angular_velocity_error_scale"])
        )
        circularity_score = exponential_similarity(
            prediction_orbit.circle.radial_cv,
            scale=float(config["radial_cv_scale"]),
        )
        uniformity_score = exponential_similarity(
            prediction_orbit.angular_fit_rmse_rad,
            scale=float(config["angular_fit_rmse_scale_rad"]),
        )
        object_metrics.append(
            {
                "angle_trajectory_score": angle_score,
                "angular_velocity_score": omega_score,
                "circularity_score": circularity_score,
                "uniformity_score": uniformity_score,
                "angle_trajectory_rmse_rad": angle_rmse,
                "reference_angular_velocity_rad_s": (
                    reference_orbit.angular_velocity_rad_s
                ),
                "prediction_angular_velocity_rad_s": (
                    prediction_orbit.angular_velocity_rad_s
                ),
                "angular_velocity_relative_error": omega_error,
                "prediction_radial_cv": prediction_orbit.circle.radial_cv,
                "prediction_angular_fit_rmse_rad": (
                    prediction_orbit.angular_fit_rmse_rad
                ),
            }
        )
    reference_radii = np.asarray(
        [trace.circle.radius_px for trace in reference], dtype=np.float64
    )
    prediction_radii = np.asarray(
        [trace.circle.radius_px for trace in prediction], dtype=np.float64
    )
    reference_configuration = reference_radii / max(reference_radii.max(), 1e-9)
    prediction_configuration = prediction_radii / max(
        prediction_radii.max(), 1e-9
    )
    radius_configuration_error = float(
        np.sqrt(
            np.mean(
                np.square(reference_configuration - prediction_configuration)
            )
        )
    )
    radius_score = exponential_similarity(
        radius_configuration_error,
        scale=float(config["radius_configuration_scale"]),
    )
    components = {
        "angular_trajectory": float(
            np.mean([item["angle_trajectory_score"] for item in object_metrics])
        ),
        "angular_velocity": float(
            np.mean([item["angular_velocity_score"] for item in object_metrics])
        ),
        "orbit_geometry": float(
            0.5
            * (
                np.mean([item["circularity_score"] for item in object_metrics])
                + radius_score
            )
        ),
        "uniform_motion": float(
            np.mean([item["uniformity_score"] for item in object_metrics])
        ),
    }
    weights = {key: float(value) for key, value in config["weights"].items()}
    denominator = sum(weights[name] for name in components)
    score = sum(weights[name] * value for name, value in components.items())
    return {
        "score": float(score / denominator),
        "components": components,
        "objects": object_metrics,
        "reference_normalized_radii": reference_configuration.tolist(),
        "prediction_normalized_radii": prediction_configuration.tolist(),
        "radius_configuration_error": radius_configuration_error,
        "weights_used": {
            name: weights[name] / denominator for name in components
        },
    }
