from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...common.artifacts.curves import save_iou_curve as _save_iou_curve
from ...common.masks.quality import observed_mask_iou
from ...common.similarity import (
    exponential_delta_similarity,
    scaled_delta,
)


class TraceQualityError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PendulumTrace:
    times_s: np.ndarray
    angle_rad: np.ndarray
    bob_xy: np.ndarray
    frame_pivot_xy: np.ndarray
    pivot_xy: np.ndarray
    length_px: np.ndarray
    valid: np.ndarray
    valid_ratio: float
    pivot_drift_ratio: float
    length_cv: float
    period_s: float | None
    amplitude_rad: float


def _interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = values.astype(np.float64, copy=True)
    indices = np.arange(len(values))
    if output.ndim == 1:
        output[~valid] = np.interp(indices[~valid], indices[valid], output[valid])
        return output
    for column in range(output.shape[1]):
        output[~valid, column] = np.interp(
            indices[~valid],
            indices[valid],
            output[valid, column],
        )
    return output


def _estimate_period(
    angle: np.ndarray,
    times_s: np.ndarray,
    *,
    minimum_s: float,
    maximum_s: float,
) -> float | None:
    if len(angle) < 5:
        return None
    centered = angle - np.mean(angle)
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return None
    step = float(np.median(np.diff(times_s)))
    minimum_lag = max(2, int(math.ceil(minimum_s / step)))
    maximum_lag = min(len(angle) - 2, int(math.floor(maximum_s / step)))
    if minimum_lag > maximum_lag:
        return None
    correlations = np.correlate(centered, centered, mode="full")[len(angle) - 1 :]
    normalized = correlations / max(correlations[0], 1e-12)
    for lag in range(minimum_lag + 1, maximum_lag):
        if (
            normalized[lag] >= normalized[lag - 1]
            and normalized[lag] > normalized[lag + 1]
            and normalized[lag] >= 0.1
        ):
            return lag * step
    window = normalized[minimum_lag : maximum_lag + 1]
    if not len(window):
        return None
    lag = minimum_lag + int(np.argmax(window))
    return lag * step if normalized[lag] >= 0.1 else None


def extract_trace(
    masks: list[np.ndarray],
    times_s: list[float],
    *,
    quality_config: dict[str, Any],
    period_config: dict[str, Any],
) -> PendulumTrace:
    if len(masks) != len(times_s):
        raise ValueError("mask/time length mismatch")
    frame_area = float(masks[0].shape[0] * masks[0].shape[1])
    minimum_area = max(
        int(quality_config["minimum_mask_pixels"]),
        int(round(frame_area * float(quality_config["minimum_mask_area_ratio"]))),
    )
    maximum_area = int(
        round(frame_area * float(quality_config["maximum_mask_area_ratio"]))
    )
    valid = np.zeros(len(masks), dtype=bool)
    pivots = np.full((len(masks), 2), np.nan, dtype=np.float64)
    bobs = np.full((len(masks), 2), np.nan, dtype=np.float64)
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        area = len(xs)
        if area < minimum_area or area > maximum_area:
            continue
        top_cut = np.quantile(ys, 0.06)
        bottom_cut = np.quantile(ys, 0.76)
        top = ys <= top_cut
        bottom = ys >= bottom_cut
        if top.sum() < 2 or bottom.sum() < 3:
            continue
        pivots[index] = [float(np.median(xs[top])), float(np.min(ys[top]))]
        bobs[index] = [float(np.mean(xs[bottom])), float(np.mean(ys[bottom]))]
        valid[index] = True
    valid_ratio = float(np.mean(valid))
    required_ratio = float(quality_config["minimum_valid_frame_ratio"])
    if valid_ratio < required_ratio or int(valid.sum()) < 3:
        raise TraceQualityError(
            "insufficient_valid_masks",
            f"valid pendulum masks {int(valid.sum())}/{len(valid)} "
            f"({valid_ratio:.3f}) below required {required_ratio:.3f}",
        )
    pivots = _interpolate(pivots, valid)
    bobs = _interpolate(bobs, valid)
    pivot = np.median(pivots[valid], axis=0)
    displacement = bobs - pivot
    length = np.linalg.norm(displacement, axis=1)
    median_length = float(np.median(length[valid]))
    if median_length <= 1.0:
        raise TraceQualityError(
            "invalid_pendulum_length", "detected pendulum length is too small"
        )
    angle = np.arctan2(displacement[:, 0], displacement[:, 1])
    pivot_drift = np.linalg.norm(pivots - pivot, axis=1)
    pivot_drift_ratio = float(np.std(pivot_drift[valid]) / median_length)
    length_cv = float(np.std(length[valid]) / max(np.mean(length[valid]), 1e-12))
    times = np.asarray(times_s, dtype=np.float64)
    period = _estimate_period(
        angle,
        times,
        minimum_s=float(period_config["minimum_s"]),
        maximum_s=float(period_config["maximum_s"]),
    )
    amplitude = float(np.quantile(np.abs(angle - np.median(angle)), 0.95))
    return PendulumTrace(
        times_s=times,
        angle_rad=angle,
        bob_xy=bobs,
        frame_pivot_xy=pivots,
        pivot_xy=pivot,
        length_px=length,
        valid=valid,
        valid_ratio=valid_ratio,
        pivot_drift_ratio=pivot_drift_ratio,
        length_cv=length_cv,
        period_s=period,
        amplitude_rad=amplitude,
    )


def score_traces(
    reference: PendulumTrace,
    prediction: PendulumTrace,
    *,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    if not np.allclose(reference.times_s, prediction.times_s, atol=1e-9):
        raise ValueError("reference and prediction traces use different timelines")
    difference = prediction.angle_rad - reference.angle_rad
    angle_rmse = float(np.sqrt(np.mean(np.square(difference))))
    angle_scale = max(
        reference.amplitude_rad,
        math.radians(float(scoring_config["minimum_angle_scale_deg"])),
    )
    trajectory_score = float(math.exp(-angle_rmse / angle_scale))

    amplitude_error = abs(
        prediction.amplitude_rad - reference.amplitude_rad
    ) / max(reference.amplitude_rad, math.radians(1.0))
    amplitude_score = float(math.exp(-amplitude_error))

    if reference.period_s is None:
        period_score: float | None = None
        period_relative_error: float | None = None
    elif prediction.period_s is None:
        period_score = 0.0
        period_relative_error = None
    else:
        period_relative_error = abs(
            prediction.period_s - reference.period_s
        ) / reference.period_s
        period_score = float(math.exp(-period_relative_error))

    drift_scale = float(scoring_config["pivot_drift_scale"])
    length_scale = float(scoring_config["length_cv_scale"])
    pivot_drift_error = scaled_delta(
        reference.pivot_drift_ratio,
        prediction.pivot_drift_ratio,
        scale=drift_scale,
    )
    length_cv_error = scaled_delta(
        reference.length_cv,
        prediction.length_cv,
        scale=length_scale,
    )
    pivot_score = exponential_delta_similarity(
        reference.pivot_drift_ratio,
        prediction.pivot_drift_ratio,
        scale=drift_scale,
    )
    length_score = exponential_delta_similarity(
        reference.length_cv,
        prediction.length_cv,
        scale=length_scale,
    )
    structural_score = 0.5 * (pivot_score + length_score)

    configured_weights = {
        key: float(value)
        for key, value in scoring_config["weights"].items()
    }
    components: dict[str, float] = {
        "angle_trajectory": trajectory_score,
        "amplitude": amplitude_score,
        "structural_consistency": structural_score,
    }
    if period_score is not None:
        components["period"] = period_score
    denominator = sum(configured_weights[name] for name in components)
    score = sum(
        components[name] * configured_weights[name] for name in components
    ) / denominator
    return {
        "score": float(score),
        "components": components,
        "angle_rmse_rad": angle_rmse,
        "reference_amplitude_rad": reference.amplitude_rad,
        "prediction_amplitude_rad": prediction.amplitude_rad,
        "amplitude_relative_error": amplitude_error,
        "reference_period_s": reference.period_s,
        "prediction_period_s": prediction.period_s,
        "period_relative_error": period_relative_error,
        "reference_pivot_drift_ratio": reference.pivot_drift_ratio,
        "prediction_pivot_drift_ratio": prediction.pivot_drift_ratio,
        "pivot_drift_scaled_error": pivot_drift_error,
        "reference_length_cv": reference.length_cv,
        "prediction_length_cv": prediction.length_cv,
        "length_cv_scaled_error": length_cv_error,
        "weights_used": {
            name: configured_weights[name] / denominator for name in components
        },
    }


def write_per_frame_csv(
    path: Path,
    *,
    times_s: list[float],
    reference_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    reference: PendulumTrace,
    prediction: PendulumTrace,
) -> list[float | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ious: list[float | None] = []
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(times_s):
        iou = observed_mask_iou(
            reference_masks[index], prediction_masks[index]
        )
        ious.append(iou)
        rows.append(
            {
                "frame": index,
                "time_s": time_s,
                "physical_subject_iou": iou,
                "reference_angle_rad": float(reference.angle_rad[index]),
                "prediction_angle_rad": float(prediction.angle_rad[index]),
                "reference_mask_area": int(
                    np.count_nonzero(reference_masks[index])
                ),
                "prediction_mask_area": int(
                    np.count_nonzero(prediction_masks[index])
                ),
                "reference_tracking_valid": bool(reference.valid[index]),
                "prediction_tracking_valid": bool(prediction.valid[index]),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return ious


def save_iou_curve(
    path: Path,
    *,
    times_s: list[float],
    ious: list[float | None],
    case_id: str,
) -> None:
    _save_iou_curve(
        path,
        times_s=times_s,
        ious=ious,
        case_id=case_id,
        scene_name="Pendulum",
    )
