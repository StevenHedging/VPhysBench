from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .artifacts import save_iou_curve
from .errors import SceneAnalysisError
from .masks.quality import mask_centroid


@dataclass(frozen=True)
class SubjectComparison:
    """Reference-aware physical-subject position, shape, and appearance score."""

    score: float
    components: dict[str, float | None]
    per_frame: list[dict[str, Any]]
    reference_observed_ratio: float
    prediction_observed_ratio: float
    comparable_reference: str

    def to_metric(self, *, weights: dict[str, float]) -> dict[str, Any]:
        return {
            "score": self.score,
            "components": self.components,
            "weights_used": weights,
            "reference_observed_frame_ratio": self.reference_observed_ratio,
            "prediction_observed_frame_ratio": self.prediction_observed_ratio,
            "comparable_reference": self.comparable_reference,
        }


def infer_reference_mode(case: dict[str, Any]) -> str:
    return (
        "same_case_reference"
        if case.get("has_real_reference_video", False)
        else "parent_physics_reference"
    )


def _safe_score(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _mask_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference_binary = reference > 0
    prediction_binary = prediction > 0
    union = int(np.logical_or(reference_binary, prediction_binary).sum())
    if union == 0:
        return 0.0
    return float(
        np.logical_and(reference_binary, prediction_binary).sum() / union
    )


def _canonical_subject(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    binary = (mask > 0).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width <= 0 or height <= 0:
        return None
    crop_frame = frame[y : y + height, x : x + width]
    crop_mask = binary[y : y + height, x : x + width]
    scale = min(size / width, size / height)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    image = cv2.resize(
        crop_frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    resized_mask = cv2.resize(
        crop_mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output_image = np.zeros((size, size, 3), dtype=np.uint8)
    output_mask = np.zeros((size, size), dtype=np.uint8)
    offset_x = (size - target_width) // 2
    offset_y = (size - target_height) // 2
    output_image[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = image
    output_mask[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = resized_mask
    return output_image, output_mask


def _boundary_f(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    tolerance_px: int,
) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_boundary = cv2.morphologyEx(
        (reference > 0).astype(np.uint8),
        cv2.MORPH_GRADIENT,
        kernel,
    )
    prediction_boundary = cv2.morphologyEx(
        (prediction > 0).astype(np.uint8),
        cv2.MORPH_GRADIENT,
        kernel,
    )
    reference_count = int(reference_boundary.sum())
    prediction_count = int(prediction_boundary.sum())
    if reference_count == 0 or prediction_count == 0:
        return 1.0 if reference_count == prediction_count else 0.0
    radius = max(1, int(tolerance_px))
    tolerance = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    reference_dilated = cv2.dilate(reference_boundary, tolerance)
    prediction_dilated = cv2.dilate(prediction_boundary, tolerance)
    precision = float(
        np.logical_and(prediction_boundary > 0, reference_dilated > 0).sum()
        / prediction_count
    )
    recall = float(
        np.logical_and(reference_boundary > 0, prediction_dilated > 0).sum()
        / reference_count
    )
    if precision + recall <= 1e-12:
        return 0.0
    return _safe_score(2.0 * precision * recall / (precision + recall))


def _histogram_similarity(
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
) -> float:
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    prediction_lab = cv2.cvtColor(prediction, cv2.COLOR_BGR2LAB)
    scores = []
    for channel in range(3):
        reference_hist = cv2.calcHist(
            [reference_lab], [channel], reference_mask, [16], [0, 256]
        ).reshape(-1)
        prediction_hist = cv2.calcHist(
            [prediction_lab], [channel], prediction_mask, [16], [0, 256]
        ).reshape(-1)
        reference_hist /= max(float(reference_hist.sum()), 1e-12)
        prediction_hist /= max(float(prediction_hist.sum()), 1e-12)
        scores.append(float(np.minimum(reference_hist, prediction_hist).sum()))
    return _safe_score(float(np.mean(scores)))


def _global_ssim(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = mask > 0
    if int(selected.sum()) < 4:
        return 0.0
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(
        np.float64
    )
    prediction_gray = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY).astype(
        np.float64
    )
    x = reference_gray[selected]
    y = prediction_gray[selected]
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    variance_x = float(x.var())
    variance_y = float(y.var())
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (
        (mean_x * mean_x + mean_y * mean_y + c1)
        * (variance_x + variance_y + c2)
    )
    if denominator <= 1e-12:
        return 0.0
    return _safe_score((numerator / denominator + 1.0) / 2.0)


def _gradient_histogram(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = 8,
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=False)
    selected = mask > 0
    if not selected.any():
        return np.zeros(bins, dtype=np.float64)
    indices = np.floor(angle[selected] * bins / (2.0 * np.pi)).astype(int) % bins
    histogram = np.bincount(
        indices,
        weights=magnitude[selected],
        minlength=bins,
    ).astype(np.float64)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm > 1e-12 else histogram


def _gradient_similarity(
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
) -> float:
    first = _gradient_histogram(reference, reference_mask)
    second = _gradient_histogram(prediction, prediction_mask)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 and second_norm <= 1e-12:
        return 1.0
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return 0.0
    return _safe_score(float(np.dot(first, second) / (first_norm * second_norm)))


def _appearance(
    reference_frame: np.ndarray,
    prediction_frame: np.ndarray,
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    canonical_size: int,
) -> tuple[float, dict[str, float]]:
    reference = _canonical_subject(
        reference_frame, reference_mask, size=canonical_size
    )
    prediction = _canonical_subject(
        prediction_frame, prediction_mask, size=canonical_size
    )
    if reference is None or prediction is None:
        return 0.0, {"color": 0.0, "ssim": 0.0, "gradient": 0.0}
    reference_image, reference_canonical_mask = reference
    prediction_image, prediction_canonical_mask = prediction
    intersection = cv2.bitwise_and(
        reference_canonical_mask, prediction_canonical_mask
    )
    color = _histogram_similarity(
        reference_image,
        prediction_image,
        reference_canonical_mask,
        prediction_canonical_mask,
    )
    ssim = _global_ssim(reference_image, prediction_image, intersection)
    gradient = _gradient_similarity(
        reference_image,
        prediction_image,
        reference_canonical_mask,
        prediction_canonical_mask,
    )
    score = 0.45 * color + 0.35 * ssim + 0.20 * gradient
    return _safe_score(score), {
        "color": color,
        "ssim": ssim,
        "gradient": gradient,
    }


def _shape(
    reference_frame: np.ndarray,
    prediction_frame: np.ndarray,
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    canonical_size: int,
    boundary_tolerance_px: int,
) -> tuple[float, float, float]:
    reference = _canonical_subject(
        reference_frame, reference_mask, size=canonical_size
    )
    prediction = _canonical_subject(
        prediction_frame, prediction_mask, size=canonical_size
    )
    if reference is None or prediction is None:
        return 0.0, 0.0, 0.0
    _, reference_canonical = reference
    _, prediction_canonical = prediction
    aligned_iou = _mask_iou(reference_canonical, prediction_canonical)
    boundary = _boundary_f(
        reference_canonical,
        prediction_canonical,
        tolerance_px=boundary_tolerance_px,
    )
    return _safe_score(0.6 * aligned_iou + 0.4 * boundary), aligned_iou, boundary


def compare_subjects(
    *,
    reference_frames: list[np.ndarray],
    prediction_frames: list[np.ndarray],
    reference_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    reference_mode: str,
    config: dict[str, Any],
    condition_frame: np.ndarray | None = None,
    condition_mask: np.ndarray | None = None,
) -> SubjectComparison:
    lengths = {
        len(reference_frames),
        len(prediction_frames),
        len(reference_masks),
        len(prediction_masks),
    }
    if len(lengths) != 1 or not reference_frames:
        raise SceneAnalysisError(
            "subject_sequence_mismatch",
            "subject comparison requires equal non-empty frame and mask sequences",
        )
    minimum_area = int(config.get("minimum_observed_pixels", 4))
    position_scale = max(float(config.get("position_distance_scale", 0.08)), 1e-6)
    canonical_size = int(config.get("canonical_crop_size", 64))
    boundary_tolerance = int(config.get("boundary_tolerance_px", 2))
    weights = {
        key: float(value)
        for key, value in config.get(
            "weights",
            {"position": 0.5, "shape": 0.2, "appearance": 0.3},
        ).items()
    }
    reference_valid = np.asarray(
        [_mask_area(mask) >= minimum_area for mask in reference_masks],
        dtype=bool,
    )
    prediction_valid = np.asarray(
        [_mask_area(mask) >= minimum_area for mask in prediction_masks],
        dtype=bool,
    )
    if not reference_valid.any():
        raise SceneAnalysisError(
            "reference_subject_unobserved",
            "reference contains no observable physical-subject mask",
        )

    if (condition_frame is None) != (condition_mask is None):
        raise SceneAnalysisError(
            "condition_subject_mismatch",
            "condition frame and mask must either both be supplied or both be absent",
        )
    if condition_frame is not None:
        if (
            condition_frame.shape[:2] != prediction_frames[0].shape[:2]
            or condition_mask is None
            or condition_mask.shape[:2] != prediction_masks[0].shape[:2]
        ):
            raise SceneAnalysisError(
                "condition_subject_shape_mismatch",
                "condition frame and mask must use the normalized prediction canvas",
            )
        conditioned_frame = condition_frame
        conditioned_mask = condition_mask
        conditioned_source = "case_condition_first_frame"
    else:
        conditioned_frame = prediction_frames[0]
        conditioned_mask = prediction_masks[0]
        conditioned_source = "prediction_frame_zero_fallback"
    conditioned_valid = _mask_area(conditioned_mask) >= minimum_area

    per_frame: list[dict[str, Any]] = []
    for index, (
        reference_frame,
        prediction_frame,
        reference_mask,
        prediction_mask,
    ) in enumerate(
        zip(
            reference_frames,
            prediction_frames,
            reference_masks,
            prediction_masks,
        )
    ):
        row: dict[str, Any] = {
            "frame": index,
            "reference_observed": bool(reference_valid[index]),
            "prediction_observed": bool(prediction_valid[index]),
        }
        if reference_mode == "parent_physics_reference":
            row["position"] = None
            row["mask_iou"] = None
            row["conditioned_source"] = conditioned_source
            if not conditioned_valid or not prediction_valid[index]:
                row.update(
                    {
                        "shape": 0.0,
                        "appearance": 0.0,
                        "appearance_color": 0.0,
                        "appearance_ssim": 0.0,
                        "appearance_gradient": 0.0,
                        "subject": 0.0,
                    }
                )
            else:
                shape, aligned_iou, boundary = _shape(
                    conditioned_frame,
                    prediction_frame,
                    conditioned_mask,
                    prediction_mask,
                    canonical_size=canonical_size,
                    boundary_tolerance_px=boundary_tolerance,
                )
                appearance, appearance_parts = _appearance(
                    conditioned_frame,
                    prediction_frame,
                    conditioned_mask,
                    prediction_mask,
                    canonical_size=canonical_size,
                )
                row.update(
                    {
                        "shape": shape,
                        "aligned_mask_iou": aligned_iou,
                        "boundary_f": boundary,
                        "appearance": appearance,
                        "appearance_color": appearance_parts["color"],
                        "appearance_ssim": appearance_parts["ssim"],
                        "appearance_gradient": appearance_parts["gradient"],
                        "subject": appearance,
                    }
                )
            per_frame.append(row)
            continue

        if not reference_valid[index]:
            row.update(
                {
                    "position": None,
                    "mask_iou": None,
                    "shape": None,
                    "appearance": None,
                    "subject": None,
                }
            )
            per_frame.append(row)
            continue
        if not prediction_valid[index]:
            row.update(
                {
                    "position": 0.0,
                    "mask_iou": 0.0,
                    "centroid_similarity": 0.0,
                    "shape": 0.0,
                    "aligned_mask_iou": 0.0,
                    "boundary_f": 0.0,
                    "appearance": 0.0,
                    "appearance_color": 0.0,
                    "appearance_ssim": 0.0,
                    "appearance_gradient": 0.0,
                    "subject": 0.0,
                }
            )
            per_frame.append(row)
            continue

        reference_centroid = mask_centroid(reference_mask)
        prediction_centroid = mask_centroid(prediction_mask)
        if reference_centroid is None or prediction_centroid is None:
            centroid_similarity = 0.0
        else:
            height, width = reference_mask.shape[:2]
            diagonal = max(float(math.hypot(width, height)), 1.0)
            distance = float(
                np.linalg.norm(prediction_centroid - reference_centroid)
            )
            centroid_similarity = float(
                math.exp(-distance / (position_scale * diagonal))
            )
            row["centroid_distance_normalized"] = distance / diagonal
        iou = _mask_iou(reference_mask, prediction_mask)
        position = _safe_score(0.6 * centroid_similarity + 0.4 * iou)
        shape, aligned_iou, boundary = _shape(
            reference_frame,
            prediction_frame,
            reference_mask,
            prediction_mask,
            canonical_size=canonical_size,
            boundary_tolerance_px=boundary_tolerance,
        )
        appearance, appearance_parts = _appearance(
            reference_frame,
            prediction_frame,
            reference_mask,
            prediction_mask,
            canonical_size=canonical_size,
        )
        denominator = sum(weights.values())
        subject = (
            weights["position"] * position
            + weights["shape"] * shape
            + weights["appearance"] * appearance
        ) / max(denominator, 1e-12)
        row.update(
            {
                "position": position,
                "mask_iou": iou,
                "centroid_similarity": centroid_similarity,
                "shape": shape,
                "aligned_mask_iou": aligned_iou,
                "boundary_f": boundary,
                "appearance": appearance,
                "appearance_color": appearance_parts["color"],
                "appearance_ssim": appearance_parts["ssim"],
                "appearance_gradient": appearance_parts["gradient"],
                "subject": _safe_score(subject),
            }
        )
        per_frame.append(row)

    if reference_mode == "parent_physics_reference":
        component_names = ("appearance",)
        comparable_reference = (
            "case_condition_first_frame_appearance_only"
            if condition_frame is not None
            else "prediction_frame_zero_appearance_fallback"
        )
        score_values = [float(row["appearance"]) for row in per_frame]
    else:
        component_names = ("position", "shape", "appearance")
        comparable_reference = "same_case_ground_truth_video"
        score_values = [
            float(row["subject"])
            for row in per_frame
            if row["subject"] is not None
        ]
    components: dict[str, float | None] = {}
    for name in component_names:
        values = [
            float(row[name])
            for row in per_frame
            if row.get(name) is not None
        ]
        components[name] = float(np.mean(values)) if values else None
    return SubjectComparison(
        score=_safe_score(float(np.mean(score_values)) if score_values else 0.0),
        components=components,
        per_frame=per_frame,
        reference_observed_ratio=float(reference_valid.mean()),
        prediction_observed_ratio=float(prediction_valid.mean()),
        comparable_reference=comparable_reference,
    )


def compose_subject_and_state_score(
    *,
    state_score: float,
    subject: SubjectComparison,
    reference_mode: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if reference_mode == "parent_physics_reference":
        weights = {
            key: float(value)
            for key, value in config.get(
                "parent_weights",
                {"physics_state": 0.7, "conditioned_appearance": 0.3},
            ).items()
        }
        components = {
            "physics_state": _safe_score(state_score),
            "conditioned_appearance": subject.score,
        }
    else:
        weights = {
            key: float(value)
            for key, value in config.get(
                "case_weights",
                {"physics_state": 0.4, "subject": 0.6},
            ).items()
        }
        components = {
            "physics_state": _safe_score(state_score),
            "subject": subject.score,
        }
    denominator = sum(weights[name] for name in components)
    score = sum(weights[name] * value for name, value in components.items())
    return {
        "score": _safe_score(score / max(denominator, 1e-12)),
        "components": components,
        "weights_used": {
            name: weights[name] / max(denominator, 1e-12)
            for name in components
        },
        "reference_mode": reference_mode,
    }


def write_subject_artifacts(
    *,
    directory: Path,
    times_s: list[float],
    comparison: SubjectComparison,
    case_id: str,
    scene_name: str,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "subject_components.csv"
    curve_path = directory / "subject_similarity_curve.png"
    rows = []
    for time_s, row in zip(times_s, comparison.per_frame):
        rows.append({"time_s": time_s, **row})
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_iou_curve(
        curve_path,
        times_s=times_s,
        ious=[
            float(row["subject"]) if row.get("subject") is not None else None
            for row in comparison.per_frame
        ],
        case_id=case_id,
        scene_name=f"{scene_name} subject similarity",
        series_label="Physical-subject position/shape/appearance similarity",
        y_label="Similarity",
        metric_name="score",
    )
    return {
        "subject_components_csv": str(csv_path),
        "subject_similarity_curve": str(curve_path),
    }


def subject_weights(config: dict[str, Any]) -> dict[str, float]:
    weights = {
        key: float(value)
        for key, value in config.get(
            "weights",
            {"position": 0.5, "shape": 0.2, "appearance": 0.3},
        ).items()
    }
    denominator = sum(weights.values())
    return {
        key: value / max(denominator, 1e-12)
        for key, value in weights.items()
    }


def degraded_subject_metric(
    *,
    reference_mode: str,
    code: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "score": 0.0,
        "components": {
            (
                "conditioned_appearance"
                if reference_mode == "parent_physics_reference"
                else "subject"
            ): 0.0,
            "physics_state": 0.0,
        },
        "weights_used": {},
        "reference_mode": reference_mode,
        "degraded": True,
        "degradation_code": code,
        "degradation_reason": reason,
    }


__all__ = [
    "SubjectComparison",
    "compare_subjects",
    "compose_subject_and_state_score",
    "degraded_subject_metric",
    "infer_reference_mode",
    "subject_weights",
    "write_subject_artifacts",
]
