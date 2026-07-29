from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...io import sha256_file
from ..contracts import CaseEvaluationRequest
from .artifacts import save_iou_curve, write_rows_csv
from .base import SceneAnalysis
from .errors import ReferenceAnalysisError, SceneAnalysisError
from .subject import (
    SubjectComparison,
    compare_subjects,
    compose_subject_and_state_score,
    degraded_subject_metric,
    infer_reference_mode,
    subject_weights,
    write_subject_artifacts,
)


def robust_subject_enabled(config: dict[str, Any]) -> bool:
    return config.get("evaluator_contract") == "robust_subject_v3"


def reference_failure(
    exc: Exception,
    *,
    stage: str,
) -> ReferenceAnalysisError:
    code = getattr(exc, "code", "reference_analysis_failed")
    return ReferenceAnalysisError(
        f"reference_{code}",
        f"reference {stage} failed: {type(exc).__name__}: {exc}",
    )


def add_subject_comparison(
    analysis: SceneAnalysis,
    request: CaseEvaluationRequest,
    *,
    times_s: list[float],
    reference_frames: list[np.ndarray],
    prediction_frames: list[np.ndarray],
    reference_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    scene_name: str,
) -> SceneAnalysis:
    config = request.evaluator_config["subject_scoring"]
    reference_mode = infer_reference_mode(request.case)
    condition_frame: np.ndarray | None = None
    condition_mask: np.ndarray | None = None
    condition_provenance: dict[str, Any] = {}
    if reference_mode == "parent_physics_reference":
        value = request.case.get("assets", {}).get("first_frame")
        if not value:
            raise ReferenceAnalysisError(
                "reference_condition_frame_missing",
                "parent-reference case has no conditioned first-frame asset",
            )
        asset_root = request.asset_root.resolve()
        condition_path = (asset_root / value).resolve()
        try:
            condition_path.relative_to(asset_root)
        except ValueError as exc:
            raise ReferenceAnalysisError(
                "reference_condition_path_escape",
                f"conditioned first frame escapes dataset root: {value}",
            ) from exc
        source = cv2.imread(str(condition_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ReferenceAnalysisError(
                "reference_condition_frame_unreadable",
                f"cannot read conditioned first frame: {condition_path}",
            )
        target_height, target_width = prediction_frames[0].shape[:2]
        source_height, source_width = source.shape[:2]
        scale = min(
            target_width / max(source_width, 1),
            target_height / max(source_height, 1),
        )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            source,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        condition_frame = np.full(
            (target_height, target_width, 3),
            int(request.evaluator_config["spatial"].get("pad_value", 0)),
            dtype=np.uint8,
        )
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        condition_frame[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        # The generated frame-zero subject mask is deliberately reused only
        # as an ROI on the immutable condition image. A missing/wrong
        # prediction-side mask therefore remains a conservative zero.
        condition_mask = prediction_masks[0]
        condition_provenance = {
            "condition_frame": str(condition_path),
            "condition_frame_sha256": sha256_file(condition_path),
            "condition_mask_source": "prediction_frame_zero_roi",
            "condition_spatial_transform": {
                "policy": "preserve_aspect_ratio_letterbox",
                "scale": scale,
                "offset_xy": [offset_x, offset_y],
                "source_size": [source_width, source_height],
                "target_size": [target_width, target_height],
            },
        }
    try:
        comparison = compare_subjects(
            reference_frames=reference_frames,
            prediction_frames=prediction_frames,
            reference_masks=reference_masks,
            prediction_masks=prediction_masks,
            reference_mode=reference_mode,
            config=config,
            condition_frame=condition_frame,
            condition_mask=condition_mask,
        )
    except SceneAnalysisError as exc:
        if exc.code.startswith("reference_"):
            raise ReferenceAnalysisError(exc.code, str(exc)) from exc
        raise
    composite = compose_subject_and_state_score(
        state_score=float(analysis.score),
        subject=comparison,
        reference_mode=reference_mode,
        config=config,
    )
    analysis.score = float(composite["score"])
    analysis.metrics["scene_subject_state_similarity"] = composite
    metric_weights = (
        {"appearance": 1.0}
        if reference_mode == "parent_physics_reference"
        else subject_weights(config)
    )
    analysis.metrics["physical_subject_similarity"] = comparison.to_metric(
        weights=metric_weights
    )
    if "physical_subject_mask_iou" in analysis.metrics:
        analysis.metrics["physical_subject_mask_iou"]["role"] = (
            "position_component_and_required_diagnostic"
            if reference_mode == "same_case_reference"
            else "parent_reference_visual_diagnostic_not_scored"
        )
    analysis.quality.update(
        {
            "degraded": False,
            "degradation_codes": [],
            "subject_reference_observed_ratio": (
                comparison.reference_observed_ratio
            ),
            "subject_prediction_observed_ratio": (
                comparison.prediction_observed_ratio
            ),
        }
    )
    analysis.artifacts.update(
        write_subject_artifacts(
            directory=request.artifact_dir,
            times_s=times_s,
            comparison=comparison,
            case_id=request.case["case_id"],
            scene_name=scene_name,
        )
    )
    analysis.provenance["subject_comparison"] = {
        "contract": "position_shape_appearance_v1",
        "reference_mode": reference_mode,
        "same_case_components": ["position", "shape", "appearance"],
        "parent_case_components": [
            "physics_state",
            "conditioned_first_frame_appearance",
        ],
        **condition_provenance,
    }
    return analysis


def degraded_prediction_analysis(
    request: CaseEvaluationRequest,
    *,
    times_s: list[float],
    reference_masks: list[np.ndarray],
    code: str,
    reason: str,
    scene_name: str,
) -> SceneAnalysis:
    reference_mode = infer_reference_mode(request.case)
    observed = [
        bool(np.count_nonzero(mask))
        for mask in reference_masks
    ]
    ious = [0.0 if value else None for value in observed]
    rows = [
        {
            "frame": index,
            "time_s": time_s,
            "reference_observed": observed[index],
            "prediction_observed": False,
            "physical_subject_iou": ious[index],
            "position": 0.0 if observed[index] else None,
            "shape": 0.0 if observed[index] else None,
            "appearance": 0.0 if observed[index] else None,
            "subject": 0.0 if observed[index] else None,
            "degradation_code": code,
        }
        for index, time_s in enumerate(times_s)
    ]
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    csv_path = request.artifact_dir / "per_frame.csv"
    subject_csv_path = request.artifact_dir / "subject_components.csv"
    iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
    subject_curve_path = request.artifact_dir / "subject_similarity_curve.png"
    write_rows_csv(csv_path, rows)
    write_rows_csv(subject_csv_path, rows)
    save_iou_curve(
        iou_path,
        times_s=times_s,
        ious=ious,
        case_id=request.case["case_id"],
        scene_name=scene_name,
    )
    save_iou_curve(
        subject_curve_path,
        times_s=times_s,
        ious=ious,
        case_id=request.case["case_id"],
        scene_name=f"{scene_name} subject similarity",
        series_label="Physical-subject position/shape/appearance similarity",
        y_label="Similarity",
        metric_name="score",
    )
    observed_values = [float(value) for value in ious if value is not None]
    subject_metric = degraded_subject_metric(
        reference_mode=reference_mode,
        code=code,
        reason=reason,
    )
    return SceneAnalysis(
        score=0.0,
        metrics={
            "scene_subject_state_similarity": subject_metric,
            "physical_subject_similarity": {
                "score": 0.0,
                "components": {
                    "position": 0.0,
                    "shape": 0.0,
                    "appearance": 0.0,
                },
                "comparable_reference": (
                    "same_case_ground_truth_video"
                    if reference_mode == "same_case_reference"
                    else "conditioned_first_frame_appearance_only"
                ),
                "degraded": True,
            },
            "physical_subject_mask_iou": {
                "mean": (
                    float(np.mean(observed_values))
                    if observed_values
                    else 0.0
                ),
                "minimum": (
                    float(np.min(observed_values))
                    if observed_values
                    else 0.0
                ),
                "maximum": (
                    float(np.max(observed_values))
                    if observed_values
                    else 0.0
                ),
                "observed_frame_ratio": (
                    float(len(observed_values) / len(times_s))
                    if times_s
                    else 0.0
                ),
                "role": "conservative_zero_for_prediction_observation_failure",
            },
        },
        quality={
            "degraded": True,
            "degradation_codes": [code],
            "degradation_reason": reason,
            "prediction_observation_available": False,
            "subject_reference_observed_ratio": (
                float(np.mean(observed)) if observed else 0.0
            ),
            "subject_prediction_observed_ratio": 0.0,
        },
        artifacts={
            "per_frame_csv": str(csv_path),
            "subject_components_csv": str(subject_csv_path),
            "physical_subject_iou_curve": str(iou_path),
            "subject_similarity_curve": str(subject_curve_path),
        },
        provenance={
            "degradation": {
                "origin": "prediction",
                "code": code,
                "reason": reason,
                "policy": "conservative_zero_not_evaluator_failure",
            }
        },
    )


def prediction_failure_code(exc: Exception, *, stage: str) -> tuple[str, str]:
    if isinstance(exc, SceneAnalysisError):
        code = exc.code
    else:
        code = f"prediction_{stage}_failed"
    return code, f"{type(exc).__name__}: {exc}"


__all__ = [
    "add_subject_comparison",
    "degraded_prediction_analysis",
    "prediction_failure_code",
    "reference_failure",
    "robust_subject_enabled",
]
