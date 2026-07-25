from __future__ import annotations

from typing import Any

import numpy as np

from ...common.artifacts import (
    save_iou_curve,
    save_series_comparison,
    write_rows_csv,
)
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.errors import SceneAnalysisError
from ...common.geometry import rectify_axis_masks
from ...common.masks.motion import build_motion_prompt
from ...common.masks.quality import mask_iou
from ...common.masks.sam2 import Sam2VideoSegmenter
from ...common.tracking import extract_centroid_trace
from ...contracts import CaseEvaluationRequest
from .scoring import extract_incline_trace, score_incline


class InclinedPlaneCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "inclined_plane_state"
    evaluator_version = "1.0"
    scene_id = "inclined_plane_slide"
    primary_score = "inclined_plane_state_similarity"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": "sliding_block",
            "segmentation": self._segmenter.describe(),
            "coordinate_system": "independently_fitted_plane_axis",
            "state": [
                "along_plane_trajectory",
                "normalized_acceleration",
                "descent_time",
                "cross_track_contact",
                "pose_stability",
            ],
        }

    def _segment(
        self, frames: list[np.ndarray], *, fallback_prompt=None
    ) -> tuple[list[np.ndarray], dict[str, Any], Any]:
        proposal = self.config["motion_proposal"]
        try:
            prompt = build_motion_prompt(
                frames,
                threshold=float(proposal["threshold"]),
                minimum_area=int(proposal["minimum_area"]),
                box_expand=float(proposal["box_expand"]),
                minimum_box_side=int(proposal["minimum_box_side"]),
            )
            source = "independent_motion_proposal"
        except SceneAnalysisError:
            if fallback_prompt is None:
                raise
            prompt = fallback_prompt
            source = "reference_geometry_fallback"
        masks, metadata = self._segmenter.segment(
            frames, prompt=prompt, temporary_prefix="physbench_incline_"
        )
        metadata["prompt_source"] = source
        return masks, metadata, prompt

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        reference_masks, reference_segmentation, reference_prompt = self._segment(
            reference_video.frames
        )
        prediction_masks, prediction_segmentation, _ = self._segment(
            prediction_video.frames, fallback_prompt=reference_prompt
        )
        quality = self.config["quality"]
        centroid_arguments = {
            "minimum_area": int(quality["minimum_mask_pixels"]),
            "maximum_area_ratio": float(quality["maximum_mask_area_ratio"]),
            "minimum_valid_ratio": float(quality["minimum_valid_frame_ratio"]),
        }
        reference_centroid = extract_centroid_trace(
            reference_masks, **centroid_arguments
        )
        prediction_centroid = extract_centroid_trace(
            prediction_masks, **centroid_arguments
        )
        trace_arguments = {
            "times_s": times_s,
            "minimum_span_px": float(quality["minimum_motion_span_px"]),
        }
        reference_trace = extract_incline_trace(
            reference_centroid, reference_masks, **trace_arguments
        )
        prediction_trace = extract_incline_trace(
            prediction_centroid, prediction_masks, **trace_arguments
        )
        state_score = score_incline(
            reference_trace,
            prediction_trace,
            config=self.config["scoring"],
        )
        original_ious = [
            mask_iou(reference_masks[index], prediction_masks[index])
            for index in range(len(times_s))
        ]
        reference_rectified = rectify_axis_masks(
            reference_masks,
            axis=reference_trace.axis,
            span_px=reference_trace.span_px,
        )
        prediction_rectified = rectify_axis_masks(
            prediction_masks,
            axis=prediction_trace.axis,
            span_px=prediction_trace.span_px,
        )
        rectified_ious = [
            mask_iou(reference_rectified[index], prediction_rectified[index])
            for index in range(len(times_s))
        ]
        rows = [
            {
                "frame": index,
                "time_s": time_s,
                "physical_subject_iou": original_ious[index],
                "plane_rectified_iou": rectified_ious[index],
                "reference_along_normalized": (
                    reference_trace.normalized_along_displacement[index]
                ),
                "prediction_along_normalized": (
                    prediction_trace.normalized_along_displacement[index]
                ),
                "reference_cross_px": (
                    reference_trace.cross_displacement_px[index]
                ),
                "prediction_cross_px": (
                    prediction_trace.cross_displacement_px[index]
                ),
                "reference_tracking_valid": bool(reference_trace.valid[index]),
                "prediction_tracking_valid": bool(prediction_trace.valid[index]),
            }
            for index, time_s in enumerate(times_s)
        ]
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        trajectory_path = request.artifact_dir / "along_plane_trajectory_curve.png"
        write_rows_csv(csv_path, rows)
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=original_ious,
            case_id=request.case["case_id"],
            scene_name="Inclined-plane slide",
        )
        save_series_comparison(
            trajectory_path,
            times_s=times_s,
            reference=reference_trace.normalized_along_displacement,
            prediction=prediction_trace.normalized_along_displacement,
            ylabel="Normalized along-plane displacement",
            title=f"Inclined-plane trajectory — {request.case['case_id']}",
        )
        physics = request.case.get("physics", {})
        theoretical = physics.get("theoretical_acceleration", {}).get("value")
        calibration = physics.get("calibration_length", {}).get("value")
        reference_acceleration_m_s2 = None
        if calibration and reference_trace.span_px > 0:
            reference_acceleration_m_s2 = (
                reference_trace.acceleration_px_s2
                / (reference_trace.span_px / float(calibration))
            )
        return SceneAnalysis(
            score=state_score["score"],
            metrics={
                "inclined_plane_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    "mean": float(np.mean(original_ious)),
                    "minimum": float(np.min(original_ious)),
                    "maximum": float(np.max(original_ious)),
                    "role": "diagnostic_not_primary_score",
                },
                "plane_rectified_mask_iou": {
                    "mean": float(np.mean(rectified_ious)),
                    "minimum": float(np.min(rectified_ious)),
                    "maximum": float(np.max(rectified_ious)),
                    "role": "viewpoint_normalized_diagnostic",
                },
                "reference_physics_diagnostic": {
                    "theoretical_acceleration_m_s2": theoretical,
                    "fitted_acceleration_m_s2": reference_acceleration_m_s2,
                    "initial_velocity_treatment": "fitted_not_trusted_from_label",
                },
            },
            quality={
                "reference_valid_mask_ratio": reference_trace.valid_ratio,
                "prediction_valid_mask_ratio": prediction_trace.valid_ratio,
                "reference_axis_explained_ratio": (
                    reference_trace.axis.explained_ratio
                ),
                "prediction_axis_explained_ratio": (
                    prediction_trace.axis.explained_ratio
                ),
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "along_plane_trajectory_curve": str(trajectory_path),
            },
            provenance={
                "segmentation": {
                    "reference": reference_segmentation,
                    "prediction": prediction_segmentation,
                },
                "rectification": {
                    "type": "independent_motion_axis_to_canonical_strip",
                    "reference_axis_direction_xy": (
                        reference_trace.axis.direction_xy.tolist()
                    ),
                    "prediction_axis_direction_xy": (
                        prediction_trace.axis.direction_xy.tolist()
                    ),
                },
            },
        )
