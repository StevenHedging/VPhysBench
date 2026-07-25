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
from ...common.masks.motion import build_motion_prompt
from ...common.masks.quality import mask_iou
from ...common.masks.sam2 import Sam2VideoSegmenter
from ...common.tracking import extract_centroid_trace
from ...contracts import CaseEvaluationRequest
from .scoring import extract_free_fall_trace, score_free_fall


class FreeFallCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "free_fall_state"
    evaluator_version = "1.0"
    scene_id = "free_fall"
    primary_score = "free_fall_state_similarity"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": "falling_ball",
            "segmentation": self._segmenter.describe(),
            "state": [
                "vertical_trajectory",
                "normalized_acceleration",
                "impact_time",
                "horizontal_drift",
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
            frames, prompt=prompt, temporary_prefix="physbench_free_fall_"
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
        trace_arguments = {
            "minimum_area": int(quality["minimum_mask_pixels"]),
            "maximum_area_ratio": float(quality["maximum_mask_area_ratio"]),
            "minimum_valid_ratio": float(quality["minimum_valid_frame_ratio"]),
        }
        reference_centroid = extract_centroid_trace(
            reference_masks, **trace_arguments
        )
        prediction_centroid = extract_centroid_trace(
            prediction_masks, **trace_arguments
        )
        minimum_span = float(quality["minimum_vertical_span_px"])
        reference_trace = extract_free_fall_trace(
            reference_centroid,
            times_s,
            minimum_vertical_span_px=minimum_span,
        )
        prediction_trace = extract_free_fall_trace(
            prediction_centroid,
            times_s,
            minimum_vertical_span_px=minimum_span,
        )
        state_score = score_free_fall(
            reference_trace,
            prediction_trace,
            config=self.config["scoring"],
        )
        ious = [
            mask_iou(reference_masks[index], prediction_masks[index])
            for index in range(len(times_s))
        ]
        rows = [
            {
                "frame": index,
                "time_s": time_s,
                "physical_subject_iou": ious[index],
                "reference_x_px": reference_trace.xy[index, 0],
                "reference_y_px": reference_trace.xy[index, 1],
                "prediction_x_px": prediction_trace.xy[index, 0],
                "prediction_y_px": prediction_trace.xy[index, 1],
                "reference_vertical_normalized": (
                    reference_trace.normalized_vertical_displacement[index]
                ),
                "prediction_vertical_normalized": (
                    prediction_trace.normalized_vertical_displacement[index]
                ),
                "reference_tracking_valid": bool(reference_trace.valid[index]),
                "prediction_tracking_valid": bool(prediction_trace.valid[index]),
            }
            for index, time_s in enumerate(times_s)
        ]
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        trajectory_path = request.artifact_dir / "vertical_trajectory_curve.png"
        write_rows_csv(csv_path, rows)
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=ious,
            case_id=request.case["case_id"],
            scene_name="Free fall",
        )
        save_series_comparison(
            trajectory_path,
            times_s=times_s,
            reference=reference_trace.normalized_vertical_displacement,
            prediction=prediction_trace.normalized_vertical_displacement,
            ylabel="Normalized downward displacement",
            title=f"Free-fall vertical trajectory — {request.case['case_id']}",
        )
        initial_height = (
            request.case.get("physics", {})
            .get("initial_height", {})
            .get("value")
        )
        reference_acceleration_m_s2 = None
        if initial_height and reference_trace.vertical_span_px > 0:
            pixels_per_meter = reference_trace.vertical_span_px / float(
                initial_height
            )
            reference_acceleration_m_s2 = (
                reference_trace.acceleration_px_s2 / pixels_per_meter
            )
        return SceneAnalysis(
            score=state_score["score"],
            metrics={
                "free_fall_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    "mean": float(np.mean(ious)),
                    "minimum": float(np.min(ious)),
                    "maximum": float(np.max(ious)),
                    "role": "diagnostic_not_primary_score",
                },
                "reference_physics_diagnostic": {
                    "initial_height_m": initial_height,
                    "fitted_acceleration_m_s2": reference_acceleration_m_s2,
                    "role": "reference_calibration_diagnostic",
                },
            },
            quality={
                "reference_valid_mask_ratio": reference_trace.valid_ratio,
                "prediction_valid_mask_ratio": prediction_trace.valid_ratio,
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "vertical_trajectory_curve": str(trajectory_path),
            },
            provenance={
                "segmentation": {
                    "reference": reference_segmentation,
                    "prediction": prediction_segmentation,
                }
            },
        )
