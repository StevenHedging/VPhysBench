from __future__ import annotations

from typing import Any
import math

import numpy as np

from ...common.artifacts import (
    save_iou_curve,
    save_series_comparison,
    write_rows_csv,
)
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.errors import SceneAnalysisError
from ...common.geometry import rectify_circle_masks
from ...common.masks.quality import observed_mask_iou, summarize_mask_ious
from ...common.robustness import (
    add_subject_comparison,
    degraded_prediction_analysis,
    prediction_failure_code,
    reference_failure,
    robust_subject_enabled,
)
from ...common.tracking import extract_instance_tracks
from ...contracts import CaseEvaluationRequest
from ....datasets.physics import flat_physics_quantities
from .observation import green_disk_object_masks
from .scoring import extract_orbit_traces, score_orbits


class CircularMotionCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "uniform_circular_motion_state"
    evaluator_version = "1.1"
    scene_id = "uniform_circular_motion"
    primary_score = "uniform_circular_motion_state_similarity"

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": "moving_orbit_objects",
            "segmentation": {
                "backend": "green_disk_interior_color_contrast",
                "instance_policy": "continuity_assignment_then_radius_order",
            },
            "coordinate_system": "independently_fitted_orbit_center_and_radius",
            "state": [
                "relative_angular_trajectory",
                "angular_velocity",
                "orbit_circularity",
                "radius_configuration",
            ],
        }

    def _observe(self, frames, *, expected_count: int):
        proposal = self.config["color_observation"]
        masks, observation = green_disk_object_masks(
            frames, config=proposal
        )
        tracks = extract_instance_tracks(
            masks,
            expected_count=expected_count,
            minimum_component_area=int(proposal["minimum_component_area"]),
            maximum_component_area_ratio=float(
                proposal["maximum_component_area_ratio"]
            ),
            minimum_valid_ratio=float(proposal["minimum_valid_frame_ratio"]),
            maximum_candidates=int(proposal["maximum_candidates"]),
        )
        return tracks, {
            **observation,
            "expected_objects": expected_count,
            "valid_track_ratios": tracks.valid_ratio.tolist(),
        }

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        expected_count = int(
            request.case.get("appearance", {}).get("object_count", 1)
        )
        robust = robust_subject_enabled(self.config)
        try:
            reference_tracks, reference_observation = self._observe(
                reference_video.frames, expected_count=expected_count
            )
            reference_orbits = extract_orbit_traces(
                reference_tracks, times_s
            )
        except SceneAnalysisError as exc:
            if robust:
                raise reference_failure(
                    exc, stage="circular-motion observation"
                ) from exc
            raise
        try:
            prediction_tracks, prediction_observation = self._observe(
                prediction_video.frames, expected_count=expected_count
            )
            prediction_orbits = extract_orbit_traces(
                prediction_tracks, times_s
            )
            state_score = score_orbits(
                reference_orbits,
                prediction_orbits,
                config=self.config["scoring"],
            )
        except SceneAnalysisError as exc:
            if robust:
                code, reason = prediction_failure_code(
                    exc, stage="circular_motion_observation"
                )
                return degraded_prediction_analysis(
                    request,
                    times_s=times_s,
                    reference_masks=reference_tracks.union_masks,
                    code=code,
                    reason=reason,
                    scene_name="Uniform circular motion",
                )
            raise
        original_ious = [
            observed_mask_iou(
                reference_tracks.union_masks[index],
                prediction_tracks.union_masks[index],
            )
            for index in range(len(times_s))
        ]
        reference_outer = reference_orbits[-1].circle
        prediction_outer = prediction_orbits[-1].circle
        reference_rectified = rectify_circle_masks(
            reference_tracks.union_masks,
            center_xy=reference_outer.center_xy,
            radius_px=reference_outer.radius_px,
        )
        prediction_rectified = rectify_circle_masks(
            prediction_tracks.union_masks,
            center_xy=prediction_outer.center_xy,
            radius_px=prediction_outer.radius_px,
        )
        rectified_ious = [
            observed_mask_iou(
                reference_rectified[index], prediction_rectified[index]
            )
            for index in range(len(times_s))
        ]
        original_iou_summary = summarize_mask_ious(original_ious)
        rectified_iou_summary = summarize_mask_ious(rectified_ious)
        rows = []
        for frame_index, time_s in enumerate(times_s):
            row = {
                "frame": frame_index,
                "time_s": time_s,
                "physical_subject_iou": original_ious[frame_index],
                "orbit_rectified_iou": rectified_ious[frame_index],
            }
            for object_index in range(expected_count):
                row[f"reference_angle_{object_index + 1}_rad"] = (
                    reference_orbits[object_index].relative_angle_rad[frame_index]
                )
                row[f"prediction_angle_{object_index + 1}_rad"] = (
                    prediction_orbits[object_index].relative_angle_rad[frame_index]
                )
                row[f"reference_tracking_{object_index + 1}_valid"] = bool(
                    reference_orbits[object_index].valid[frame_index]
                )
                row[f"prediction_tracking_{object_index + 1}_valid"] = bool(
                    prediction_orbits[object_index].valid[frame_index]
                )
            rows.append(row)
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        angle_path = request.artifact_dir / "angular_trajectory_curve.png"
        write_rows_csv(csv_path, rows)
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=original_ious,
            case_id=request.case["case_id"],
            scene_name="Uniform circular motion",
        )
        save_series_comparison(
            angle_path,
            times_s=times_s,
            reference=reference_orbits[-1].relative_angle_rad,
            prediction=prediction_orbits[-1].relative_angle_rad,
            ylabel="Relative unwrapped angle (rad)",
            title=f"Circular-motion angular trajectory — {request.case['case_id']}",
        )
        omega_deg_s = flat_physics_quantities(request.case).get(
            "angular_velocity", {}
        ).get("value")
        annotated_omega = (
            math.radians(float(omega_deg_s))
            if omega_deg_s is not None
            else None
        )
        analysis = SceneAnalysis(
            score=state_score["score"],
            metrics={
                "uniform_circular_motion_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    **original_iou_summary,
                    "role": "diagnostic_not_primary_score",
                },
                "orbit_rectified_mask_iou": {
                    **rectified_iou_summary,
                    "role": "center_and_scale_normalized_diagnostic",
                },
                "annotation_diagnostic": {
                    "annotated_angular_velocity_rad_s": annotated_omega,
                    "initial_angle_treatment": "relative_to_first_observation",
                },
            },
            quality={
                "expected_object_count": expected_count,
                "reference_valid_track_ratios": (
                    reference_tracks.valid_ratio.tolist()
                ),
                "prediction_valid_track_ratios": (
                    prediction_tracks.valid_ratio.tolist()
                ),
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "angular_trajectory_curve": str(angle_path),
            },
            provenance={
                "observation": {
                    "reference": reference_observation,
                    "prediction": prediction_observation,
                },
                "rectification": {
                    "type": "independent_orbit_center_and_radius",
                },
            },
        )
        if robust:
            return add_subject_comparison(
                analysis,
                request,
                times_s=times_s,
                reference_frames=reference_video.frames,
                prediction_frames=prediction_video.frames,
                reference_masks=reference_tracks.union_masks,
                prediction_masks=prediction_tracks.union_masks,
                scene_name="Uniform circular motion",
            )
        return analysis
