from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...common.artifacts import (
    save_iou_curve,
    save_series_comparison,
    write_rows_csv,
)
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.errors import SceneAnalysisError
from ...common.masks.quality import mask_iou
from ...common.masks.sam2 import Sam2VideoSegmenter
from ...common.tracking import extract_centroid_trace
from ...contracts import CaseEvaluationRequest
from .observation import build_collision_prompts
from .scoring import extract_collision_trace, score_collision


class CollisionCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "collision_1d_state"
    evaluator_version = "1.0"
    scene_id = "collision_1d"
    primary_score = "collision_1d_state_similarity"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": "three_ball_instances",
            "segmentation": self._segmenter.describe(),
            "coordinate_system": "fitted_track_axis",
            "state": [
                "instance_trajectories",
                "contact_event",
                "pre_post_velocities",
                "momentum_residual",
                "effective_restitution",
            ],
        }

    def _observe(self, frames):
        prompts, prompt_metadata = build_collision_prompts(
            frames[0], config=self.config["frame_zero_observation"]
        )
        masks, segmentation = self._segmenter.segment_instances(
            frames,
            prompts=prompts,
            temporary_prefix="physbench_collision_",
        )
        quality = self.config["quality"]
        centroid_traces = [
            extract_centroid_trace(
                instance,
                minimum_area=int(quality["minimum_mask_pixels"]),
                maximum_area_ratio=float(quality["maximum_mask_area_ratio"]),
                minimum_valid_ratio=float(quality["minimum_valid_frame_ratio"]),
            )
            for instance in masks
        ]
        xy = np.stack([trace.xy for trace in centroid_traces], axis=1)
        valid = np.stack([trace.valid for trace in centroid_traces], axis=1)
        union_masks = []
        for frame_index in range(len(frames)):
            union = np.zeros_like(masks[0][frame_index])
            for object_index in range(len(masks)):
                union = cv2.bitwise_or(
                    union, masks[object_index][frame_index]
                )
            union_masks.append(union)
        return xy, valid, masks, union_masks, {
            "prompt_builder": prompt_metadata,
            "segmentation": segmentation,
        }

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        (
            reference_xy,
            reference_valid,
            reference_instances,
            reference_union,
            reference_observation,
        ) = self._observe(reference_video.frames)
        (
            prediction_xy,
            prediction_valid,
            prediction_instances,
            prediction_union,
            prediction_observation,
        ) = self._observe(prediction_video.frames)
        physics = request.case.get("physics", {})
        masses = np.asarray(
            [
                float(physics[f"ball_{index}_mass"]["value"])
                for index in range(1, 4)
            ],
            dtype=np.float64,
        )
        quality = self.config["quality"]
        trace_arguments = {
            "times_s": times_s,
            "masses_kg": masses,
            "minimum_span_px": float(quality["minimum_motion_span_px"]),
            "velocity_window_fraction": float(
                quality["velocity_window_fraction"]
            ),
        }
        reference_trace = extract_collision_trace(
            reference_xy, reference_valid, **trace_arguments
        )
        prediction_trace = extract_collision_trace(
            prediction_xy, prediction_valid, **trace_arguments
        )
        state_score = score_collision(
            reference_trace,
            prediction_trace,
            config=self.config["scoring"],
        )
        union_ious = [
            mask_iou(reference_union[index], prediction_union[index])
            for index in range(len(times_s))
        ]
        instance_ious = [
            [
                mask_iou(
                    reference_instances[object_index][frame_index],
                    prediction_instances[object_index][frame_index],
                )
                for object_index in range(3)
            ]
            for frame_index in range(len(times_s))
        ]
        rows = []
        for frame_index, time_s in enumerate(times_s):
            row = {
                "frame": frame_index,
                "time_s": time_s,
                "physical_subject_iou": union_ious[frame_index],
                "mean_instance_iou": float(np.mean(instance_ious[frame_index])),
            }
            for object_index in range(3):
                row[f"instance_{object_index + 1}_iou"] = instance_ious[
                    frame_index
                ][object_index]
                row[f"reference_s_{object_index + 1}_normalized"] = (
                    reference_trace.normalized_position[
                        frame_index, object_index
                    ]
                )
                row[f"prediction_s_{object_index + 1}_normalized"] = (
                    prediction_trace.normalized_position[
                        frame_index, object_index
                    ]
                )
            rows.append(row)
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        trajectory_path = request.artifact_dir / "striker_trajectory_curve.png"
        write_rows_csv(csv_path, rows)
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=union_ious,
            case_id=request.case["case_id"],
            scene_name="One-dimensional collision",
        )
        save_series_comparison(
            trajectory_path,
            times_s=times_s,
            reference=reference_trace.normalized_position[:, 0],
            prediction=prediction_trace.normalized_position[:, 0],
            ylabel="Normalized striker track position",
            title=f"Collision striker trajectory — {request.case['case_id']}",
        )
        return SceneAnalysis(
            score=state_score["score"],
            metrics={
                "collision_1d_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    "mean": float(np.mean(union_ious)),
                    "minimum": float(np.min(union_ious)),
                    "maximum": float(np.max(union_ious)),
                    "role": "diagnostic_not_primary_score",
                },
                "matched_instance_mask_iou": {
                    "mean": float(np.mean(instance_ious)),
                    "per_instance_mean": np.mean(
                        np.asarray(instance_ious), axis=0
                    ).tolist(),
                    "role": "identity_preservation_diagnostic",
                },
                "annotation_policy": {
                    "masses_kg": masses.tolist(),
                    "restitution_source": (
                        "estimated_from_reference_pre_post_velocities"
                    ),
                    "material_based_restitution_assumption": False,
                },
            },
            quality={
                "reference_valid_track_ratios": (
                    reference_trace.valid_ratio.tolist()
                ),
                "prediction_valid_track_ratios": (
                    prediction_trace.valid_ratio.tolist()
                ),
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
                "striker_trajectory_curve": str(trajectory_path),
            },
            provenance={
                "observation": {
                    "reference": reference_observation,
                    "prediction": prediction_observation,
                }
            },
        )
