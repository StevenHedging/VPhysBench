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
from ...common.masks.quality import observed_mask_iou, summarize_mask_ious
from ...common.masks.sam2 import Sam2VideoSegmenter
from ...common.robustness import (
    add_subject_comparison,
    degraded_prediction_analysis,
    prediction_failure_code,
    reference_failure,
    robust_subject_enabled,
)
from ...common.tracking import extract_centroid_trace
from ...contracts import CaseEvaluationRequest
from ....datasets.physics import flat_physics_quantities
from .observation import (
    build_collision_prompts,
    build_multiframe_collision_prompts,
    stabilize_collision_instance_masks,
)
from .scoring import extract_collision_trace, score_collision
from .visualization import write_collision_visualization


class CollisionCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "collision_1d_state"
    evaluator_version = "1.1"
    scene_id = "collision_1d"
    primary_score = "collision_1d_state_similarity"

    def __init__(self, config: dict[str, Any]):
        self.robust_evaluator_version = (
            "1.4"
            if config.get("type") == "collision_1d_state_v3"
            else "1.3"
        )
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": "three_ball_instances",
            "segmentation": self._segmenter.describe(),
            "prompting": (
                "multiframe_circle_motion_proposals"
                if "multi_frame_observation" in self.config
                else "frame_zero_color_components"
            ),
            "propagation": (
                "shared_seed_forward_and_reverse"
                if "multi_frame_observation" in self.config
                else "frame_zero_forward"
            ),
            "instance_overlap": (
                "highest_positive_sam2_logit_wins"
                if "multi_frame_observation" in self.config
                else "independent_binary_masks"
            ),
            "coordinate_system": "fitted_track_axis",
            "state": [
                "instance_trajectories",
                "contact_event",
                "pre_post_velocities",
                "momentum_residual",
                "effective_restitution",
            ],
        }

    def _observe(
        self,
        frames,
        *,
        minimum_valid_ratio: float | None = None,
    ):
        multi_frame = self.config.get("multi_frame_observation")
        if multi_frame is not None:
            prompts, prompt_metadata = build_multiframe_collision_prompts(
                frames, config=multi_frame
            )
        else:
            prompts, prompt_metadata = build_collision_prompts(
                frames[0], config=self.config["frame_zero_observation"]
            )
        masks, segmentation = self._segmenter.segment_instances(
            frames,
            prompts=prompts,
            temporary_prefix="physbench_collision_",
            exclusive_masks=multi_frame is not None,
        )
        quality = self.config["quality"]
        stabilization: dict[str, Any] | None = None
        if multi_frame is not None:
            masks, stabilization = stabilize_collision_instance_masks(
                masks,
                prompts,
                config=quality,
            )
        required_ratio = (
            float(minimum_valid_ratio)
            if minimum_valid_ratio is not None
            else float(quality["minimum_valid_frame_ratio"])
        )
        centroid_traces = [
            extract_centroid_trace(
                instance,
                minimum_area=int(quality["minimum_mask_pixels"]),
                maximum_area_ratio=float(quality["maximum_mask_area_ratio"]),
                minimum_valid_ratio=required_ratio,
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
            **(
                {"mask_stabilization": stabilization}
                if stabilization is not None
                else {}
            ),
        }

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        robust = robust_subject_enabled(self.config)
        physics = flat_physics_quantities(request.case)
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
        try:
            (
                reference_xy,
                reference_valid,
                reference_instances,
                reference_union,
                reference_observation,
            ) = self._observe(
                reference_video.frames,
                minimum_valid_ratio=quality.get(
                    "minimum_reference_valid_frame_ratio"
                ),
            )
            reference_trace = extract_collision_trace(
                reference_xy, reference_valid, **trace_arguments
            )
        except SceneAnalysisError as exc:
            if robust:
                raise reference_failure(
                    exc, stage="collision observation"
                ) from exc
            raise
        try:
            (
                prediction_xy,
                prediction_valid,
                prediction_instances,
                prediction_union,
                prediction_observation,
            ) = self._observe(
                prediction_video.frames,
                minimum_valid_ratio=quality.get(
                    "minimum_prediction_valid_frame_ratio"
                ),
            )
            prediction_trace = extract_collision_trace(
                prediction_xy, prediction_valid, **trace_arguments
            )
            state_score = score_collision(
                reference_trace,
                prediction_trace,
                config=self.config["scoring"],
            )
            if "multi_frame_observation" in self.config:
                coverage_relative_to_reference = np.clip(
                    prediction_trace.valid_ratio
                    / np.maximum(reference_trace.valid_ratio, 1e-6),
                    0.0,
                    1.0,
                )
                observation_reliability = float(
                    np.min(coverage_relative_to_reference)
                )
                state_score["raw_score_before_observation_reliability"] = (
                    state_score["score"]
                )
                state_score["observation_reliability"] = (
                    observation_reliability
                )
                state_score["per_role_coverage_relative_to_reference"] = (
                    coverage_relative_to_reference.tolist()
                )
                state_score["score"] = float(
                    state_score["score"] * observation_reliability
                )
        except SceneAnalysisError as exc:
            if robust:
                code, reason = prediction_failure_code(
                    exc, stage="collision_observation"
                )
                analysis = degraded_prediction_analysis(
                    request,
                    times_s=times_s,
                    reference_masks=reference_union,
                    code=code,
                    reason=reason,
                    scene_name="One-dimensional collision",
                )
                visualization_config = self.config.get("visualization")
                if visualization_config is not None:
                    height, width = reference_union[0].shape
                    empty_instances = [
                        [
                            np.zeros((height, width), dtype=np.uint8)
                            for _ in times_s
                        ]
                        for _ in range(3)
                    ]
                    empty_union = [
                        np.zeros((height, width), dtype=np.uint8)
                        for _ in times_s
                    ]
                    union_ious = [
                        (
                            0.0
                            if np.count_nonzero(mask) > 0
                            else None
                        )
                        for mask in reference_union
                    ]
                    instance_ious = [
                        [
                            (
                                0.0
                                if np.count_nonzero(
                                    reference_instances[object_index][
                                        frame_index
                                    ]
                                )
                                > 0
                                else None
                            )
                            for object_index in range(3)
                        ]
                        for frame_index in range(len(times_s))
                    ]
                    degraded_rows = [
                        {
                            "frame": frame_index,
                            "time_s": time_s,
                            "physical_subject_iou": union_ious[frame_index],
                            "degradation_code": code,
                            **{
                                (
                                    f"reference_s_{object_index + 1}"
                                    "_normalized"
                                ): reference_trace.normalized_position[
                                    frame_index, object_index
                                ]
                                for object_index in range(3)
                            },
                        }
                        for frame_index, time_s in enumerate(times_s)
                    ]
                    analysis.artifacts.update(
                        write_collision_visualization(
                            request,
                            config=visualization_config,
                            times_s=times_s,
                            reference_frames=reference_video.frames,
                            prediction_frames=prediction_video.frames,
                            reference_masks=reference_instances,
                            prediction_masks=empty_instances,
                            reference_union=reference_union,
                            prediction_union=empty_union,
                            reference_xy=reference_xy,
                            prediction_xy=np.full_like(
                                reference_xy, np.nan
                            ),
                            reference_normalized=(
                                reference_trace.normalized_position
                            ),
                            prediction_normalized=np.zeros_like(
                                reference_trace.normalized_position
                            ),
                            union_ious=union_ious,
                            instance_ious=instance_ious,
                            rows=degraded_rows,
                            reference_observation=reference_observation,
                            prediction_observation={
                                "prompt_builder": {
                                    "seed_source": (
                                        "unavailable_prediction_observation"
                                    )
                                },
                                "error": {"code": code, "reason": reason},
                            },
                            reference_event_frame=(
                                reference_trace.event_frame
                            ),
                            prediction_event_frame=None,
                        )
                    )
                return analysis
            raise
        union_ious = [
            observed_mask_iou(
                reference_union[index], prediction_union[index]
            )
            for index in range(len(times_s))
        ]
        instance_ious = [
            [
                observed_mask_iou(
                    reference_instances[object_index][frame_index],
                    prediction_instances[object_index][frame_index],
                )
                for object_index in range(3)
            ]
            for frame_index in range(len(times_s))
        ]
        union_iou_summary = summarize_mask_ious(union_ious)
        per_instance_summaries = [
            summarize_mask_ious(
                [
                    instance_ious[frame_index][object_index]
                    for frame_index in range(len(times_s))
                ]
            )
            for object_index in range(3)
        ]
        all_observed_instance_ious = [
            value
            for frame_values in instance_ious
            for value in frame_values
            if value is not None
        ]
        rows = []
        for frame_index, time_s in enumerate(times_s):
            row = {
                "frame": frame_index,
                "time_s": time_s,
                "physical_subject_iou": union_ious[frame_index],
                "mean_instance_iou": (
                    float(
                        np.mean(
                            [
                                value
                                for value in instance_ious[frame_index]
                                if value is not None
                            ]
                        )
                    )
                    if any(
                        value is not None
                        for value in instance_ious[frame_index]
                    )
                    else None
                ),
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
        visualization_artifacts: dict[str, Any] = {}
        visualization_config = self.config.get("visualization")
        if visualization_config is not None:
            visualization_artifacts = write_collision_visualization(
                request,
                config=visualization_config,
                times_s=times_s,
                reference_frames=reference_video.frames,
                prediction_frames=prediction_video.frames,
                reference_masks=reference_instances,
                prediction_masks=prediction_instances,
                reference_union=reference_union,
                prediction_union=prediction_union,
                reference_xy=reference_xy,
                prediction_xy=prediction_xy,
                reference_normalized=reference_trace.normalized_position,
                prediction_normalized=prediction_trace.normalized_position,
                union_ious=union_ious,
                instance_ious=instance_ious,
                rows=rows,
                reference_observation=reference_observation,
                prediction_observation=prediction_observation,
                reference_event_frame=reference_trace.event_frame,
                prediction_event_frame=prediction_trace.event_frame,
            )
        analysis = SceneAnalysis(
            score=state_score["score"],
            metrics={
                "collision_1d_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    **union_iou_summary,
                    "role": "diagnostic_not_primary_score",
                },
                "matched_instance_mask_iou": {
                    "mean": float(np.mean(all_observed_instance_ious)),
                    "per_instance_mean": [
                        summary["mean"]
                        for summary in per_instance_summaries
                    ],
                    "per_instance_observed_frame_ratio": [
                        summary["observed_frame_ratio"]
                        for summary in per_instance_summaries
                    ],
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
                **(
                    {
                        "observation_reliability": state_score[
                            "observation_reliability"
                        ]
                    }
                    if "observation_reliability" in state_score
                    else {}
                ),
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "striker_trajectory_curve": str(trajectory_path),
                **visualization_artifacts,
            },
            provenance={
                "observation": {
                    "reference": reference_observation,
                    "prediction": prediction_observation,
                }
            },
        )
        if robust:
            return add_subject_comparison(
                analysis,
                request,
                times_s=times_s,
                reference_frames=reference_video.frames,
                prediction_frames=prediction_video.frames,
                reference_masks=reference_union,
                prediction_masks=prediction_union,
                scene_name="One-dimensional collision",
            )
        return analysis
