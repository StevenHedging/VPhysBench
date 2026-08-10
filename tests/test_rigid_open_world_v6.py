from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.entities import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
    ReferenceCapability,
    build_common_time_grid,
)
from physbench.evaluation.scenes.rigid_body_open_world import (
    RigidBodyOpenWorldCaseEvaluatorBase,
    build_expected_rigid_body_timeline,
    build_condition_incline_apparatus_seed,
    build_rigid_body_motion_prompts,
    build_rigid_body_reference,
    classify_coupled_rigid_body_artifacts,
    coalesce_condition_directed_identity,
    default_rigid_body_config,
    discover_rigid_body_objects,
    evaluate_rigid_body_open_world,
    freeze_condition_incline_axis,
    merge_disconnected_directed_fragments,
    observation_from_mask_channels,
    reexpress_same_case_reference_on_condition_axis,
    recover_rigid_body_mask_gaps,
    safe_compare_rigid_body_open_world,
    select_rigid_body_reference_hypothesis,
    score_rigid_body_reference_hypothesis,
    write_rigid_body_audit_json,
)


_CSTI_CONFIG = CSTIConfig.from_mapping(
    {
        "enabled": True,
        "algorithm": "exact_full_tube_edt",
        "spatial_tolerance_fraction": 0.005,
        "temporal_tolerance_s": 0.05,
        "condition_frame_policy": "exclude_initial_samples",
        "initial_frames_excluded": 1,
        "score_aggregation": "full_tube",
        "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
        "case_aggregation": "mean_gt_entities",
        "timeline_policy": "physical_overlap",
        "mask_resolution": "scene_analysis_native",
    }
)


def _grid(frame_count: int, fps: float = 10.0):
    return build_common_time_grid(
        np.arange(frame_count, dtype=np.float64) / fps
    )


def _circle(x: float, y: float, *, shape=(80, 80), radius=4):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (round(x), round(y)), radius, 255, -1)
    return mask


def _block(
    x: float,
    y: float,
    *,
    shape=(80, 80),
    width=9,
    height=7,
):
    mask = np.zeros(shape, dtype=np.uint8)
    left = round(x - width / 2)
    top = round(y - height / 2)
    cv2.rectangle(
        mask,
        (left, top),
        (left + width, top + height),
        255,
        -1,
    )
    return mask


def _frames(masks, *, extras=None):
    output = []
    extras = extras or [np.zeros_like(masks[0]) for _ in masks]
    for mask, extra in zip(masks, extras):
        frame = np.zeros((*mask.shape, 3), dtype=np.uint8)
        frame[np.logical_or(mask > 0, extra > 0)] = (230, 230, 230)
        output.append(frame)
    return output


def _entity(entity_class: str):
    return SimpleNamespace(
        entity_id="subject",
        role_id="subject",
        entity_class=entity_class,
        parts=(),
        exchangeability_group=None,
        lifecycle="may_exit",
        condition_anchor={"source": "condition_frame"},
    )


def _reference(
    masks,
    *,
    scene_kind: str,
    entity_class: str,
):
    config = default_rigid_body_config(scene_kind)
    return build_rigid_body_reference(
        masks,
        entity_id="subject",
        entity_class=entity_class,
        scene_kind=scene_kind,
        frame_shape=masks[0].shape,
        minimum_area=8,
        maximum_area_ratio=0.1,
        minimum_span_px=6.0,
        config=config,
    )


def _timeline(reference, grid, entity_class):
    return build_expected_rigid_body_timeline(
        reference=reference,
        scoring_reference=reference,
        manifest_entity=_entity(entity_class),
        time_grid=grid,
        capability=ReferenceCapability.SAME_CASE_GT,
        condition_mask=reference.masks[0],
    )


def _compare(reference, observation, *, scene_kind: str):
    grid = _grid(len(reference.masks))
    timeline = _timeline(reference, grid, reference.entity_class)
    comparison, retained = safe_compare_rigid_body_open_world(
        expected_timeline=timeline,
        prediction_factory=lambda: observation,
        time_grid=grid,
        frame_shape=reference.masks[0].shape,
        minimum_match_position_similarity=0.1,
    )
    result = evaluate_rigid_body_open_world(
        reference=reference,
        scoring_reference=reference,
        observation=retained,
        comparison=comparison,
        time_grid=grid,
        frame_shape=reference.masks[0].shape,
        scene_kind=scene_kind,
        scoring_config=(
            {
                "trajectory_error_scale": 0.15,
                "acceleration_error_scale": 0.35,
                "impact_time_error_scale": 0.15,
                "horizontal_drift_scale": 0.08,
                "weights": {
                    "vertical_trajectory": 0.5,
                    "normalized_acceleration": 0.25,
                    "impact_time": 0.15,
                    "motion_constraints": 0.1,
                },
            }
            if scene_kind == "free_fall"
            else {
                "trajectory_error_scale": 0.18,
                "acceleration_error_scale": 0.4,
                "descent_time_error_scale": 0.18,
                "cross_track_scale": 0.05,
                "orientation_std_scale_deg": 12.0,
                "weights": {
                    "along_plane_trajectory": 0.5,
                    "normalized_acceleration": 0.25,
                    "descent_time": 0.15,
                    "contact_and_pose_constraints": 0.1,
                },
            }
        ),
        observer_config=default_rigid_body_config(scene_kind),
    )
    return comparison, result


def _visualization_inputs():
    count = 8
    masks = [_circle(30, 10 + 5 * index) for index in range(count)]
    reference = _reference(
        masks,
        scene_kind="free_fall",
        entity_class="ball",
    )
    grid = _grid(count)
    timeline = _timeline(reference, grid, "ball")
    observation = observation_from_mask_channels(
        directed_masks=masks,
        residual_instance_masks=[],
        entity_class="ball",
        time_grid=grid,
    )
    comparison, result = _compare(
        reference,
        observation,
        scene_kind="free_fall",
    )
    return {
        "times_s": grid.times_s.tolist(),
        "reference_frames": _frames(masks),
        "prediction_frames": _frames(masks),
        "expected_timeline": timeline,
        "observation": result.observation,
        "comparison": comparison,
        "reference_union_masks": reference.masks,
        "prediction_union_masks": result.prediction_union_masks,
        "full_subject_ious": [
            row["physical_subject_iou"] for row in result.per_frame
        ],
        "prediction_available": [True] * count,
        "reference_role": "REFERENCE",
        "score_summary": {"score": result.composition["score"]},
        "per_frame_diagnostics": [],
        "has_issues": False,
    }, result


class RigidBodyOpenWorldV6Tests(unittest.TestCase):
    def test_inclined_plane_csti_uses_aligned_subject_masks(self) -> None:
        count = 8
        masks = [_block(12 + 4 * index, 12 + 3 * index) for index in range(count)]
        reference = _reference(
            masks,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        observation = observation_from_mask_channels(
            directed_masks=masks,
            residual_instance_masks=[],
            entity_class="block",
            time_grid=_grid(count),
        )
        _, result = _compare(reference, observation, scene_kind="inclined_plane")
        evaluator = object.__new__(RigidBodyOpenWorldCaseEvaluatorBase)

        value = evaluator._build_csti_input(
            capability=ReferenceCapability.SAME_CASE_GT,
            times_s=_grid(count).times_s.tolist(),
            entity=_entity("block"),
            reference=reference,
            result=result,
        )
        metric = evaluate_csti(
            value,
            expected_entities=(("subject", "subject"),),
            config=_CSTI_CONFIG,
        )

        self.assertAlmostEqual(1.0, metric["score"], places=12)
        self.assertEqual(count, metric["frame_count"])

    def test_csti_does_not_extend_legal_exit_lifecycle(self) -> None:
        count = 12
        visible = [_block(20 + 6 * index, 20 + 3 * index) for index in range(8)]
        masks = visible + [np.zeros_like(visible[0]) for _ in range(count - len(visible))]
        reference = _reference(
            masks,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        observation = observation_from_mask_channels(
            directed_masks=masks,
            residual_instance_masks=[],
            entity_class="block",
            time_grid=_grid(count),
        )
        _, result = _compare(reference, observation, scene_kind="inclined_plane")
        evaluator = object.__new__(RigidBodyOpenWorldCaseEvaluatorBase)

        value = evaluator._build_csti_input(
            capability=ReferenceCapability.SAME_CASE_GT,
            times_s=_grid(count).times_s.tolist(),
            entity=_entity("block"),
            reference=reference,
            result=result,
        )

        self.assertEqual(count, len(value.entities[0].reference_masks))
        prediction = value.entities[0].prediction_masks
        assert prediction is not None
        self.assertEqual(count, len(prediction))
        self.assertTrue(all(not np.any(mask) for mask in prediction[8:]))

    @staticmethod
    def _strict_reference_config(scene_kind: str) -> dict:
        config = default_rigid_body_config(scene_kind)
        config.update(
            {
                "reference_hypothesis_minimum_raw_coverage": (
                    0.65 if scene_kind == "free_fall" else 0.55
                ),
                "reference_hypothesis_minimum_adjacent_steps": 3,
                "reference_hypothesis_minimum_direction_consistency": (
                    0.75 if scene_kind == "free_fall" else 0.60
                ),
                "reference_hypothesis_maximum_internal_gap_frames": (
                    1 if scene_kind == "free_fall" else 4
                ),
                "reference_hypothesis_minimum_area_stability": (
                    0.30 if scene_kind == "free_fall" else 0.25
                ),
                "reference_hypothesis_minimum_score": (
                    0.55 if scene_kind == "free_fall" else 0.50
                ),
                "reference_hypothesis_minimum_free_fall_downward_span_fraction": (
                    0.12 if scene_kind == "free_fall" else 0.0
                ),
                "exit_terminal_geometry_policy": (
                    "scene_terminal_geometry_v2"
                ),
                "exit_minimum_observed_tail_frames": 2,
            }
        )
        return config

    def test_v7_reference_candidates_reject_larger_noncompact_motion(
        self,
    ) -> None:
        frames = []
        for index in range(10):
            frame = np.zeros((120, 120, 3), dtype=np.uint8)
            # A large articulated distractor moves near the top.
            cv2.rectangle(
                frame,
                (12 + index, 8),
                (72 + index, 18),
                (160, 160, 160),
                -1,
            )
            # The physical ball is compact and moves down.
            cv2.circle(
                frame,
                (82, 22 + 7 * index),
                5,
                (230, 230, 230),
                -1,
            )
            frames.append(frame)
        config = default_rigid_body_config("free_fall")
        config.update(
            {
                "reference_candidate_minimum_compact_score": 0.12,
                "reference_candidate_maximum_area_ratio": 0.1,
            }
        )
        prompts = build_rigid_body_motion_prompts(
            frames,
            scene_kind="free_fall",
            threshold=8.0,
            minimum_area=8,
            box_expand=1.2,
            minimum_box_side=12,
            config=config,
            maximum_candidates=3,
        )
        self.assertTrue(prompts)
        self.assertLess(
            abs(float(prompts[0].points_xy[0, 0]) - 82.0),
            6.0,
        )

    def test_v7_reference_scale_consensus_rejects_tiny_continuous_decoy(
        self,
    ) -> None:
        candidates = [
            {
                "candidate_index": 0,
                "status": "accepted",
                "validation": {"score": 0.781},
                "scale_observation": {
                    "robust_log_area": math.log(3650.0),
                    "internal_consistency": 0.94,
                },
            },
            {
                "candidate_index": 1,
                "status": "accepted",
                "validation": {"score": 0.782},
                "scale_observation": {
                    "robust_log_area": math.log(3600.0),
                    "internal_consistency": 0.96,
                },
            },
            {
                "candidate_index": 2,
                "status": "accepted",
                "validation": {"score": 0.793},
                "scale_observation": {
                    "robust_log_area": math.log(425.0),
                    "internal_consistency": 0.90,
                },
            },
        ]
        selected, diagnostics = (
            select_rigid_body_reference_hypothesis(
                candidates,
                policy="validation_plus_consensus_scale_v2",
                config={},
            )
        )
        self.assertIn(selected, {0, 1})
        self.assertEqual(3, diagnostics["consensus_candidate_count"])
        rows = {
            row["candidate_index"]: row
            for row in diagnostics["candidate_selection"]
        }
        self.assertGreater(
            rows[selected]["adjusted_score"],
            rows[2]["adjusted_score"],
        )

    def test_v7_incline_condition_seed_and_axis_use_condition_pixels(
        self,
    ) -> None:
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.line(frame, (35, 205), (300, 65), (55, 55, 55), 9)
        cv2.rectangle(frame, (245, 37), (292, 70), (120, 160, 190), -1)
        seed, diagnostics = build_condition_incline_apparatus_seed(
            frame,
            minimum_area=20,
            maximum_area=12000,
            config={},
        )
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertFalse(
            diagnostics["condition_future_pixels_used"]
        )
        centroid = np.asarray(diagnostics["seed_center_xy"])
        axis = freeze_condition_incline_axis(
            frame,
            condition_centroid_xy=centroid,
            fallback=None,
            allow_reference_fallback=False,
        )
        self.assertEqual(
            "condition_apparatus_hough_axis",
            axis.source,
        )
        self.assertGreater(axis.direction_xy[1], 0.0)
        self.assertGreater(axis.span_px, 100.0)

    def test_v7_reference_hypothesis_prefers_continuous_body(
        self,
    ) -> None:
        masks = [_circle(30, 10 + 4 * index) for index in range(12)]
        reference = _reference(
            masks,
            scene_kind="free_fall",
            entity_class="ball",
        )
        score, diagnostics = score_rigid_body_reference_hypothesis(
            reference,
            frames=_frames(masks),
            scene_kind="free_fall",
            config=default_rigid_body_config("free_fall"),
        )
        self.assertTrue(diagnostics["accepted"])
        self.assertGreater(score, 0.85)
        self.assertEqual(1.0, diagnostics["continuity_score"])

    def test_v7_sparse_reference_hypothesis_is_rejected(self) -> None:
        masks = [
            (
                _circle(30, 10 + 3 * index)
                if index in {0, 9, 19}
                else np.zeros((80, 80), dtype=np.uint8)
            )
            for index in range(20)
        ]
        config = self._strict_reference_config("free_fall")
        reference = build_rigid_body_reference(
            masks,
            entity_id="subject",
            entity_class="ball",
            scene_kind="free_fall",
            frame_shape=masks[0].shape,
            minimum_area=8,
            maximum_area_ratio=0.1,
            minimum_span_px=6.0,
            config=config,
        )
        _, diagnostics = score_rigid_body_reference_hypothesis(
            reference,
            frames=_frames(masks),
            scene_kind="free_fall",
            config=config,
        )
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(0, diagnostics["adjacent_observation_steps"])
        self.assertLess(
            diagnostics["observed_ratio"],
            diagnostics["minimum_raw_coverage"],
        )

    def test_v7_reverse_free_fall_reference_is_rejected(self) -> None:
        masks = [_circle(30, 70 - 3 * index) for index in range(20)]
        config = self._strict_reference_config("free_fall")
        reference = build_rigid_body_reference(
            masks,
            entity_id="subject",
            entity_class="ball",
            scene_kind="free_fall",
            frame_shape=masks[0].shape,
            minimum_area=8,
            maximum_area_ratio=0.1,
            minimum_span_px=6.0,
            config=config,
        )
        _, diagnostics = score_rigid_body_reference_hypothesis(
            reference,
            frames=_frames(masks),
            scene_kind="free_fall",
            config=config,
        )
        self.assertFalse(diagnostics["accepted"])
        self.assertLess(diagnostics["total_displacement_px"], 0.0)
        self.assertGreater(
            diagnostics["minimum_free_fall_displacement_px"],
            0.0,
        )

    def test_v7_internal_reference_gap_is_rejected(self) -> None:
        masks = [_circle(30, 10 + 2 * index) for index in range(20)]
        for index in (7, 8, 9):
            masks[index] = np.zeros_like(masks[index])
        config = self._strict_reference_config("free_fall")
        reference = build_rigid_body_reference(
            masks,
            entity_id="subject",
            entity_class="ball",
            scene_kind="free_fall",
            frame_shape=masks[0].shape,
            minimum_area=8,
            maximum_area_ratio=0.1,
            minimum_span_px=6.0,
            config=config,
        )
        _, diagnostics = score_rigid_body_reference_hypothesis(
            reference,
            frames=_frames(masks),
            scene_kind="free_fall",
            config=config,
        )
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(3, diagnostics["internal_gap_frames"])
        self.assertEqual(1, diagnostics["maximum_internal_gap_frames"])

    def test_v7_high_speed_directed_samples_keep_one_frozen_id(
        self,
    ) -> None:
        tracks = []
        for index in range(8):
            mask = _circle(20, 8 + 9 * index)
            tracks.append(
                OpenWorldTrack(
                    track_id=f"generic_birth_{index}",
                    detections=(
                        ObjectDetection(
                            frame_index=index,
                            detection_id=f"directed_{index}",
                            xy=np.asarray([20.0, 8.0 + 9 * index]),
                            area_px2=float(np.count_nonzero(mask)),
                            entity_class="ball",
                            confidence=1.0,
                            evidence_tier=EvidenceTier.PARTICIPANT,
                            sources=("directed_sam2",),
                            mask=mask,
                            metadata={
                                "exclusive_tracking_partition": (
                                    "rigid_condition_directed"
                                ),
                                "identity_anchor_valid": True,
                            },
                        ),
                    ),
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                )
            )
        coalesced = coalesce_condition_directed_identity(
            OpenWorldObservation(
                tracks=tuple(tracks),
                overflow_counts=np.zeros(8, dtype=np.float64),
            ),
            config={
                "condition_directed_track_policy": (
                    "frozen_condition_identity_v2"
                )
            },
        )
        self.assertEqual(1, len(coalesced.tracks))
        self.assertEqual("condition_identity", coalesced.tracks[0].track_id)
        self.assertEqual(
            list(range(8)),
            [
                detection.frame_index
                for detection in coalesced.tracks[0].detections
            ],
        )

    def test_v7_coupled_shadow_is_not_a_body_but_compact_copy_is(
        self,
    ) -> None:
        primary_detections = []
        shadow_detections = []
        copy_detections = []
        for frame in range(8):
            primary_xy = np.asarray([30.0, 10.0 + 5.0 * frame])
            primary_detections.append(
                ObjectDetection(
                    frame_index=frame,
                    detection_id=f"primary_{frame}",
                    xy=primary_xy,
                    area_px2=50.0,
                    entity_class="ball",
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("directed_sam2",),
                    metadata={"identity_anchor_valid": True},
                )
            )
            if frame == 0:
                continue
            common = {
                "frame_index": frame,
                "area_px2": 45.0,
                "entity_class": "ball",
                "confidence": 0.9,
                "evidence_tier": EvidenceTier.PARTICIPANT,
            }
            shadow_sources = (
                ("condition_difference", "independent_compact_shape")
                if frame == 2
                else ("condition_difference", "temporal_motion")
            )
            shadow_detections.append(
                ObjectDetection(
                    detection_id=f"shadow_{frame}",
                    xy=primary_xy + np.asarray([1.0, 12.0]),
                    sources=shadow_sources,
                    metadata={"residual": True},
                    **common,
                )
            )
            copy_detections.append(
                ObjectDetection(
                    detection_id=f"copy_{frame}",
                    xy=primary_xy + np.asarray([18.0, 0.0]),
                    sources=(
                        "condition_difference",
                        "independent_compact_shape",
                    ),
                    metadata={"residual": True},
                    **common,
                )
            )
        observation = OpenWorldObservation(
            tracks=(
                OpenWorldTrack(
                    track_id="primary",
                    detections=tuple(primary_detections),
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
                OpenWorldTrack(
                    track_id="shadow",
                    detections=tuple(shadow_detections),
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
                OpenWorldTrack(
                    track_id="copy",
                    detections=tuple(copy_detections),
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
            ),
            overflow_counts=np.zeros(8, dtype=np.float64),
        )
        config = {
            "coupled_optical_artifact_policy": (
                "motion_correlation_with_compact_veto_v1"
            ),
            "coupled_artifact_minimum_overlap_frames": 4,
            "coupled_artifact_maximum_compact_support_ratio": 0.3,
            "coupled_artifact_maximum_offset_std_radii": 0.8,
            "coupled_artifact_maximum_velocity_error_radii": 0.75,
            "coupled_artifact_minimum_offset_radii": 0.6,
            "coupled_artifact_maximum_offset_radii": 6.0,
        }
        classified = classify_coupled_rigid_body_artifacts(
            observation,
            body_radius_px=4.0,
            config=config,
        )
        tiers = {
            track.track_id: track.evidence_tier
            for track in classified.tracks
        }
        # Kinematic coupling alone cannot erase a real duplicate: the shadow
        # hypothesis keeps a conservative tentative exposure.
        self.assertIs(EvidenceTier.TENTATIVE, tiers["shadow"])
        self.assertIs(EvidenceTier.PARTICIPANT, tiers["copy"])
        self.assertEqual(
            ["shadow"],
            [
                row["track_id"]
                for row in classified.diagnostics[
                    "coupled_optical_artifacts"
                ]
            ],
        )

    def test_v7_gap_recovery_requires_current_pixel_evidence(
        self,
    ) -> None:
        visible = [_circle(30, 12 + 5 * index) for index in range(7)]
        missing_mask = np.zeros_like(visible[0])
        directed = [*visible[:-1], missing_mask]
        frames_with_body = _frames(visible)
        frames_without_body = _frames(
            [*visible[:-1], missing_mask]
        )
        config = default_rigid_body_config("free_fall")
        config.update(
            {
                "directed_recovery_policy": (
                    "one_step_compact_or_change_v1"
                ),
                "directed_recovery_maximum_gap_frames": 1,
                "directed_recovery_maximum_distance_radii": 2.5,
                "directed_recovery_minimum_area_ratio": 0.35,
                "directed_recovery_maximum_area_ratio": 3.0,
                "directed_recovery_minimum_color_similarity": 0.1,
            }
        )
        recovered, diagnostics = recover_rigid_body_mask_gaps(
            directed,
            frames=frames_with_body,
            scene_kind="free_fall",
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=config,
        )
        self.assertGreater(np.count_nonzero(recovered[-1]), 0)
        self.assertEqual(
            [len(directed) - 1],
            [
                row["frame_index"]
                for row in diagnostics["recovered_frames"]
            ],
        )
        absent, absent_diagnostics = recover_rigid_body_mask_gaps(
            directed,
            frames=frames_without_body,
            scene_kind="free_fall",
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=config,
        )
        self.assertEqual(0, np.count_nonzero(absent[-1]))
        self.assertEqual([], absent_diagnostics["recovered_frames"])

    def test_local_audit_keeps_per_id_detection_tracks(self) -> None:
        values, result = _visualization_inputs()
        reference = _reference(
            values["reference_union_masks"],
            scene_kind="free_fall",
            entity_class="ball",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            write_rigid_body_audit_json(
                path,
                reference=reference,
                expected_timeline=values["expected_timeline"],
                result=result,
                prediction_failures=[],
            )
            audit = json.loads(path.read_text(encoding="utf-8"))
        tracks = audit["observation"]["tracks"]
        self.assertEqual(1, len(tracks))
        self.assertEqual("ball", tracks[0]["entity_class"])
        self.assertEqual(
            len(values["times_s"]),
            len(tracks[0]["detections"]),
        )

    def test_shared_evaluator_forwards_complete_open_world_artifacts(
        self,
    ) -> None:
        values, _ = _visualization_inputs()
        evaluator = SimpleNamespace(
            scene_name="Open-world free fall",
            config={
                "visualization": {
                    "enabled": True,
                    "namespace": "scene_default_v6",
                }
            },
        )
        request = SimpleNamespace()
        expected = {
            "open_world_v2_artifact_manifest": "manifest.json"
        }
        with patch(
            "physbench.evaluation.scenes.rigid_body_open_world."
            "write_open_world_v2_artifacts",
            return_value=expected,
        ) as writer:
            artifacts = (
                RigidBodyOpenWorldCaseEvaluatorBase
                ._write_open_world_visualization(
                    evaluator,
                    request,
                    **values,
                )
            )
        self.assertEqual(expected, artifacts)
        writer.assert_called_once()
        keyword = writer.call_args.kwargs
        self.assertIs(request, writer.call_args.args[0])
        self.assertEqual(
            "Open-world free fall",
            keyword["scene_name"],
        )
        self.assertEqual(
            [values["expected_timeline"]],
            keyword["expected_timelines"],
        )
        self.assertIs(
            values["observation"],
            keyword["prediction_observation"],
        )
        self.assertIs(values["comparison"], keyword["comparison"])
        self.assertIs(
            values["reference_union_masks"],
            keyword["reference_union_masks"],
        )
        self.assertIs(
            values["prediction_union_masks"],
            keyword["prediction_union_masks"],
        )
        self.assertEqual(
            values["full_subject_ious"],
            keyword["full_subject_ious"],
        )
        self.assertEqual(
            values["prediction_available"],
            keyword["prediction_available"],
        )
        self.assertEqual(values["reference_role"], keyword["reference_role"])
        self.assertEqual(values["score_summary"], keyword["score_summary"])
        self.assertEqual(
            values["per_frame_diagnostics"],
            keyword["per_frame_diagnostics"],
        )
        self.assertEqual(values["has_issues"], keyword["has_issues"])
        self.assertEqual(
            evaluator.config["visualization"],
            keyword["config"],
        )

    def test_visualization_failure_does_not_change_rigid_body_score(
        self,
    ) -> None:
        values, result = _visualization_inputs()
        evaluator = SimpleNamespace(
            scene_name="Open-world free fall",
            config={"visualization": {"enabled": True}},
        )
        score_before = result.composition["score"]
        with patch(
            "physbench.evaluation.scenes.rigid_body_open_world."
            "write_open_world_v2_artifacts",
            side_effect=RuntimeError("synthetic renderer failure"),
        ):
            artifacts = (
                RigidBodyOpenWorldCaseEvaluatorBase
                ._write_open_world_visualization(
                    evaluator,
                    SimpleNamespace(),
                    **values,
                )
            )
        self.assertEqual({}, artifacts)
        self.assertEqual(score_before, result.composition["score"])

    def test_second_body_and_static_extra_lower_integrity(self) -> None:
        count = 16
        expected = [_circle(30, 10 + 2 * index) for index in range(count)]
        extra = [_circle(45, 20) for _ in range(count)]
        reference = _reference(
            expected, scene_kind="free_fall", entity_class="ball"
        )
        grid = _grid(count)
        clean = observation_from_mask_channels(
            directed_masks=expected,
            residual_instance_masks=[],
            entity_class="ball",
            time_grid=grid,
        )
        duplicated = observation_from_mask_channels(
            directed_masks=expected,
            residual_instance_masks=[extra],
            entity_class="ball",
            time_grid=grid,
        )
        clean_comparison, _ = _compare(
            reference, clean, scene_kind="free_fall"
        )
        extra_comparison, _ = _compare(
            reference, duplicated, scene_kind="free_fall"
        )
        self.assertAlmostEqual(
            1.0, clean_comparison.integrity.integrity_gate
        )
        self.assertLess(
            extra_comparison.integrity.integrity_gate,
            clean_comparison.integrity.integrity_gate,
        )
        self.assertTrue(extra_comparison.per_frame[5]["extra_track_ids"])

    def test_moving_extra_is_discovered_from_pixels(self) -> None:
        count = 10
        directed = [_circle(28, 10 + 3 * index) for index in range(count)]
        extras = [_circle(42 + index, 24 + index) for index in range(count)]
        condition = _frames([directed[0]])[0]
        frames = _frames(directed, extras=extras)
        reference = _reference(
            directed, scene_kind="free_fall", entity_class="ball"
        )
        config = default_rigid_body_config("free_fall")
        observation = discover_rigid_body_objects(
            frames,
            directed_masks=directed,
            condition_frame=condition,
            condition_mask=directed[0],
            reference_axis=reference.axis,
            entity_class="ball",
            scene_kind="free_fall",
            time_grid=_grid(count),
            available=None,
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=config,
        )
        self.assertGreaterEqual(len(observation.tracks), 2)
        self.assertGreater(
            observation.diagnostics["source_counts"]["residual"], 0
        )

    def test_condition_only_top_edge_apparatus_is_diagnostic_only(
        self,
    ) -> None:
        count = 10
        directed = [_circle(24, 12 + 3 * index) for index in range(count)]
        top_edge_apparatus = [
            _block(58, 1, width=9, height=7) for _ in range(count)
        ]
        reference = _reference(
            directed, scene_kind="free_fall", entity_class="ball"
        )
        config = default_rigid_body_config("free_fall")
        # Keep this probe condition-difference-only: an apparatus-shaped
        # boundary change must not receive compact-body corroboration.
        config["minimum_duplicate_color_similarity"] = 1.01
        config["outside_roi_minimum_anchor_color_similarity"] = 1.01
        observation = discover_rigid_body_objects(
            _frames(directed, extras=top_edge_apparatus),
            directed_masks=directed,
            condition_frame=_frames([directed[0]])[0],
            condition_mask=directed[0],
            reference_axis=reference.axis,
            entity_class="ball",
            scene_kind="free_fall",
            time_grid=_grid(count),
            available=None,
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=config,
        )
        comparison, _ = _compare(
            reference, observation, scene_kind="free_fall"
        )
        formal = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        rejected = observation.diagnostics["rejected_candidates"]
        self.assertEqual(1, len(formal))
        self.assertAlmostEqual(1.0, comparison.integrity.integrity_gate)
        self.assertGreaterEqual(len(rejected), count)
        self.assertTrue(
            all(
                row["sources"] == ["condition_difference"]
                and "top" in row["touched_edges"]
                and row["formal_exposure_weight"] == 0.0
                for row in rejected
            )
        )

    def test_nearby_extra_without_mask_overlap_is_not_apparatus(
        self,
    ) -> None:
        primary = OpenWorldTrack(
            track_id="primary",
            detections=tuple(
                ObjectDetection(
                    frame_index=index,
                    detection_id=f"primary_{index}",
                    xy=np.asarray([20.0, 12.0 + 3 * index]),
                    area_px2=49.0,
                    entity_class="ball",
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("directed_sam2",),
                    mask=_circle(20, 12 + 3 * index),
                    metadata={"identity_anchor_valid": True},
                )
                for index in range(8)
            ),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        )
        apparatus_anchor = OpenWorldTrack(
            track_id="condition_apparatus",
            detections=(
                ObjectDetection(
                    frame_index=0,
                    detection_id="condition_apparatus_0",
                    xy=np.asarray([48.0, 40.0]),
                    area_px2=49.0,
                    entity_class="ball",
                    confidence=0.85,
                    evidence_tier=EvidenceTier.AMBIGUOUS,
                    sources=("independent_compact_shape",),
                    mask=_circle(48, 40),
                    metadata={
                        "suppress_persistence_only_promotion": True
                    },
                ),
            ),
            confirmed=False,
            evidence_tier=EvidenceTier.AMBIGUOUS,
        )
        nearby_extra = OpenWorldTrack(
            track_id="nearby_extra",
            detections=tuple(
                ObjectDetection(
                    frame_index=index,
                    detection_id=f"nearby_extra_{index}",
                    xy=np.asarray([58.0, 40.0]),
                    area_px2=49.0,
                    entity_class="ball",
                    confidence=0.85,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("independent_compact_shape",),
                    mask=_circle(58, 40),
                    metadata={},
                )
                for index in range(1, 8)
            ),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        )
        classified = classify_coupled_rigid_body_artifacts(
            OpenWorldObservation(
                tracks=(primary, apparatus_anchor, nearby_extra),
                overflow_counts=np.zeros(8, dtype=np.float64),
            ),
            body_radius_px=4.0,
            config={
                "coupled_optical_artifact_policy": (
                    "motion_correlation_with_compact_veto_v1"
                ),
                "coupled_artifact_minimum_overlap_frames": 4,
                "coupled_artifact_maximum_compact_support_ratio": 0.3,
                "coupled_artifact_maximum_offset_std_radii": 0.8,
                "coupled_artifact_maximum_velocity_error_radii": 0.75,
                "coupled_artifact_minimum_offset_radii": 0.6,
                "coupled_artifact_maximum_offset_radii": 6.0,
                "condition_present_apparatus_policy": (
                    "frozen_condition_track_v2"
                ),
                "condition_apparatus_maximum_fragment_span_radii": 3.0,
                "condition_apparatus_maximum_anchor_distance_radii": 3.0,
                "condition_apparatus_minimum_compact_support_ratio": 0.5,
                "condition_apparatus_minimum_mask_containment": 0.35,
            },
        )
        tiers = {
            track.track_id: track.evidence_tier
            for track in classified.tracks
        }
        self.assertIs(EvidenceTier.AMBIGUOUS, tiers["condition_apparatus"])
        self.assertIs(EvidenceTier.PARTICIPANT, tiers["nearby_extra"])
        self.assertEqual(
            [],
            classified.diagnostics["condition_apparatus_fragments"],
        )

    def test_edge_entering_extra_with_compact_support_is_penalized(
        self,
    ) -> None:
        count = 10
        directed = [_circle(24, 12 + 3 * index) for index in range(count)]
        edge_entering_extra = [
            _circle(58, 2 * index) for index in range(count)
        ]
        reference = _reference(
            directed, scene_kind="free_fall", entity_class="ball"
        )
        observation = discover_rigid_body_objects(
            _frames(directed, extras=edge_entering_extra),
            directed_masks=directed,
            condition_frame=_frames([directed[0]])[0],
            condition_mask=directed[0],
            reference_axis=reference.axis,
            entity_class="ball",
            scene_kind="free_fall",
            time_grid=_grid(count),
            available=None,
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=default_rigid_body_config("free_fall"),
        )
        comparison, _ = _compare(
            reference, observation, scene_kind="free_fall"
        )
        formal = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        source_counts = observation.diagnostics["source_counts"]
        self.assertGreaterEqual(len(formal), 2)
        self.assertGreater(
            source_counts["independent_compact_shape"]
            + source_counts["temporal_motion"],
            0,
        )
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(
            any(row["extra_track_ids"] for row in comparison.per_frame)
        )

    def test_static_duplicate_left_at_condition_position_is_discovered(
        self,
    ) -> None:
        count = 10
        directed = [_circle(28, 10 + 3 * index) for index in range(count)]
        duplicate = [_circle(28, 10) for _ in range(count)]
        condition = _frames([directed[0]])[0]
        frames = _frames(directed, extras=duplicate)
        reference = _reference(
            directed, scene_kind="free_fall", entity_class="ball"
        )
        observation = discover_rigid_body_objects(
            frames,
            directed_masks=directed,
            condition_frame=condition,
            condition_mask=directed[0],
            reference_axis=reference.axis,
            entity_class="ball",
            scene_kind="free_fall",
            time_grid=_grid(count),
            available=None,
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=default_rigid_body_config("free_fall"),
        )
        self.assertGreaterEqual(len(observation.tracks), 2)
        self.assertGreater(
            observation.diagnostics["source_counts"][
                "independent_compact_shape"
            ],
            0,
        )

    def test_v7_near_extra_in_prediction_frame_zero_is_not_apparatus(
        self,
    ) -> None:
        count = 12
        direction = np.asarray([3.0, 2.0], dtype=np.float64)
        normal = np.asarray([-direction[1], direction[0]])
        normal /= np.linalg.norm(normal)
        directed = []
        extras = []
        for index in range(count):
            center = np.asarray([18.0, 15.0]) + index * direction
            extra_center = center + 12.0 * normal
            directed.append(_block(*center))
            extras.append(_block(*extra_center))
        reference = _reference(
            directed,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        config = default_rigid_body_config("inclined_plane")
        config.update(
            {
                "exclusive_tracking_partition_policy": (
                    "condition_directed_vs_residual_v1"
                ),
                "condition_directed_track_policy": (
                    "frozen_condition_identity_v2"
                ),
                "compact_only_evidence_policy": (
                    "condition_frame_apparatus_only_v1"
                ),
                "condition_frame_apparatus_maximum_change_fraction": 0.25,
                "condition_present_apparatus_policy": (
                    "frozen_condition_track_v2"
                ),
                "coupled_optical_artifact_policy": (
                    "motion_correlation_with_compact_veto_v1"
                ),
                "residual_dilation_body_fraction": 2.0,
            }
        )
        observation = discover_rigid_body_objects(
            _frames(directed, extras=extras),
            directed_masks=directed,
            condition_frame=_frames([directed[0]])[0],
            condition_mask=directed[0],
            reference_axis=reference.axis,
            entity_class="block",
            scene_kind="inclined_plane",
            time_grid=_grid(count),
            available=None,
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=config,
        )
        formal = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        self.assertEqual(2, len(formal))
        self.assertEqual(
            [],
            observation.diagnostics[
                "condition_present_apparatus_tracks"
            ],
        )
        comparison, _ = _compare(
            reference,
            observation,
            scene_kind="inclined_plane",
        )
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(comparison.per_frame[0]["extra_track_ids"])

    def test_second_incline_block_is_formal_extra(self) -> None:
        count = 12
        expected = [
            _block(12 + 3 * index, 12 + 2 * index)
            for index in range(count)
        ]
        second = [
            _block(26 + 2 * index, 18 + 2 * index)
            for index in range(count)
        ]
        reference = _reference(
            expected,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        observation = observation_from_mask_channels(
            directed_masks=expected,
            residual_instance_masks=[second],
            entity_class="block",
            time_grid=_grid(count),
        )
        comparison, _ = _compare(
            reference, observation, scene_kind="inclined_plane"
        )
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(comparison.per_frame[4]["extra_track_ids"])

    def test_v7_small_directed_fragment_merges_but_second_block_does_not(
        self,
    ) -> None:
        shape = (100, 140)
        selected = _block(
            40,
            50,
            shape=shape,
            width=28,
            height=18,
        )
        # Mimic a SAM mask split by a narrow internal/disconnected seam.
        fragment = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(fragment, (56, 45), (61, 55), 255, -1)
        second = _block(
            90,
            50,
            shape=shape,
            width=28,
            height=18,
        )
        components = [
            (
                np.asarray([58.5, 50.0]),
                float(np.count_nonzero(fragment)),
                fragment,
            ),
            (
                np.asarray([90.0, 50.0]),
                float(np.count_nonzero(second)),
                second,
            ),
        ]
        merged, retained, diagnostics = (
            merge_disconnected_directed_fragments(
                selected,
                components,
                anchor_area_px2=float(np.count_nonzero(selected)),
                config={
                    "disconnected_directed_component_policy": (
                        "merge_same_body_fragments_v2"
                    ),
                    "directed_fragment_maximum_anchor_area_fraction": 0.30,
                    "directed_fragment_maximum_gap_body_radii": 0.30,
                    "directed_fragment_maximum_union_span_body_radii": 4.8,
                    "directed_fragment_maximum_union_anchor_area_ratio": 1.55,
                },
            )
        )
        self.assertEqual(1, len(diagnostics))
        self.assertEqual(1, len(retained))
        self.assertTrue(np.array_equal(retained[0][2], second))
        self.assertGreater(
            np.count_nonzero(merged),
            np.count_nonzero(selected),
        )

    def test_v7_same_case_state_keeps_real_cross_track_on_causal_axis(
        self,
    ) -> None:
        masks = [
            _block(15 + 4 * index, 18 + 3 * index)
            for index in range(10)
        ]
        reference = _reference(
            masks,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        direction = np.asarray([1.0, 0.5], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        causal_axis = type(reference.axis)(
            origin_xy=reference.xy[0],
            direction_xy=direction,
            normal_xy=np.asarray([-direction[1], direction[0]]),
            span_px=55.0,
            source="condition_apparatus_hough_axis",
            explained_ratio=1.0,
        )
        expressed = reexpress_same_case_reference_on_condition_axis(
            reference,
            condition_axis=causal_axis,
        )
        self.assertIs(causal_axis, expressed.axis)
        self.assertTrue(np.array_equal(reference.xy, expressed.xy))
        self.assertTrue(
            all(
                np.array_equal(first, second)
                for first, second in zip(
                    reference.masks,
                    expressed.masks,
                )
            )
        )
        _, cross = causal_axis.project(expressed.xy[expressed.expected])
        self.assertGreater(float(np.ptp(cross)), 0.0)
        along, _ = causal_axis.project(
            expressed.xy[expressed.expected]
        )
        np.testing.assert_allclose(
            expressed.normalized_progress[expressed.expected],
            (along - along[0]) / causal_axis.span_px,
        )

    def test_far_off_axis_incline_duplicate_cannot_hide_outside_scene_roi(
        self,
    ) -> None:
        count = 16
        shape = (480, 640)
        direction = np.asarray([14.0, 10.0], dtype=np.float64)
        normal = np.asarray([-direction[1], direction[0]])
        normal /= np.linalg.norm(normal)
        expected = []
        extras = []
        for index in range(count):
            center = np.asarray([220.0, 100.0]) + index * direction
            extra_center = center + 160.0 * normal
            expected.append(
                _block(
                    *center,
                    shape=shape,
                    width=20,
                    height=14,
                )
            )
            extras.append(
                _block(
                    *extra_center,
                    shape=shape,
                    width=20,
                    height=14,
                )
            )
        reference = _reference(
            expected,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        observation = discover_rigid_body_objects(
            _frames(expected, extras=extras),
            directed_masks=expected,
            condition_frame=_frames([expected[0]])[0],
            condition_mask=expected[0],
            reference_axis=reference.axis,
            entity_class="block",
            scene_kind="inclined_plane",
            time_grid=_grid(count),
            available=None,
            minimum_area=8,
            maximum_area_ratio=0.1,
            config=default_rigid_body_config("inclined_plane"),
        )
        formal = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        comparison, _ = _compare(
            reference,
            observation,
            scene_kind="inclined_plane",
        )
        self.assertGreaterEqual(len(formal), 2)
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(comparison.per_frame[5]["extra_track_ids"])

    def test_missing_10_25_50_percent_is_monotonic(self) -> None:
        count = 20
        expected = [_circle(30, 10 + 2 * index) for index in range(count)]
        reference = _reference(
            expected, scene_kind="free_fall", entity_class="ball"
        )
        scores = []
        gates = []
        for missing_count in (2, 5, 10):
            prediction = list(expected)
            for index in range(count - missing_count, count):
                prediction[index] = np.zeros_like(prediction[index])
            observation = observation_from_mask_channels(
                directed_masks=prediction,
                residual_instance_masks=[],
                entity_class="ball",
                time_grid=_grid(count),
            )
            comparison, result = _compare(
                reference, observation, scene_kind="free_fall"
            )
            gates.append(comparison.integrity.integrity_gate)
            scores.append(result.composition["score"])
        self.assertGreater(gates[0], gates[1])
        self.assertGreater(gates[1], gates[2])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_short_prediction_tail_is_missing_not_repeated(self) -> None:
        count = 16
        expected = [_circle(30, 10 + 2 * index) for index in range(count)]
        reference = _reference(
            expected, scene_kind="free_fall", entity_class="ball"
        )
        available = [True] * 10 + [False] * 6
        observation = observation_from_mask_channels(
            directed_masks=expected,
            residual_instance_masks=[],
            entity_class="ball",
            time_grid=_grid(count),
            available=available,
        )
        comparison, result = _compare(
            reference, observation, scene_kind="free_fall"
        )
        self.assertEqual(
            ["subject"], comparison.per_frame[-1]["missing_entity_ids"]
        )
        self.assertLess(
            result.state_metric["matched_reference_exposure_ratio"], 1.0
        )
        self.assertTrue(math.isfinite(result.composition["score"]))

    def test_replacement_cannot_fill_missing_directed_identity(self) -> None:
        count = 12
        expected = [_circle(30, 10 + 3 * index) for index in range(count)]
        reference = _reference(
            expected, scene_kind="free_fall", entity_class="ball"
        )
        identity = [True] * 6 + [False] * 6
        observation = observation_from_mask_channels(
            directed_masks=expected,
            residual_instance_masks=[],
            entity_class="ball",
            time_grid=_grid(count),
            identity_valid=identity,
        )
        comparison, _ = _compare(
            reference, observation, scene_kind="free_fall"
        )
        self.assertEqual(
            ["subject"], comparison.per_frame[-1]["missing_entity_ids"]
        )
        self.assertTrue(comparison.per_frame[-1]["extra_track_ids"])
        self.assertLess(comparison.integrity.integrity_gate, 0.5)

    def test_legal_exit_is_not_missing_but_reappearance_is_extra(self) -> None:
        count = 12
        visible = [_circle(30, 10 + 6 * index) for index in range(8)]
        reference_masks = visible + [
            np.zeros_like(visible[0]) for _ in range(count - len(visible))
        ]
        reference = _reference(
            reference_masks,
            scene_kind="free_fall",
            entity_class="ball",
        )
        self.assertEqual(8, reference.legal_exit_frame)
        clean = observation_from_mask_channels(
            directed_masks=reference_masks,
            residual_instance_masks=[],
            entity_class="ball",
            time_grid=_grid(count),
        )
        reappeared = list(reference_masks)
        reappeared[10] = _circle(30, 65)
        reappeared[11] = _circle(30, 68)
        with_return = observation_from_mask_channels(
            directed_masks=reference_masks,
            residual_instance_masks=[reappeared],
            entity_class="ball",
            time_grid=_grid(count),
        )
        clean_comparison, _ = _compare(
            reference, clean, scene_kind="free_fall"
        )
        return_comparison, _ = _compare(
            reference, with_return, scene_kind="free_fall"
        )
        self.assertEqual(
            ["subject"],
            clean_comparison.per_frame[9]["legally_absent_entity_ids"],
        )
        self.assertEqual(
            [], clean_comparison.per_frame[9]["missing_entity_ids"]
        )
        self.assertLess(
            return_comparison.integrity.integrity_gate,
            clean_comparison.integrity.integrity_gate,
        )

    def test_short_reference_detector_gap_is_not_declared_legal_exit(self) -> None:
        count = 12
        visible = [_circle(30, 10 + index) for index in range(8)]
        masks = visible + [
            np.zeros_like(visible[0]) for _ in range(count - len(visible))
        ]
        reference = _reference(
            masks, scene_kind="free_fall", entity_class="ball"
        )
        self.assertIsNone(reference.legal_exit_frame)
        np.testing.assert_array_equal(
            reference.expected, np.ones(count, dtype=bool)
        )

    def test_mid_sequence_reference_gap_cannot_become_legal_exit(
        self,
    ) -> None:
        count = 16
        masks = [_circle(30, 10 + 3 * index) for index in range(count)]
        for index in range(6, 10):
            masks[index] = np.zeros_like(masks[index])
        config = self._strict_reference_config("free_fall")
        reference = build_rigid_body_reference(
            masks,
            entity_id="subject",
            entity_class="ball",
            scene_kind="free_fall",
            frame_shape=masks[0].shape,
            minimum_area=8,
            maximum_area_ratio=0.1,
            minimum_span_px=6.0,
            config=config,
        )
        self.assertIsNone(reference.legal_exit_frame)
        np.testing.assert_array_equal(
            reference.expected,
            np.ones(count, dtype=bool),
        )
        self.assertTrue(np.all(reference.expected[6:10]))

    def test_wrong_prediction_axis_is_penalized(self) -> None:
        count = 14
        reference_masks = [
            _block(12 + 3 * index, 12 + 2 * index)
            for index in range(count)
        ]
        correct = list(reference_masks)
        wrong = [
            _block(12 + 3 * index, 12)
            for index in range(count)
        ]
        reference = _reference(
            reference_masks,
            scene_kind="inclined_plane",
            entity_class="block",
        )
        correct_observation = observation_from_mask_channels(
            directed_masks=correct,
            residual_instance_masks=[],
            entity_class="block",
            time_grid=_grid(count),
        )
        wrong_observation = observation_from_mask_channels(
            directed_masks=wrong,
            residual_instance_masks=[],
            entity_class="block",
            time_grid=_grid(count),
        )
        _, correct_result = _compare(
            reference, correct_observation, scene_kind="inclined_plane"
        )
        _, wrong_result = _compare(
            reference, wrong_observation, scene_kind="inclined_plane"
        )
        self.assertGreater(
            correct_result.state_metric["score"],
            wrong_result.state_metric["score"],
        )
        self.assertFalse(
            wrong_result.state_metric["prediction_axis_refit"]
        )

    def test_overflow_is_formal_false_exposure(self) -> None:
        count = 8
        expected = [_circle(30, 10 + 4 * index) for index in range(count)]
        extra = [_circle(44, 10 + 4 * index) for index in range(count)]
        reference = _reference(
            expected, scene_kind="free_fall", entity_class="ball"
        )
        observation = observation_from_mask_channels(
            directed_masks=expected,
            residual_instance_masks=[extra],
            entity_class="ball",
            time_grid=_grid(count),
            maximum_tracks=1,
        )
        comparison, _ = _compare(
            reference, observation, scene_kind="free_fall"
        )
        self.assertIn("__overflow__", comparison.prediction_exposure)
        self.assertLess(comparison.integrity.integrity_gate, 1.0)

    def test_prediction_failure_is_finite_fail_closed(self) -> None:
        count = 10
        expected = [_circle(30, 10 + 3 * index) for index in range(count)]
        reference = _reference(
            expected, scene_kind="free_fall", entity_class="ball"
        )
        grid = _grid(count)
        timeline = _timeline(reference, grid, "ball")

        def failed():
            raise RuntimeError("synthetic residual failure")

        comparison, observation = safe_compare_rigid_body_open_world(
            expected_timeline=timeline,
            prediction_factory=failed,
            time_grid=grid,
            frame_shape=expected[0].shape,
            minimum_match_position_similarity=0.1,
        )
        result = evaluate_rigid_body_open_world(
            reference=reference,
            scoring_reference=reference,
            observation=observation,
            comparison=comparison,
            time_grid=grid,
            frame_shape=expected[0].shape,
            scene_kind="free_fall",
            scoring_config={
                "trajectory_error_scale": 0.15,
                "acceleration_error_scale": 0.35,
                "impact_time_error_scale": 0.15,
                "horizontal_drift_scale": 0.08,
                "weights": {
                    "vertical_trajectory": 0.5,
                    "normalized_acceleration": 0.25,
                    "impact_time": 0.15,
                    "motion_constraints": 0.1,
                },
            },
            observer_config=default_rigid_body_config("free_fall"),
        )
        self.assertTrue(comparison.failed)
        self.assertEqual(0.0, comparison.integrity.integrity_gate)
        self.assertEqual(0.0, result.composition["score"])
        self.assertTrue(math.isfinite(result.composition["score"]))


if __name__ == "__main__":
    unittest.main()
