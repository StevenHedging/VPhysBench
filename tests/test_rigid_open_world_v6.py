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

from physbench.evaluation.common.entities import (
    ReferenceCapability,
    build_common_time_grid,
)
from physbench.evaluation.scenes.rigid_body_open_world import (
    RigidBodyOpenWorldCaseEvaluatorBase,
    build_expected_rigid_body_timeline,
    build_rigid_body_reference,
    default_rigid_body_config,
    discover_rigid_body_objects,
    evaluate_rigid_body_open_world,
    observation_from_mask_channels,
    safe_compare_rigid_body_open_world,
    write_rigid_body_audit_json,
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
    }, result


class RigidBodyOpenWorldV6Tests(unittest.TestCase):
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
