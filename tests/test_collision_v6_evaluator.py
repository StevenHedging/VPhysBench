from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    OpenWorldObservation,
    compare_open_world_tracks,
    detections_from_instance_masks,
    track_open_world_detections,
)
from physbench.evaluation.common.errors import SceneAnalysisError
from physbench.evaluation.common.masks.sam2 import MaskPrompt
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.scenes.collision.v6_evaluator import (
    CollisionFailClosedCaseEvaluator,
    CollisionIdentityContext,
)
from physbench.evaluation.scenes.collision.v6_identity import (
    CollisionIdentityAnchor,
    latch_collision_identity_masks,
)
from physbench.evaluation.scenes.collision.open_world import (
    prediction_observation_from_instance_masks,
    reference_tracks_from_instance_masks,
)
import physbench.evaluation.scenes.collision.v5_evaluator as v5_module
import physbench.evaluation.scenes.collision.v6_evaluator as v6_module


_FRAME_COUNT = 5
_HEIGHT = 60
_WIDTH = 160
_TIMES = [0.0, 0.1, 0.2, 0.3, 0.4]
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


def _quantity(value: float, unit: str) -> dict[str, object]:
    return {"value": value, "unit": unit, "annotated": True}


def _case() -> dict[str, object]:
    return {
        "case_id": "collision_v6_two_body",
        "scene_id": "collision_1d",
        "appearance": {
            "ball_sequence": ["dark", "light"],
            "ball_materials": ["steel", "steel"],
        },
        "physics": {
            "ball_1_mass": _quantity(0.01, "kg"),
            "ball_1_radius": _quantity(0.005, "m"),
            "ball_1_initial_velocity": _quantity(0.2, "m/s"),
            "ball_2_mass": _quantity(0.02, "kg"),
            "ball_2_radius": _quantity(0.005, "m"),
            "ball_2_initial_velocity": _quantity(0.0, "m/s"),
        },
        "has_real_reference_video": True,
        "provenance": {"parent_case_id": None},
    }


def _frames() -> list[np.ndarray]:
    return [
        np.full((_HEIGHT, _WIDTH, 3), 80, dtype=np.uint8)
        for _ in range(_FRAME_COUNT)
    ]


def _masks(*, reappear: bool = False) -> list[list[np.ndarray]]:
    output = []
    for x in (30, 120):
        instance = []
        for frame_index in range(_FRAME_COUNT):
            mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
            if not reappear or frame_index not in {2}:
                cv2.circle(mask, (x + frame_index, 48), 5, 255, -1)
            instance.append(mask)
        output.append(instance)
    return output


def _prompts() -> list[MaskPrompt]:
    output = []
    for entity_id, x in zip(("ball_1", "ball_2"), (30.0, 120.0)):
        output.append(
            MaskPrompt(
                frame_index=0,
                box_xyxy=np.asarray([x - 8, 40, x + 8, 56]),
                points_xy=np.asarray([[x, 48.0]]),
                point_labels=np.asarray([1], dtype=np.int32),
                metadata={
                    "entity_id": entity_id,
                    "circle_xyr": [x, 48.0, 5.0],
                },
            )
        )
    return output


def _anchors() -> tuple[CollisionIdentityAnchor, ...]:
    frame = _frames()[0]
    values = []
    for order, (entity_id, x) in enumerate(
        zip(("ball_1", "ball_2"), (30, 120)), start=1
    ):
        mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
        cv2.circle(mask, (x, 48), 5, 255, -1)
        mask.setflags(write=False)
        histogram = np.zeros(512, dtype=np.float64)
        histogram[0] = 1.0
        histogram.setflags(write=False)
        values.append(
            CollisionIdentityAnchor(
                logical_entity_id=entity_id,
                dataset_object_id=f"object_{order}",
                mask=mask,
                centroid_xy=np.asarray([float(x), 48.0]),
                equivalent_radius_px=5.0,
                histogram=histogram,
                provenance={
                    "policy": "reference_motion_validated_frame_zero_v1"
                },
            )
        )
    del frame
    return tuple(values)


def _observation_tuple(instance_masks):
    count = len(instance_masks)
    xy = np.full((_FRAME_COUNT, count, 2), np.nan, dtype=np.float64)
    valid = np.zeros((_FRAME_COUNT, count), dtype=bool)
    radii = np.full((_FRAME_COUNT, count), 5.0, dtype=np.float64)
    union = []
    for frame_index in range(_FRAME_COUNT):
        current_union = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
        for object_index, instance in enumerate(instance_masks):
            mask = instance[frame_index]
            current_union = cv2.bitwise_or(current_union, mask)
            moments = cv2.moments(mask, binaryImage=True)
            if moments["m00"]:
                xy[frame_index, object_index] = [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ]
                valid[frame_index, object_index] = True
        union.append(current_union)
    return (
        xy,
        valid,
        radii,
        instance_masks,
        union,
        {"status": "mock_confirmed"},
    )


def _tracked_observation(instance_masks, *, time_grid):
    detections = detections_from_instance_masks(
        instance_masks,
        entity_class="ball",
        source="mock_residual",
        evidence_tier=EvidenceTier.PARTICIPANT,
        confidence=1.0,
        minimum_area_px2=8,
    )
    return track_open_world_detections(
        detections,
        time_grid=time_grid,
        maximum_gap_s=0.25,
        minimum_scale_px=4.0,
    )


class _Segmenter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.prompt_frames: list[list[int]] = []

    def segment_instances(self, frames, *, prompts, **kwargs):
        del kwargs
        self.prompt_frames.append([value.frame_index for value in prompts])
        masks = self.outcomes.pop(0)
        return (
            [[np.asarray(mask).copy() for mask in role] for role in masks],
            {"backend": "mock"},
        )


class CollisionV6EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol = json.loads(
            (
                Path(__file__).parents[1]
                / "configs/evaluation/protocols/scene_default_v1.json"
            ).read_text(encoding="utf-8")
        )
        cls.config = copy.deepcopy(protocol["scenes"]["collision_1d"])
        cls.config["type"] = "collision_1d_v1"
        cls.config["subject_identity"] = {
            "anchor_policy": "reference_motion_validated_frame_zero_v1",
            "failure_policy": "fail_closed_v1",
            "reference_localization": {
                "minimum_motion_fraction": 0.20,
                "minimum_mean_change": 8.0,
                "maximum_entity_y_spread_px": 32.0,
                "maximum_radius_ratio": 4.5,
                "minimum_gap_radius_fraction": 0.45,
                "minimum_two_body_gap_radius_fraction": 0.75,
                "minimum_set_score_margin": 0.20,
            },
            "prediction_binding": {
                "box_expand": 1.5,
                "minimum_box_side": 24,
                "maximum_center_distance_radii": 3.0,
                "minimum_position_similarity": 0.2,
                "minimum_scale_similarity": 0.45,
                "minimum_appearance_similarity": 0.15,
                "minimum_binding_score": 0.55,
                "minimum_assignment_margin": 0.05,
                "weights": {
                    "position": 0.5,
                    "scale": 0.2,
                    "appearance": 0.3,
                },
            },
            "tracking": {
                "minimum_mask_pixels": 12,
                "maximum_mask_area_ratio": 0.02,
                "maximum_centroid_jump_px": 120.0,
            },
            "unexpected_participant": {
                "minimum_observed_frames": 3,
                "minimum_observed_frame_fraction": 0.25,
            },
        }

    def _evaluator(self) -> CollisionFailClosedCaseEvaluator:
        evaluator = object.__new__(CollisionFailClosedCaseEvaluator)
        evaluator.config = copy.deepcopy(self.config)
        evaluator.csti_enabled = True
        evaluator._segmenter = _Segmenter([])
        return evaluator

    def test_reference_and_prediction_roles_are_seeded_only_at_frame_zero(
        self,
    ) -> None:
        evaluator = self._evaluator()
        masks = _masks()
        evaluator._segmenter = _Segmenter([masks, masks])
        context = CollisionIdentityContext(
            anchors=_anchors(),
            reference_localization={"status": "confirmed"},
        )
        with patch.object(
            v5_module,
            "build_multiframe_collision_entity_prompts",
            side_effect=AssertionError("v6 searched a future frame"),
        ), patch.object(
            v6_module,
            "build_prediction_collision_identity_prompts",
            return_value=(
                _prompts(),
                {"seed_frame": 0, "status": "confirmed"},
            ),
        ):
            reference = evaluator._observe_expected_role(
                _frames(),
                expected_count=2,
                entity_ids=("ball_1", "ball_2"),
                observation_role="reference",
                identity_context=context,
            )
            prediction = evaluator._observe_expected_role(
                _frames(),
                expected_count=2,
                entity_ids=("ball_1", "ball_2"),
                observation_role="prediction",
                identity_context=context,
                available=[True] * _FRAME_COUNT,
            )

        self.assertEqual([[0, 0], [0, 0]], evaluator._segmenter.prompt_frames)
        self.assertTrue(reference[1][0].all())
        self.assertTrue(prediction[1][0].all())
        self.assertEqual(
            "causal_terminal_identity_latch_v1",
            prediction[5]["identity_latch"]["policy"],
        )

    def test_identity_dropout_is_terminal_and_cannot_reappear(self) -> None:
        masks = _masks(reappear=True)
        latched, metadata = latch_collision_identity_masks(
            masks,
            _prompts(),
            config=self.config["subject_identity"]["tracking"],
        )

        self.assertGreater(np.count_nonzero(latched[0][1]), 0)
        self.assertEqual(0, np.count_nonzero(latched[0][2]))
        self.assertEqual(0, np.count_nonzero(latched[0][3]))
        self.assertEqual(2, metadata["roles"][0]["terminated_frame"])
        self.assertEqual(
            "mask_missing_or_invalid",
            metadata["roles"][0]["termination_reason"],
        )

    def test_direct_identity_failure_cannot_be_rescued_by_residual_tracks(
        self,
    ) -> None:
        evaluator = self._evaluator()
        reference_masks = _masks()
        perfect_residual = _tracked_observation(
            reference_masks,
            time_grid=v5_module.build_common_time_grid(_TIMES),
        )
        case = _case()
        request = CaseEvaluationRequest(
            job={"job_id": "collision_v6_terminal_identity"},
            case=case,
            case_catalog={case["case_id"]: case},
            prediction={"status": "complete", "video_path": "mock.mp4"},
            asset_root=Path("/tmp"),
            artifact_dir=Path("/tmp/collision_v6_terminal_identity"),
            evaluator_config=evaluator.config,
        )
        observations = [
            _observation_tuple(reference_masks),
            SceneAnalysisError(
                "collision_prediction_identity_ambiguous",
                "frame-zero assignment is not unique",
            ),
        ]

        def observe(*args, **kwargs):
            del args, kwargs
            value = observations.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            request = CaseEvaluationRequest(
                **{
                    **request.__dict__,
                    "artifact_dir": Path(temporary) / "artifacts",
                }
            )
            stack.enter_context(
                patch.object(evaluator, "_prepare_identity_context")
            )
            stack.enter_context(
                patch.object(
                    evaluator,
                    "_observe_expected_role",
                    side_effect=observe,
                )
            )
            stack.enter_context(
                patch.object(
                    v5_module,
                    "discover_prediction_objects",
                    return_value=perfect_residual,
                )
            )
            stack.enter_context(patch.object(v5_module, "write_rows_csv"))
            stack.enter_context(patch.object(v5_module, "save_iou_curve"))
            stack.enter_context(
                patch.object(v5_module, "save_series_comparison")
            )
            stack.enter_context(
                patch.object(
                    v5_module,
                    "write_collision_v5_visualization",
                    return_value={},
                )
            )
            analysis = evaluator.analyze(
                request,
                times_s=_TIMES,
                reference_video=SimpleNamespace(frames=_frames()),
                prediction_video=SimpleNamespace(
                    frames=_frames(),
                    available=[True] * _FRAME_COUNT,
                ),
            )

        self.assertEqual(0.0, analysis.score)
        self.assertEqual(0, analysis.quality["prediction_track_count"])
        self.assertEqual(
            2,
            analysis.quality[
                "prediction_residual_after_identity_failure"
            ]["track_count"],
        )
        self.assertTrue(
            analysis.quality["directed_identity_failure_terminal"]
        )
        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=(
                ("ball_1", "body_1"),
                ("ball_2", "body_2"),
            ),
            config=_CSTI_CONFIG,
        )
        self.assertEqual(0.0, metric["score"])

    def test_exact_directed_tracks_retain_unit_score_and_csti(self) -> None:
        evaluator = self._evaluator()
        masks = _masks()
        observations = [
            _observation_tuple(masks),
            _observation_tuple(masks),
        ]

        def observe(*args, **kwargs):
            del args, kwargs
            return observations.pop(0)

        def discovery(frames, **kwargs):
            del frames
            return prediction_observation_from_instance_masks(
                kwargs["directed_instance_masks"],
                time_grid=kwargs["time_grid"],
                quality_config=self.config["quality"],
                available=kwargs["available"],
            )

        case = _case()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            request = CaseEvaluationRequest(
                job={"job_id": "collision_v6_exact"},
                case=case,
                case_catalog={case["case_id"]: case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=evaluator.config,
            )
            stack.enter_context(
                patch.object(evaluator, "_prepare_identity_context")
            )
            stack.enter_context(
                patch.object(
                    evaluator,
                    "_observe_expected_role",
                    side_effect=observe,
                )
            )
            stack.enter_context(
                patch.object(
                    v5_module,
                    "discover_prediction_objects",
                    side_effect=discovery,
                )
            )
            comparison = stack.enter_context(
                patch.object(
                    v5_module,
                    "compare_open_world_tracks",
                    wraps=v5_module.compare_open_world_tracks,
                )
            )
            stack.enter_context(patch.object(v5_module, "write_rows_csv"))
            stack.enter_context(patch.object(v5_module, "save_iou_curve"))
            stack.enter_context(
                patch.object(v5_module, "save_series_comparison")
            )
            stack.enter_context(
                patch.object(
                    v5_module,
                    "write_collision_v5_visualization",
                    return_value={},
                )
            )
            analysis = evaluator.analyze(
                request,
                times_s=_TIMES,
                reference_video=SimpleNamespace(frames=_frames()),
                prediction_video=SimpleNamespace(
                    frames=_frames(),
                    available=[True] * _FRAME_COUNT,
                ),
            )

        self.assertAlmostEqual(1.0, analysis.score, places=12)
        self.assertFalse(
            analysis.quality["directed_identity_failure_terminal"]
        )
        self.assertIsNone(
            analysis.quality["prediction_residual_after_identity_failure"]
        )
        for call in comparison.call_args_list:
            self.assertEqual(
                {
                    "ball_1": "direct_000",
                    "ball_2": "direct_001",
                },
                call.kwargs["fixed_entity_track_ids"],
            )
        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=(
                ("ball_1", "body_1"),
                ("ball_2", "body_2"),
            ),
            config=_CSTI_CONFIG,
        )
        self.assertAlmostEqual(1.0, metric["score"], places=12)

    def test_residual_after_latch_cannot_reactivate_declared_role(self) -> None:
        grid = v5_module.build_common_time_grid(_TIMES)
        reference_masks = _masks()
        directed_masks = _masks()
        for frame_index in range(2, _FRAME_COUNT):
            directed_masks[0][frame_index] = np.zeros(
                (_HEIGHT, _WIDTH), dtype=np.uint8
            )
        direct = prediction_observation_from_instance_masks(
            directed_masks,
            time_grid=grid,
            quality_config=self.config["quality"],
        )
        residual_masks = [[
            (
                np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
                if frame_index < 2
                else reference_masks[0][frame_index]
            )
            for frame_index in range(_FRAME_COUNT)
        ]]
        residual = _tracked_observation(residual_masks, time_grid=grid)
        observation = OpenWorldObservation(
            tracks=(*direct.tracks, *residual.tracks),
            overflow_counts=np.zeros(_FRAME_COUNT, dtype=np.float64),
            diagnostics={"status": "mock_latched_with_residual"},
        )
        terminal = self._evaluator()._terminal_prediction_observation_failure(
            observation,
            entity_ids=("ball_1", "ball_2"),
            frame_count=_FRAME_COUNT,
        )
        self.assertIsNotNone(terminal)
        self.assertEqual(
            "collision_prediction_unexpected_participant",
            terminal[0]["code"],
        )
        reference = reference_tracks_from_instance_masks(
            reference_masks,
            entity_ids=("ball_1", "ball_2"),
            entity_class="ball",
            time_grid=grid,
            minimum_area=8,
            maximum_area_ratio=0.08,
        )

        comparison = compare_open_world_tracks(
            reference_tracks=reference,
            prediction_observation=observation,
            time_grid=grid,
            frame_diagonal_px=float(np.hypot(_WIDTH, _HEIGHT)),
            fixed_entity_track_ids={
                "ball_1": "direct_000",
                "ball_2": "direct_001",
            },
        )

        ball_1_matches = [
            value for value in comparison.matches
            if value.entity_id == "ball_1"
        ]
        self.assertEqual([0, 1], [value.frame_index for value in ball_1_matches])
        self.assertTrue(
            all(value.track_id == "direct_000" for value in ball_1_matches)
        )
        self.assertTrue(
            all(
                comparison.per_frame[index]["missing_entity_ids"]
                == ["ball_1"]
                for index in range(2, _FRAME_COUNT)
            )
        )


if __name__ == "__main__":
    unittest.main()
