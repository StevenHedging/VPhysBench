from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.entities.contracts import EntityMatch
from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    detections_from_instance_masks,
    track_open_world_detections,
)
from physbench.evaluation.common.errors import (
    ReferenceAnalysisError,
    SceneAnalysisError,
)
from physbench.evaluation.common.masks.sam2 import MaskPrompt
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.scenes.collision import open_world
from physbench.evaluation.scenes.collision.v5_evaluator import (
    CollisionOpenWorldCaseEvaluator,
)
import physbench.evaluation.scenes.collision.v5_evaluator as v5_module


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
    return {
        "value": value,
        "unit": unit,
        "annotated": True,
    }


def _collision_case(count: int) -> dict[str, object]:
    physics: dict[str, object] = {}
    for index in range(1, count + 1):
        physics[f"ball_{index}_mass"] = _quantity(
            0.01 * index, "kg"
        )
        physics[f"ball_{index}_radius"] = _quantity(0.005, "m")
        physics[f"ball_{index}_initial_velocity"] = _quantity(
            0.2 if index in {1, count} else 0.0,
            "m/s",
        )
    return {
        "case_id": f"collision_v5_{count}_body",
        "scene_id": "collision_1d",
        "appearance": {
            "ball_sequence": [
                f"ball_appearance_{index}" for index in range(count)
            ],
            "ball_materials": ["steel"] * count,
        },
        "physics": physics,
        "has_real_reference_video": True,
        "provenance": {"parent_case_id": None},
    }


def _frames(count: int = _FRAME_COUNT) -> list[np.ndarray]:
    return [
        np.full((_HEIGHT, _WIDTH, 3), 80, dtype=np.uint8)
        for _ in range(count)
    ]


def _instance_masks(
    count: int,
    *,
    missing: set[int] | None = None,
    x_values: list[int] | None = None,
) -> list[list[np.ndarray]]:
    missing = missing or set()
    xs = x_values or np.linspace(20, _WIDTH - 20, count).astype(int).tolist()
    output: list[list[np.ndarray]] = []
    for object_index, x in enumerate(xs):
        instance = []
        for _ in range(_FRAME_COUNT):
            mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
            if object_index not in missing:
                cv2.circle(mask, (int(x), 48), 5, 255, -1)
            instance.append(mask)
        output.append(instance)
    return output


def _prompt_builder(
    frames,
    *,
    expected_count,
    config,
    entity_ids=None,
    available=None,
):
    del config
    ids = tuple(entity_ids or [
        f"ball_{index + 1}" for index in range(expected_count)
    ])
    availability = (
        [True] * len(frames) if available is None else list(available)
    )
    try:
        seed = next(
            index for index, value in enumerate(availability) if value
        )
    except StopIteration as exc:
        raise SceneAnalysisError(
            "collision_prediction_timeline_unavailable",
            "mock timeline contains no available frame",
        ) from exc
    xs = np.linspace(20, _WIDTH - 20, expected_count)
    prompts = []
    for entity_id, x in zip(ids, xs):
        prompts.append(
            MaskPrompt(
                frame_index=seed,
                box_xyxy=np.asarray([x - 7, 41, x + 7, 55]),
                points_xy=np.asarray([[x, 48.0]]),
                point_labels=np.asarray([1], dtype=np.int32),
                metadata={
                    "entity_id": entity_id,
                    "circle_xyr": [float(x), 48.0, 5.0],
                },
            )
        )
    return prompts, {
        "expected_count": expected_count,
        "selected_entity_ids": list(ids),
        "seed_frame": seed,
    }


class _ScriptedSegmenter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[int] = []

    def describe(self):
        return {"backend": "mock_segmenter"}

    def segment_instances(self, frames, *, prompts, **kwargs):
        del kwargs
        self.calls.append(len(prompts))
        if not self.outcomes:
            raise AssertionError("mock segmenter received an unexpected call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        masks = [
            [np.asarray(mask).copy() for mask in instance]
            for instance in outcome
        ]
        if any(len(instance) != len(frames) for instance in masks):
            return masks, {"backend": "mock_segmenter"}
        return masks, {"backend": "mock_segmenter"}


def _observation_from_masks(instance_masks, *, time_grid):
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


class CollisionV5EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol = json.loads(
            (
                Path(__file__).parents[1]
                / "configs/evaluation/protocols/scene_default_v1.json"
            ).read_text(encoding="utf-8")
        )
        cls.config = copy.deepcopy(protocol["scenes"]["collision_1d"])
        # These unit tests exercise the retained legacy online-observation
        # implementation in isolation.  The public V14 protocol is covered by
        # frozen-reference routing tests instead.
        cls.config.pop("reference_observation_policy", None)

    def _evaluator(
        self,
        outcomes,
        *,
        csti_enabled: bool = False,
    ) -> CollisionOpenWorldCaseEvaluator:
        evaluator = object.__new__(CollisionOpenWorldCaseEvaluator)
        evaluator.config = copy.deepcopy(self.config)
        evaluator.csti_enabled = csti_enabled
        evaluator._segmenter = _ScriptedSegmenter(outcomes)
        return evaluator

    def test_current_search_band_seeds_real_supplement_balls(self) -> None:
        root = Path(__file__).parents[1]
        protocol = json.loads(
            (
                root
                / "configs/evaluation/protocols/scene_default_v1.json"
            ).read_text(encoding="utf-8")
        )
        config = protocol["scenes"]["collision_1d"][
            "multi_frame_observation"
        ]
        reference = (
            root
            / "datasets/assets/collision_1d"
            / "collision_n2-single_b1-steelL-d25mm-m64p77g-v0mps_"
            "b2-steelM-d20mm-m33p13g-vneg0p2847mps_img0989"
            / "canonical/reference.mp4"
        )
        if not reference.is_file():
            self.skipTest("full collision media assets are not published")
        capture = cv2.VideoCapture(str(reference))
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        frame = cv2.resize(frame, (936, 540))

        prompts, metadata = (
            open_world.build_multiframe_collision_entity_prompts(
                [frame],
                expected_count=2,
                entity_ids=["ball_1", "ball_2"],
                config=config,
            )
        )

        circles = metadata["selected_circles_xyr"]
        self.assertEqual(2, len(prompts))
        self.assertEqual(2, len(circles))
        self.assertTrue(all(float(circle[0]) > 800.0 for circle in circles))
        self.assertTrue(all(float(circle[1]) < 400.0 for circle in circles))
        self.assertEqual("multiframe_hough_circle", metadata["seed_source"])

    def _analyze(
        self,
        *,
        count: int,
        outcomes,
        prediction_frames=None,
        prediction_available=None,
        discovery=None,
        case=None,
        csti_enabled: bool = False,
    ):
        evaluator = self._evaluator(outcomes, csti_enabled=csti_enabled)
        reference_frames = _frames()
        if prediction_frames is None:
            prediction_frames = _frames()
        if prediction_available is None:
            prediction_available = [True] * len(prediction_frames)
        case_value = copy.deepcopy(case or _collision_case(count))
        with tempfile.TemporaryDirectory() as temporary:
            request = CaseEvaluationRequest(
                job={"job_id": "collision_v5_unit"},
                case=case_value,
                case_catalog={case_value["case_id"]: case_value},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=evaluator.config,
            )
            reference_video = SimpleNamespace(
                frames=reference_frames,
                available=None,
            )
            prediction_video = SimpleNamespace(
                frames=prediction_frames,
                available=prediction_available,
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        v5_module,
                        "build_multiframe_collision_entity_prompts",
                        side_effect=_prompt_builder,
                    )
                )
                stack.enter_context(
                    patch.object(v5_module, "write_rows_csv")
                )
                stack.enter_context(
                    patch.object(v5_module, "save_iou_curve")
                )
                stack.enter_context(
                    patch.object(v5_module, "save_series_comparison")
                )
                stack.enter_context(
                    patch.object(
                        v5_module,
                        "write_collision_v5_visualization",
                        return_value={
                            "collision_v5_visualization_manifest": (
                                "mock_visualization_manifest.json"
                            )
                        },
                    )
                )
                if discovery is None:
                    def discovery(frames, **kwargs):
                        del frames
                        return _observation_from_masks(
                            kwargs["directed_instance_masks"],
                            time_grid=kwargs["time_grid"],
                        )
                stack.enter_context(
                    patch.object(
                        v5_module,
                        "discover_prediction_objects",
                        side_effect=discovery,
                    )
                )
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=reference_video,
                    prediction_video=prediction_video,
                )
        return evaluator, analysis

    def test_csti_keeps_collision_entities_separate_and_averages_gt(self) -> None:
        reference = _instance_masks(2)
        prediction = _instance_masks(2, missing={1})

        _, analysis = self._analyze(
            count=2,
            outcomes=[reference, prediction],
            csti_enabled=True,
        )
        self.assertIsNotNone(analysis.csti_input)
        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=(("ball_1", "body_1"), ("ball_2", "body_2")),
            config=_CSTI_CONFIG,
        )

        self.assertEqual(["ball_1", "ball_2"], [item["entity_id"] for item in metric["objects"]])
        self.assertAlmostEqual(1.0, metric["objects"][0]["score"], places=12)
        self.assertEqual(0.0, metric["objects"][1]["score"])
        self.assertFalse(metric["objects"][1]["matched"])
        self.assertAlmostEqual(0.5, metric["score"], places=12)

    def test_mock_segmenter_supports_two_and_four_manifest_bodies(self) -> None:
        for count in (2, 4):
            with self.subTest(count=count):
                masks = _instance_masks(count)
                evaluator, analysis = self._analyze(
                    count=count,
                    outcomes=[masks, masks],
                )
                self.assertEqual([count, count], evaluator._segmenter.calls)
                self.assertEqual(
                    count, analysis.quality["expected_entity_count"]
                )
                self.assertTrue(math.isfinite(analysis.score))
                self.assertAlmostEqual(1.0, analysis.score, places=6)

    def test_declared_primary_metric_exists_in_normal_result(self) -> None:
        contract_evaluator = CollisionOpenWorldCaseEvaluator(
            copy.deepcopy(self.config)
        )
        description = contract_evaluator.describe()
        self.assertEqual("2.2", description["version"])
        self.assertEqual(
            0.1,
            description["observation"]["assignment"][
                "minimum_match_position_similarity"
            ],
        )
        declared_primary = description["primary_score"]
        masks = _instance_masks(2)
        _, analysis = self._analyze(
            count=2,
            outcomes=[masks, masks],
        )
        self.assertIn(declared_primary, analysis.metrics)
        self.assertAlmostEqual(
            analysis.score,
            analysis.metrics[declared_primary]["score"],
            places=7,
        )
        self.assertEqual(
            analysis.metrics[declared_primary],
            analysis.metrics["collision_1d_open_world_similarity"],
        )

    def test_v5_passes_frozen_null_assignment_threshold(self) -> None:
        masks = _instance_masks(2)
        implementation = v5_module.compare_open_world_tracks
        with patch.object(
            v5_module,
            "compare_open_world_tracks",
            wraps=implementation,
        ) as comparison:
            _, analysis = self._analyze(
                count=2,
                outcomes=[masks, masks],
            )
        self.assertAlmostEqual(1.0, analysis.score, places=6)
        self.assertGreaterEqual(comparison.call_count, 1)
        for call in comparison.call_args_list:
            self.assertEqual(
                0.1,
                call.kwargs["minimum_match_position_similarity"],
            )

    def test_direct_runtime_failure_continues_with_residual_tracks(self) -> None:
        masks = _instance_masks(2)

        def residual_discovery(frames, **kwargs):
            del frames, kwargs
            return _observation_from_masks(
                masks,
                time_grid=v5_module.build_common_time_grid(_TIMES),
            )

        _, analysis = self._analyze(
            count=2,
            outcomes=[masks, RuntimeError("direct SAM failure")],
            discovery=residual_discovery,
        )
        self.assertIn(
            "prediction_directed_observation_failed",
            analysis.quality["degradation_codes"],
        )
        self.assertEqual(2, analysis.quality["prediction_track_count"])
        self.assertTrue(math.isfinite(analysis.score))
        self.assertGreater(analysis.score, 0.0)

    def test_residual_observation_failure_is_conservative_zero(self) -> None:
        masks = _instance_masks(2)

        def failed_discovery(frames, **kwargs):
            del frames, kwargs
            raise RuntimeError("open-world observer unavailable")

        _, analysis = self._analyze(
            count=2,
            outcomes=[masks, masks],
            discovery=failed_discovery,
        )
        self.assertEqual(0.0, analysis.score)
        self.assertTrue(analysis.quality["degraded"])
        self.assertIn(
            "prediction_residual_observation_failed",
            analysis.quality["degradation_codes"],
        )
        self.assertEqual(0, analysis.quality["prediction_track_count"])
        self.assertEqual(
            "failed_as_empty_prediction",
            analysis.quality["prediction_open_world_observation"][
                "status"
            ],
        )
        self.assertNotEqual(
            "failed_but_residual_discovery_continued",
            analysis.quality["prediction_directed_observation"].get(
                "status"
            ),
        )

    def test_empty_and_short_prediction_are_finite_missing_evidence(self) -> None:
        reference = _instance_masks(2)
        _, empty = self._analyze(
            count=2,
            outcomes=[reference],
            prediction_frames=[],
            prediction_available=[],
        )
        self.assertEqual(0.0, empty.score)
        self.assertIn(
            "prediction_partial_timeline",
            empty.quality["degradation_codes"],
        )
        self.assertEqual(0, empty.quality["available_prediction_frames"])

        prediction = _instance_masks(2)
        _, short = self._analyze(
            count=2,
            outcomes=[reference, prediction],
            prediction_frames=_frames(1),
            prediction_available=[True],
        )
        self.assertTrue(math.isfinite(short.score))
        self.assertLess(short.score, 1.0)
        self.assertEqual(1, short.quality["available_prediction_frames"])

    def test_extra_and_missing_have_conservative_integrity_semantics(
        self,
    ) -> None:
        reference = _instance_masks(2)
        missing = _instance_masks(2, missing={1})
        _, missing_analysis = self._analyze(
            count=2,
            outcomes=[reference, missing],
        )
        missing_integrity = missing_analysis.metrics[
            "object_centric_integrity"
        ]
        self.assertAlmostEqual(
            0.5,
            missing_integrity["presence_detection_accuracy"],
            places=7,
        )
        self.assertAlmostEqual(
            0.0,
            missing_analysis.metrics[
                "collision_nbody_state_similarity"
            ]["score"],
            places=7,
        )
        self.assertEqual(0.0, missing_analysis.score)

        extra_masks = _instance_masks(1, x_values=[80])

        def extra_discovery(frames, **kwargs):
            del frames
            return _observation_from_masks(
                [
                    *kwargs["directed_instance_masks"],
                    *extra_masks,
                ],
                time_grid=kwargs["time_grid"],
            )

        _, extra_analysis = self._analyze(
            count=2,
            outcomes=[reference, reference],
            discovery=extra_discovery,
        )
        extra_integrity = extra_analysis.metrics[
            "object_centric_integrity"
        ]
        self.assertAlmostEqual(
            2.0 / 3.0,
            extra_integrity["presence_detection_accuracy"],
            places=7,
        )
        self.assertAlmostEqual(
            1.0,
            extra_analysis.metrics[
                "collision_nbody_state_similarity"
            ]["score"],
            places=7,
        )
        self.assertAlmostEqual(
            extra_integrity["integrity_gate"],
            extra_analysis.score,
            places=6,
        )

    def test_manifest_and_reference_channel_mismatch_are_reference_failures(
        self,
    ) -> None:
        malformed = _collision_case(2)
        del malformed["physics"]["ball_2_radius"]
        evaluator = self._evaluator([])
        request = CaseEvaluationRequest(
            job={"job_id": "collision_v5_bad_manifest"},
            case=malformed,
            case_catalog={malformed["case_id"]: malformed},
            prediction=None,
            asset_root=Path("/tmp"),
            artifact_dir=Path("/tmp/collision_v5_bad_manifest"),
            evaluator_config=evaluator.config,
        )
        with self.assertRaisesRegex(
            ReferenceAnalysisError, "entity manifest is invalid"
        ) as context:
            evaluator.analyze(
                request,
                times_s=_TIMES,
                reference_video=SimpleNamespace(frames=_frames()),
                prediction_video=SimpleNamespace(
                    frames=_frames(),
                    available=[True] * _FRAME_COUNT,
                ),
            )
        self.assertEqual(
            "reference_entity_manifest_invalid",
            context.exception.code,
        )

        masks = _instance_masks(2)
        with self.assertRaises(ReferenceAnalysisError) as mismatch:
            self._analyze(
                count=2,
                outcomes=[[masks[0]]],
            )
        self.assertEqual(
            "reference_collision_open_world_observation_failed",
            mismatch.exception.code,
        )

    def test_prompt_builder_has_no_three_body_special_case(self) -> None:
        config = copy.deepcopy(
            self.config["multi_frame_observation"]
        )
        circles = [
            (20.0, 48.0, 5.0),
            (60.0, 48.0, 5.0),
            (100.0, 48.0, 5.0),
            (140.0, 48.0, 5.0),
        ]
        with patch.object(
            open_world,
            "_hough_candidates",
            return_value=(circles, (40, 56)),
        ):
            for count in (3, 4):
                entity_ids = tuple(
                    f"custom_{index}" for index in range(count)
                )
                prompts, metadata = (
                    open_world.build_multiframe_collision_entity_prompts(
                        _frames(),
                        expected_count=count,
                        entity_ids=entity_ids,
                        config=config,
                    )
                )
                self.assertEqual(count, len(prompts))
                self.assertEqual(
                    list(entity_ids), metadata["selected_entity_ids"]
                )

    def test_only_full_participant_residual_enters_nbody_physics(
        self,
    ) -> None:
        grid = v5_module.build_common_time_grid([0.0, 0.1, 0.2])

        def residual_detections(
            *,
            x: float,
            tier: EvidenceTier,
            source: str,
        ) -> list[ObjectDetection]:
            output = []
            for frame_index in range(3):
                mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
                cv2.circle(mask, (int(x), 48), 5, 255, -1)
                output.append(
                    ObjectDetection(
                        frame_index=frame_index,
                        detection_id=f"{source}_{frame_index}",
                        xy=np.asarray([x, 48.0]),
                        area_px2=float(np.count_nonzero(mask)),
                        entity_class="ball",
                        mask=mask,
                        confidence=0.95,
                        evidence_tier=tier,
                        sources=(source,),
                    )
                )
            return output

        hough = residual_detections(
            x=90.0,
            tier=EvidenceTier.AMBIGUOUS,
            source="hough_circle",
        )
        motion = residual_detections(
            x=120.0,
            tier=EvidenceTier.PARTICIPANT,
            source="motion_circle",
        )
        observation = track_open_world_detections(
            [
                [hough[frame_index], motion[frame_index]]
                for frame_index in range(3)
            ],
            time_grid=grid,
        )
        centers, valid, _, ids, _ = v5_module._prediction_nbody_inputs(
            comparison_matches=(),
            observation=observation,
            entity_ids=("ball_1", "ball_2"),
            reference_radii=np.full((3, 2), 5.0),
            reference_masses=np.asarray([0.01, 0.02]),
            time_grid=grid,
        )
        self.assertEqual(3, centers.shape[1])
        self.assertEqual(3, valid.shape[1])
        residual_ids = [
            value for value in ids if value.startswith("residual:")
        ]
        self.assertEqual(1, len(residual_ids))
        participant_track = next(
            track
            for track in observation.tracks
            if track.evidence_tier is EvidenceTier.PARTICIPANT
        )
        self.assertEqual(
            [f"residual:{participant_track.track_id}"],
            residual_ids,
        )

    def test_partially_matched_participant_conserves_unmatched_detections(
        self,
    ) -> None:
        grid = v5_module.build_common_time_grid(_TIMES)
        masks = _instance_masks(1, x_values=[100])[0]
        detections = detections_from_instance_masks(
            [masks],
            entity_class="ball",
            source="participant",
            evidence_tier=EvidenceTier.PARTICIPANT,
            confidence=1.0,
            minimum_area_px2=8,
        )
        observation = track_open_world_detections(
            detections,
            time_grid=grid,
            maximum_gap_s=0.25,
            minimum_scale_px=4.0,
        )
        self.assertEqual(1, len(observation.tracks))
        track = observation.tracks[0]
        match = EntityMatch(
            frame_index=0,
            entity_id="ball_1",
            track_id=track.track_id,
            localization_quality=1.0,
            normalized_distance=0.0,
            weight=float(grid.cell_weights_s[0]),
            association_eligible=True,
        )
        _, valid, _, ids, _ = v5_module._prediction_nbody_inputs(
            comparison_matches=(match,),
            observation=observation,
            entity_ids=("ball_1", "ball_2"),
            reference_radii=np.full((_FRAME_COUNT, 2), 5.0),
            reference_masses=np.asarray([0.01, 0.02]),
            time_grid=grid,
        )
        residual_index = ids.index(f"residual:{track.track_id}")
        self.assertEqual(
            [True, False, False, False, False],
            valid[:, 0].tolist(),
        )
        self.assertEqual(
            [False, True, True, True, True],
            valid[:, residual_index].tolist(),
        )
        self.assertFalse(
            np.any(valid[:, 0] & valid[:, residual_index])
        )
        self.assertEqual(
            track.to_object_track(
                frame_count=_FRAME_COUNT,
                time_weights_s=grid.cell_weights_s,
            ).observed.tolist(),
            (valid[:, 0] | valid[:, residual_index]).tolist(),
        )

    def test_alternating_matches_preserve_singleton_residual_slots(
        self,
    ) -> None:
        grid = v5_module.build_common_time_grid(_TIMES)
        masks = _instance_masks(1, x_values=[100])[0]
        observation = track_open_world_detections(
            detections_from_instance_masks(
                [masks],
                entity_class="ball",
                source="participant",
                evidence_tier=EvidenceTier.PARTICIPANT,
                confidence=1.0,
                minimum_area_px2=8,
            ),
            time_grid=grid,
            maximum_gap_s=0.25,
            minimum_scale_px=4.0,
        )
        track = observation.tracks[0]
        matches = tuple(
            EntityMatch(
                frame_index=frame_index,
                entity_id="ball_1",
                track_id=track.track_id,
                localization_quality=1.0,
                normalized_distance=0.0,
                weight=float(grid.cell_weights_s[frame_index]),
                association_eligible=True,
            )
            for frame_index in (0, 2, 4)
        )
        _, valid, _, ids, _ = v5_module._prediction_nbody_inputs(
            comparison_matches=matches,
            observation=observation,
            entity_ids=("ball_1", "ball_2"),
            reference_radii=np.full((_FRAME_COUNT, 2), 5.0),
            reference_masses=np.asarray([0.01, 0.02]),
            time_grid=grid,
        )
        residual_index = ids.index(f"residual:{track.track_id}")
        self.assertEqual(
            [True, False, True, False, True],
            valid[:, 0].tolist(),
        )
        self.assertEqual(
            [False, True, False, True, False],
            valid[:, residual_index].tolist(),
        )
        self.assertTrue(
            np.all(valid[:, 0] | valid[:, residual_index])
        )
        self.assertFalse(
            np.any(valid[:, 0] & valid[:, residual_index])
        )


if __name__ == "__main__":
    unittest.main()
