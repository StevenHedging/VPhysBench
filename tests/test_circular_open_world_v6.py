from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.entities import (
    PositionEvidence,
    ReferenceCapability,
    build_common_time_grid,
    materialize_entity_manifest,
    resolve_expected_entity_timeline,
    safe_compare_open_world_v2,
)
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.scenes.circular_motion.open_world import (
    empty_open_world_observation,
    freeze_reference_identities,
    freeze_prediction_identity,
    observe_circular_objects,
    polar_position_similarity,
)
from physbench.evaluation.scenes.circular_motion.v6_evaluator import (
    CircularMotionOpenWorldCaseEvaluator,
)


_FRAME_COUNT = 9
_TIMES = [index / 8.0 for index in range(_FRAME_COUNT)]
_GRID = build_common_time_grid(_TIMES)
_COLOR_CONFIG = {
    "green_hsv_lower": [35, 35, 25],
    "green_hsv_upper": [100, 255, 255],
    "disk_erosion_kernel": 5,
    "minimum_disk_area_ratio": 0.05,
    "minimum_component_area": 6,
    "maximum_component_area_ratio": 0.03,
    # Legacy expected-N selector cap: v6 must not use this value.
    "maximum_candidates": 2,
    "minimum_valid_frame_ratio": 0.9,
    "open_world_maximum_gap_s": 0.4,
    "open_world_maximum_candidates_per_frame": 64,
    "open_world_maximum_tracks": 64,
}


def _case(count: int = 2) -> dict[str, object]:
    physics: dict[str, object] = {
        "angular_velocity": {
            "value": 1.2,
            "unit": "rad/s",
            "annotated": True,
        },
        "angular_velocity_rad_s": {
            "value": 1.2,
            "unit": "rad/s",
            "annotated": False,
        },
    }
    for index in range(count):
        physics[f"object_{index + 1}_orbit_radius"] = {
            "value": 0.1 + 0.1 * index,
            "unit": "m",
            "annotated": True,
        }
    return {
        "case_id": f"circular_{count}",
        "scene_id": "uniform_circular_motion",
        "has_real_reference_video": True,
        "appearance": {
            "object_count": count,
            "moving_objects": [
                "silver" if index % 2 == 0 else "wood"
                for index in range(count)
            ],
        },
        "physics": physics,
    }


def _frames(
    *,
    radii: tuple[int, ...] = (22, 38),
    present: dict[int, set[int]] | None = None,
    extra_windows: tuple[tuple[int, int, int, float], ...] = (),
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for frame_index in range(_FRAME_COUNT):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(frame, (64, 64), 52, (0, 180, 0), -1)
        for object_index, radius in enumerate(radii):
            if (
                present is not None
                and frame_index not in present.get(
                    object_index, set(range(_FRAME_COUNT))
                )
            ):
                continue
            angle = 0.15 * frame_index + 1.4 * object_index
            _draw_object(frame, radius=radius, angle=angle, index=object_index)
        for radius, first, last, phase in extra_windows:
            if first <= frame_index <= last:
                _draw_object(
                    frame,
                    radius=radius,
                    angle=0.15 * frame_index + phase,
                    index=7,
                )
        output.append(frame)
    return output


def _draw_object(
    frame: np.ndarray,
    *,
    radius: int,
    angle: float,
    index: int,
) -> None:
    x = int(round(64 + radius * math.cos(angle)))
    y = int(round(64 + radius * math.sin(angle)))
    color = (255, 255, 255) if index % 2 == 0 else (40, 100, 180)
    cv2.rectangle(frame, (x - 4, y - 3), (x + 4, y + 3), color, -1)


def _reference():
    observation = observe_circular_objects(
        _frames(), time_grid=_GRID, config=_COLOR_CONFIG
    )
    manifest = materialize_entity_manifest(_case())
    frozen = freeze_reference_identities(
        observation,
        entities=manifest.entities,
        time_grid=_GRID,
    )
    return observation, frozen


def _integrity(frames: list[np.ndarray]):
    _, frozen = _reference()
    manifest = materialize_entity_manifest(_case())
    prediction = observe_circular_objects(
        frames, time_grid=_GRID, config=_COLOR_CONFIG
    )
    reference_by_id = {
        track.matched_entity_id: track for track in frozen.tracks
    }
    masks_by_id = {
        entity_id: masks
        for entity_id, masks in zip(
            frozen.entity_ids, frozen.instance_masks
        )
    }
    timelines = [
        resolve_expected_entity_timeline(
            entity.to_entity_spec(),
            time_grid=_GRID,
            capability=ReferenceCapability.SAME_CASE_GT,
            reference_track=reference_by_id[entity.entity_id],
            reference_masks=masks_by_id[entity.entity_id],
        )
        for entity in manifest.entities
    ]
    assignment = freeze_prediction_identity(
        prediction.objects,
        entities=manifest.entities,
        condition_anchors=frozen.anchors,
    )

    def position(timeline, detection, frame_index, _diagonal):
        value = polar_position_similarity(
            timeline.reference_xy[frame_index], detection.xy
        )
        return PositionEvidence(
            score=float(value["score"]),
            normalized_distance=float(value["normalized_distance"]),
            diagnostics={"mode": value["mode"]},
        )

    comparison = safe_compare_open_world_v2(
        expected_timelines=timelines,
        prediction_factory=lambda: prediction.objects,
        time_grid=_GRID,
        frame_diagonal_px=2.0 * math.sqrt(2.0) * 100.0,
        frozen_identity=assignment,
        minimum_match_position_similarity=0.1,
        position_similarity_callback=position,
    )
    return comparison, prediction


class CircularOpenWorldObservationTests(unittest.TestCase):
    def test_preserves_one_two_and_n_candidates_without_expected_n_crop(
        self,
    ) -> None:
        for radii in ((30,), (22, 38), (13, 23, 33, 43)):
            with self.subTest(count=len(radii)):
                observation = observe_circular_objects(
                    _frames(radii=radii),
                    time_grid=_GRID,
                    config=_COLOR_CONFIG,
                )
                self.assertEqual(
                    {len(radii)}, set(observation.candidate_counts.tolist())
                )
                self.assertEqual(len(radii), len(observation.objects.tracks))
                self.assertTrue(
                    all(
                        track.formal_exposure_weight == 1.0
                        for track in observation.objects.tracks
                    )
                )

    def test_disk_is_apparatus_not_a_physical_object(self) -> None:
        frames = []
        for _ in range(_FRAME_COUNT):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.circle(frame, (64, 64), 52, (0, 180, 0), -1)
            frames.append(frame)
        observation = observe_circular_objects(
            frames, time_grid=_GRID, config=_COLOR_CONFIG
        )
        self.assertEqual({0}, set(observation.candidate_counts.tolist()))
        self.assertEqual(0, len(observation.objects.tracks))

    def test_third_orbiter_is_formal_extra_and_lowers_integrity(self) -> None:
        baseline, _ = _integrity(_frames())
        extra, observation = _integrity(
            _frames(extra_windows=((30, 0, _FRAME_COUNT - 1, 2.6),))
        )
        self.assertEqual(3, len(observation.objects.tracks))
        self.assertEqual(1.0, baseline.integrity.integrity_gate)
        self.assertLess(extra.integrity.integrity_gate, 1.0)
        self.assertTrue(
            all(row["extra_track_ids"] for row in extra.per_frame)
        )

    def test_missing_penalty_is_monotonic_with_duration(self) -> None:
        scores = []
        for missing_count in (0, 2, 4):
            visible = set(range(_FRAME_COUNT - missing_count))
            comparison, _ = _integrity(
                _frames(
                    present={
                        0: set(range(_FRAME_COUNT)),
                        1: visible,
                    }
                )
            )
            scores.append(comparison.integrity.integrity_gate)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_extra_penalty_is_monotonic_with_duration(self) -> None:
        scores = []
        for extra_count in (0, 3, 6):
            first = _FRAME_COUNT - extra_count
            windows = (
                ()
                if extra_count == 0
                else ((30, first, _FRAME_COUNT - 1, 2.6),)
            )
            comparison, _ = _integrity(
                _frames(extra_windows=windows)
            )
            scores.append(comparison.integrity.integrity_gate)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_far_replacement_cannot_take_over_without_identity_penalty(
        self,
    ) -> None:
        present = {
            0: set(range(_FRAME_COUNT)),
            1: set(range(5)),
        }
        frames = _frames(
            present=present,
            extra_windows=((38, 5, _FRAME_COUNT - 1, math.pi),),
        )
        comparison, _ = _integrity(frames)
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(
            comparison.integrity.association_accuracy < 1.0
            or any(
                row["missing_entity_ids"]
                or row["rejected_candidate_matches"]
                for row in comparison.per_frame[5:]
            )
        )

    def test_appearance_identity_swap_is_split_and_penalized(self) -> None:
        frames = _frames()
        for frame_index in range(5, _FRAME_COUNT):
            angle = 0.15 * frame_index + 1.4
            x = int(round(64 + 38 * math.cos(angle)))
            y = int(round(64 + 38 * math.sin(angle)))
            cv2.rectangle(
                frames[frame_index],
                (x - 4, y - 3),
                (x + 4, y + 3),
                (255, 255, 255),
                -1,
            )
        comparison, observation = _integrity(frames)
        self.assertGreater(
            int(
                observation.diagnostics[
                    "appearance_identity_switch_splits"
                ]
            ),
            0,
        )
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(
            any(
                row["missing_entity_ids"] and row["extra_track_ids"]
                for row in comparison.per_frame[5:]
            )
        )

    def test_short_occlusion_bridges_one_causal_track_id(self) -> None:
        present = {
            0: set(range(_FRAME_COUNT)),
            1: set(range(_FRAME_COUNT)) - {4},
        }
        observation = observe_circular_objects(
            _frames(present=present),
            time_grid=_GRID,
            config=_COLOR_CONFIG,
        )
        outer = max(
            observation.objects.tracks,
            key=lambda track: float(
                np.median(
                    [
                        detection.metadata["normalized_orbit_radius"]
                        for detection in track.detections
                    ]
                )
            ),
        )
        self.assertEqual(_FRAME_COUNT - 1, len(outer.detections))
        self.assertEqual(
            [0, 1, 2, 3, 5, 6, 7, 8],
            [value.frame_index for value in outer.detections],
        )

    def test_center_fallback_and_overflow_are_finite_and_auditable(
        self,
    ) -> None:
        center = polar_position_similarity(
            [0.0, 0.0], [2.0, -1.0]
        )
        self.assertEqual("cartesian_center_fallback", center["mode"])
        self.assertTrue(math.isfinite(float(center["score"])))
        config = {
            **_COLOR_CONFIG,
            "open_world_maximum_candidates_per_frame": 2,
        }
        observation = observe_circular_objects(
            _frames(radii=(13, 23, 33, 43)),
            time_grid=_GRID,
            config=config,
        )
        self.assertTrue(
            np.all(observation.objects.overflow_counts >= 2.0)
        )
        self.assertTrue(
            np.isfinite(observation.objects.overflow_counts).all()
        )


class CircularOpenWorldEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "type": "uniform_circular_motion_state_v6",
            "evaluator_contract": "robust_subject_v3",
            "timeline": {"decode_policy": "sequential_forward"},
            "spatial": {"width": 128, "height": 128, "pad_value": 0},
            "color_observation": copy.deepcopy(_COLOR_CONFIG),
            "scoring": {
                "angular_trajectory_scale_rad": 0.35,
                "angular_velocity_error_scale": 0.25,
                "radial_cv_scale": 0.08,
                "angular_fit_rmse_scale_rad": 0.2,
                "radius_configuration_scale": 0.12,
                "weights": {
                    "angular_trajectory": 0.5,
                    "angular_velocity": 0.25,
                    "orbit_geometry": 0.15,
                    "uniform_motion": 0.1,
                },
            },
            "subject_scoring": {
                "minimum_observed_pixels": 4,
                "position_distance_scale": 0.08,
                "canonical_crop_size": 32,
                "boundary_tolerance_px": 2,
                "weights": {
                    "position": 0.5,
                    "shape": 0.2,
                    "appearance": 0.3,
                },
            },
            "object_centric_scoring": {
                "assignment": {
                    "minimum_match_position_similarity": 0.1
                }
            },
        }

    def test_normal_result_has_unified_metrics_and_finite_score(self) -> None:
        config = self._config()
        evaluator = CircularMotionOpenWorldCaseEvaluator(
            copy.deepcopy(config)
        )
        case = _case()
        with tempfile.TemporaryDirectory() as temporary:
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v6_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=config,
            )
            video = SimpleNamespace(
                frames=_frames(),
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=video,
                    prediction_video=video,
                )
        self.assertTrue(math.isfinite(analysis.score))
        self.assertGreater(analysis.score, 0.99)
        for name in (
            "scene_subject_state_similarity",
            "uniform_circular_motion_open_world_similarity",
            "object_centric_integrity",
            "entity_manifest",
        ):
            self.assertIn(name, analysis.metrics)
        self.assertEqual(
            analysis.metrics["scene_subject_state_similarity"],
            analysis.metrics[
                "uniform_circular_motion_open_world_similarity"
            ],
        )
        self.assertIn("object_centric_audit", analysis.artifacts)
        self.assertEqual(
            "open_world_v2",
            analysis.provenance["open_world_protocol"]["id"],
        )

    def test_empty_prediction_is_finite_conservative_zero(self) -> None:
        _, frozen = _reference()
        manifest = materialize_entity_manifest(_case())
        reference_by_id = {
            track.matched_entity_id: track for track in frozen.tracks
        }
        timelines = [
            resolve_expected_entity_timeline(
                entity.to_entity_spec(),
                time_grid=_GRID,
                capability=ReferenceCapability.SAME_CASE_GT,
                reference_track=reference_by_id[entity.entity_id],
            )
            for entity in manifest.entities
        ]
        comparison = safe_compare_open_world_v2(
            expected_timelines=timelines,
            prediction_factory=lambda: empty_open_world_observation(
                _FRAME_COUNT,
                code="prediction_observation_failed",
                reason="synthetic empty prediction",
            ),
            time_grid=_GRID,
            frame_diagonal_px=2.0 * math.sqrt(2.0) * 100.0,
            minimum_match_position_similarity=0.1,
        )
        self.assertTrue(math.isfinite(comparison.integrity.integrity_gate))
        self.assertEqual(0.0, comparison.integrity.integrity_gate)

    def test_physics_parent_uses_condition_identity_not_future_pixels(
        self,
    ) -> None:
        config = self._config()
        evaluator = CircularMotionOpenWorldCaseEvaluator(
            copy.deepcopy(config)
        )
        case = _case()
        case["has_real_reference_video"] = False
        case["provenance"] = {"parent_case_id": "physics_parent"}
        case["assets"] = {"first_frame": "condition.png"}
        frames = _frames()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(
                cv2.imwrite(str(root / "condition.png"), frames[0])
            )
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v6_parent_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config=config,
            )
            video = SimpleNamespace(
                frames=frames,
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=video,
                    prediction_video=video,
                )
        timelines = analysis.provenance["expected_timelines"]
        self.assertTrue(
            all(
                value["capability"] == "physics_parent"
                for value in timelines
            )
        )
        self.assertTrue(
            all(
                value["localization_supervised"][0]
                and not any(value["localization_supervised"][1:])
                for value in timelines
            )
        )
        self.assertTrue(
            all(
                all(
                    all(coordinate is None for coordinate in position)
                    for position in value["reference_xy"][1:]
                )
                for value in timelines
            )
        )
        self.assertGreater(analysis.score, 0.8)


if __name__ == "__main__":
    unittest.main()
