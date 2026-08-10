from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.entities import materialize_entity_manifest
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.common.entities.contracts import (
    EntitySpec,
    LifecyclePolicy,
    ReferenceCapability,
)
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.parabolic_motion.evaluator import (
    BallCandidate,
    ParabolicMotionCaseEvaluator,
    _compose_parabolic_scores,
    observe_projectile,
    score_parabolic_observations,
)
from physbench.evaluation.scenes.parabolic_motion.visualization import (
    write_parabolic_visualization,
)


def _frames(*, shift_xy: tuple[int, int] = (0, 0)) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    shift_x, shift_y = shift_xy
    for index in range(12):
        frame = np.full((160, 240, 3), 224, dtype=np.uint8)
        x = 218 - 12 * index + shift_x
        y = 28 + round(0.65 * index * index) + shift_y
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x, y), 8, (25, 25, 25), -1)
            cv2.circle(frame, (x - 2, y - 2), 2, (210, 210, 210), -1)
        frames.append(frame)
    return frames


def _observation_config() -> dict:
    return {
        "initial_hough_thresholds": [18, 14, 10],
        "hough_thresholds": [16, 12, 9],
        "minimum_radius_px": 4,
        "maximum_radius_px": 14,
        "initial_minimum_x_ratio": 0.65,
        "initial_maximum_y_ratio": 0.45,
        "minimum_local_contrast": 5.0,
        "foreground_threshold": 8.0,
        "maximum_interpolation_gap_frames": 2,
    }


def _scoring_config() -> dict:
    return {
        "trajectory_distance_scale": 0.06,
        "weights": {
            "time_parameterized_trajectory": 0.45,
            "horizontal_uniform_motion": 0.15,
            "vertical_uniform_acceleration": 0.15,
            "parabolic_geometry": 0.15,
            "lifecycle_and_cardinality": 0.1,
        },
    }


_CSTI_MAPPING = {
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
_CSTI_CONFIG = CSTIConfig.from_mapping(_CSTI_MAPPING)


def _case() -> dict[str, object]:
    return {
        "case_id": "synthetic_parabolic",
        "scene_id": "parabolic_motion",
        "physics": {
            "ball_mass": {"value": 0.014, "unit": "kg", "annotated": True},
            "ball_radius": {"value": 0.0075, "unit": "m", "annotated": True},
            "initial_horizontal_velocity": {
                "value": 1.0,
                "unit": "m/s",
                "annotated": True,
            },
            "launch_height": {"value": 0.77, "unit": "m", "annotated": True},
        },
        "appearance": {"ball_material": "steel", "ball_size_class": "small"},
        "assets": {"reference_video": "reference.mp4"},
    }


class ParabolicMotionEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _false_high_components() -> dict[str, float]:
        return {
            "time_parameterized_trajectory": 0.6493891526642883,
            "horizontal_uniform_motion": 0.40136498136363213,
            "vertical_uniform_acceleration": 0.1140319428328521,
            "parabolic_geometry": 0.31531938280236077,
            "lifecycle_and_cardinality": 0.9305362023265322,
        }

    def test_strict_composition_does_not_compensate_failed_motion_law(
        self,
    ) -> None:
        composed = _compose_parabolic_scores(
            self._false_high_components(),
            subject_score=0.8143368124741036,
            integrity_score=0.9305362023265322,
            config={
                "composition": "strict_multiplicative_v2",
                "weights": _scoring_config()["weights"],
                "case_weights": {"physics_state": 0.75, "subject": 0.25},
            },
        )

        self.assertEqual("strict_multiplicative_v2", composed["composition"])
        self.assertLess(composed["motion_law_gate"], 0.25)
        self.assertLess(composed["physics_score"], 0.16)
        self.assertLess(composed["score"], 0.15)

    def test_strict_composition_preserves_all_one_exact_control(self) -> None:
        composed = _compose_parabolic_scores(
            {name: 1.0 for name in self._false_high_components()},
            subject_score=1.0,
            integrity_score=1.0,
            config={"composition": "strict_multiplicative_v2"},
        )

        self.assertEqual(1.0, composed["motion_law_gate"])
        self.assertEqual(1.0, composed["physics_score"])
        self.assertEqual(1.0, composed["score"])

    def test_default_composition_retains_v1_arithmetic_score(self) -> None:
        composed = _compose_parabolic_scores(
            self._false_high_components(),
            subject_score=0.8143368124741036,
            integrity_score=0.9305362023265322,
            config={
                "weights": _scoring_config()["weights"],
                "case_weights": {"physics_state": 0.75, "subject": 0.25},
            },
        )

        self.assertEqual("weighted_arithmetic_v1", composed["composition"])
        self.assertIsNone(composed["motion_law_gate"])
        self.assertAlmostEqual(0.5656459893608466, composed["score"], places=12)

    @staticmethod
    def _candidate(
        xy: tuple[float, float],
        *,
        shape: tuple[int, int] = (160, 240),
    ) -> BallCandidate:
        mask = np.zeros(shape, dtype=np.uint8)
        center = tuple(np.rint(xy).astype(int).tolist())
        cv2.circle(mask, center, 8, 255, -1)
        histogram = np.zeros(24, dtype=np.float64)
        histogram[0] = 1.0
        return BallCandidate(
            xy=np.asarray(xy, dtype=np.float64),
            radius=8.0,
            mask=mask,
            histogram=histogram,
            contrast=30.0,
            source="fixture",
        )

    def test_ambiguous_assignment_terminates_identity_without_reacquisition(
        self,
    ) -> None:
        frames = [
            np.full((160, 240, 3), 224, dtype=np.uint8)
            for _ in range(4)
        ]
        seed = self._candidate((120.0, 35.0))
        candidates = [
            [self._candidate((108.0, 48.0)), self._candidate((132.0, 48.0))],
            [self._candidate((96.0, 65.0))],
            [self._candidate((84.0, 84.0))],
        ]
        config = {
            **_observation_config(),
            "minimum_assignment_cost_margin": 0.15,
            "latch_identity_loss": True,
        }

        with (
            patch(
                "physbench.evaluation.scenes.parabolic_motion.evaluator."
                "_hough_candidates",
                side_effect=candidates,
            ),
            patch(
                "physbench.evaluation.scenes.parabolic_motion.evaluator."
                "_foreground_candidates",
                return_value=[],
            ),
        ):
            observation = observe_projectile(
                frames,
                available=np.ones(len(frames), dtype=bool),
                config=config,
                seed_override=seed,
                seed_provenance={"policy": "fixture_anchor"},
            )

        self.assertEqual([True, False, False, False], observation.observed.tolist())
        self.assertEqual("terminated_uncertain", observation.diagnostics["identity_state"])
        self.assertEqual(1, observation.diagnostics["identity_termination_frame"])
        self.assertEqual(
            "ambiguous_assignment",
            observation.diagnostics["identity_termination_reason"],
        )
        self.assertAlmostEqual(
            0.0, observation.diagnostics["assignment_margins"][1]
        )
        self.assertEqual(
            {"policy": "fixture_anchor"},
            observation.diagnostics["seed_provenance"],
        )

    def test_unique_assignments_keep_anchored_identity_active(self) -> None:
        frames = [
            np.full((160, 240, 3), 224, dtype=np.uint8)
            for _ in range(4)
        ]
        seed = self._candidate((120.0, 35.0))
        candidates = [
            [self._candidate((108.0, 48.0))],
            [self._candidate((96.0, 65.0))],
            [self._candidate((84.0, 84.0))],
        ]
        config = {
            **_observation_config(),
            "minimum_assignment_cost_margin": 0.15,
            "latch_identity_loss": True,
        }

        with (
            patch(
                "physbench.evaluation.scenes.parabolic_motion.evaluator."
                "_hough_candidates",
                side_effect=candidates,
            ),
            patch(
                "physbench.evaluation.scenes.parabolic_motion.evaluator."
                "_foreground_candidates",
                return_value=[],
            ),
        ):
            observation = observe_projectile(
                frames,
                available=np.ones(len(frames), dtype=bool),
                config=config,
                seed_override=seed,
            )

        self.assertEqual([True, True, True, True], observation.observed.tolist())
        self.assertEqual("active", observation.diagnostics["identity_state"])
        self.assertIsNone(observation.diagnostics["identity_termination_frame"])

    def test_identical_projectile_analysis_exposes_unit_csti(self) -> None:
        protocol = load_evaluation_protocol("scene_default_v10")
        config = copy.deepcopy(protocol["scenes"]["parabolic_motion"])
        config["general_metrics"] = {"csti": _CSTI_MAPPING}
        evaluator = ParabolicMotionCaseEvaluator(config)
        case = _case()
        manifest = materialize_entity_manifest(case)
        frames = _frames()
        times_s = (np.arange(len(frames)) / 24.0).tolist()
        video = SimpleNamespace(
            frames=frames,
            available=np.ones(len(frames), dtype=bool),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = CaseEvaluationRequest(
                job={"job_id": "parabolic_csti_unit"},
                case=case,
                case_catalog={case["case_id"]: case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config=config,
            )
            with (
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.write_json"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.write_parabolic_visualization",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=times_s,
                    reference_video=video,
                    prediction_video=video,
                )

        self.assertIsNotNone(analysis.csti_input)
        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=tuple(
                (entity.entity_id, entity.role_id)
                for entity in manifest.entities
            ),
            config=_CSTI_CONFIG,
        )
        self.assertAlmostEqual(1.0, metric["score"], places=12)
        self.assertTrue(metric["objects"][0]["matched"])
        self.assertEqual(
            ["bound_projectile"],
            metric["objects"][0]["matched_prediction_track_ids"],
        )

    def test_rejected_projectile_binding_is_csti_unmatched_zero(self) -> None:
        protocol = load_evaluation_protocol("scene_default_v10")
        config = copy.deepcopy(protocol["scenes"]["parabolic_motion"])
        config["general_metrics"] = {"csti": _CSTI_MAPPING}
        evaluator = ParabolicMotionCaseEvaluator(config)
        case = _case()
        manifest = materialize_entity_manifest(case)
        reference_frames = _frames()
        prediction_frames = [np.full_like(frame, 224) for frame in reference_frames]
        times_s = (np.arange(len(reference_frames)) / 24.0).tolist()
        reference = SimpleNamespace(
            frames=reference_frames,
            available=np.ones(len(reference_frames), dtype=bool),
        )
        prediction = SimpleNamespace(
            frames=prediction_frames,
            available=np.ones(len(prediction_frames), dtype=bool),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = CaseEvaluationRequest(
                job={"job_id": "parabolic_csti_unmatched_unit"},
                case=case,
                case_catalog={case["case_id"]: case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config=config,
            )
            with (
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.write_json"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.parabolic_motion."
                    "evaluator.write_parabolic_visualization",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=times_s,
                    reference_video=reference,
                    prediction_video=prediction,
                )

        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=tuple(
                (entity.entity_id, entity.role_id)
                for entity in manifest.entities
            ),
            config=_CSTI_CONFIG,
        )
        self.assertEqual(0.0, metric["score"])
        self.assertFalse(metric["objects"][0]["matched"])
        self.assertIsNone(metric["objects"][0]["diagnostic_prefix_curve"])

    def test_unified_parabolic_overlay_is_run_owned_and_manifested(self) -> None:
        frames = _frames()
        observation = observe_projectile(
            frames,
            available=np.ones(len(frames), dtype=bool),
            config=_observation_config(),
        )
        times_s = (np.arange(len(frames)) / 24.0).tolist()
        scored = score_parabolic_observations(
            observation,
            observation,
            times_s=times_s,
            frame_shape=frames[0].shape[:2],
            available=np.ones(len(frames), dtype=bool),
            config=_scoring_config(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local",
                case={"case_id": "projectile", "scene_id": "parabolic_motion"},
                job={"job_id": "projectile_job"},
                evaluator_config={"type": "parabolic_motion_state_v1"},
                prediction={"video_sha256": "2" * 64},
                run_id="baseline_parabolic_run",
                save_visualizations=True,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            artifacts = write_parabolic_visualization(
                request,
                config={
                    "enabled": True,
                    "mode": "all",
                    "layout": "quad",
                    "panel_width": 128,
                    "panel_height": 96,
                    "fps": 12.0,
                },
                times_s=times_s,
                reference_frames=frames,
                prediction_frames=frames,
                entity=EntitySpec(
                    entity_id="projectile_ball",
                    role_id="projectile",
                    entity_class="ball",
                    lifecycle=LifecyclePolicy.MAY_EXIT,
                ),
                capability=ReferenceCapability.SAME_CASE_GT,
                reference=observation,
                prediction=observation,
                scored=scored,
                prediction_available=[True] * len(frames),
                reference_role="REFERENCE",
                score_summary={"score": 1.0, "components": {}},
                has_issues=False,
            )
            manifest_path = Path(
                artifacts["open_world_v2_artifact_manifest"]
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual("complete", manifest["status"])
            self.assertIn(
                "parabolic_motion/projectile",
                manifest["visualization_relative_directory"],
            )
            self.assertTrue(
                Path(manifest["files"]["overlay_video"]["path"]).is_file()
            )

    def test_manifest_guided_observer_tracks_synthetic_projectile(self) -> None:
        frames = _frames()
        observation = observe_projectile(
            frames,
            available=np.ones(len(frames), dtype=bool),
            config=_observation_config(),
        )
        self.assertIsNotNone(observation.seed)
        self.assertGreaterEqual(observation.observed_count, 9)
        valid = observation.xy[observation.observed]
        self.assertLess(valid[-1, 0], valid[0, 0])
        self.assertGreater(valid[-1, 1], valid[0, 1])
        self.assertNotIn("identity_state", observation.diagnostics)
        self.assertNotIn("assignment_margins", observation.diagnostics)

    def test_identical_observation_scores_exactly_one(self) -> None:
        frames = _frames()
        observation = observe_projectile(
            frames,
            available=np.ones(len(frames), dtype=bool),
            config=_observation_config(),
        )
        score = score_parabolic_observations(
            observation,
            observation,
            times_s=(np.arange(len(frames)) / 24.0).tolist(),
            frame_shape=frames[0].shape[:2],
            available=np.ones(len(frames), dtype=bool),
            config=_scoring_config(),
        )
        self.assertEqual(1.0, score["score"])
        self.assertEqual(1.0, score["physics_score"])
        self.assertEqual(1.0, score["subject_score"])
        self.assertEqual(1.0, score["integrity_score"])

    def test_shifted_trajectory_is_penalized_without_error(self) -> None:
        reference_frames = _frames()
        prediction_frames = _frames(shift_xy=(-18, 8))
        reference = observe_projectile(
            reference_frames,
            available=np.ones(len(reference_frames), dtype=bool),
            config=_observation_config(),
        )
        prediction = observe_projectile(
            prediction_frames,
            available=np.ones(len(prediction_frames), dtype=bool),
            config=_observation_config(),
        )
        score = score_parabolic_observations(
            reference,
            prediction,
            times_s=(np.arange(len(reference_frames)) / 24.0).tolist(),
            frame_shape=reference_frames[0].shape[:2],
            available=np.ones(len(reference_frames), dtype=bool),
            config=_scoring_config(),
        )
        self.assertGreaterEqual(score["score"], 0.0)
        self.assertLess(score["score"], 1.0)

    def test_missing_prediction_record_is_fail_closed_evaluated_zero(self) -> None:
        protocol = load_evaluation_protocol("scene_default_v8")
        evaluator = SceneEvaluatorRegistry(protocol).resolve(
            "parabolic_motion"
        )
        case = {
            "case_id": "synthetic_parabolic",
            "scene_id": "parabolic_motion",
            "physics": {
                "ball_mass": {"value": 0.014, "unit": "kg", "annotated": True},
                "ball_radius": {"value": 0.0075, "unit": "m", "annotated": True},
                "initial_horizontal_velocity": {
                    "value": 1.0,
                    "unit": "m/s",
                    "annotated": True,
                },
                "launch_height": {"value": 0.77, "unit": "m", "annotated": True},
            },
            "appearance": {"ball_material": "steel", "ball_size_class": "small"},
            "assets": {},
            "has_real_reference_video": False,
            "provenance": {"parent_case_id": None},
        }
        with tempfile.TemporaryDirectory() as temporary:
            request = CaseEvaluationRequest(
                job={
                    "job_id": "missing_prediction",
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                },
                case=case,
                case_catalog={case["case_id"]: case},
                prediction=None,
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=protocol["scenes"]["parabolic_motion"],
            )
            result = evaluator.evaluate(request)
        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual("prediction_record_missing", result.reason_code)

    def test_protocol_registry_routes_parabolic_scene(self) -> None:
        protocol = load_evaluation_protocol("scene_default_v8")
        evaluator = SceneEvaluatorRegistry(protocol).resolve(
            "parabolic_motion"
        )
        self.assertEqual("parabolic_motion", evaluator.scene_id)
        self.assertEqual("parabolic_motion_state", evaluator.evaluator_id)
        self.assertIn(
            "parabolic_motion_state_v1",
            SceneEvaluatorRegistry.supported_evaluator_types(),
        )


if __name__ == "__main__":
    unittest.main()
