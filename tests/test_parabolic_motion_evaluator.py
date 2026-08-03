from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.common.entities.contracts import (
    EntitySpec,
    LifecyclePolicy,
    ReferenceCapability,
)
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.parabolic_motion.evaluator import (
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


class ParabolicMotionEvaluatorTests(unittest.TestCase):
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
