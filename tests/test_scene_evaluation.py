from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physbench.evaluation.contracts import CaseEvaluationResult
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.task_evaluator import evaluate_task

try:
    import numpy as np

    from physbench.evaluation.common.geometry import fit_axis, fit_circle
    from physbench.evaluation.common.tracking import CentroidTrace, InstanceTracks
    from physbench.evaluation.scenes.circular_motion.scoring import (
        extract_orbit_traces,
        score_orbits,
    )
    from physbench.evaluation.scenes.collision.scoring import (
        extract_collision_trace,
        score_collision,
    )
    from physbench.evaluation.scenes.free_fall.scoring import (
        extract_free_fall_trace,
        score_free_fall,
    )
    from physbench.evaluation.scenes.inclined_plane.scoring import (
        extract_incline_trace,
        score_incline,
    )
    from physbench.evaluation.scenes.pendulum.scoring import (
        extract_trace,
        mask_iou,
        save_iou_curve,
        score_traces,
    )
except ModuleNotFoundError as exc:
    np = None
    _EVALUATION_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _EVALUATION_IMPORT_ERROR = None


class _FakeEvaluator:
    evaluator_id = "fake"
    evaluator_version = "1.0"

    def __init__(self, scene_id: str):
        self.scene_id = scene_id

    def describe(self):
        return {
            "id": self.evaluator_id,
            "version": self.evaluator_version,
            "scene_id": self.scene_id,
        }

    def evaluate(self, request):
        if request.prediction is None:
            return CaseEvaluationResult(
                request.job["job_id"],
                request.case["case_id"],
                self.scene_id,
                self.describe(),
                "unavailable",
                None,
                "prediction_record_missing",
            )
        return CaseEvaluationResult(
            request.job["job_id"],
            request.case["case_id"],
            self.scene_id,
            self.describe(),
            "evaluated",
            float(request.prediction["test_score"]),
        )


class _FakeRegistry:
    def resolve(self, scene_id):
        return _FakeEvaluator(scene_id)

    def describe(self):
        return {"pendulum": _FakeEvaluator("pendulum").describe()}


@unittest.skipIf(
    _EVALUATION_IMPORT_ERROR is not None,
    f"scene-evaluation extras unavailable: {_EVALUATION_IMPORT_ERROR}",
)
class SceneEvaluationTests(unittest.TestCase):
    @staticmethod
    def _pendulum_masks() -> tuple[list[np.ndarray], list[float]]:
        times = (np.arange(81) / 16.0).tolist()
        masks = []
        for time_s in times:
            angle = 0.35 * np.cos(2 * np.pi * time_s)
            pivot_x, pivot_y, length = 48, 12, 48
            bob_x = int(round(pivot_x + length * np.sin(angle)))
            bob_y = int(round(pivot_y + length * np.cos(angle)))
            mask = np.zeros((80, 96), np.uint8)
            count = max(abs(bob_x - pivot_x), abs(bob_y - pivot_y), 1)
            xs = np.rint(np.linspace(pivot_x, bob_x, count + 1)).astype(int)
            ys = np.rint(np.linspace(pivot_y, bob_y, count + 1)).astype(int)
            mask[ys, xs] = 255
            yy, xx = np.ogrid[:80, :96]
            mask[(xx - bob_x) ** 2 + (yy - bob_y) ** 2 <= 5**2] = 255
            masks.append(mask)
        return masks, times

    def test_identical_pendulum_trace_scores_one(self) -> None:
        masks, times = self._pendulum_masks()
        quality = {
            "minimum_mask_pixels": 10,
            "minimum_mask_area_ratio": 0.0,
            "maximum_mask_area_ratio": 0.5,
            "minimum_valid_frame_ratio": 0.9,
        }
        period = {"minimum_s": 0.5, "maximum_s": 2.0}
        trace = extract_trace(
            masks, times, quality_config=quality, period_config=period
        )
        result = score_traces(
            trace,
            trace,
            scoring_config={
                "minimum_angle_scale_deg": 5.0,
                "pivot_drift_scale": 0.03,
                "length_cv_scale": 0.05,
                "weights": {
                    "angle_trajectory": 0.6,
                    "period": 0.2,
                    "amplitude": 0.1,
                    "structural_consistency": 0.1,
                },
            },
        )
        self.assertGreater(result["score"], 0.95)
        self.assertAlmostEqual(
            1.0, result["components"]["angle_trajectory"]
        )
        self.assertAlmostEqual(1.0, result["components"]["period"])
        self.assertAlmostEqual(1.0, result["components"]["amplitude"])
        self.assertIsNotNone(trace.period_s)

    def test_empty_masks_do_not_receive_perfect_iou(self) -> None:
        empty = np.zeros((8, 8), np.uint8)
        self.assertEqual(0.0, mask_iou(empty, empty))

    def test_jensen_style_iou_curve_is_written(self) -> None:
        _, times = self._pendulum_masks()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "iou.png"
            save_iou_curve(
                path,
                times_s=times,
                ious=[0.5] * len(times),
                case_id="pendulum_test",
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    @staticmethod
    def _free_fall_trace(*, upward: bool = False):
        times = np.arange(17, dtype=np.float64) / 32.0
        direction = -1.0 if upward else 1.0
        xy = np.column_stack(
            [
                np.full(len(times), 40.0),
                10.0 + direction * 200.0 * np.square(times),
            ]
        )
        centroid = CentroidTrace(
            xy=xy,
            area=np.full(len(times), 50.0),
            valid=np.ones(len(times), dtype=bool),
            valid_ratio=1.0,
        )
        return extract_free_fall_trace(
            centroid,
            times.tolist(),
            minimum_vertical_span_px=5.0,
        )

    def test_identical_free_fall_trace_scores_one(self) -> None:
        trace = self._free_fall_trace()
        result = score_free_fall(
            trace,
            trace,
            config={
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
        )
        self.assertAlmostEqual(1.0, result["score"])

    def test_upward_motion_is_penalized_as_non_free_fall(self) -> None:
        reference = self._free_fall_trace()
        upward = self._free_fall_trace(upward=True)
        result = score_free_fall(
            reference,
            upward,
            config={
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
        )
        self.assertLess(result["score"], 0.35)

    def test_common_axis_and_circle_geometry(self) -> None:
        axis = fit_axis(np.asarray([[0, 0], [1, 2], [2, 4], [3, 6]]))
        along, cross = axis.project(np.asarray([[0, 0], [3, 6]]))
        self.assertGreater(axis.explained_ratio, 0.999)
        self.assertLess(float(np.ptp(cross)), 1e-9)
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        points = np.column_stack(
            [4.0 + 3.0 * np.cos(angles), -2.0 + 3.0 * np.sin(angles)]
        )
        circle = fit_circle(points)
        np.testing.assert_allclose(circle.center_xy, [4.0, -2.0], atol=1e-9)
        self.assertAlmostEqual(3.0, circle.radius_px)

    def test_identical_incline_trace_scores_one(self) -> None:
        times = np.arange(33, dtype=np.float64) / 16.0
        progress = 25.0 * np.square(times)
        xy = np.column_stack([30.0 + progress, 25.0 + 0.5 * progress])
        masks = []
        for x, y in xy:
            mask = np.zeros((180, 220), np.uint8)
            x0, y0 = int(round(x)), int(round(y))
            mask[max(0, y0 - 4) : y0 + 5, max(0, x0 - 6) : x0 + 7] = 255
            masks.append(mask)
        centroid = CentroidTrace(
            xy=xy,
            area=np.full(len(times), 117.0),
            valid=np.ones(len(times), dtype=bool),
            valid_ratio=1.0,
        )
        trace = extract_incline_trace(
            centroid,
            masks,
            times.tolist(),
            minimum_span_px=10.0,
        )
        result = score_incline(
            trace,
            trace,
            config={
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
            },
        )
        self.assertGreater(result["score"], 0.99)
        self.assertGreater(trace.axis.explained_ratio, 0.999)

    def test_circular_motion_uses_relative_angle_and_scores_identity(self) -> None:
        times = np.arange(41, dtype=np.float64) / 8.0
        angle = 0.7 + 0.95 * times
        xy = np.column_stack(
            [100.0 + 35.0 * np.cos(angle), 80.0 + 35.0 * np.sin(angle)]
        )[:, None, :]
        masks = [[np.zeros((160, 200), np.uint8) for _ in times]]
        tracks = InstanceTracks(
            xy=xy,
            valid=np.ones((len(times), 1), dtype=bool),
            valid_ratio=np.ones(1),
            instance_masks=masks,
            union_masks=[np.zeros((160, 200), np.uint8) for _ in times],
        )
        trace = extract_orbit_traces(tracks, times.tolist())
        result = score_orbits(
            trace,
            trace,
            config={
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
        )
        self.assertGreater(result["score"], 0.999)
        self.assertAlmostEqual(0.95, trace[0].angular_velocity_rad_s, places=3)
        self.assertAlmostEqual(0.0, trace[0].relative_angle_rad[0])

    def test_identical_collision_trace_scores_one(self) -> None:
        times = np.arange(33, dtype=np.float64) / 16.0
        before = np.minimum(times, 1.0)
        after = np.maximum(times - 1.0, 0.0)
        scalar = np.column_stack(
            [
                10.0 + 40.0 * before,
                np.full(len(times), 55.0),
                65.0 + 40.0 * after,
            ]
        )
        xy = np.stack([scalar, np.full_like(scalar, 30.0)], axis=2)
        valid = np.ones((len(times), 3), dtype=bool)
        trace = extract_collision_trace(
            xy,
            valid,
            times.tolist(),
            masses_kg=np.ones(3),
            minimum_span_px=10.0,
            velocity_window_fraction=0.2,
        )
        result = score_collision(
            trace,
            trace,
            config={
                "trajectory_error_scale": 0.12,
                "event_time_error_scale": 0.1,
                "velocity_error_scale": 0.25,
                "momentum_residual_scale": 0.2,
                "restitution_error_scale": 0.25,
                "cross_track_scale": 0.03,
                "weights": {
                    "instance_trajectories": 0.45,
                    "contact_event_time": 0.15,
                    "pre_post_velocities": 0.2,
                    "collision_physics": 0.15,
                    "one_dimensional_constraint": 0.05,
                },
            },
        )
        self.assertGreater(result["score"], 0.999)
        self.assertAlmostEqual(1.0, trace.effective_restitution, places=6)

    def test_task_evaluator_uses_frozen_jobs_as_primary_table(self) -> None:
        plan = {
            "task_id": "task",
            "family": "direct_eval",
            "conditioning": "generic",
            "scene_ids": ["pendulum"],
            "jobs": [
                {
                    "job_id": "job-a",
                    "case_id": "case-a",
                    "scene_id": "pendulum",
                    "evaluation_partition": "group_1",
                    "conditioning": "generic",
                    "seed": 42,
                },
                {
                    "job_id": "job-b",
                    "case_id": "case-b",
                    "scene_id": "pendulum",
                    "evaluation_partition": "group_1",
                    "conditioning": "generic",
                    "seed": 42,
                },
            ],
        }
        cases = [
            {"case_id": "case-a", "scene_id": "pendulum"},
            {"case_id": "case-b", "scene_id": "pendulum"},
        ]
        protocol = {
            "protocol_id": "test",
            "fingerprint": "f" * 64,
            "path": "/test/protocol.json",
            "scenes": {"pendulum": {"type": "fake"}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            results, summary = evaluate_task(
                plan=plan,
                cases=cases,
                predictions=[
                    {
                        "job_id": "job-a",
                        "status": "complete",
                        "test_score": 0.8,
                    }
                ],
                asset_root=temporary,
                protocol=protocol,
                output_dir=Path(temporary) / "evaluation",
                registry=_FakeRegistry(),
            )
            self.assertEqual(2, len(results))
            self.assertEqual(
                ["evaluated", "unavailable"],
                [item["status"] for item in results],
            )
            self.assertEqual(0.5, summary["coverage"])
            self.assertIsNone(summary["score"])
            self.assertAlmostEqual(0.8, summary["observed_mean_score"])

    def test_default_protocol_resolves_all_five_scene_evaluators(self) -> None:
        protocol = load_evaluation_protocol("scene_default_v1")
        registry = SceneEvaluatorRegistry(protocol)
        expected = {
            "pendulum": "pendulum_state_similarity",
            "free_fall": "free_fall_state_similarity",
            "inclined_plane_slide": "inclined_plane_state_similarity",
            "uniform_circular_motion": (
                "uniform_circular_motion_state_similarity"
            ),
            "collision_1d": "collision_1d_state_similarity",
        }
        for scene_id, primary_score in expected.items():
            description = registry.resolve(scene_id).describe()
            self.assertTrue(description["implemented"])
            self.assertEqual(primary_score, description["primary_score"])


if __name__ == "__main__":
    unittest.main()
