from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from physbench.evaluation.contracts import CaseEvaluationResult
from physbench.evaluation.scenes.pendulum.scoring import (
    extract_trace,
    mask_iou,
    save_iou_curve,
    score_traces,
)
from physbench.evaluation.task_evaluator import evaluate_task


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


if __name__ == "__main__":
    unittest.main()
