from __future__ import annotations

import unittest

from _paths import FIXTURES, METRICS, SCENES
from physbench.io import load_json, load_jsonl
from physbench.metrics import evaluate_cases
from physbench.validation import load_scene_configs


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(FIXTURES / "cases.jsonl")
        cls.scenes = load_scene_configs(SCENES)
        cls.config = load_json(METRICS)

    def test_all_real_reference_dimensions_can_aggregate(self) -> None:
        prediction = [{
            "job_id": "job-id", "case_id": "pend_id_001", "baseline_id": "manual",
            "status": "complete", "video_path": "generated.mp4",
            "manual_scores": {"common_sense": 0.8, "prediction": 0.6, "visual_judgment": 0.9},
        }]
        results, summary = evaluate_cases(self.cases, prediction, self.scenes, self.config)
        self.assertAlmostEqual(0.74, results[0]["final_score"])
        self.assertEqual(1.0, results[0]["metric_coverage"])
        self.assertEqual(1, summary["scored_jobs"])

    def test_ood_without_real_continuation_excludes_visual(self) -> None:
        prediction = [{
            "job_id": "job-ood", "case_id": "pend_ood_bg_001", "baseline_id": "manual",
            "status": "complete", "video_path": "generated.mp4",
            "manual_scores": {"common_sense": 0.8, "prediction": 0.6},
        }]
        results, _ = evaluate_cases(self.cases, prediction, self.scenes, self.config)
        result = results[0]
        self.assertEqual("not_applicable", result["metrics"]["visual_judgment"]["status"])
        self.assertAlmostEqual((0.8 * 0.25 + 0.6 * 0.45) / 0.70, result["final_score"])

    def test_missing_plugins_do_not_become_zero(self) -> None:
        prediction = [{
            "job_id": "job-placeholder", "case_id": "pend_id_001", "baseline_id": "dummy",
            "status": "placeholder", "video_path": None, "manual_scores": {},
        }]
        results, _ = evaluate_cases(self.cases, prediction, self.scenes, self.config)
        self.assertIsNone(results[0]["final_score"])
        self.assertEqual(0.0, results[0]["metric_coverage"])

    def test_train_seen_does_not_change_official_id_ood_mean(self) -> None:
        predictions = [
            {
                "job_id": "official", "case_id": "pend_id_001", "baseline_id": "manual",
                "evaluation_partition": "test_id", "status": "complete", "video_path": "id.mp4",
                "manual_scores": {"common_sense": 0.2, "prediction": 0.2, "visual_judgment": 0.2},
            },
            {
                "job_id": "seen", "case_id": "pend_train_001", "baseline_id": "manual",
                "evaluation_partition": "train_seen", "status": "complete", "video_path": "seen.mp4",
                "manual_scores": {"common_sense": 1.0, "prediction": 1.0, "visual_judgment": 1.0},
            },
        ]
        _, summary = evaluate_cases(self.cases, predictions, self.scenes, self.config)
        self.assertEqual(1, summary["jobs"])
        self.assertEqual(2, summary["all_jobs"])
        self.assertEqual(1, summary["auxiliary_train_seen_jobs"])
        self.assertAlmostEqual(0.2, summary["mean_score"])
        self.assertAlmostEqual(1.0, summary["auxiliary_train_seen_mean_score"])

    def test_prompt_profiles_are_reported_separately(self) -> None:
        predictions = [
            {
                "job_id": "generic", "case_id": "pend_id_001", "baseline_id": "manual",
                "prompt_profile_id": "generic", "evaluation_partition": "test_id",
                "status": "complete", "video_path": "generic.mp4",
                "manual_scores": {
                    "common_sense": 0.2, "prediction": 0.2, "visual_judgment": 0.2
                },
            },
            {
                "job_id": "physics", "case_id": "pend_id_001", "baseline_id": "manual",
                "prompt_profile_id": "physics_natural", "evaluation_partition": "test_id",
                "status": "complete", "video_path": "physics.mp4",
                "manual_scores": {
                    "common_sense": 0.8, "prediction": 0.8, "visual_judgment": 0.8
                },
            },
        ]
        _, summary = evaluate_cases(self.cases, predictions, self.scenes, self.config)
        self.assertEqual(0.2, summary["prompt_breakdown"]["generic"]["mean_score"])
        self.assertEqual(
            0.8, summary["prompt_breakdown"]["physics_natural"]["mean_score"]
        )
        self.assertIn("pendulum/test_id/generic", summary["breakdown"])
        self.assertIn("pendulum/test_id/physics_natural", summary["breakdown"])


if __name__ == "__main__":
    unittest.main()
