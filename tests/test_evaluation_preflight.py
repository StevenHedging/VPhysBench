from __future__ import annotations

import math
import unittest

from physbench.evaluation.preflight import validate_preflight_results


class EvaluationPreflightValidationTests(unittest.TestCase):
    @staticmethod
    def _plan() -> dict[str, object]:
        return {
            "task_id": "fixture",
            "jobs": [
                {
                    "job_id": "fixture__case_a__seed000042",
                    "case_id": "case_a",
                    "scene_id": "pendulum",
                    "evaluation_partition": "test",
                    "seed": 42,
                }
            ],
        }

    @staticmethod
    def _result() -> dict[str, object]:
        return {
            "job_id": "fixture__case_a__seed000042",
            "case_id": "case_a",
            "scene_id": "pendulum",
            "status": "evaluated",
            "score": 0.75,
            "metrics": {
                "csti": {
                    "status": "evaluated",
                    "score": 0.80,
                }
            },
        }

    def test_complete_finite_result_passes(self) -> None:
        summary = validate_preflight_results(
            plan=self._plan(),
            case_results=[self._result()],
        )

        self.assertEqual("complete", summary["status"])
        self.assertEqual(1, summary["expected_jobs"])
        self.assertEqual(1, summary["evaluated_jobs"])
        self.assertEqual([], summary["issues"])

    def test_non_evaluated_case_is_rejected(self) -> None:
        result = self._result()
        result.update(status="unavailable", score=None)

        summary = validate_preflight_results(
            plan=self._plan(),
            case_results=[result],
        )

        self.assertEqual("failed", summary["status"])
        self.assertEqual("case_not_evaluated", summary["issues"][0]["code"])

    def test_nonfinite_primary_or_csti_score_is_rejected(self) -> None:
        for field in ("primary", "csti"):
            with self.subTest(field=field):
                result = self._result()
                if field == "primary":
                    result["score"] = math.nan
                else:
                    result["metrics"]["csti"]["score"] = math.inf

                summary = validate_preflight_results(
                    plan=self._plan(),
                    case_results=[result],
                )

                self.assertEqual("failed", summary["status"])
                self.assertIn(
                    summary["issues"][0]["code"],
                    {"primary_score_invalid", "csti_score_invalid"},
                )

    def test_not_applicable_csti_with_null_score_passes(self) -> None:
        result = self._result()
        result["metrics"]["csti"] = {
            "status": "not_applicable",
            "score": None,
        }

        summary = validate_preflight_results(
            plan=self._plan(),
            case_results=[result],
        )

        self.assertEqual("complete", summary["status"])

    def test_missing_and_duplicate_jobs_are_rejected(self) -> None:
        result = self._result()
        duplicate_summary = validate_preflight_results(
            plan=self._plan(),
            case_results=[result, dict(result)],
        )
        missing_summary = validate_preflight_results(
            plan=self._plan(),
            case_results=[],
        )

        self.assertEqual("duplicate_result", duplicate_summary["issues"][0]["code"])
        self.assertEqual("missing_result", missing_summary["issues"][0]["code"])


if __name__ == "__main__":
    unittest.main()
