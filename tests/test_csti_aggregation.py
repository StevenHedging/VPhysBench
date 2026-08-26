from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from physbench.evaluation.task_evaluator import (
    aggregate_task_results,
    evaluate_task,
)


FIXED_CSTI = {
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


def plan_for_scenes(*scene_ids: str, family: str = "direct_eval") -> dict:
    return {
        "task_id": "csti_test",
        "family": family,
        "scene_ids": list(scene_ids),
    }


def case(
    job_id: str,
    scene_id: str,
    *,
    expert: float | None,
    csti: float | None,
    status: str = "evaluated",
    csti_status: str = "evaluated",
    partition: str = "test",
    init_failure_reason: str | None = None,
) -> dict:
    metric = {
        "status": csti_status,
        "score": csti,
    }
    if csti_status == "evaluator_init_failure":
        metric.update(
            {
                "evaluator_init_success": False,
                "evaluator_init_failure_reason": {
                    "code": init_failure_reason or "insufficient_detections",
                    "message": "synthetic initialization failure",
                    "details": {},
                },
            }
        )
    elif csti_status == "evaluated":
        metric["evaluator_init_success"] = True
    return {
        "job_id": job_id,
        "case_id": job_id,
        "scene_id": scene_id,
        "evaluation_partition": partition,
        "status": status,
        "score": expert,
        "metrics": {
            "csti": metric
        },
        "quality": {},
        "reason_code": None,
    }


class CSTIAggregationTest(unittest.TestCase):
    def test_expert_and_csti_are_independent_scene_macro_means(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("pendulum", "collision_1d"),
            case_results=[
                case("p1", "pendulum", expert=0.8, csti=0.2),
                case("c1", "collision_1d", expert=0.4, csti=0.8),
                case("c2", "collision_1d", expert=0.6, csti=1.0),
            ],
            general_metrics={"csti": FIXED_CSTI},
        )

        self.assertAlmostEqual(0.65, aggregation["score"])
        self.assertAlmostEqual(0.65, aggregation["dimensions"]["expert"]["score"])
        self.assertAlmostEqual(
            2.0 / 3.0, aggregation["dimensions"]["csti"]["score"]
        )
        self.assertAlmostEqual(0.2, aggregation["dimensions"]["csti"]["by_scene"]["pendulum"]["score"])
        self.assertAlmostEqual(0.9, aggregation["dimensions"]["csti"]["by_scene"]["collision_1d"]["score"])

    def test_init_failures_are_excluded_from_score_and_reported(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("collision_1d"),
            case_results=[
                case("good", "collision_1d", expert=0.7, csti=0.8),
                case(
                    "init_failed",
                    "collision_1d",
                    expert=0.6,
                    csti=None,
                    csti_status="evaluator_init_failure",
                    init_failure_reason="initial_match_below_threshold",
                ),
            ],
            general_metrics={"csti": FIXED_CSTI},
        )
        csti = aggregation["dimensions"]["csti"]

        self.assertEqual("complete", csti["status"])
        self.assertAlmostEqual(0.8, csti["score"])
        self.assertEqual(1, csti["valid_video_count"])
        self.assertEqual(1, csti["evaluator_init_failure_video_count"])
        self.assertAlmostEqual(0.5, csti["init_coverage"])
        self.assertEqual(
            {"initial_match_below_threshold": 1},
            csti["evaluator_init_failure_reason_counts"],
        )
        self.assertEqual(2, csti["expected_jobs"])
        self.assertEqual(1, csti["evaluated_jobs"])
        self.assertEqual(1.0, csti["coverage"])

    def test_all_initialization_failures_have_visible_null_score(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("collision_1d"),
            case_results=[
                case(
                    "failed",
                    "collision_1d",
                    expert=0.5,
                    csti=None,
                    csti_status="evaluator_init_failure",
                )
            ],
            general_metrics={"csti": FIXED_CSTI},
        )
        csti = aggregation["dimensions"]["csti"]

        self.assertEqual("evaluator_init_failure", csti["status"])
        self.assertIsNone(csti["score"])
        self.assertEqual(0.0, csti["init_coverage"])
        self.assertEqual(0, csti["valid_video_count"])
        self.assertEqual(1, csti["evaluator_init_failure_video_count"])

    def test_incomplete_coverage_has_null_strict_and_finite_observed_means(self) -> None:
        failed = case("c1", "collision_1d", expert=None, csti=None, status="error")
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("pendulum", "collision_1d"),
            case_results=[
                case("p1", "pendulum", expert=0.8, csti=0.2),
                failed,
            ],
            general_metrics={"csti": FIXED_CSTI},
        )

        self.assertIsNone(aggregation["score"])
        self.assertAlmostEqual(0.8, aggregation["observed_mean_score"])
        self.assertIsNone(aggregation["dimensions"]["csti"]["score"])
        self.assertAlmostEqual(0.2, aggregation["dimensions"]["csti"]["observed_mean_score"])

    def test_finetune_dimension_macro_averages_required_partitions(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("pendulum", family="finetune_eval"),
            case_results=[
                case("id", "pendulum", expert=0.4, csti=0.2, partition="test_id"),
                case("ood", "pendulum", expert=0.8, csti=1.0, partition="test_ood"),
            ],
            general_metrics={"csti": FIXED_CSTI},
        )

        self.assertAlmostEqual(0.6, aggregation["score"])
        self.assertAlmostEqual(0.6, aggregation["dimensions"]["csti"]["score"])
        self.assertEqual(
            "macro_mean_required_partitions",
            aggregation["dimensions"]["csti"]["by_scene"]["pendulum"]["aggregation_policy"],
        )

    def test_not_applicable_cases_are_excluded_from_csti_denominators(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("pendulum"),
            case_results=[
                case("same", "pendulum", expert=0.7, csti=0.6),
                case(
                    "parent",
                    "pendulum",
                    expert=0.9,
                    csti=None,
                    csti_status="not_applicable",
                ),
            ],
            general_metrics={"csti": FIXED_CSTI},
        )
        csti = aggregation["dimensions"]["csti"]

        self.assertEqual("complete", csti["status"])
        self.assertEqual(1, csti["expected_jobs"])
        self.assertEqual(1, csti["evaluated_jobs"])
        self.assertEqual(1, csti["not_applicable_jobs"])
        self.assertEqual(1.0, csti["coverage"])
        self.assertAlmostEqual(0.6, csti["score"])

    def test_all_not_applicable_dimension_has_explicit_status(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("pendulum", "collision_1d"),
            case_results=[
                case("p", "pendulum", expert=0.7, csti=None, csti_status="not_applicable"),
                case("c", "collision_1d", expert=0.9, csti=None, csti_status="not_applicable"),
            ],
            general_metrics={"csti": FIXED_CSTI},
        )
        csti = aggregation["dimensions"]["csti"]

        self.assertEqual("not_applicable", csti["status"])
        self.assertIsNone(csti["score"])
        self.assertEqual(0, csti["expected_jobs"])
        self.assertEqual(2, csti["not_applicable_jobs"])
        self.assertTrue(
            all(value["status"] == "not_applicable" for value in csti["by_scene"].values())
        )

    def test_v10_call_retains_the_exact_legacy_key_set(self) -> None:
        aggregation = aggregate_task_results(
            plan=plan_for_scenes("pendulum"),
            case_results=[case("p", "pendulum", expert=0.8, csti=0.2)],
        )

        self.assertEqual(
            {
                "status",
                "expected_jobs",
                "evaluated_jobs",
                "coverage",
                "status_counts",
                "score",
                "observed_mean_score",
                "aggregation_policy",
                "by_scene",
                "breakdown",
            },
            set(aggregation),
        )

    def test_prediction_preflight_zero_contains_manifest_complete_csti(self) -> None:
        plan = {
            "task_id": "csti_preflight",
            "family": "direct_eval",
            "scene_ids": ["pendulum"],
            "jobs": [
                {
                    "job_id": "missing_prediction",
                    "case_id": "pendulum_case",
                    "scene_id": "pendulum",
                    "evaluation_partition": "test",
                    "seed": 1,
                }
            ],
        }
        case_value = {
            "case_id": "pendulum_case",
            "scene_id": "pendulum",
            "physics": {},
            "entities": [
                {
                    "entity_id": "bob",
                    "role_id": "moving_bob",
                    "entity_class": "pendulum_bob",
                    "physical_attributes": {},
                    "condition_anchor": {},
                    "lifecycle": "persistent",
                }
            ],
            "apparatus": [],
            "assets": {"reference_video": "reference.mp4"},
        }
        protocol = {
            "protocol_id": "scene_default_v1",
            "fingerprint": "f" * 64,
            "path": "scene_default_v1.json",
            "scenes": {"pendulum": {"type": "pendulum_v1"}},
            "robustness": {
                "prediction_record_failure_policy": "evaluated_zero"
            },
            "general_metrics": {"csti": FIXED_CSTI},
        }

        with tempfile.TemporaryDirectory() as temporary:
            results, task_result = evaluate_task(
                plan=plan,
                cases=[case_value],
                predictions=[],
                asset_root=temporary,
                protocol=protocol,
                output_dir=Path(temporary) / "evaluation",
            )

        self.assertEqual("evaluated", results[0]["status"])
        self.assertEqual(0.0, results[0]["score"])
        self.assertEqual(0.0, results[0]["metrics"]["csti"]["score"])
        self.assertEqual(
            ["bob"],
            [item["entity_id"] for item in results[0]["metrics"]["csti"]["objects"]],
        )
        self.assertEqual(0.0, task_result["dimensions"]["csti"]["score"])


if __name__ == "__main__":
    unittest.main()
