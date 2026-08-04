from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from physbench.data_layout import LATEST_DATASET, V51_DATASET, V6_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.task_evaluator import aggregate_task_results
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/six_scene_finetune_eval.json"
DIRECT_TASK = ROOT / "tasks/official/six_scene_direct_eval.json"


class SixSceneDatasetV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v51 = load_dataset(V51_DATASET)
        cls.v6 = load_dataset(V6_DATASET, check_assets=True)
        cls.view = cls.v6.views["view_a"]

    def test_v6_is_metadata_only_latest_release(self) -> None:
        self.assertNotEqual(V6_DATASET, LATEST_DATASET)
        self.assertEqual("4.0", self.v6.descriptor["schema_version"])
        self.assertEqual("physics_video_six_scene_v6", self.v6.dataset_id)
        self.assertEqual("6.0.0", self.v6.descriptor["release"])
        self.assertEqual(604, len(self.v6.cases))
        self.assertEqual(
            {case["case_id"] for case in self.v51.cases},
            {case["case_id"] for case in self.v6.cases},
        )
        self.assertEqual(
            self.v51.asset_lock["files_digest"],
            self.v6.asset_lock["files_digest"],
        )
        for case in self.v6.cases:
            self.assertEqual("4.0", case["schema_version"])
            self.assertNotIn("ood", case)

    def test_view_a_has_only_complete_train_test_partitions(self) -> None:
        self.assertEqual("3.0", self.view["schema_version"])
        self.assertEqual("complete", self.view["coverage"])
        expected = {
            "collision_1d": (230, 100),
            "free_fall": (7, 4),
            "inclined_plane_slide": (58, 37),
            "parabolic_motion": (70, 27),
            "pendulum": (27, 8),
            "uniform_circular_motion": (18, 18),
        }
        all_ids = []
        test_ids = []
        for scene_id, (train_count, test_count) in expected.items():
            groups = self.view["scenes"][scene_id]
            self.assertEqual({"train", "test"}, set(groups))
            self.assertEqual(train_count, len(groups["train"]))
            self.assertEqual(test_count, len(groups["test"]))
            all_ids.extend(groups["train"])
            all_ids.extend(groups["test"])
            test_ids.extend(groups["test"])
        self.assertEqual(604, len(all_ids))
        self.assertEqual(604, len(set(all_ids)))
        self.assertEqual(set(test_ids), set(self.view["test_annotations"]))

    def test_generalization_annotations_are_relative_and_factorized(self) -> None:
        annotations = self.view["test_annotations"]
        self.assertEqual(
            {"id": 118, "ood": 70, "mixed": 6},
            dict(Counter(
                value["generalization_regime"]
                for value in annotations.values()
            )),
        )
        factor_counts = Counter(
            (factor["category"], factor["name"])
            for value in annotations.values()
            for factor in value["ood_factors"]
        )
        self.assertEqual(42, factor_counts[
            ("object_composition", "ball_spec_composition")
        ])
        self.assertEqual(34, factor_counts[
            ("interaction_structure", "collision_structure")
        ])
        self.assertEqual(22, factor_counts[("environment", "background")])
        self.assertEqual(8, factor_counts[
            ("object_composition", "ball_material")
        ])
        self.assertEqual(12, factor_counts[
            ("object_composition", "moving_object_composition")
        ])
        mixed = [
            value for value in annotations.values()
            if value["generalization_regime"] == "mixed"
        ]
        self.assertTrue(all(
            value["co_varying_factors"] == [{
                "name": "incline_angle",
                "category": "physical_parameter",
            }]
            for value in mixed
        ))

    def test_v4_tasks_plan_overall_test_and_preserve_direct_groups(self) -> None:
        finetune = plan_atomic_task(load_task(FINETUNE_TASK), self.v6).value
        self.assertEqual("4.0", finetune["schema_version"])
        self.assertEqual(410, len(finetune["train_case_ids"]))
        self.assertEqual(194, len(finetune["jobs"]))
        self.assertEqual(
            {"test"},
            {job["evaluation_partition"] for job in finetune["jobs"]},
        )
        self.assertEqual(194, len(finetune["evaluation_annotations"]))

        direct = plan_atomic_task(load_task(DIRECT_TASK), self.v6).value
        self.assertEqual([], direct["train_case_ids"])
        self.assertEqual(604, len(direct["jobs"]))
        self.assertEqual(
            {"group_1", "group_2", "group_3", "group_4", "group_5"},
            {job["evaluation_partition"] for job in direct["jobs"]},
        )

    def test_overall_score_is_primary_and_small_subgroups_are_na(self) -> None:
        plan = plan_atomic_task(load_task(FINETUNE_TASK), self.v6).value
        results = [{
            **job,
            "status": "evaluated",
            "score": 1.0,
        } for job in plan["jobs"]]
        summary = aggregate_task_results(plan=plan, case_results=results)
        self.assertEqual(1.0, summary["score"])
        self.assertEqual(
            "mean_all_test_jobs_regimes_are_diagnostics",
            summary["by_scene"]["collision_1d"]["aggregation_policy"],
        )
        diagnostics = summary["generalization_breakdown"]
        self.assertEqual(1.0, diagnostics["by_regime"]["ood"]["score"])
        self.assertEqual(
            "insufficient_samples",
            diagnostics["by_scene_regime"]["free_fall/id"]["status"],
        )
        self.assertIsNone(
            diagnostics["by_scene_regime"]["free_fall/id"]["score"]
        )
        self.assertEqual(
            "not_applicable",
            diagnostics["by_scene_regime"]["pendulum/ood"]["status"],
        )
        self.assertEqual(
            42,
            diagnostics["by_ood_factor"][
                "object_composition/ball_spec_composition"
            ]["expected_jobs"],
        )

    def test_collision_split_is_balanced_and_replicate_safe(self) -> None:
        collision = self.view["scenes"]["collision_1d"]
        annotations = self.view["test_annotations"]
        cases = {case["case_id"]: case for case in self.v6.cases}
        self.assertEqual(230, len(collision["train"]))
        self.assertEqual(
            {"id": 58, "ood": 42},
            dict(Counter(
                annotations[case_id]["generalization_regime"]
                for case_id in collision["test"]
            )),
        )
        split_audit = self.v6.root / "split_audit.json"
        audit = json.loads(split_audit.read_text(encoding="utf-8"))
        collision_audit = audit["split_details"]["collision"]
        self.assertEqual(
            "collision_stratified_replicate_safe_v2",
            collision_audit["policy"],
        )
        self.assertEqual(
            0,
            collision_audit["train_test_replicate_component_overlap"],
        )
        self.assertTrue(collision_audit["velocity_endpoints_forced_to_train"])
        self.assertEqual(
            {"train": 230, "test_id": 58, "test_ood": 42, "test": 100},
            collision_audit["counts"],
        )

        train_by_stratum: dict[tuple[tuple[str, ...], str], list[float]] = {}
        for case_id in collision["train"]:
            case = cases[case_id]
            stratum = (
                tuple(case["appearance"]["ball_sequence"]),
                case["appearance"]["collision_structure"],
            )
            train_by_stratum.setdefault(stratum, []).append(abs(
                case["physics"]["striker_initial_velocity"]["value"]
            ))
        ood_reasons = Counter()
        for case_id in collision["test"]:
            case = cases[case_id]
            regime = annotations[case_id]["generalization_regime"]
            stratum = (
                tuple(case["appearance"]["ball_sequence"]),
                case["appearance"]["collision_structure"],
            )
            if regime == "id":
                self.assertIn(stratum, train_by_stratum)
                velocity = abs(
                    case["physics"]["striker_initial_velocity"]["value"]
                )
                self.assertGreaterEqual(velocity, min(train_by_stratum[stratum]))
                self.assertLessEqual(velocity, max(train_by_stratum[stratum]))
            if case["appearance"]["collision_structure"] == (
                "two_ball_opposed_incident"
            ):
                ood_reasons["opposed_incident"] += 1
            if "glass" in case["appearance"]["ball_materials"]:
                ood_reasons["glass"] += 1
        self.assertEqual(
            {"opposed_incident": 34, "glass": 8},
            dict(ood_reasons),
        )


if __name__ == "__main__":
    unittest.main()
