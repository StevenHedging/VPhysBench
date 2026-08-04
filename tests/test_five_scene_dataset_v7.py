from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from physbench.data_layout import LATEST_DATASET, V6_DATASET, V7_DATASET, V8_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.task_evaluator import aggregate_task_results
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/five_scene_finetune_eval.json"
DIRECT_TASK = ROOT / "tasks/official/five_scene_direct_eval.json"


class FiveSceneDatasetV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v6 = load_dataset(V6_DATASET)
        cls.v7 = load_dataset(V7_DATASET, check_assets=True)
        cls.v8 = load_dataset(V8_DATASET, check_assets=True)
        cls.view = cls.v7.views["view_a"]

    def test_v7_is_latest_five_scene_release(self) -> None:
        self.assertNotEqual(V7_DATASET, LATEST_DATASET)
        self.assertEqual(V8_DATASET, LATEST_DATASET)
        self.assertEqual("4.0", self.v7.descriptor["schema_version"])
        self.assertEqual("physics_video_five_scene_v7", self.v7.dataset_id)
        self.assertEqual("7.0.0", self.v7.descriptor["release"])
        self.assertEqual(593, len(self.v7.cases))
        self.assertEqual(1617, len(self.v7.asset_lock["files"]))
        self.assertEqual(
            11,
            len({case["case_id"] for case in self.v6.cases}
                - {case["case_id"] for case in self.v7.cases}),
        )
        self.assertEqual(
            {case["case_id"] for case in self.v7.cases},
            {case["case_id"] for case in self.v6.cases}
            & {case["case_id"] for case in self.v7.cases},
        )
        for case in self.v7.cases:
            self.assertEqual("4.0", case["schema_version"])
            self.assertNotIn("ood", case)

    def test_view_a_has_complete_train_test_partitions(self) -> None:
        self.assertEqual("3.0", self.view["schema_version"])
        self.assertEqual("complete", self.view["coverage"])
        expected = {
            "collision_1d": (230, 100),
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
        self.assertEqual(593, len(all_ids))
        self.assertEqual(593, len(set(all_ids)))
        self.assertEqual(set(test_ids), set(self.view["test_annotations"]))

    def test_generalization_annotations_are_preserved(self) -> None:
        annotations = self.view["test_annotations"]
        self.assertEqual(
            {"id": 114, "ood": 70, "mixed": 6},
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

    def test_v4_tasks_plan_current_release(self) -> None:
        finetune = plan_atomic_task(load_task(FINETUNE_TASK), self.v8).value
        self.assertEqual("4.0", finetune["schema_version"])
        self.assertEqual(582, len(finetune["train_case_ids"]))
        self.assertEqual(76, len(finetune["jobs"]))
        self.assertEqual(
            {"test"},
            {job["evaluation_partition"] for job in finetune["jobs"]},
        )
        self.assertEqual(76, len(finetune["evaluation_annotations"]))

        direct = plan_atomic_task(load_task(DIRECT_TASK), self.v8).value
        self.assertEqual([], direct["train_case_ids"])
        self.assertEqual(658, len(direct["jobs"]))
        self.assertEqual(
            {"group_1", "group_2", "group_3", "group_4", "group_5"},
            {job["evaluation_partition"] for job in direct["jobs"]},
        )

    def test_overall_score_and_breakdowns_remain_well_defined(self) -> None:
        plan = plan_atomic_task(load_task(FINETUNE_TASK), self.v8).value
        results = [{
            **job,
            "status": "evaluated",
            "score": 1.0,
        } for job in plan["jobs"]]
        summary = aggregate_task_results(plan=plan, case_results=results)
        self.assertEqual(1.0, summary["score"])
        diagnostics = summary["generalization_breakdown"]
        self.assertEqual(1.0, diagnostics["by_regime"]["id"]["score"])
        self.assertEqual(
            "not_applicable",
            diagnostics["by_scene_regime"]["pendulum/ood"]["status"],
        )
        self.assertEqual({}, diagnostics["by_ood_factor"])

    def test_collision_split_remains_replicate_safe(self) -> None:
        collision = self.view["scenes"]["collision_1d"]
        annotations = self.view["test_annotations"]
        self.assertEqual(230, len(collision["train"]))
        self.assertEqual(
            {"id": 58, "ood": 42},
            dict(Counter(
                annotations[case_id]["generalization_regime"]
                for case_id in collision["test"]
            )),
        )
        audit = json.loads(
            (self.v7.root / "split_audit.json").read_text(encoding="utf-8")
        )
        collision_audit = audit["split_details"]["collision"]
        self.assertEqual(
            "collision_stratified_replicate_safe_v2",
            collision_audit["policy"],
        )
        self.assertEqual(
            0,
            collision_audit["train_test_replicate_component_overlap"],
        )
        self.assertEqual(
            {"train": 230, "test_id": 58, "test_ood": 42, "test": 100},
            collision_audit["counts"],
        )


if __name__ == "__main__":
    unittest.main()
