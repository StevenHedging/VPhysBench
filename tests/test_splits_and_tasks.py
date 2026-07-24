from __future__ import annotations

import copy
import unittest

from _paths import FIXTURES
from physbench.io import load_json, load_jsonl
from physbench.splitters import build_view_a, build_view_b
from physbench.task_planner import plan_task, validate_task


class SplitAndTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(FIXTURES / "cases.jsonl")

    def test_view_a_is_disjoint_and_complete(self) -> None:
        split = build_view_a(self.cases)
        ids = [case_id for scene in split["scenes"].values() for group in scene.values() for case_id in group]
        self.assertEqual(len(self.cases), len(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_view_b_is_reproducible_and_balanced(self) -> None:
        first = build_view_b(self.cases, groups=2, seed=42)
        second = build_view_b(self.cases, groups=2, seed=42)
        self.assertEqual(first, second)
        for scene in first["scenes"].values():
            sizes = [len(group) for group in scene.values()]
            self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_view_a_plan(self) -> None:
        plan = plan_task(
            load_json(FIXTURES / "task_view_a.json"), self.cases, build_view_a(self.cases)
        )
        self.assertEqual(2, len(plan["train_case_ids"]))
        self.assertEqual(4, len(plan["jobs"]))
        self.assertFalse(plan["train_preview"]["enabled"])
        self.assertEqual("physics_natural", plan["prompt_profiles"]["train"])
        self.assertEqual(["physics_natural"], plan["prompt_profiles"]["eval"])

    def test_same_cases_expand_over_two_prompt_profiles(self) -> None:
        plan = plan_task(
            load_json(FIXTURES / "task_view_a.json"),
            self.cases,
            build_view_a(self.cases),
            eval_prompt_profiles=["generic", "physics_natural"],
        )
        self.assertEqual(8, len(plan["jobs"]))
        self.assertEqual(
            {"generic", "physics_natural"},
            {job["prompt_profile_id"] for job in plan["jobs"]},
        )
        grouped = {}
        for job in plan["jobs"]:
            key = (job["case_id"], job["evaluation_partition"], job["seed"])
            grouped.setdefault(key, set()).add(job["prompt_profile_id"])
        self.assertTrue(
            all(value == {"generic", "physics_natural"} for value in grouped.values())
        )

    def test_view_a_can_add_deterministic_seen_training_previews(self) -> None:
        task = load_json(FIXTURES / "task_view_a.json")
        split = build_view_a(self.cases)
        first = plan_task(
            task, self.cases, split,
            train_preview_per_scene=1, train_preview_seed=123,
        )
        second = plan_task(
            task, self.cases, split,
            train_preview_per_scene=1, train_preview_seed=123,
        )
        previews = [job for job in first["jobs"] if job["evaluation_partition"] == "train_seen"]
        self.assertEqual(first, second)
        self.assertEqual(2, len(previews))
        self.assertEqual({"pendulum", "free_fall"}, {job["scene_id"] for job in previews})
        self.assertTrue({job["case_id"] for job in previews} <= set(first["train_case_ids"]))
        self.assertEqual(6, len(first["jobs"]))

    def test_stale_or_incomplete_split_is_rejected(self) -> None:
        split = build_view_a(self.cases)
        split["scenes"]["pendulum"]["test_id"] = []
        with self.assertRaisesRegex(ValueError, "exactly once"):
            plan_task(load_json(FIXTURES / "task_view_a.json"), self.cases, split)

    def test_view_b_jobs_keep_group_partition(self) -> None:
        plan = plan_task(
            load_json(FIXTURES / "task_view_b.json"), self.cases, build_view_b(self.cases, 2, 42)
        )
        self.assertEqual(0, len(plan["train_case_ids"]))
        self.assertEqual({"group_1", "group_2"}, {job["evaluation_partition"] for job in plan["jobs"]})

    def test_ood2_is_rejected(self) -> None:
        task = copy.deepcopy(load_json(FIXTURES / "task_view_a.json"))
        task["ood2"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "OOD2"):
            validate_task(task)


if __name__ == "__main__":
    unittest.main()
