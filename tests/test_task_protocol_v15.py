from __future__ import annotations

import unittest

from _paths import ROOT
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task, plan_atomic_task


FINETUNE_TASK = (
    ROOT / "tasks/official/six_scene_train_six_scene_eval_v15.json"
)
DIRECT_TASK = ROOT / "tasks/official/six_scene_direct_eval_v15.json"
SIX_SCENES = {
    "pendulum",
    "collision_1d",
    "inclined_plane_slide",
    "uniform_circular_motion",
    "parabolic_motion",
    "vertical_spring_oscillator",
}


class TaskProtocolV15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)

    def test_finetune_task_trains_and_evaluates_six_scenes(self) -> None:
        task = load_task(FINETUNE_TASK)
        plan = plan_atomic_task(task, self.dataset).value

        self.assertEqual("six_scene_train_six_scene_eval_v15", task.task_id)
        self.assertEqual("scene_default_v15", task.value["evaluation"]["protocol"])
        self.assertEqual(SIX_SCENES, set(plan["training_scene_ids"]))
        self.assertEqual(SIX_SCENES, set(plan["scene_ids"]))
        self.assertEqual(679, len(plan["train_case_ids"]))
        self.assertEqual(96, len(plan["jobs"]))

    def test_direct_task_evaluates_all_six_scenes(self) -> None:
        task = load_task(DIRECT_TASK)
        plan = plan_atomic_task(task, self.dataset).value

        self.assertEqual("six_scene_direct_eval_v15", task.task_id)
        self.assertEqual("scene_default_v15", task.value["evaluation"]["protocol"])
        self.assertEqual(SIX_SCENES, set(plan["scene_ids"]))
        self.assertEqual([], plan["train_case_ids"])
        self.assertEqual(775, len(plan["jobs"]))

    def test_push_bottle_remains_outside_official_v15(self) -> None:
        for path in (FINETUNE_TASK, DIRECT_TASK):
            with self.subTest(task=path.name):
                task = load_task(path)
                selected = task.value["selection"].get(
                    "evaluation_scene_ids",
                    task.value["selection"].get("scene_ids", []),
                )
                self.assertNotIn("push_bottle", selected)


if __name__ == "__main__":
    unittest.main()
