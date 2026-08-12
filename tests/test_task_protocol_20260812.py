from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from _paths import ROOT
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.io import write_json
from physbench.tasks import load_task, plan_atomic_task


SPLIT_TASK = (
    ROOT
    / "tasks"
    / "experiments"
    / "six_scene_train_five_scene_eval_v14.json"
)
LEGACY_TASK = (
    ROOT
    / "tasks"
    / "experiments"
    / "seven_scene_entity_vector_finetune_eval.json"
)
DIRECT_TASK = (
    ROOT / "tasks" / "official" / "five_scene_direct_eval_csti_identity_v3.json"
)


class TaskProtocol20260812Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)

    def test_split_task_trains_six_scenes_and_evaluates_supported_five(self) -> None:
        task = load_task(SPLIT_TASK)
        plan = plan_atomic_task(task, self.dataset).value

        self.assertEqual("six_scene_train_five_scene_eval_v14", task.task_id)
        self.assertEqual("scene_default_v14", task.value["evaluation"]["protocol"])
        self.assertEqual(
            {
                "pendulum",
                "collision_1d",
                "inclined_plane_slide",
                "uniform_circular_motion",
                "parabolic_motion",
                "vertical_spring_oscillator",
            },
            set(plan["training_scene_ids"]),
        )
        self.assertEqual(
            {
                "pendulum",
                "collision_1d",
                "inclined_plane_slide",
                "uniform_circular_motion",
                "parabolic_motion",
            },
            set(plan["scene_ids"]),
        )
        self.assertEqual(679, len(plan["train_case_ids"]))
        self.assertEqual(76, len(plan["jobs"]))
        self.assertNotIn("push_bottle", plan["training_scene_ids"])
        self.assertNotIn("push_bottle", plan["scene_ids"])

    def test_legacy_single_scene_selector_keeps_its_frozen_plan_shape(self) -> None:
        task = load_task(LEGACY_TASK)
        plan = plan_atomic_task(task, self.dataset).value

        self.assertEqual(806, len(plan["train_case_ids"]))
        self.assertEqual(110, len(plan["jobs"]))
        self.assertNotIn("training_scene_ids", plan)

    def test_split_and_legacy_scene_selectors_are_mutually_exclusive(self) -> None:
        value = copy.deepcopy(load_task(SPLIT_TASK).value)
        value["selection"]["scene_ids"] = value["selection"][
            "evaluation_scene_ids"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.json"
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "unknown fields|exactly"):
                load_task(path)

    def test_direct_task_accepts_explicit_evaluation_scene_selector(self) -> None:
        value = copy.deepcopy(load_task(DIRECT_TASK).value)
        value["task_id"] = "split_direct_selector_test"
        value["selection"]["evaluation_scene_ids"] = value["selection"].pop(
            "scene_ids"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "direct.json"
            write_json(path, value)
            task = load_task(path)
            plan = plan_atomic_task(task, self.dataset).value

        self.assertEqual(658, len(plan["jobs"]))
        self.assertEqual(
            set(value["selection"]["evaluation_scene_ids"]),
            set(plan["scene_ids"]),
        )


if __name__ == "__main__":
    unittest.main()
