from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/seven_scene_train_five_scene_eval_v1.json"
DIRECT_TASK = ROOT / "tasks/official/five_scene_direct_eval_v1.json"


class CurrentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cls.view = cls.dataset.views["view_a"]

    def test_v13_is_current_and_complete(self) -> None:
        self.assertEqual(
            ROOT / "datasets/releases/13.0.0/dataset.json",
            LATEST_DATASET,
        )
        self.assertEqual("physics_video_seven_scene_v13", self.dataset.dataset_id)
        self.assertEqual("13.0.0", self.dataset.descriptor["release"])
        self.assertEqual(916, len(self.dataset.cases))
        self.assertIsNone(self.dataset.asset_lock)
        self.assertEqual(
            {
                "collision_1d",
                "inclined_plane_slide",
                "parabolic_motion",
                "pendulum",
                "push_bottle",
                "uniform_circular_motion",
                "vertical_spring_oscillator",
            },
            set(self.dataset.scene_configs),
        )

    def test_view_a_is_train_plus_id_test_only(self) -> None:
        expected = {
            "collision_1d": (310, 20),
            "inclined_plane_slide": (80, 15),
            "parabolic_motion": (82, 15),
            "pendulum": (80, 20),
            "push_bottle": (127, 14),
            "uniform_circular_motion": (30, 6),
            "vertical_spring_oscillator": (97, 20),
        }
        all_ids: list[str] = []
        test_ids: list[str] = []
        for scene_id, (train_count, test_count) in expected.items():
            groups = self.view["scenes"][scene_id]
            self.assertEqual(train_count, len(groups["train"]))
            self.assertEqual(test_count, len(groups["test"]))
            all_ids.extend(groups["train"])
            all_ids.extend(groups["test"])
            test_ids.extend(groups["test"])
        self.assertEqual(916, len(all_ids))
        self.assertEqual(916, len(set(all_ids)))
        self.assertEqual(set(test_ids), set(self.view["test_annotations"]))
        self.assertEqual(
            {"id": 110},
            dict(Counter(
                value["generalization_regime"]
                for value in self.view["test_annotations"].values()
            )),
        )

    def test_official_v1_tasks_use_seven_train_and_five_eval_scenes(self) -> None:
        finetune_task = load_task(FINETUNE_TASK)
        direct_task = load_task(DIRECT_TASK)
        self.assertEqual(
            "seven_scene_train_five_scene_eval_v1",
            finetune_task.task_id,
        )
        self.assertEqual("five_scene_direct_eval_v1", direct_task.task_id)
        finetune = plan_atomic_task(finetune_task, self.dataset).value
        direct = plan_atomic_task(direct_task, self.dataset).value
        self.assertEqual(5, len(finetune["scene_ids"]))
        self.assertEqual(5, len(direct["scene_ids"]))
        self.assertEqual(7, len(finetune["training_scene_ids"]))
        self.assertEqual(806, len(finetune["train_case_ids"]))
        self.assertEqual(76, len(finetune["jobs"]))
        self.assertEqual(658, len(direct["jobs"]))
        self.assertNotIn("push_bottle", direct["scene_ids"])
        self.assertNotIn("vertical_spring_oscillator", direct["scene_ids"])


if __name__ == "__main__":
    unittest.main()
