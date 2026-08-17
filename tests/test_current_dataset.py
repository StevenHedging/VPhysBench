from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/six_scene_train_six_scene_eval_v1.json"
DIRECT_TASK = ROOT / "tasks/official/six_scene_direct_eval_v1.json"


class CurrentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cls.view = cls.dataset.views["view_a"]

    def test_v14_is_current_and_complete(self) -> None:
        self.assertEqual(
            ROOT / "datasets/releases/14.0.0/dataset.json",
            LATEST_DATASET,
        )
        self.assertEqual("physics_video_seven_scene_v14", self.dataset.dataset_id)
        self.assertEqual("14.0.0", self.dataset.descriptor["release"])
        self.assertEqual(903, len(self.dataset.cases))
        self.assertIsNotNone(self.dataset.asset_lock)
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
            "collision_1d": (297, 20),
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
        self.assertEqual(903, len(all_ids))
        self.assertEqual(903, len(set(all_ids)))
        self.assertEqual(set(test_ids), set(self.view["test_annotations"]))
        self.assertEqual(
            {"id": 110},
            dict(Counter(
                value["generalization_regime"]
                for value in self.view["test_annotations"].values()
            )),
        )
        self.assertEqual(127, len(self.view["scenes"]["push_bottle"]["train"]))
        self.assertEqual(14, len(self.view["scenes"]["push_bottle"]["test"]))

    def test_official_v1_tasks_score_six_scenes_and_exclude_push_bottle(self) -> None:
        finetune_task = load_task(FINETUNE_TASK)
        direct_task = load_task(DIRECT_TASK)
        self.assertEqual(
            "six_scene_train_six_scene_eval_v1",
            finetune_task.task_id,
        )
        self.assertEqual("six_scene_direct_eval_v1", direct_task.task_id)
        finetune = plan_atomic_task(finetune_task, self.dataset).value
        direct = plan_atomic_task(direct_task, self.dataset).value
        self.assertEqual(6, len(finetune["scene_ids"]))
        self.assertEqual(6, len(direct["scene_ids"]))
        self.assertEqual(6, len(finetune["training_scene_ids"]))
        self.assertEqual(666, len(finetune["train_case_ids"]))
        self.assertEqual(96, len(finetune["jobs"]))
        self.assertEqual(762, len(direct["jobs"]))
        selected_training_scenes = {
            case["scene_id"]
            for case in self.dataset.cases
            if case["case_id"] in set(finetune["train_case_ids"])
        }
        self.assertNotIn("push_bottle", finetune["training_scene_ids"])
        self.assertNotIn("push_bottle", selected_training_scenes)
        self.assertNotIn("push_bottle", finetune["scene_ids"])
        self.assertNotIn(
            "push_bottle",
            {job["scene_id"] for job in finetune["jobs"]},
        )
        self.assertNotIn("push_bottle", direct["scene_ids"])
        self.assertIn("vertical_spring_oscillator", direct["scene_ids"])
        self.assertEqual(
            117,
            sum(
                job["scene_id"] == "vertical_spring_oscillator"
                for job in direct["jobs"]
            ),
        )
        self.assertEqual(
            20,
            sum(
                job["scene_id"] == "vertical_spring_oscillator"
                for job in finetune["jobs"]
            ),
        )


if __name__ == "__main__":
    unittest.main()
