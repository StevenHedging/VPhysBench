from __future__ import annotations

import unittest
from pathlib import Path

from _paths import ROOT
from physbench.data_layout import LATEST_DATASET, V14_DATASET
from physbench.io import load_json
from physbench.tasks import load_task


DATASET_ID = "physics_video_seven_scene_v14"
TASKS = (
    ROOT / "tasks/official/six_scene_direct_eval_v1.json",
    ROOT / "tasks/official/six_scene_train_six_scene_eval_v1.json",
)


class TaskProtocol20260817Tests(unittest.TestCase):
    def test_current_dataset_is_v14_only(self) -> None:
        self.assertEqual(Path("datasets/releases/14.0.0/dataset.json"), V14_DATASET.relative_to(ROOT))
        self.assertEqual(V14_DATASET, LATEST_DATASET)

    def test_official_v1_tasks_bind_v14(self) -> None:
        for path in TASKS:
            with self.subTest(task=path.name):
                task = load_task(path)
                self.assertEqual(DATASET_ID, task.value["dataset_id"])
                self.assertEqual(
                    "scene_default_v1",
                    task.value["evaluation"]["protocol"],
                )

    def test_v14_case_schema_requires_frozen_reference_roles(self) -> None:
        schema = load_json(ROOT / "schemas/v6/case.schema.json")
        required = set(schema["properties"]["assets"]["required"])
        self.assertIn("reference_observation_manifest", required)
        self.assertIn(
            "reference_observation_visualization_manifest",
            required,
        )


if __name__ == "__main__":
    unittest.main()
