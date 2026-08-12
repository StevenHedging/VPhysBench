from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from _paths import ROOT
from physbench.baseline_api import TaskBuilder
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.tasks import load_task, plan_atomic_task


OFFICIAL_TASKS = ROOT / "tasks" / "official"
PROTOCOLS = ROOT / "configs" / "evaluation" / "protocols"
DIRECT_TASK = OFFICIAL_TASKS / "six_scene_direct_eval_v1.json"
FINETUNE_TASK = (
    OFFICIAL_TASKS / "six_scene_train_six_scene_eval_v1.json"
)
SCORED_SCENES = {
    "pendulum",
    "collision_1d",
    "inclined_plane_slide",
    "uniform_circular_motion",
    "parabolic_motion",
    "vertical_spring_oscillator",
}
TRAINING_SCENES = SCORED_SCENES
PUBLIC_EVALUATOR_TYPES = {
    "pendulum_v1",
    "collision_1d_v1",
    "inclined_plane_slide_v1",
    "uniform_circular_motion_v1",
    "parabolic_motion_v1",
    "vertical_spring_oscillator_v1",
    "unsupported",
}


class ReleaseV1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)

    def test_release_exposes_exactly_two_v1_tasks(self) -> None:
        self.assertEqual(
            {
                "six_scene_direct_eval_v1.json",
                "six_scene_train_six_scene_eval_v1.json",
            },
            {path.name for path in OFFICIAL_TASKS.glob("*.json")},
        )

    def test_direct_v1_plans_all_six_scene_cases(self) -> None:
        self.assertTrue(DIRECT_TASK.is_file(), DIRECT_TASK)
        task = load_task(DIRECT_TASK)
        plan = plan_atomic_task(task, self.dataset).value

        self.assertEqual("1.0", task.value["schema_version"])
        self.assertEqual("six_scene_direct_eval_v1", task.task_id)
        self.assertEqual("scene_default_v1", task.value["evaluation"]["protocol"])
        self.assertEqual([], plan["training_scene_ids"])
        self.assertEqual(SCORED_SCENES, set(plan["scene_ids"]))
        self.assertEqual([], plan["train_case_ids"])
        self.assertEqual(775, len(plan["jobs"]))
        self.assertEqual(
            {
                "collision_1d": 330,
                "inclined_plane_slide": 95,
                "parabolic_motion": 97,
                "pendulum": 100,
                "uniform_circular_motion": 36,
                "vertical_spring_oscillator": 117,
            },
            dict(Counter(job["scene_id"] for job in plan["jobs"])),
        )

    def test_finetune_v1_scores_six_scenes_and_excludes_push_bottle(self) -> None:
        self.assertTrue(FINETUNE_TASK.is_file(), FINETUNE_TASK)
        task = load_task(FINETUNE_TASK)
        plan = plan_atomic_task(task, self.dataset).value
        by_case = {case["case_id"]: case for case in self.dataset.cases}

        self.assertEqual("1.0", task.value["schema_version"])
        self.assertEqual("six_scene_train_six_scene_eval_v1", task.task_id)
        self.assertEqual("scene_default_v1", task.value["evaluation"]["protocol"])
        self.assertEqual(TRAINING_SCENES, set(plan["training_scene_ids"]))
        self.assertEqual(SCORED_SCENES, set(plan["scene_ids"]))
        self.assertEqual(679, len(plan["train_case_ids"]))
        self.assertEqual(96, len(plan["jobs"]))
        self.assertEqual(
            {
                "collision_1d": 20,
                "inclined_plane_slide": 15,
                "parabolic_motion": 15,
                "pendulum": 20,
                "uniform_circular_motion": 6,
                "vertical_spring_oscillator": 20,
            },
            dict(Counter(job["scene_id"] for job in plan["jobs"])),
        )
        self.assertEqual(
            TRAINING_SCENES,
            {
                by_case[case_id]["scene_id"]
                for case_id in plan["train_case_ids"]
            },
        )
        self.assertNotIn(
            "push_bottle",
            {by_case[case_id]["scene_id"] for case_id in plan["train_case_ids"]},
        )
        self.assertNotIn(
            "push_bottle",
            {job["scene_id"] for job in plan["jobs"]},
        )
        self.assertEqual(
            {"id": 96},
            dict(Counter(
                annotation["generalization_regime"]
                for annotation in plan["evaluation_annotations"].values()
            )),
        )

    def test_task_loader_rejects_pre_v1_schema(self) -> None:
        source = DIRECT_TASK
        value = json.loads(source.read_text(encoding="utf-8"))
        value["schema_version"] = "4.0"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-task.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version=1.0"):
                load_task(path)

    def test_baseline_compiler_cannot_replan_the_task(self) -> None:
        self.assertFalse(hasattr(TaskBuilder, "build"))

    def test_release_exposes_one_latest_only_protocol(self) -> None:
        self.assertEqual(
            {"scene_default_v1.json"},
            {path.name for path in PROTOCOLS.glob("*.json")},
        )
        protocol = load_evaluation_protocol("scene_default_v1")
        self.assertEqual(
            {
                "pendulum": "pendulum_v1",
                "collision_1d": "collision_1d_v1",
                "inclined_plane_slide": "inclined_plane_slide_v1",
                "uniform_circular_motion": "uniform_circular_motion_v1",
                "parabolic_motion": "parabolic_motion_v1",
                "vertical_spring_oscillator": "vertical_spring_oscillator_v1",
            },
            {
                scene_id: config["type"]
                for scene_id, config in protocol["scenes"].items()
            },
        )
        self.assertEqual(
            "exact_full_tube_edt",
            protocol["general_metrics"]["csti"]["algorithm"],
        )

    def test_registry_exposes_only_v1_public_types(self) -> None:
        self.assertEqual(
            PUBLIC_EVALUATOR_TYPES,
            set(SceneEvaluatorRegistry.supported_evaluator_types()),
        )


if __name__ == "__main__":
    unittest.main()
