from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _baseline_fixtures import create_generic_baseline_pair
from _paths import ROOT
from physbench.baseline_api import load_baseline_bundle, load_baseline_plugin
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.datasets.physics import flat_physics_quantities
from physbench.domain import BaselineTaskInstance
from physbench.io import load_json, load_jsonl
from physbench.orchestration.atomic_runner import run_atomic
from physbench.tasks import load_task


DIRECT_TASK = (
    ROOT / "tasks" / "official" / "five_scene_direct_eval.json"
)
FINETUNE_TASK = (
    ROOT / "tasks" / "official" / "five_scene_finetune_eval.json"
)
def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


class TaskBuilderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.generic_path, cls.physics_path = create_generic_baseline_pair(
            Path(cls.temporary.name)
        )
        cls.dataset_path = LATEST_DATASET
        cls.dataset = load_dataset(
            cls.dataset_path, check_assets=False
        )
        cls.generic_bundle = load_baseline_bundle(cls.generic_path)
        cls.generic_plugin = load_baseline_plugin(cls.generic_bundle)
        cls.physics_bundle = load_baseline_bundle(cls.physics_path)
        cls.physics_plugin = load_baseline_plugin(cls.physics_bundle)

    def test_builder_is_deterministic_and_instance_is_sealed(self) -> None:
        task = load_task(FINETUNE_TASK)
        first = self.generic_plugin.task_builder.build(
            self.dataset, task
        )
        second = self.generic_plugin.task_builder.build(
            self.dataset, task
        )
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.value, second.value)
        first.verify()

        tampered = first.value
        tampered["semantics"]["scene_ids"].append("invented_scene")
        with self.assertRaisesRegex(ValueError, "modified or invalid"):
            BaselineTaskInstance.from_document(tampered)

    def test_task_plan_jobs_and_instance_have_no_conditioning_axis(
        self,
    ) -> None:
        task = load_task(DIRECT_TASK)
        instance = self.generic_plugin.task_builder.build(
            self.dataset, task
        ).value
        self.assertFalse(_contains_key(task.value, "conditioning"))
        self.assertFalse(
            _contains_key(instance["canonical_plan"], "conditioning")
        )
        self.assertTrue(all(
            "conditioning" not in job
            for job in instance["canonical_plan"]["jobs"]
        ))
        self.assertTrue(all(
            "conditioning" not in job
            for job in instance["inference"]["jobs"]
        ))
        self.assertNotIn("conditioning", instance)

    def test_finetune_instance_has_train_infer_evaluate_graph(
        self,
    ) -> None:
        task = load_task(FINETUNE_TASK)
        instance = self.generic_plugin.task_builder.build(
            self.dataset, task
        ).value
        self.assertIsNotNone(instance["training"])
        self.assertEqual(
            instance["canonical_plan"]["train_case_ids"],
            instance["training"]["case_ids"],
        )
        operations = {
            item["operation_id"]: item
            for item in instance["execution_graph"]["operations"]
        }
        self.assertEqual([], operations["train"]["depends_on"])
        self.assertEqual(
            ["train"], operations["infer"]["depends_on"]
        )
        self.assertEqual(
            ["infer"], operations["evaluate"]["depends_on"]
        )
        self.assertTrue(all(
            job["model_ref"] == "artifact://train/model"
            for job in instance["inference"]["jobs"]
        ))

    def test_direct_eval_instance_has_no_training_operation(self) -> None:
        task = load_task(DIRECT_TASK)
        instance = self.generic_plugin.task_builder.build(
            self.dataset, task
        ).value
        self.assertIsNone(instance["training"])
        operation_ids = {
            item["operation_id"]
            for item in instance["execution_graph"]["operations"]
        }
        self.assertNotIn("train", operation_ids)
        self.assertTrue(all(
            job["model_ref"] == "baseline://frozen_model"
            for job in instance["inference"]["jobs"]
        ))

    def test_baselines_share_plan_but_own_distinct_adaptation(
        self,
    ) -> None:
        task = load_task(FINETUNE_TASK)
        generic = self.generic_plugin.task_builder.build(
            self.dataset, task
        )
        physics = self.physics_plugin.task_builder.build(
            self.dataset, task
        )
        generic_value = generic.value
        physics_value = physics.value

        self.assertNotEqual(generic.digest, physics.digest)
        self.assertEqual(
            generic_value["canonical_plan"],
            physics_value["canonical_plan"],
        )
        generic_adapter = generic_value["identity"]["data_adapter"]
        physics_adapter = physics_value["identity"]["data_adapter"]
        self.assertNotEqual(
            generic_adapter["fingerprint"],
            physics_adapter["fingerprint"],
        )
        self.assertEqual(
            generic_adapter["materialization_fingerprint"],
            physics_adapter["materialization_fingerprint"],
        )
        self.assertEqual(
            "ignored",
            self.generic_bundle.value["input_policy"]["physics"][
                "usage"
            ],
        )
        self.assertEqual(
            "required",
            self.physics_bundle.value["input_policy"]["physics"][
                "usage"
            ],
        )

    def test_eval_source_case_exposes_conditions_not_ground_truth(
        self,
    ) -> None:
        task = load_task(DIRECT_TASK)
        instance = self.physics_plugin.task_builder.build(
            self.dataset, task
        ).value
        source = instance["source"]["cases"][0]
        original = next(
            case
            for case in self.dataset.cases
            if case["case_id"] == source["case_id"]
        )

        self.assertEqual(original["text"], source["text"])
        self.assertTrue(source["physics"])
        self.assertTrue(all(
            set(quantity) == {"value", "unit", "symbol"}
            for quantity in source["physics"].values()
        ))
        self.assertEqual(flat_physics_quantities(original), source["physics"])
        self.assertNotIn("provenance", source)
        self.assertNotIn("alignment", source)
        self.assertNotIn("has_real_reference_video", source)
        self.assertNotIn("supervised_targets", source)
        for ground_truth_key in (
            "reference_video",
            "physics_reference_video",
            "source_video",
            "source_archive",
        ):
            self.assertNotIn(ground_truth_key, source["assets"])

    def test_atomic_run_executes_the_sealed_instance(self) -> None:
        pendulum = next(
            case
            for case in self.dataset.cases
            if case["scene_id"] == "pendulum"
        )
        first_frame = ROOT / "datasets" / pendulum["assets"]["first_frame"]
        if not first_frame.is_file():
            self.skipTest("full Dataset assets are not present")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = run_atomic(
                dataset_path=self.dataset_path,
                task_path=DIRECT_TASK,
                baseline_path=self.generic_path,
                output_root=temporary,
                run_id="task-builder-dryrun",
                execute=False,
                scene_ids=["pendulum"],
                case_ids=[pendulum["case_id"]],
                check_assets=False,
            )
            manifest = load_json(
                run_dir / "task_instance" / "manifest.json"
            )
            run = load_json(run_dir / "run.json")
            jobs = load_jsonl(
                run_dir
                / "task_instance"
                / "inference_jobs.jsonl"
            )
            self.assertEqual(
                manifest["instance_digest"],
                run["task_instance_digest"],
            )
            self.assertEqual(1, len(jobs))
            self.assertNotIn("conditioning", jobs[0])
            self.assertEqual("planned", run["status"])
            self.assertTrue(
                (run_dir / "task_builder.json").is_file()
            )
            self.assertFalse(
                (run_dir / "frozen" / "assets.lock.json").exists()
            )
            artifact_policy = load_json(
                run_dir / "artifact_policy.json"
            )
            self.assertFalse(
                artifact_policy[
                    "external_prediction_references_allowed"
                ]
            )
            prediction_artifacts = load_json(
                run_dir
                / "artifacts"
                / "prediction_artifacts.json"
            )
            self.assertEqual(
                [], prediction_artifacts["prediction_videos"]
            )
            self.assertTrue(
                (
                    run_dir
                    / "task_instance"
                    / "adaptations.jsonl"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
