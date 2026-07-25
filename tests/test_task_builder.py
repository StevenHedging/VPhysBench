from __future__ import annotations

import copy
import tempfile
import unittest

from _paths import ROOT
from physbench.baseline_api import load_baseline_bundle, load_baseline_plugin
from physbench.data_layout import V2_DATASET
from physbench.datasets import load_dataset_v2
from physbench.domain import BaselineTaskInstance
from physbench.io import load_json, load_jsonl
from physbench.orchestration.atomic_runner import run_atomic
from physbench.tasks import load_task_v2


class TaskBuilderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_path = V2_DATASET
        cls.baseline_path = ROOT / "baselines" / "wan22_lora" / "baseline.json"
        cls.dataset = load_dataset_v2(cls.dataset_path, check_assets=True)
        cls.bundle = load_baseline_bundle(cls.baseline_path)
        cls.plugin = load_baseline_plugin(cls.bundle)

    def test_builder_is_deterministic_and_instance_is_sealed(self) -> None:
        task = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_physics.json"
        )
        first = self.plugin.task_builder.build(self.dataset, task)
        second = self.plugin.task_builder.build(self.dataset, task)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.value, second.value)
        first.verify()

        tampered = first.value
        tampered["semantics"]["conditioning"] = "generic"
        with self.assertRaisesRegex(ValueError, "modified"):
            BaselineTaskInstance.from_document(tampered)

    def test_task1_instance_has_train_infer_evaluate_graph(self) -> None:
        task = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_generic.json"
        )
        instance = self.plugin.task_builder.build(self.dataset, task).value
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
        self.assertEqual(["train"], operations["infer"]["depends_on"])
        self.assertEqual(["infer"], operations["evaluate"]["depends_on"])
        self.assertTrue(all(
            job["model_ref"] == "artifact://train/model"
            for job in instance["inference"]["jobs"]
        ))

    def test_task2_instance_has_no_training_operation(self) -> None:
        task = load_task_v2(
            ROOT / "tasks" / "official" / "direct_eval_generic.json"
        )
        instance = self.plugin.task_builder.build(self.dataset, task).value
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

    def test_generic_and_physics_share_cases_but_build_distinct_instances(self) -> None:
        generic_task = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_generic.json"
        )
        physics_task = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_physics.json"
        )
        generic = self.plugin.task_builder.build(self.dataset, generic_task)
        physics = self.plugin.task_builder.build(self.dataset, physics_task)
        generic_value = generic.value
        physics_value = physics.value
        self.assertNotEqual(generic.digest, physics.digest)
        self.assertEqual(
            generic_value["canonical_plan"]["train_case_ids"],
            physics_value["canonical_plan"]["train_case_ids"],
        )
        self.assertEqual(
            generic_value["identity"]["data_adapter"][
                "materialization_fingerprint"
            ],
            physics_value["identity"]["data_adapter"][
                "materialization_fingerprint"
            ],
        )

    def test_atomic_run_executes_the_sealed_instance(self) -> None:
        pendulum = next(
            case for case in self.dataset.cases
            if case["scene_id"] == "pendulum"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = run_atomic(
                dataset_path=self.dataset_path,
                task_path=(
                    ROOT / "tasks" / "official" / "direct_eval_generic.json"
                ),
                baseline_path=self.baseline_path,
                output_root=temporary,
                run_id="task-builder-dryrun",
                execute=False,
                scene_ids=["pendulum"],
                case_ids=[pendulum["case_id"]],
            )
            manifest = load_json(run_dir / "task_instance" / "manifest.json")
            run = load_json(run_dir / "run.json")
            jobs = load_jsonl(
                run_dir / "task_instance" / "inference_jobs.jsonl"
            )
            self.assertEqual(manifest["instance_digest"], run["task_instance_digest"])
            self.assertEqual(1, len(jobs))
            self.assertEqual("planned", run["status"])
            self.assertTrue((run_dir / "task_builder.json").is_file())
            self.assertTrue((run_dir / "frozen" / "assets.lock.json").is_file())
            self.assertTrue(
                (
                    run_dir / "logs" / "baseline_command"
                    / "run_task.stderr.log"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
