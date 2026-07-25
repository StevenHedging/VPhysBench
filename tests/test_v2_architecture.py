from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from _paths import ROOT
from physbench.baseline_api import load_baseline_bundle, load_baseline_plugin
from physbench.data_layout import V2_DATASET
from physbench.datasets import load_dataset_v2
from physbench.domain import TaskSpec
from physbench.io import canonical_sha256, load_json, write_json
from physbench.tasks import load_task_v2, plan_atomic_task


class V2ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset_v2(
            V2_DATASET, check_assets=True
        )
        cls.bundle = load_baseline_bundle(
            ROOT / "baselines" / "wan22_lora" / "baseline.json"
        )
        cls.plugin = load_baseline_plugin(cls.bundle)

    def test_dataset_has_no_model_conditioning_payload(self) -> None:
        forbidden = {
            "prompt", "input_views", "physical_parameters", "text", "view_a_split"
        }
        for case in self.dataset.cases:
            stack = [case]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    self.assertFalse(forbidden & set(value), case["case_id"])
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)

    def test_dataset_release_locks_every_referenced_asset(self) -> None:
        self.assertIsNotNone(self.dataset.asset_lock)
        locked = {
            item["path"] for item in self.dataset.asset_lock["files"]
        }
        referenced = {
            value
            for case in self.dataset.cases
            for value in case["assets"].values()
            if value
        }
        self.assertEqual(referenced, locked)
        self.assertEqual("2.0.0", self.dataset.descriptor["release"])

    def test_paired_finetune_tasks_have_identical_data_plan(self) -> None:
        generic = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_generic.json"
        )
        physics = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_physics.json"
        )
        first = plan_atomic_task(generic, self.dataset).value
        second = plan_atomic_task(physics, self.dataset).value
        self.assertEqual(first["train_case_ids"], second["train_case_ids"])
        self.assertEqual(first["training_seed"], second["training_seed"])
        first_jobs = [
            (item["case_id"], item["evaluation_partition"], item["seed"])
            for item in first["jobs"]
        ]
        second_jobs = [
            (item["case_id"], item["evaluation_partition"], item["seed"])
            for item in second["jobs"]
        ]
        self.assertEqual(first_jobs, second_jobs)
        self.assertNotEqual(first["task_id"], second["task_id"])

    def test_data_adapter_declares_all_five_stages(self) -> None:
        description = self.plugin.task_builder.data_adapter.describe()
        self.assertEqual(
            {"spatial", "temporal", "paradigm", "text", "physics"},
            set(description["stages"]),
        )
        self.assertTrue(description["native_inputs_are_opaque_to_benchmark"])
        self.assertNotIn("condition_adapter", self.bundle.value["components"])
        self.assertNotIn("data_adapter", self.bundle.value["components"])
        self.assertIn("task_builder", self.bundle.value["components"])
        dependencies = self.plugin.task_builder.describe()[
            "runtime_dependency_fingerprints"
        ]
        self.assertIn(
            "src/physbench/baselines/wan22_media.py", dependencies
        )
        self.assertIn("scripts/wan22_generate_batch.py", dependencies)

    def test_bundle_rejects_public_condition_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "wan22"
            shutil.copytree(self.bundle.root, root)
            path = root / "baseline.json"
            value = load_json(path)
            value["components"]["condition_adapter"] = {"type": "legacy"}
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "not a public"):
                load_baseline_bundle(path)

    def test_generic_adaptation_cannot_observe_physics(self) -> None:
        case = copy.deepcopy(next(
            item for item in self.dataset.cases if item["scene_id"] == "pendulum"
        ))
        changed = copy.deepcopy(case)
        changed["physics"]["string_length"]["value"] = 999.0
        first = self.plugin.task_builder.data_adapter.adapt_case(
            case, "generic", role="eval"
        )
        second = self.plugin.task_builder.data_adapter.adapt_case(
            changed, "generic", role="eval"
        )
        self.assertEqual(first["prompt"], second["prompt"])
        self.assertEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertEqual(first["native_inputs"], second["native_inputs"])
        self.assertEqual({}, first["used_parameters"])
        self.assertFalse(first["stages"]["physics"]["enabled"])
        self.assertFalse(
            first["stages"]["text"]["contains_detailed_physics"]
        )

    def test_wan_physics_adaptation_injects_into_native_text(self) -> None:
        case = copy.deepcopy(next(
            item for item in self.dataset.cases if item["scene_id"] == "pendulum"
        ))
        changed = copy.deepcopy(case)
        changed["physics"]["string_length"]["value"] += 0.001
        first = self.plugin.task_builder.data_adapter.adapt_case(
            case, "physics", role="eval"
        )
        second = self.plugin.task_builder.data_adapter.adapt_case(
            changed, "physics", role="eval"
        )
        self.assertNotEqual(first["prompt"], second["prompt"])
        self.assertIn("string_length", first["used_parameters"])
        self.assertTrue(first["stages"]["physics"]["enabled"])
        self.assertEqual(
            "append_structured_values_to_text",
            first["stages"]["physics"]["strategy"],
        )
        self.assertEqual(
            first["prompt"], first["native_inputs"]["text"]["prompt"]
        )

    def test_text_changes_do_not_invalidate_media_materialization(self) -> None:
        description = self.plugin.task_builder.data_adapter.describe()
        material_stages = {
            name: description["stage_fingerprints"][name]
            for name in ("spatial", "temporal", "paradigm")
        }
        self.assertEqual(
            canonical_sha256(material_stages),
            self.plugin.task_builder.data_adapter.materialization_fingerprint,
        )

    def test_direct_eval_can_plan_explicit_cases(self) -> None:
        task = load_task_v2(
            ROOT / "tasks" / "official" / "direct_eval_physics.json"
        )
        value = copy.deepcopy(task.value)
        selected = self.dataset.cases[0]["case_id"]
        value["selection"]["scene_ids"] = [self.dataset.cases[0]["scene_id"]]
        value["selection"]["case_ids"] = [selected]
        effective = TaskSpec(task.path, value, canonical_sha256(value))
        plan = plan_atomic_task(effective, self.dataset)
        self.assertEqual([], plan.train_case_ids)
        self.assertEqual([selected], [job["case_id"] for job in plan.jobs])
        self.assertEqual({"explicit"}, {
            job["evaluation_partition"] for job in plan.jobs
        })


if __name__ == "__main__":
    unittest.main()
