from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from _paths import ROOT
from physbench.baseline_api import load_baseline_bundle, load_baseline_plugin
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.io import canonical_sha256, load_json, write_json
from physbench.tasks import load_task, plan_atomic_task


DIRECT_TASK = (
    ROOT / "tasks" / "official" / "five_scene_direct_eval.json"
)
FINETUNE_TASK = (
    ROOT / "tasks" / "official" / "five_scene_finetune_eval.json"
)
GENERIC_BASELINE = (
    ROOT
    / "baselines"
    / "wan22_g15_sparse_motion"
    / "baseline.json"
)
PHYSICS_BASELINE = (
    ROOT
    / "baselines"
    / "wan22_g15_sparse_motion"
    / "physics.baseline.json"
)


def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


class ArchitectureV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=True)
        cls.task = load_task(DIRECT_TASK)

        cls.generic_bundle = load_baseline_bundle(GENERIC_BASELINE)
        cls.generic_plugin = load_baseline_plugin(cls.generic_bundle)
        cls.physics_bundle = load_baseline_bundle(PHYSICS_BASELINE)
        cls.physics_plugin = load_baseline_plugin(cls.physics_bundle)

    def _pendulum_case(self) -> dict:
        return copy.deepcopy(next(
            case
            for case in self.dataset.cases
            if case["scene_id"] == "pendulum"
        ))

    def test_dataset_case_owns_canonical_text_and_structured_physics(
        self,
    ) -> None:
        forbidden = {
            "conditioning",
            "input_views",
            "physical_parameters",
            "prompt_profile_id",
            "view_a_split",
        }
        for case in self.dataset.cases:
            self.assertFalse(forbidden & set(case), case["case_id"])
            self.assertEqual(
                {"prompt"},
                set(case["text"]),
            )
            self.assertTrue(case["text"]["prompt"].strip())
            self.assertTrue(case["physics"])
            self.assertTrue(any(
                quantity["annotated"] is True
                for quantity in case["physics"].values()
            ))
            self.assertFalse(_contains_key(case, "conditioning"))

    def test_dataset_release_does_not_use_an_asset_lock(self) -> None:
        self.assertIsNone(self.dataset.asset_lock)
        self.assertEqual("12.0.0", self.dataset.descriptor["release"])

    def test_tasks_and_canonical_plans_are_model_agnostic(self) -> None:
        for path in (DIRECT_TASK, FINETUNE_TASK):
            with self.subTest(task=path.name):
                task = load_task(path)
                plan = plan_atomic_task(task, self.dataset).value
                self.assertFalse(_contains_key(task.value, "conditioning"))
                self.assertNotIn("conditioning", plan)
                self.assertTrue(all(
                    "conditioning" not in job for job in plan["jobs"]
                ))

    def test_task_rejects_baseline_conditioning_property(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            value = load_json(DIRECT_TASK)
            value["conditioning"] = "physics"
            write_json(path, value)
            with self.assertRaisesRegex(
                ValueError,
                "physics use is a Baseline input-policy property",
            ):
                load_task(path)

    def test_same_task_has_same_plan_but_baseline_owned_adaptation(
        self,
    ) -> None:
        generic = self.generic_plugin.task_builder.build(
            self.dataset, self.task
        ).value
        physics = self.physics_plugin.task_builder.build(
            self.dataset, self.task
        ).value

        self.assertEqual(
            generic["canonical_plan"],
            physics["canonical_plan"],
        )
        generic_adapter = generic["identity"]["data_adapter"]
        physics_adapter = physics["identity"]["data_adapter"]
        self.assertNotEqual(
            generic_adapter["fingerprint"],
            physics_adapter["fingerprint"],
        )
        self.assertEqual(
            generic_adapter["materialization_fingerprint"],
            physics_adapter["materialization_fingerprint"],
        )

    def test_data_adapter_declares_all_five_stages(self) -> None:
        description = (
            self.generic_plugin.task_builder.data_adapter.describe()
        )
        self.assertEqual(
            {"spatial", "temporal", "paradigm", "text", "physics"},
            set(description["stages"]),
        )
        self.assertEqual(
            "case.text.prompt",
            description["input_policy"]["text"]["source"],
        )
        self.assertEqual(
            "ignored",
            description["input_policy"]["physics"]["usage"],
        )
        dependencies = self.generic_plugin.task_builder.describe()[
            "runtime_dependency_fingerprints"
        ]
        self.assertIn(
            "src/physbench/baselines/wan22_media.py", dependencies
        )
        self.assertIn("scripts/wan22_generate_batch.py", dependencies)

    def test_generic_adapter_uses_case_prompt_and_ignores_physics(
        self,
    ) -> None:
        case = self._pendulum_case()
        changed = copy.deepcopy(case)
        changed["physics"]["string_length"]["value"] = 999.0
        adapter = self.generic_plugin.task_builder.data_adapter

        first = adapter.adapt_case(case, role="eval")
        second = adapter.adapt_case(changed, role="eval")

        self.assertEqual(case["text"]["prompt"], first["prompt"])
        self.assertEqual(first["prompt"], second["prompt"])
        self.assertEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertEqual(first["native_inputs"], second["native_inputs"])
        self.assertEqual({}, first["used_parameters"])
        self.assertEqual(
            "ignored", first["stages"]["physics"]["usage"]
        )
        self.assertEqual(
            first["prompt"], first["native_inputs"]["text"]["prompt"]
        )

    def test_physics_adapter_extends_case_prompt_and_tracks_changes(
        self,
    ) -> None:
        case = self._pendulum_case()
        changed = copy.deepcopy(case)
        changed["physics"]["string_length"]["value"] += 0.01
        adapter = self.physics_plugin.task_builder.data_adapter

        first = adapter.adapt_case(case, role="eval")
        second = adapter.adapt_case(changed, role="eval")

        self.assertTrue(first["prompt"].startswith(case["text"]["prompt"]))
        self.assertNotEqual(first["prompt"], second["prompt"])
        self.assertIn("string_length", first["used_parameters"])
        self.assertEqual(
            "required", first["stages"]["physics"]["usage"]
        )
        self.assertEqual(
            "append_structured_text_v1",
            first["stages"]["physics"]["strategy"],
        )
        self.assertEqual(
            first["prompt"], first["native_inputs"]["text"]["prompt"]
        )

    def test_text_and_physics_do_not_change_media_materialization(
        self,
    ) -> None:
        generic = self.generic_plugin.task_builder.data_adapter
        physics = self.physics_plugin.task_builder.data_adapter
        self.assertNotEqual(generic.fingerprint, physics.fingerprint)
        self.assertEqual(
            generic.materialization_fingerprint,
            physics.materialization_fingerprint,
        )

    def test_direct_eval_can_plan_explicit_cases(self) -> None:
        value = copy.deepcopy(self.task.value)
        selected = self.dataset.cases[0]["case_id"]
        value["selection"]["scene_ids"] = [
            self.dataset.cases[0]["scene_id"]
        ]
        value["selection"]["case_ids"] = [selected]
        effective = TaskSpec(
            self.task.path,
            value,
            canonical_sha256(value),
        )
        plan = plan_atomic_task(effective, self.dataset)
        self.assertEqual([], plan.train_case_ids)
        self.assertEqual(
            [selected], [job["case_id"] for job in plan.jobs]
        )
        self.assertEqual(
            {"explicit"},
            {job["evaluation_partition"] for job in plan.jobs},
        )
        self.assertTrue(all(
            "conditioning" not in job for job in plan.jobs
        ))

if __name__ == "__main__":
    unittest.main()
