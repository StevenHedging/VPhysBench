from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _paths import ROOT
from physbench.baseline_api import (
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime.task_instance_validation import (
    validate_task_instance_document,
)
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.identifiers import SAFE_ID_PATTERN
from physbench.io import canonical_sha256, load_json, write_json
from physbench.orchestration.atomic_runner import run_atomic, run_matrix
from physbench.tasks import load_task, plan_atomic_task


DIRECT_TASK = (
    ROOT / "tasks" / "official" / "five_scene_direct_eval.json"
)
FINETUNE_TASK = (
    ROOT / "tasks" / "official" / "five_scene_finetune_eval.json"
)
GENERIC_BASELINE = (
    ROOT / "baselines" / "wan22_lora" / "baseline.json"
)
PHYSICS_BASELINE = (
    ROOT / "baselines" / "wan22_lora" / "physics.baseline.json"
)


class TaskRuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cls.direct = load_task(DIRECT_TASK)
        cls.finetune = load_task(FINETUNE_TASK)
        cls.generic_bundle = load_baseline_bundle(GENERIC_BASELINE)
        cls.generic_plugin = load_baseline_plugin(cls.generic_bundle)
        cls.physics_bundle = load_baseline_bundle(PHYSICS_BASELINE)

    def _one_case_value(self, *, task_id: str = "one_case_direct") -> dict:
        value = copy.deepcopy(self.direct.value)
        case = self.dataset.cases[0]
        value["task_id"] = task_id
        value["selection"]["scene_ids"] = [case["scene_id"]]
        value["selection"]["groups"] = "all"
        value["selection"]["case_ids"] = [case["case_id"]]
        return value

    def _one_case_instance(self):
        value = self._one_case_value()
        task = TaskSpec(
            self.direct.path,
            value,
            canonical_sha256(value),
        )
        return self.generic_plugin.task_builder.build(
            self.dataset,
            task,
        )

    def test_task_rejects_invalid_nested_shapes_and_seed_values(self) -> None:
        invalid_documents: list[tuple[str, dict, str]] = []

        value = copy.deepcopy(self.direct.value)
        value["selection"] = []
        invalid_documents.append(("selection_object", value, "must be an object"))

        value = copy.deepcopy(self.direct.value)
        value["selection"]["unexpected"] = []
        invalid_documents.append(("selection_fields", value, "unknown fields"))

        value = copy.deepcopy(self.direct.value)
        value["selection"]["scene_ids"] = ["pendulum", "pendulum"]
        invalid_documents.append(("scene_unique", value, "unique identifiers"))

        value = copy.deepcopy(self.direct.value)
        value["selection"]["groups"] = []
        invalid_documents.append(("groups_nonempty", value, "non-empty array"))

        value = copy.deepcopy(self.direct.value)
        value["ood2"] = {"enabled": 1}
        invalid_documents.append((
            "ood2_boolean",
            value,
            "outside the model-agnostic contract",
        ))

        value = copy.deepcopy(self.direct.value)
        value["ood2"] = {
            "enabled": False,
            "heldout_scenes": ["pendulum"],
        }
        invalid_documents.append((
            "ood2_disabled",
            value,
            "outside the model-agnostic contract",
        ))

        value = copy.deepcopy(self.direct.value)
        value["seeds"]["training"] = [7]
        invalid_documents.append(("direct_train_seed", value, "exactly 0"))

        value = copy.deepcopy(self.direct.value)
        value["seeds"]["inference"] = [True]
        invalid_documents.append(
            ("boolean_inference_seed", value, "non-boolean integers")
        )

        value = copy.deepcopy(self.direct.value)
        value["seeds"]["inference"] = [7, 7]
        invalid_documents.append(("inference_unique", value, "unique seeds"))

        value = copy.deepcopy(self.finetune.value)
        value["seeds"]["training"] = [7, 8]
        invalid_documents.append(("one_train_seed", value, "exactly 1"))

        value = copy.deepcopy(self.finetune.value)
        value["selection"]["eval_partitions"] = ["test_id", "invented"]
        invalid_documents.append(
            ("eval_partition", value, "exactly one legacy scene_ids")
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, document, message in invalid_documents:
                with self.subTest(name=name):
                    path = root / f"{name}.json"
                    write_json(path, document)
                    with self.assertRaisesRegex(ValueError, message):
                        load_task(path)

    def test_plan_revalidates_programmatically_constructed_task(self) -> None:
        value = self._one_case_value()
        value["seeds"]["inference"] = [False]
        task = TaskSpec(
            self.direct.path,
            value,
            canonical_sha256(value),
        )
        with self.assertRaisesRegex(ValueError, "non-boolean integers"):
            plan_atomic_task(task, self.dataset)

    def test_task_and_run_entrypoints_reject_path_escaping_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_path = root / "one-case.json"
            write_json(task_path, self._one_case_value())

            invalid_task = self._one_case_value(task_id="../escaped")
            invalid_task_path = root / "invalid-task.json"
            write_json(invalid_task_path, invalid_task)
            with self.assertRaisesRegex(ValueError, "task_id"):
                load_task(invalid_task_path)

            output = root / "runs"
            escaped_run = root / "escaped-run"
            with self.assertRaisesRegex(ValueError, "run_id"):
                run_atomic(
                    dataset_path=LATEST_DATASET,
                    task_path=task_path,
                    baseline_path=GENERIC_BASELINE,
                    output_root=output,
                    run_id="../escaped-run",
                    execute=False,
                    check_assets=False,
                )
            self.assertFalse(escaped_run.exists())

            escaped_matrix = root / "escaped-matrix.matrix.json"
            with self.assertRaisesRegex(ValueError, "matrix_id"):
                run_matrix(
                    dataset_path=LATEST_DATASET,
                    task_path=task_path,
                    baseline_paths=[
                        GENERIC_BASELINE,
                        PHYSICS_BASELINE,
                    ],
                    output_root=output,
                    matrix_id="../escaped-matrix",
                    execute=False,
                )
            self.assertFalse(escaped_matrix.exists())

    def test_task_instance_rejects_unsafe_runtime_identifiers(self) -> None:
        original = self._one_case_instance().value
        mutations = {
            "instance_id": lambda value: value.__setitem__(
                "instance_id", "../instance"
            ),
            "source_case_id": lambda value: value["source"]["cases"][
                0
            ].__setitem__("case_id", "../case"),
            "adaptation_id": lambda value: value["adaptations"][
                0
            ].__setitem__("adaptation_id", "../adaptation"),
            "inference_job_id": lambda value: value["inference"]["jobs"][
                0
            ].__setitem__("job_id", "../job"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                document = copy.deepcopy(original)
                mutate(document)
                with self.assertRaisesRegex(
                    ValueError,
                    "path-safe identifier",
                ):
                    validate_task_instance_document(document)

        document = copy.deepcopy(original)
        document["canonical_plan"]["jobs"][0]["job_id"] = "../job"
        document["identity"]["canonical_plan_digest"] = canonical_sha256(
            document["canonical_plan"]
        )
        with self.assertRaisesRegex(ValueError, "path-safe identifier"):
            validate_task_instance_document(document)

    def test_compiler_emits_only_path_safe_ids_and_integer_seeds(self) -> None:
        document = self._one_case_instance().value
        identifiers = [
            document["instance_id"],
            *[
                case["case_id"]
                for case in document["source"]["cases"]
            ],
            *[
                adaptation["adaptation_id"]
                for adaptation in document["adaptations"]
            ],
            *[
                job["job_id"]
                for job in document["inference"]["jobs"]
            ],
        ]
        self.assertTrue(all(
            SAFE_ID_PATTERN.fullmatch(identifier)
            for identifier in identifiers
        ))
        self.assertTrue(all(
            isinstance(job["seed"], int)
            and not isinstance(job["seed"], bool)
            for job in document["canonical_plan"]["jobs"]
        ))

    def test_one_case_dry_matrix_records_plan_and_orchestration_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_path = root / "one-case.json"
            write_json(
                task_path,
                self._one_case_value(task_id="matrix_one_case"),
            )
            output = root / "runs"
            run_dirs = run_matrix(
                dataset_path=LATEST_DATASET,
                task_path=task_path,
                baseline_paths=[GENERIC_BASELINE, PHYSICS_BASELINE],
                output_root=output,
                matrix_id="matrix-dry",
                execute=False,
            )

            index = load_json(output / "matrix-dry.matrix.json")
            self.assertEqual("complete", index["orchestration_status"])
            self.assertEqual("planned", index["status"])
            self.assertEqual(2, len(run_dirs))
            self.assertEqual(
                {
                    self.generic_bundle.baseline_id: "planned",
                    self.physics_bundle.baseline_id: "planned",
                },
                index["atomic_run_statuses"],
            )
            expected_digests = {
                baseline["baseline_id"]: instance["digest"]
                for baseline, instance in zip(
                    index["baselines"],
                    index["task_instances"],
                    strict=True,
                )
            }
            for run_dir in run_dirs:
                run = load_json(run_dir / "run.json")
                self.assertEqual(
                    expected_digests[run["baseline_id"]],
                    run["task_instance_digest"],
                )

    def test_matrix_failure_is_persisted_as_orchestration_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_path = root / "one-case.json"
            write_json(
                task_path,
                self._one_case_value(task_id="matrix_failure_case"),
            )
            output = root / "runs"
            with patch(
                "physbench.orchestration.atomic_runner.run_atomic",
                side_effect=RuntimeError("fixture failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    run_matrix(
                        dataset_path=LATEST_DATASET,
                        task_path=task_path,
                        baseline_paths=[
                            GENERIC_BASELINE,
                            PHYSICS_BASELINE,
                        ],
                        output_root=output,
                        matrix_id="matrix-failure",
                        execute=False,
                    )

            index = load_json(output / "matrix-failure.matrix.json")
            self.assertEqual("failed", index["orchestration_status"])
            self.assertEqual("failed", index["status"])
            self.assertIn("fixture failure", index["error"])


if __name__ == "__main__":
    unittest.main()
