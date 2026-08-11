from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _paths import ROOT
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime import create_baseline_scaffold
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.io import (
    canonical_sha256,
    load_json,
    write_json,
    write_jsonl,
)
from physbench.orchestration import compile_task_instance
from physbench.tasks import load_task


DRIVER = """\
from pathlib import Path
from physbench.baseline_runtime import DirectManagedDriver


class Driver(DirectManagedDriver):
    def prepare_job(
        self, *, job, case, adaptation, source_root, run_dir
    ):
        return {
            "job_id": job["job_id"],
            "output_video": str(
                run_dir / "predictions" / f"{job['job_id']}.mp4"
            ),
        }

    def execute_job(self, spec, *, log_path):
        output = Path(spec["output_video"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fixture-video")
        return {"return_code": 0, "log_path": str(log_path)}
"""


def _create_run_directories(run_dir: Path) -> None:
    for child in (
        "adaptations",
        "artifacts",
        "jobs",
        "logs",
        "predictions",
        "provenance",
        "task_instance",
        "training",
    ):
        (run_dir / child).mkdir(parents=True)


def _input_policy(physics_usage: str = "ignored") -> dict:
    return {
        "schema_version": "1.0",
        "case_view": "conditionable_case_v1",
        "text": {
            "source": "case.text.prompt",
            "usage": "required",
        },
        "physics": {
            "source": "case.physics",
            "usage": physics_usage,
            "representations": (
                [] if physics_usage == "ignored" else ["structured_text"]
            ),
        },
    }


def _adapter(physics_usage: str = "ignored") -> dict:
    return {
        "kind": "standard",
        "preset": "standard_i2v_v1",
        "first_frame_policy": "require_asset",
        "physics_transform": (
            {"type": "none"}
            if physics_usage == "ignored"
            else {
                "type": "append_structured_text_v1",
                "template_set": "six_scene_physics_clauses_v2",
            }
        ),
        "spatial": {
            "scene_profiles": {
                scene_id: {"width": 832, "height": 480}
                for scene_id in (
                    "pendulum",
                    "free_fall",
                    "collision_1d",
                    "inclined_plane_slide",
                    "uniform_circular_motion",
                    "parabolic_motion",
                    "push_bottle",
                )
            }
        },
        "temporal": {
            "fps": 24,
            "num_frames": 121,
            "valid_frame_rule": "4n+1",
        },
    }


def _manifest(
    baseline_id: str,
    *,
    kind: str,
    physics_usage: str = "ignored",
) -> dict:
    value = {
        "schema_version": "5.0",
        "baseline_id": baseline_id,
        "baseline_version": "0.1.0",
        "description": f"Test fixture Baseline: {baseline_id}.",
        "implementation": {
            "kind": kind,
            "fingerprint_paths": [],
        },
        "supported_scenes": "all",
        "capabilities": {
            "task_families": ["direct_eval"],
            "generation_modes": ["i2v"],
            "train": False,
            "finetune": False,
            "generate": True,
        },
        "input_policy": _input_policy(physics_usage),
        "model": {},
        "runtime": {},
        "adapter": _adapter(physics_usage),
    }
    if kind == "managed":
        value["implementation"]["driver"] = "driver.py"
        value["runner"] = {"type": "fixture", "config": {}}
    return value


def _bundle(
    root: Path,
    name: str,
    kind: str,
    *,
    physics_usage: str = "ignored",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    write_json(
        directory / "baseline.json",
        _manifest(
            name,
            kind=kind,
            physics_usage=physics_usage,
        ),
    )
    if kind == "managed":
        (directory / "driver.py").write_text(DRIVER, encoding="utf-8")
    return directory


class ManagedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cls.direct = load_task(
            ROOT
            / "tasks"
            / "official"
            / "five_scene_direct_eval_v1.json"
        )
        cls.finetune = load_task(
            ROOT
            / "tasks"
            / "official"
            / "seven_scene_train_five_scene_eval_v1.json"
        )

    def _one_case_task(self) -> TaskSpec:
        source = self.direct
        value = copy.deepcopy(self.direct.value)
        case = self.dataset.cases[0]
        value["selection"]["evaluation_scene_ids"] = [case["scene_id"]]
        value["selection"]["case_ids"] = [case["case_id"]]
        return TaskSpec(source.path, value, canonical_sha256(value))

    def test_release_tree_has_no_integrated_baselines(self) -> None:
        self.assertEqual({}, discover_baseline_bundles(ROOT / "baselines"))

    def test_v5_managed_and_submission_are_discovered_without_endpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            managed = _bundle(root, "managed_fixture", "managed")
            submission = _bundle(
                root, "submission_fixture", "submission"
            )
            self.assertEqual(
                {"managed_fixture", "submission_fixture"},
                set(discover_baseline_bundles(root)),
            )
            self.assertFalse((managed / "plugin" / "main.py").exists())
            self.assertFalse((submission / "plugin" / "main.py").exists())
            self.assertEqual(
                "managed",
                load_baseline_bundle(managed).value["implementation"]["kind"],
            )
            self.assertEqual(
                "submission",
                load_baseline_bundle(submission).value[
                    "implementation"
                ]["kind"],
            )

    def test_managed_builder_is_sealed_deterministic_and_does_not_leak(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _bundle(Path(temporary), "fixture", "managed")
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            first = compile_task_instance(plugin, self.dataset, self.direct)
            second = compile_task_instance(plugin, self.dataset, self.direct)
            self.assertEqual(first.digest, second.digest)
            first.verify()
            self.assertEqual(
                canonical_sha256(first.value["canonical_plan"]),
                first.value["identity"]["canonical_plan_digest"],
            )
            self.assertNotIn("conditioning", first.value)
            self.assertTrue(all(
                item["used_parameters"] == {}
                for item in first.value["adaptations"]
            ))
            case = copy.deepcopy(self.dataset.cases[0])
            changed = copy.deepcopy(case)
            parameter = next(iter(changed["physics"].values()))
            parameter["value"] = 999
            adapter = plugin.task_builder.data_adapter
            self.assertEqual(
                adapter.adapt_case(case, role="eval")["native_inputs"],
                adapter.adapt_case(changed, role="eval")["native_inputs"],
            )
            self.assertEqual(
                case["text"]["prompt"],
                adapter.adapt_case(case, role="eval")["prompt"],
            )

    def test_physics_usage_is_baseline_owned_and_shares_media_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            generic_root = _bundle(
                parent, "fixture_generic", "managed"
            )
            physics_root = _bundle(
                parent,
                "fixture_physics",
                "managed",
                physics_usage="required",
            )
            generic = load_baseline_plugin(
                load_baseline_bundle(generic_root)
            )
            physics = load_baseline_plugin(
                load_baseline_bundle(physics_root)
            )
            task = self._one_case_task()
            generic_instance = compile_task_instance(generic,
                self.dataset, task
            )
            physics_instance = compile_task_instance(physics,
                self.dataset, task
            )
            self.assertEqual(
                generic_instance.value["canonical_plan"],
                physics_instance.value["canonical_plan"],
            )
            generic_adapter = generic.task_builder.data_adapter
            physics_adapter = physics.task_builder.data_adapter
            self.assertNotEqual(
                generic_adapter.fingerprint,
                physics_adapter.fingerprint,
            )
            self.assertEqual(
                generic_adapter.materialization_fingerprint,
                physics_adapter.materialization_fingerprint,
            )
            case = self.dataset.cases[0]
            generic_adaptation = generic_adapter.adapt_case(
                case, role="eval"
            )
            physics_adaptation = physics_adapter.adapt_case(
                case, role="eval"
            )
            self.assertEqual(
                case["text"]["prompt"], generic_adaptation["prompt"]
            )
            self.assertNotEqual(
                generic_adaptation["prompt"],
                physics_adaptation["prompt"],
            )
            self.assertEqual({}, generic_adaptation["used_parameters"])
            self.assertTrue(physics_adaptation["used_parameters"])

    def test_managed_driver_is_fingerprinted_and_path_escape_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = _bundle(parent, "fixture", "managed")
            first = load_baseline_bundle(root)
            driver = root / "driver.py"
            driver.write_text(
                driver.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )
            second = load_baseline_bundle(root)
            self.assertNotEqual(first.digest, second.digest)
            value = load_json(root / "baseline.json")
            value["implementation"]["driver"] = "../outside.py"
            write_json(root / "baseline.json", value)
            with self.assertRaisesRegex(ValueError, "bundle-relative"):
                load_baseline_bundle(root)

    def test_managed_capability_rejection_happens_before_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _bundle(Path(temporary), "fixture", "managed")
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            with self.assertRaisesRegex(
                ValueError, "does not support task family"
            ):
                compile_task_instance(plugin, self.dataset, self.finetune)

    def test_finetune_requires_support_for_all_training_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _bundle(Path(temporary), "fixture", "managed")
            value = load_json(root / "baseline.json")
            value["supported_scenes"] = [
                "pendulum",
                "collision_1d",
                "inclined_plane_slide",
                "uniform_circular_motion",
                "parabolic_motion",
            ]
            value["capabilities"].update({
                "task_families": ["direct_eval", "finetune_eval"],
                "train": True,
                "finetune": True,
            })
            value["trainer"] = {"type": "fixture", "config": {}}
            write_json(root / "baseline.json", value)
            plugin = load_baseline_plugin(load_baseline_bundle(root))

            with self.assertRaisesRegex(
                ValueError,
                "push_bottle.*vertical_spring_oscillator",
            ):
                compile_task_instance(plugin, self.dataset, self.finetune)

    def test_managed_plugin_rejects_driver_output_identity_and_coverage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = _bundle(parent, "fixture", "managed")
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            instance = compile_task_instance(plugin,
                self.dataset, self._one_case_task()
            )
            job = instance.value["inference"]["jobs"][0]
            valid = {
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "baseline_id": plugin.bundle.baseline_id,
                "evaluation_partition": job["evaluation_partition"],
                "seed": int(job["seed"]),
                "status": "planned",
                "video_path": None,
            }
            training = {"status": "not_requested"}

            invalid_identities = {}
            missing_seed = copy.deepcopy(valid)
            missing_seed.pop("seed")
            invalid_identities["missing seed"] = missing_seed
            wrong_seed = copy.deepcopy(valid)
            wrong_seed["seed"] += 1
            invalid_identities["wrong seed"] = wrong_seed
            wrong_identity = copy.deepcopy(valid)
            wrong_identity["baseline_id"] = "different_baseline"
            invalid_identities["wrong identity"] = wrong_identity
            for label, prediction in invalid_identities.items():
                with self.subTest(label=label):
                    run_dir = parent / label.replace(" ", "_")
                    _create_run_directories(run_dir)
                    with patch.object(
                        plugin.driver,
                        "run_task",
                        return_value=(training, [prediction]),
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "identity mismatch"
                        ):
                            plugin.run_task(
                                instance=instance,
                                run_dir=run_dir,
                                execute=False,
                                stop_after_training=False,
                            )

            coverage_cases = {
                "duplicate job": [valid, copy.deepcopy(valid)],
                "missing job": [],
            }
            for label, predictions in coverage_cases.items():
                with self.subTest(label=label):
                    run_dir = parent / label.replace(" ", "_")
                    _create_run_directories(run_dir)
                    with patch.object(
                        plugin.driver,
                        "run_task",
                        return_value=(training, predictions),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            (
                                "duplicate prediction"
                                if predictions
                                else "coverage mismatch"
                            ),
                        ):
                            plugin.run_task(
                                instance=instance,
                                run_dir=run_dir,
                                execute=False,
                                stop_after_training=False,
                            )

    def test_submission_requires_exact_identity_and_imports_run_local(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = _bundle(parent, "fixture", "submission")
            task = self._one_case_task()
            provisional = load_baseline_plugin(
                load_baseline_bundle(root)
            )
            instance = compile_task_instance(provisional, self.dataset, task)
            job = instance.value["inference"]["jobs"][0]
            source = parent / "external.mp4"
            source.write_bytes(b"submitted-video")
            manifest = parent / "submission.jsonl"
            write_jsonl(manifest, [{
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "seed": int(job["seed"]),
                "video_path": str(source),
            }])
            write_json(root / "baseline.local.json", {
                "runtime": {"submission_manifest": str(manifest)}
            })
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            instance = compile_task_instance(plugin, self.dataset, task)
            run_dir = parent / "run"
            for child in ("predictions", "provenance"):
                (run_dir / child).mkdir(parents=True)
            _, predictions = plugin.run_task(
                instance=instance,
                run_dir=run_dir,
                execute=True,
                stop_after_training=False,
            )
            destination = Path(predictions[0]["video_path"])
            self.assertTrue(destination.is_file())
            self.assertEqual(b"submitted-video", destination.read_bytes())
            self.assertTrue(destination.is_relative_to(run_dir))

            bad = [{
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "seed": int(job["seed"]) + 1,
                "video_path": str(source),
            }]
            write_jsonl(manifest, bad)
            with self.assertRaisesRegex(
                ValueError, "changed after Baseline deployment"
            ):
                plugin.run_task(
                    instance=instance,
                    run_dir=run_dir,
                    execute=True,
                    stop_after_training=False,
                )
            reloaded = load_baseline_plugin(load_baseline_bundle(root))
            bad_instance = compile_task_instance(reloaded,
                self.dataset, task
            )
            with self.assertRaisesRegex(
                ValueError, "identity mismatch"
            ):
                reloaded.run_task(
                    instance=bad_instance,
                    run_dir=run_dir,
                    execute=True,
                    stop_after_training=False,
                )

    def test_submission_rejects_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = _bundle(parent, "fixture", "submission")
            manifest = parent / "submission.jsonl"
            manifest.write_text("", encoding="utf-8")
            write_json(root / "baseline.local.json", {
                "runtime": {"submission_manifest": str(manifest)}
            })
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            instance = compile_task_instance(plugin,
                self.dataset, self._one_case_task()
            )
            with self.assertRaisesRegex(
                ValueError, "coverage mismatch"
            ):
                plugin.run_task(
                    instance=instance,
                    run_dir=parent / "run",
                    execute=True,
                    stop_after_training=False,
                )

    def test_scaffolds_pass_registry_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backend in ("managed-i2v", "submission"):
                directory = create_baseline_scaffold(
                    name=backend.replace("-", "_"),
                    backend=backend,
                    root=root,
                )
                example = load_json(
                    directory / "baseline.local.example.json"
                )
                self.assertFalse(
                    Path(example["model"]["checkpoint"]).is_absolute()
                )
                if backend == "submission":
                    self.assertFalse(
                        Path(
                            example["runtime"]["submission_manifest"]
                        ).is_absolute()
                    )
                plugin = load_baseline_plugin(
                    load_baseline_bundle(directory)
                )
                self.assertIsNotNone(plugin.task_builder.fingerprint)


if __name__ == "__main__":
    unittest.main()
