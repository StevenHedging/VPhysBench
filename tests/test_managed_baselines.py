from __future__ import annotations

import csv
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _paths import ROOT
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime import create_baseline_scaffold
from physbench.baseline_runtime.drivers.wan22 import Wan22ManagedDriver
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.io import (
    canonical_sha256,
    load_json,
    write_json,
    write_jsonl,
)
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
            / "five_scene_direct_eval.json"
        )
        cls.finetune = load_task(
            ROOT
            / "tasks"
            / "official"
            / "five_scene_finetune_eval.json"
        )

    def _one_case_task(self) -> TaskSpec:
        source = self.direct
        value = copy.deepcopy(self.direct.value)
        case = self.dataset.cases[0]
        value["selection"]["scene_ids"] = [case["scene_id"]]
        value["selection"]["case_ids"] = [case["case_id"]]
        return TaskSpec(source.path, value, canonical_sha256(value))

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
            first = plugin.task_builder.build(self.dataset, self.direct)
            second = plugin.task_builder.build(self.dataset, self.direct)
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
            generic_instance = generic.task_builder.build(
                self.dataset, task
            )
            physics_instance = physics.task_builder.build(
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
                plugin.task_builder.build(self.dataset, self.finetune)

    def test_managed_plugin_rejects_driver_output_identity_and_coverage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = _bundle(parent, "fixture", "managed")
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            instance = plugin.task_builder.build(
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

    def test_wan_lora_finetune_dry_run_plans_metadata_and_predictions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            bundle_root = parent / "wan22_lora"
            bundle_root.mkdir()
            source_root = ROOT / "baselines" / "wan22_lora"
            manifest = load_json(source_root / "baseline.json")
            manifest["model"]["frozen_lora_checkpoint"] = None
            manifest["runtime"].update({
                "project_root": str(parent / "wan_project"),
                "python": sys.executable,
                "model_base": str(parent / "models"),
                "cuda_visible_devices": "0",
                "accelerate_config": None,
            })
            write_json(bundle_root / "baseline.json", manifest)
            (bundle_root / "driver.py").write_text(
                (source_root / "driver.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )

            task_value = copy.deepcopy(self.finetune.value)
            task_value["task_id"] = "pendulum_finetune_dry_run"
            task_value["selection"]["scene_ids"] = ["pendulum"]
            task = TaskSpec(
                self.finetune.path,
                task_value,
                canonical_sha256(task_value),
            )
            instance = plugin.task_builder.build(self.dataset, task)
            run_dir = parent / "run"
            _create_run_directories(run_dir)
            training, predictions = plugin.run_task(
                instance=instance,
                run_dir=run_dir,
                execute=False,
                stop_after_training=False,
            )

            self.assertEqual("planned", training["status"])
            metadata_path = Path(training["metadata"])
            self.assertTrue(metadata_path.is_file())
            with metadata_path.open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                metadata_rows = list(reader)
                self.assertIn("text_transform_id", reader.fieldnames)
                self.assertNotIn(
                    "prompt_profile_id", reader.fieldnames
                )
            self.assertTrue(metadata_rows)
            self.assertTrue(all(row["prompt"] for row in metadata_rows))

            frozen_jobs = {
                job["job_id"]: job
                for job in instance.value["inference"]["jobs"]
            }
            self.assertEqual(set(frozen_jobs), {
                prediction["job_id"] for prediction in predictions
            })
            forbidden = {
                "conditioning",
                "prompt_profile_id",
                "evaluation_reference_video",
                "visual_reference_video",
                "reference_video",
                "physics_reference_video",
            }
            for prediction in predictions:
                job = frozen_jobs[prediction["job_id"]]
                self.assertEqual(int(job["seed"]), prediction["seed"])
                self.assertEqual("planned", prediction["status"])
                self.assertIsNone(prediction["video_path"])
                self.assertFalse(forbidden & set(prediction))

    def test_wan_frozen_checkpoint_requires_and_verifies_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.safetensors"
            checkpoint.write_bytes(b"frozen-checkpoint")
            model = {
                "frozen_lora_checkpoint": str(checkpoint),
                "checkpoint_sha256": None,
            }
            bundle = SimpleNamespace(value={
                "model": model,
                "runtime": {"python": sys.executable},
            })
            driver = Wan22ManagedDriver(bundle)

            with self.assertRaisesRegex(
                ValueError, "requires model.checkpoint_sha256"
            ):
                driver.validate_deployment()

            model["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                driver.validate_deployment()

            model["checkpoint_sha256"] = (
                "eb34e0c243f4f0947df3c07e185e05a43b87b3f"
                "353a31fb5c018293d2ff2f138"
            )
            driver.validate_deployment()

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
            instance = provisional.task_builder.build(self.dataset, task)
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
            instance = plugin.task_builder.build(self.dataset, task)
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
            bad_instance = reloaded.task_builder.build(
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
            instance = plugin.task_builder.build(
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
                plugin = load_baseline_plugin(
                    load_baseline_bundle(directory)
                )
                self.assertIsNotNone(plugin.task_builder.fingerprint)


if __name__ == "__main__":
    unittest.main()
