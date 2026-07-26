from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from _paths import ROOT
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime import create_baseline_scaffold
from physbench.data_layout import V2_DATASET
from physbench.datasets import load_dataset_v2
from physbench.domain import TaskSpec
from physbench.io import (
    canonical_sha256,
    load_json,
    write_json,
    write_jsonl,
)
from physbench.tasks import load_task_v2


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


def _adapter() -> dict:
    return {
        "preset": "standard_i2v_v1",
        "profile_set": "five_scene_i2v_v1",
        "first_frame_policy": "asset_or_reference_frame0",
        "spatial": {
            "scene_profiles": {
                scene_id: {"width": 832, "height": 480}
                for scene_id in (
                    "pendulum",
                    "free_fall",
                    "collision_1d",
                    "inclined_plane_slide",
                    "uniform_circular_motion",
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
) -> dict:
    value = {
        "schema_version": "4.0",
        "baseline_id": baseline_id,
        "baseline_version": "0.1.0",
        "implementation": {
            "kind": kind,
            "fingerprint_paths": [],
        },
        "supported_scenes": "all",
        "capabilities": {
            "task_families": ["direct_eval"],
            "conditioning": ["generic", "physics"],
        },
        "model": {},
        "runtime": {},
        "adapter": _adapter(),
    }
    if kind == "managed":
        value["implementation"]["driver"] = "driver.py"
        value["runner"] = {"type": "fixture", "config": {}}
    return value


def _bundle(root: Path, name: str, kind: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    write_json(directory / "baseline.json", _manifest(name, kind=kind))
    if kind == "managed":
        (directory / "driver.py").write_text(DRIVER, encoding="utf-8")
    return directory


class ManagedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset_v2(V2_DATASET, check_assets=False)
        cls.direct = load_task_v2(
            ROOT / "tasks" / "official" / "direct_eval_generic.json"
        )
        cls.finetune = load_task_v2(
            ROOT / "tasks" / "official" / "finetune_eval_generic.json"
        )

    def _one_case_task(self, *, conditioning: str = "generic") -> TaskSpec:
        source = load_task_v2(
            ROOT
            / "tasks"
            / "official"
            / f"direct_eval_{conditioning}.json"
        )
        value = copy.deepcopy(source.value)
        case = self.dataset.cases[0]
        value["selection"]["scene_ids"] = [case["scene_id"]]
        value["selection"]["case_ids"] = [case["case_id"]]
        return TaskSpec(source.path, value, canonical_sha256(value))

    def test_v4_managed_and_submission_are_discovered_without_endpoints(
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
            self.assertTrue(all(
                item["used_parameters"] == {}
                for item in first.value["adaptations"]
            ))
            case = copy.deepcopy(self.dataset.cases[0])
            changed = copy.deepcopy(case)
            changed["physics"] = {"impossible": {"value": 999}}
            adapter = plugin.task_builder.data_adapter
            self.assertEqual(
                adapter.adapt_case(
                    case, "generic", role="eval"
                )["native_inputs"],
                adapter.adapt_case(
                    changed, "generic", role="eval"
                )["native_inputs"],
            )

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
                "conditioning": "generic",
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
                "conditioning": "physics",
                "seed": int(job["seed"]),
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
