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
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.io import canonical_sha256, load_json, write_json
from physbench.tasks import load_task, plan_atomic_task


SCENE_IDS = [
    "pendulum",
    "collision_1d",
    "inclined_plane_slide",
    "uniform_circular_motion",
    "parabolic_motion",
    "push_bottle",
]


DRIVER = """\
from pathlib import Path

from physbench.baseline_runtime import DirectManagedDriver

from .helper import OUTPUT_SUFFIX


class Driver(DirectManagedDriver):
    def prepare_job(
        self, *, job, case, adaptation, source_root, run_dir
    ):
        return {
            "job_id": job["job_id"],
            "output_video": str(
                run_dir
                / "predictions"
                / f"{job['job_id']}{OUTPUT_SUFFIX}"
            ),
        }

    def execute_job(self, spec, *, log_path):
        output = Path(spec["output_video"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fixture-video")
        return {"return_code": 0, "log_path": str(log_path)}
"""


HELPER = """\
OUTPUT_SUFFIX = ".mp4"
"""


def _manifest(baseline_id: str) -> dict:
    return {
        "schema_version": "5.0",
        "baseline_id": baseline_id,
        "baseline_version": "0.1.0",
        "implementation": {
            "kind": "managed",
            "driver": "driver.py",
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
        "input_policy": {
            "schema_version": "1.0",
            "case_view": "conditionable_case_v1",
            "text": {
                "source": "case.text.prompt",
                "usage": "required",
            },
            "physics": {
                "source": "case.physics",
                "usage": "ignored",
                "representations": [],
            },
        },
        "model": {
            "model_id": "fixture/model",
            "checkpoint": None,
        },
        "runtime": {"device": "cpu"},
        "adapter": {
            "kind": "standard",
            "preset": "standard_i2v_v1",
            "first_frame_policy": "require_asset",
            "physics_transform": {"type": "none"},
            "spatial": {
                "scene_profiles": {
                    scene_id: {"width": 832, "height": 480}
                    for scene_id in SCENE_IDS
                }
            },
            "temporal": {
                "fps": 24,
                "num_frames": 121,
                "valid_frame_rule": "4n+1",
            },
        },
        "runner": {
            "type": "fixture_i2v_v1",
            "config": {},
        },
    }


def _create_bundle(root: Path, name: str, baseline_id: str) -> Path:
    bundle = root / name
    bundle.mkdir(parents=True)
    (bundle / "driver.py").write_text(DRIVER, encoding="utf-8")
    (bundle / "helper.py").write_text(HELPER, encoding="utf-8")
    write_json(bundle / "baseline.json", _manifest(baseline_id))
    return bundle


class BaselineBundleV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cls.task = load_task(
            ROOT
            / "tasks"
            / "official"
            / "five_scene_direct_eval.json"
        )

    def test_directory_discovery_and_id_resolution_require_no_registry_edit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(
                root,
                "fixture_directory",
                "fixture_managed",
            )

            discovered = discover_baseline_bundles(root)

            self.assertEqual(
                bundle_root / "baseline.json",
                discovered["fixture_managed"],
            )
            bundle = load_baseline_bundle(
                "fixture_managed",
                baselines_root=root,
            )
            plugin = load_baseline_plugin(bundle)
            self.assertEqual("fixture_managed", bundle.baseline_id)
            self.assertEqual(
                "managed_task_builder_v1",
                plugin.task_builder.describe()["type"],
            )

    def test_duplicate_baseline_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _create_bundle(root, "first", "duplicate")
            _create_bundle(root, "second", "duplicate")

            with self.assertRaisesRegex(ValueError, "duplicate baseline_id"):
                discover_baseline_bundles(root)

    def test_driver_and_python_helpers_enter_portable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = _create_bundle(
                Path(temporary),
                "fixture",
                "fixture",
            )
            first = load_baseline_bundle(bundle_root)

            driver = bundle_root / "driver.py"
            driver.write_text(
                driver.read_text(encoding="utf-8") + "\n# driver changed\n",
                encoding="utf-8",
            )
            after_driver_change = load_baseline_bundle(bundle_root)

            helper = bundle_root / "helper.py"
            helper.write_text(
                helper.read_text(encoding="utf-8") + "\n# helper changed\n",
                encoding="utf-8",
            )
            after_helper_change = load_baseline_bundle(bundle_root)

            self.assertNotEqual(first.digest, after_driver_change.digest)
            self.assertNotEqual(
                after_driver_change.digest,
                after_helper_change.digest,
            )

    def test_local_override_changes_only_deployment_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = _create_bundle(
                Path(temporary),
                "fixture",
                "fixture",
            )
            first = load_baseline_bundle(bundle_root)
            portable_policy = copy.deepcopy(first.value["input_policy"])
            write_json(
                bundle_root / "baseline.local.json",
                {
                    "runtime": {"device": "cuda:7"},
                    "model": {
                        "checkpoint": "/models/fixture.safetensors",
                    },
                },
            )

            second = load_baseline_bundle(bundle_root)

            self.assertEqual(first.digest, second.digest)
            self.assertNotEqual(
                first.deployment_digest,
                second.deployment_digest,
            )
            self.assertEqual("cuda:7", second.value["runtime"]["device"])
            self.assertEqual(
                "/models/fixture.safetensors",
                second.value["model"]["checkpoint"],
            )
            self.assertEqual(portable_policy, second.value["input_policy"])

    def test_bundle_relative_paths_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = _create_bundle(
                Path(temporary),
                "fixture",
                "fixture",
            )
            manifest_path = bundle_root / "baseline.json"

            invalid_driver = load_json(manifest_path)
            invalid_driver["implementation"]["driver"] = "../outside.py"
            write_json(manifest_path, invalid_driver)
            with self.assertRaisesRegex(ValueError, "bundle-relative"):
                load_baseline_bundle(bundle_root)

            invalid_glob = _manifest("fixture")
            invalid_glob["implementation"]["fingerprint_paths"] = [
                "../outside.py"
            ]
            write_json(manifest_path, invalid_glob)
            with self.assertRaisesRegex(ValueError, "bundle-relative"):
                load_baseline_bundle(bundle_root)

    def test_task_builder_preserves_frozen_canonical_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = _create_bundle(
                Path(temporary),
                "fixture",
                "fixture",
            )
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )
            canonical_plan = plan_atomic_task(self.task, self.dataset)
            frozen_plan = copy.deepcopy(canonical_plan.value)
            frozen_dataset = canonical_sha256(
                {
                    "descriptor": self.dataset.descriptor,
                    "cases": self.dataset.cases,
                    "views": self.dataset.views,
                }
            )

            instance = plugin.task_builder.compile(
                self.dataset,
                self.task,
                canonical_plan,
            )
            value = instance.value

            self.assertEqual(frozen_plan, canonical_plan.value)
            self.assertEqual(frozen_plan, value["canonical_plan"])
            self.assertEqual(
                canonical_sha256(frozen_plan),
                value["identity"]["canonical_plan_digest"],
            )
            self.assertEqual(
                frozen_dataset,
                canonical_sha256(
                    {
                        "descriptor": self.dataset.descriptor,
                        "cases": self.dataset.cases,
                        "views": self.dataset.views,
                    }
                ),
            )
            self.assertNotIn("conditioning", value["semantics"])
            self.assertTrue(
                all(
                    "conditioning" not in job
                    for job in value["canonical_plan"]["jobs"]
                )
            )

    def test_local_override_cannot_change_input_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = _create_bundle(
                Path(temporary),
                "fixture",
                "fixture",
            )
            changed_policy = copy.deepcopy(_manifest("fixture")["input_policy"])
            changed_policy["physics"] = {
                "source": "case.physics",
                "usage": "required",
                "representations": ["structured_text"],
            }
            write_json(
                bundle_root / "baseline.local.json",
                {"input_policy": changed_policy},
            )

            with self.assertRaisesRegex(ValueError, "may only override"):
                load_baseline_bundle(bundle_root)

    def test_standard_adapter_manifest_is_strict(self) -> None:
        mutations = {
            "unknown": (
                lambda value: value["adapter"].__setitem__(
                    "unexpected", True
                ),
                "unknown fields",
            ),
            "preset": (
                lambda value: value["adapter"].pop("preset"),
                "unsupported standard adapter preset",
            ),
            "spatial": (
                lambda value: value["adapter"].pop("spatial"),
                "requires a spatial object",
            ),
            "temporal": (
                lambda value: value["adapter"].pop("temporal"),
                "requires a temporal object",
            ),
            "first_frame_policy": (
                lambda value: value["adapter"].pop(
                    "first_frame_policy"
                ),
                "first_frame_policy=require_asset",
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, (mutate, message) in mutations.items():
                with self.subTest(name=name):
                    bundle_root = _create_bundle(
                        root,
                        name,
                        f"strict_{name}",
                    )
                    path = bundle_root / "baseline.json"
                    value = load_json(path)
                    mutate(value)
                    write_json(path, value)
                    with self.assertRaisesRegex(ValueError, message):
                        load_baseline_bundle(bundle_root)


if __name__ == "__main__":
    unittest.main()
