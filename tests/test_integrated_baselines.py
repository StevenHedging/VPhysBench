from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from _paths import ROOT
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.io import canonical_sha256, load_json
from physbench.tasks import load_task


COSMOS_ROOT = ROOT / "baselines" / "cosmos3_nano_i2v"
G15_ROOT = ROOT / "baselines" / "wan22_g15_sparse_motion"
WAN_LORA_ROOT = ROOT / "baselines" / "wan22_lora"
GENERIC_MANIFESTS = (
    COSMOS_ROOT / "baseline.json",
    G15_ROOT / "baseline.json",
    WAN_LORA_ROOT / "baseline.json",
)
PHYSICS_MANIFESTS = (
    COSMOS_ROOT / "physics.baseline.json",
    G15_ROOT / "physics.baseline.json",
    WAN_LORA_ROOT / "physics.baseline.json",
)
ALL_MANIFESTS = (*GENERIC_MANIFESTS, *PHYSICS_MANIFESTS)


def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


class IntegratedBaselineTests(unittest.TestCase):
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

    def test_expected_baseline_identities_are_discovered(
        self,
    ) -> None:
        discovered = discover_baseline_bundles()
        self.assertTrue(
            {
                "cosmos3_nano_i2v_generic",
                "cosmos3_nano_i2v_physics",
                "wan22_g15_sparse_motion_r32_e20_generic",
                "wan22_g15_sparse_motion_r32_e20_physics",
                "wan22_ti2v_5b_lora_r32_v3_generic",
                "wan22_ti2v_5b_lora_r32_v3_physics",
                "wan22_ti2v_5b_lora_r32_quantity_embedding_v1",
            }
            <= set(discovered),
        )

    def test_all_integrated_bundles_use_managed_runtime(self) -> None:
        wan_driver = (WAN_LORA_ROOT / "driver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Wan22ManagedDriver", wan_driver)
        g15_driver = (G15_ROOT / "driver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Wan22ManagedDriver", g15_driver)
        for root in (COSMOS_ROOT, G15_ROOT, WAN_LORA_ROOT):
            self.assertFalse((root / "plugin" / "main.py").exists())
        self.assertTrue(
            (
                ROOT / "src" / "physbench" / "baseline_plugins" / "wan22.py"
            ).is_file()
        )
        self.assertTrue(
            (
                ROOT / "src" / "physbench" / "baseline_runtime"
                / "drivers" / "wan22.py"
            ).is_file()
        )
        for path in ALL_MANIFESTS:
            manifest = load_json(path)
            self.assertEqual("5.0", manifest["schema_version"])
            self.assertEqual(
                "managed", manifest["implementation"]["kind"]
            )
            self.assertNotIn(
                "conditioning", manifest["capabilities"]
            )
        for generic_path, physics_path in zip(
            GENERIC_MANIFESTS, PHYSICS_MANIFESTS
        ):
            generic = load_json(generic_path)
            physics = load_json(physics_path)
            self.assertEqual(
                "ignored",
                generic["input_policy"]["physics"]["usage"],
            )
            self.assertEqual(
                {"type": "none"},
                generic["adapter"]["physics_transform"],
            )
            self.assertEqual(
                "required",
                physics["input_policy"]["physics"]["usage"],
            )
            self.assertEqual(
                "append_structured_text_v1",
                physics["adapter"]["physics_transform"]["type"],
            )

    def test_g15_is_direct_only_and_overlap_is_source_aware(self) -> None:
        manifest = load_json(G15_ROOT / "baseline.json")
        self.assertEqual(
            ["direct_eval"],
            manifest["capabilities"]["task_families"],
        )
        self.assertFalse(
            manifest["model"]["evaluation_eligibility"][
                "official_comparable"
            ]
        )
        self.assertEqual(
            "cd19f851133c8370def00991fa778a3bfb581e35713842438ba068495912906f",
            manifest["model"]["checkpoint_sha256"],
        )
        audit = load_json(
            G15_ROOT / "provenance" / "benchmark_overlap_v3.json"
        )
        self.assertEqual(176, audit["overlap"]["seen_source_cases"])
        self.assertEqual(
            121, audit["overlap"]["view_a"]["train"]["seen"]
        )
        self.assertEqual(
            "source archive member / normalized capture stem",
            audit["matching_policy"]["inclined_plane_slide"],
        )

    def test_wan_lora_checkpoint_identity_is_frozen(self) -> None:
        expected = (
            "7f8f28a36faa309431e7ea58e7de3c61cd58266b"
            "62653ee69c3b9f666745acfe"
        )
        for path in (
            WAN_LORA_ROOT / "baseline.json",
            WAN_LORA_ROOT / "physics.baseline.json",
        ):
            self.assertEqual(
                expected,
                load_json(path)["model"]["checkpoint_sha256"],
            )

    def test_direct_only_bundles_reject_finetune_eval(self) -> None:
        for path in (
            COSMOS_ROOT / "baseline.json",
            COSMOS_ROOT / "physics.baseline.json",
            G15_ROOT / "baseline.json",
            G15_ROOT / "physics.baseline.json",
        ):
            bundle = load_baseline_bundle(path)
            plugin = load_baseline_plugin(bundle)
            with self.assertRaisesRegex(
                ValueError, "does not support task family"
            ):
                plugin.task_builder.build(self.dataset, self.finetune)

    def test_cosmos_native_payload_has_valid_shape_and_no_generic_leak(
        self,
    ) -> None:
        generic_plugin = load_baseline_plugin(
            load_baseline_bundle(COSMOS_ROOT / "baseline.json")
        )
        physics_plugin = load_baseline_plugin(
            load_baseline_bundle(
                COSMOS_ROOT / "physics.baseline.json"
            )
        )
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "collision_1d"
        )
        generic = generic_plugin.task_builder.data_adapter.adapt_case(
            case, role="eval"
        )
        physics = physics_plugin.task_builder.data_adapter.adapt_case(
            case, role="eval"
        )
        shape = generic["native_inputs"]["generation_shape"]
        self.assertEqual("16,9", shape["aspect_ratio"])
        self.assertEqual("480", shape["resolution"])
        self.assertEqual(24, shape["fps"])
        self.assertEqual(0, (shape["num_frames"] - 1) % 4)
        self.assertEqual(case["text"]["prompt"], generic["prompt"])
        self.assertEqual({}, generic["used_parameters"])
        self.assertTrue(physics["used_parameters"])
        self.assertNotEqual(generic["prompt"], physics["prompt"])
        self.assertEqual(
            generic["materialization_fingerprint"],
            physics["materialization_fingerprint"],
        )

    @unittest.skipUnless(
        (COSMOS_ROOT / "baseline.local.json").is_file()
        and (G15_ROOT / "baseline.local.json").is_file()
        and (WAN_LORA_ROOT / "baseline.local.json").is_file(),
        "local model deployments are not configured",
    )
    def test_all_variants_preserve_dataset_and_share_direct_plan(
        self,
    ) -> None:
        before = canonical_sha256({
            "descriptor": self.dataset.descriptor,
            "cases": self.dataset.cases,
            "views": self.dataset.views,
        })
        instances = []
        for path in ALL_MANIFESTS:
            plugin = load_baseline_plugin(load_baseline_bundle(path))
            instance = plugin.task_builder.build(
                self.dataset, self.direct
            )
            instance.verify()
            instances.append(instance.value)
        after = canonical_sha256({
            "descriptor": self.dataset.descriptor,
            "cases": self.dataset.cases,
            "views": self.dataset.views,
        })
        self.assertEqual(before, after)
        canonical_plan = instances[0]["canonical_plan"]
        self.assertTrue(all(
            value["canonical_plan"] == canonical_plan
            for value in instances
        ))
        self.assertFalse(_contains_key(canonical_plan, "conditioning"))
        for generic, physics in zip(instances[:3], instances[3:]):
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
        self.assertTrue(all(
            job["model_ref"] == "baseline://frozen_model"
            for value in instances
            for job in value["inference"]["jobs"]
        ))

    def test_cosmos_base_checkpoint_identity_is_explicit(self) -> None:
        manifest = load_json(COSMOS_ROOT / "baseline.json")
        self.assertEqual("base_pretrained", manifest["model"]["variant"])
        self.assertEqual(
            "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
            manifest["model"]["snapshot_revision"],
        )
        self.assertEqual(
            {"config.json", "model.safetensors.index.json"},
            set(manifest["model"]["identity_files"]),
        )

    def test_cosmos_runtime_preserves_virtual_environment_boundary(
        self,
    ) -> None:
        driver_path = COSMOS_ROOT / "driver.py"
        spec = importlib.util.spec_from_file_location(
            "physbench_test_cosmos_driver",
            driver_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv = root / ".venv"
            managed = root / "managed-python" / "bin" / "python"
            managed.parent.mkdir(parents=True)
            managed.touch()
            (venv / "bin").mkdir(parents=True)
            python = venv / "bin" / "python"
            python.symlink_to(managed)
            self.assertEqual(
                venv,
                module.Driver._environment_root(python),
            )
            self.assertNotEqual(
                venv,
                python.resolve().parents[1],
            )
        driver = module.Driver.__new__(module.Driver)
        driver.bundle = SimpleNamespace(value={
            "runtime": {
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                "gpus_per_worker": 4,
            }
        })
        self.assertEqual(
            [["0", "1", "2", "3"], ["4", "5", "6", "7"]],
            driver._gpu_groups(),
        )
        first = driver._worker_index("case-a")
        self.assertEqual(first, driver._worker_index("case-a"))
        self.assertIn(first, (0, 1))

    @unittest.skipUnless(
        (COSMOS_ROOT / "baseline.local.json").is_file()
        and (G15_ROOT / "baseline.local.json").is_file()
        and (WAN_LORA_ROOT / "baseline.local.json").is_file(),
        "local model deployments are not configured",
    )
    def test_managed_variants_plan_equivalent_one_case_dry_runs(
        self,
    ) -> None:
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "collision_1d"
        )
        task_value = copy.deepcopy(self.direct.value)
        task_value["selection"]["scene_ids"] = ["collision_1d"]
        task_value["selection"]["case_ids"] = [case["case_id"]]
        task = TaskSpec(
            self.direct.path,
            task_value,
            canonical_sha256(task_value),
        )
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            instances = []
            for path in ALL_MANIFESTS:
                bundle = load_baseline_bundle(path)
                plugin = load_baseline_plugin(bundle)
                instance = plugin.task_builder.build(self.dataset, task)
                instances.append(instance)
                run_dir = parent / bundle.baseline_id
                for child in (
                    "adaptations",
                    "artifacts",
                    "jobs",
                    "logs",
                    "predictions",
                    "task_instance",
                    "training",
                ):
                    (run_dir / child).mkdir(parents=True)
                _, predictions = plugin.run_task(
                    instance=instance,
                    run_dir=run_dir,
                    execute=False,
                    stop_after_training=False,
                )
                self.assertEqual(1, len(predictions))
                self.assertEqual("planned", predictions[0]["status"])
                self.assertIsNone(predictions[0]["video_path"])
                jobs = sorted((run_dir / "jobs").glob("*.json"))
                self.assertTrue(jobs)
                if path.parent == COSMOS_ROOT:
                    payload = load_json(next(
                        (run_dir / "jobs").glob("*.payload.json")
                    ))
                    self.assertEqual("image2video", payload["model_mode"])
                    self.assertEqual(24, payload["fps"])
                    self.assertEqual(121, payload["num_frames"])
                    self.assertEqual("16,9", payload["aspect_ratio"])
                    self.assertEqual(42, payload["seed"])
                else:
                    job = load_json(jobs[0])
                    self.assertIsNone(
                        job["media_adaptation"]["reference"]
                    )
                    generation_frames = job["wan22"]["generation"][
                        "num_frames"
                    ]
                    self.assertEqual(121, generation_frames)
                    self.assertEqual(0, (generation_frames - 1) % 4)
                    self.assertIsNone(
                        job["evaluation_reference_video"]
                    )
                    self.assertIsNone(job["visual_reference_video"])
                    self.assertTrue(
                        Path(job["output_video"]).is_relative_to(run_dir)
                    )
            canonical_plan = instances[0].value["canonical_plan"]
            self.assertTrue(all(
                instance.value["canonical_plan"] == canonical_plan
                for instance in instances
            ))


if __name__ == "__main__":
    unittest.main()
