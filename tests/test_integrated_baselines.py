from __future__ import annotations

import unittest
from pathlib import Path

from _paths import ROOT
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.data_layout import V3_DATASET
from physbench.datasets import load_dataset_v2
from physbench.io import canonical_sha256, load_json
from physbench.tasks import load_task_v2


COSMOS_ROOT = ROOT / "baselines" / "cosmos3_nano_i2v"
G15_ROOT = ROOT / "baselines" / "wan22_g15_sparse_motion"


class IntegratedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset_v2(V3_DATASET, check_assets=False)
        cls.direct_generic = load_task_v2(
            ROOT / "tasks" / "official"
            / "five_scene_direct_eval_generic.json"
        )
        cls.direct_physics = load_task_v2(
            ROOT / "tasks" / "official"
            / "five_scene_direct_eval_physics.json"
        )
        cls.finetune = load_task_v2(
            ROOT / "tasks" / "official"
            / "five_scene_finetune_eval_generic.json"
        )

    def test_all_three_bundles_are_discovered_without_registry_entries(
        self,
    ) -> None:
        discovered = discover_baseline_bundles()
        self.assertEqual(
            {
                "cosmos3_nano_i2v",
                "wan22_g15_sparse_motion_r32_e20",
                "wan22_ti2v_5b_lora_r32_v3",
            },
            set(discovered),
        )

    def test_wan_bundles_share_one_model_family_implementation(self) -> None:
        for root in (ROOT / "baselines" / "wan22_lora", G15_ROOT):
            entrypoint = (root / "plugin" / "main.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "physbench.baseline_plugins.wan22", entrypoint
            )
            self.assertFalse((root / "plugin" / "implementation.py").exists())
        self.assertTrue(
            (
                ROOT / "src" / "physbench" / "baseline_plugins" / "wan22.py"
            ).is_file()
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

    def test_direct_only_bundles_reject_finetune_eval(self) -> None:
        for root in (COSMOS_ROOT, G15_ROOT):
            bundle = load_baseline_bundle(root)
            plugin = load_baseline_plugin(bundle)
            with self.assertRaisesRegex(
                ValueError, "does not support task family"
            ):
                plugin.task_builder.build(self.dataset, self.finetune)

    def test_cosmos_native_payload_has_valid_shape_and_no_generic_leak(
        self,
    ) -> None:
        bundle = load_baseline_bundle(COSMOS_ROOT)
        plugin = load_baseline_plugin(bundle)
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "collision_1d"
        )
        generic = plugin.task_builder.data_adapter.adapt_case(
            case, "generic", role="eval"
        )
        physics = plugin.task_builder.data_adapter.adapt_case(
            case, "physics", role="eval"
        )
        shape = generic["native_inputs"]["generation_shape"]
        self.assertEqual("16,9", shape["aspect_ratio"])
        self.assertEqual("480", shape["resolution"])
        self.assertEqual(24, shape["fps"])
        self.assertEqual(0, (shape["num_frames"] - 1) % 4)
        self.assertEqual({}, generic["used_parameters"])
        self.assertNotEqual(generic["prompt"], physics["prompt"])
        self.assertEqual(
            generic["materialization_fingerprint"],
            physics["materialization_fingerprint"],
        )

    @unittest.skipUnless(
        (COSMOS_ROOT / "baseline.local.json").is_file()
        and (G15_ROOT / "baseline.local.json").is_file(),
        "local model deployments are not configured",
    )
    def test_builds_preserve_dataset_and_share_canonical_direct_plan(
        self,
    ) -> None:
        before = canonical_sha256({
            "descriptor": self.dataset.descriptor,
            "cases": self.dataset.cases,
            "views": self.dataset.views,
        })
        instances = []
        for root in (COSMOS_ROOT, G15_ROOT):
            plugin = load_baseline_plugin(load_baseline_bundle(root))
            instance = plugin.task_builder.build(
                self.dataset, self.direct_physics
            )
            instance.verify()
            instances.append(instance.value)
        after = canonical_sha256({
            "descriptor": self.dataset.descriptor,
            "cases": self.dataset.cases,
            "views": self.dataset.views,
        })
        self.assertEqual(before, after)
        self.assertEqual(
            instances[0]["canonical_plan"],
            instances[1]["canonical_plan"],
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


if __name__ == "__main__":
    unittest.main()
