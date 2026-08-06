from __future__ import annotations

import unittest

from _paths import ROOT
from physbench.baseline_api import load_baseline_bundle, load_baseline_plugin
from physbench.baseline_runtime.adapter_loader import load_data_adapter
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.datasets.physics import flat_physics_quantities, is_scalar_quantity
from physbench.io import load_json


CAUSAL_MANIFEST = (
    ROOT / "baselines" / "causal_forcing_pp_2step_i2v" / "physics.baseline.json"
)
CAUSAL_RESOURCE = (
    ROOT
    / "src"
    / "physbench"
    / "baseline_plugins"
    / "resources"
    / "six_scene_physics_clauses_v2.json"
)
QUANTITY_MANIFEST = ROOT / "baselines" / "wan22_quantity_embedding" / "baseline.json"
QUANTITY_RESOURCE = (
    ROOT / "baselines" / "wan22_quantity_embedding" / "quantity_registry_v2.json"
)


class SymbolicConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET)

    def test_active_manifests_use_versioned_v2_resources(self) -> None:
        causal = load_json(CAUSAL_MANIFEST)
        quantity = load_json(QUANTITY_MANIFEST)
        self.assertEqual(
            "six_scene_physics_clauses_v2",
            causal["adapter"]["physics_transform"]["template_set"],
        )
        self.assertIn(
            "quantity_registry_v2.json",
            quantity["implementation"]["fingerprint_paths"],
        )
        self.assertEqual(
            "quantity_registry_v2.json",
            quantity["adapter"]["config"]["quantity_registry"],
        )
        self.assertEqual(
            "six_scene_physics_clauses_v2", load_json(CAUSAL_RESOURCE)["template_set_id"]
        )
        self.assertEqual(
            "six_scene_quantity_embedding_v2",
            load_json(QUANTITY_RESOURCE)["registry_id"],
        )

    def test_resources_exactly_cover_current_independent_parameter_universes(self) -> None:
        causal = load_json(CAUSAL_RESOURCE)
        quantity = load_json(QUANTITY_RESOURCE)
        for scene_id in sorted(self.dataset.scene_configs):
            expected = {
                name
                for case in self.dataset.cases
                if case["scene_id"] == scene_id
                for name, quantity in flat_physics_quantities(case).items()
                if is_scalar_quantity(quantity)
            }
            causal_names = {
                item["name"]
                for item in causal["scenes"][scene_id]["parameter_clauses"]
            }
            quantity_names = {
                item["name"]
                for item in quantity["scenes"][scene_id]["parameters"]
            }
            self.assertEqual(expected, causal_names, scene_id)
            self.assertEqual(expected, quantity_names, scene_id)

    def test_both_adapters_select_every_and_only_independent_quantity(self) -> None:
        causal_bundle = load_baseline_bundle(CAUSAL_MANIFEST)
        causal_adapter = load_data_adapter(causal_bundle)
        quantity_bundle = load_baseline_bundle(QUANTITY_MANIFEST)
        quantity_adapter = load_baseline_plugin(
            quantity_bundle
        ).task_builder.data_adapter
        for case in self.dataset.cases:
            projected = flat_physics_quantities(case)
            expected = {
                name
                for name, value in projected.items()
                if is_scalar_quantity(value)
            }
            causal = causal_adapter.adapt_case(case, role="eval")
            quantity = quantity_adapter.adapt_case(case, role="eval")
            self.assertEqual(expected, set(causal["used_parameters"]), case["case_id"])
            self.assertEqual(expected, set(quantity["used_parameters"]), case["case_id"])
            for adaptation in (causal, quantity):
                for name, audit in adaptation["used_parameters"].items():
                    self.assertEqual(projected[name]["value"], audit["value"])
                    self.assertEqual(projected[name]["unit"], audit["unit"])
                    self.assertEqual(projected[name]["symbol"], audit["symbol"])
            for item in quantity["native_inputs"]["physics"]["quantities"]:
                self.assertEqual(projected[item["name"]]["symbol"], item["symbol"])

    def test_scalar_adapters_do_not_summarize_push_force_series(self) -> None:
        case = next(
            case for case in self.dataset.cases
            if case["scene_id"] == "push_bottle"
        )
        causal = load_data_adapter(load_baseline_bundle(CAUSAL_MANIFEST)).adapt_case(
            case, role="eval"
        )
        quantity = load_baseline_plugin(
            load_baseline_bundle(QUANTITY_MANIFEST)
        ).task_builder.data_adapter.adapt_case(case, role="eval")
        for adaptation in (causal, quantity):
            self.assertEqual(
                {"bottle_mass", "bottle_height"},
                set(adaptation["used_parameters"]),
            )
            serialized = str(adaptation)
            self.assertNotIn("applied_force", serialized)
            self.assertNotIn("mean_applied_force", serialized)
            self.assertNotIn("peak_applied_force", serialized)


if __name__ == "__main__":
    unittest.main()
