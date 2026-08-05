from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from physbench.datasets.loader import _validate_case, load_dataset
from physbench.io import canonical_sha256, write_json, write_jsonl


class DatasetContractV5Tests(unittest.TestCase):
    def _case(self) -> dict:
        return {
            "schema_version": "5.0",
            "case_id": "pendulum_case_1",
            "scene_id": "pendulum",
            "text": {
                "schema_version": "1.0",
                "prompt": "A bob of mass m swings about a fixed pivot.",
                "language": "en",
                "annotation_source": "symbolic_physics_prompt_v1",
            },
            "assets": {
                "first_frame": "frame.bin",
                "reference_video": "frame.bin",
                "physics_annotation": "physics.v11.json",
            },
            "physics": {
                "bob_mass": {
                    "value": 0.2,
                    "unit": "kg",
                    "annotated": True,
                    "symbol": "m",
                }
            },
            "appearance": {},
            "temporal": {},
            "alignment": None,
            "provenance": {},
            "has_real_reference_video": True,
        }

    def _write_dataset(
        self,
        root: Path,
        *,
        case: dict | None = None,
        document_schema: str = "2.0",
    ) -> Path:
        case = copy.deepcopy(case or self._case())
        assets = root / "assets"
        assets.mkdir(parents=True)
        (assets / "frame.bin").write_bytes(b"asset\n")
        write_json(
            assets / "physics.v11.json",
            {
                "schema_version": document_schema,
                "case_id": case["case_id"],
                "scene_id": case["scene_id"],
                "physics": case["physics"],
            },
        )
        write_jsonl(root / "cases.jsonl", [case])
        (root / "scenes").mkdir()
        write_json(
            root / "scenes" / "pendulum.json",
            {
                "schema_version": "2.0",
                "scene_id": "pendulum",
                "display_name": "Pendulum",
                "structured_physics_parameters": ["bob_mass"],
                "non_conditionable_physics_parameters": [],
                "generalization_factors": [],
                "constraints": [],
                "metric_spec": {},
            },
        )
        (root / "views").mkdir()
        case_digest = canonical_sha256([case["case_id"]])
        write_json(
            root / "views" / "view_a.json",
            {
                "schema_version": "3.0",
                "view_id": "view_a",
                "coverage": "complete",
                "case_set_sha256": case_digest,
                "split_semantics": {
                    "primary_partitions": ["train", "test"],
                    "generalization_regimes": ["id", "ood", "mixed"],
                    "regime_is_relative_to": "view_a.train",
                },
                "scenes": {
                    "pendulum": {"train": [case["case_id"]], "test": []}
                },
                "test_annotations": {},
            },
        )
        write_json(
            root / "views" / "view_b.json",
            {
                "schema_version": "2.0",
                "view_id": "view_b",
                "coverage": "complete",
                "case_set_sha256": case_digest,
                "scenes": {"pendulum": {"group_1": [case["case_id"]]}},
            },
        )
        write_json(
            root / "dataset.json",
            {
                "schema_version": "5.0",
                "dataset_id": "symbolic_fixture",
                "release": "11.0.0",
                "cases": "cases.jsonl",
                "asset_root": "assets",
                "scene_catalog": "scenes",
                "views": {
                    "view_a": "views/view_a.json",
                    "view_b": "views/view_b.json",
                },
            },
        )
        return root / "dataset.json"

    def test_schema_5_case_and_matching_schema_2_document_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = load_dataset(
                self._write_dataset(Path(temporary)),
                check_assets=False,
            )
        self.assertEqual("m", snapshot.cases[0]["physics"]["bob_mass"]["symbol"])

    def test_schema_5_quantity_contract_rejects_invalid_fields_and_values(self) -> None:
        mutations = {
            "missing symbol": lambda quantity: quantity.pop("symbol"),
            "empty symbol": lambda quantity: quantity.__setitem__("symbol", " "),
            "non-string symbol": lambda quantity: quantity.__setitem__("symbol", 1),
            "unknown fifth field": lambda quantity: quantity.__setitem__("note", "x"),
            "negative value": lambda quantity: quantity.__setitem__("value", -0.2),
        }
        for label, mutate in mutations.items():
            case = self._case()
            mutate(case["physics"]["bob_mass"])
            with self.subTest(label=label), self.assertRaises(ValueError):
                _validate_case(case, {"pendulum"})

    def test_schema_5_requires_schema_2_physics_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = self._write_dataset(
                Path(temporary), document_schema="1.0"
            )
            with self.assertRaisesRegex(ValueError, "schema must be 2.0"):
                load_dataset(descriptor, check_assets=False)

    def test_schema_4_rejects_symbol_and_historical_v10_still_loads(self) -> None:
        case = self._case()
        case["schema_version"] = "4.0"
        with self.assertRaisesRegex(ValueError, "fields must be"):
            _validate_case(case, {"pendulum"})

        from physbench.data_layout import V10_DATASET

        self.assertEqual("4.0", load_dataset(V10_DATASET).descriptor["schema_version"])


if __name__ == "__main__":
    unittest.main()
