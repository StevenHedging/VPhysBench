from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from physbench.datasets.loader import _validate_case, load_dataset
from physbench.evaluation.common.reference import resolve_physics_reference
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.io import canonical_sha256, write_json, write_jsonl


class DatasetContractV5Tests(unittest.TestCase):
    def _case(self) -> dict:
        return {
            "case_id": "pendulum_case_1",
            "scene_id": "pendulum",
            "text": {
                "prompt": "A bob of mass m swings about a fixed pivot.",
            },
            "assets": {
                "caption": "caption.json",
                "first_frame": "frame.bin",
                "reference_video": "frame.bin",
                "physics_annotation": "physics.json",
            },
            "physics": {
                "objects": {
                    "object_1": {
                        "mass": {
                            "value": 0.2,
                            "unit": "kg",
                            "symbol": "m",
                        },
                        "radius": {
                            "value": 0.01,
                            "unit": "m",
                            "symbol": "r",
                        },
                        "initial_angle": {
                            "value": 0.2,
                            "unit": "rad",
                            "symbol": "theta_0",
                        },
                    }
                },
                "environment": {
                    "string_length": {
                        "value": 0.5,
                        "unit": "m",
                        "symbol": "L",
                    }
                },
            },
            "appearance": {},
            "temporal": {},
        }

    def _write_dataset(
        self,
        root: Path,
        *,
        case: dict | None = None,
        include_document_schema: bool = False,
        caption_case_id: str | None = None,
    ) -> Path:
        case = copy.deepcopy(case or self._case())
        assets = root / "assets"
        assets.mkdir(parents=True)
        (assets / "frame.bin").write_bytes(b"asset\n")
        physics_document = {
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "physics": case["physics"],
        }
        if include_document_schema:
            physics_document["schema_version"] = "2.0"
        write_json(assets / "physics.json", physics_document)
        write_json(
            assets / "caption.json",
            {
                "case_id": caption_case_id or case["case_id"],
                "scene_id": case["scene_id"],
                "caption": case["text"]["prompt"],
            },
        )
        indexed_case = copy.deepcopy(case)
        indexed_case["assets"]["caption"] = "caption.json"
        indexed_case.pop("text")
        indexed_case.pop("physics")
        write_jsonl(root / "cases.jsonl", [indexed_case])
        (root / "scenes").mkdir()
        write_json(
            root / "scenes" / "pendulum.json",
            {
                "schema_version": "2.0",
                "scene_id": "pendulum",
                "display_name": "Pendulum",
                "structured_physics_parameters": [
                    "objects.object_1.mass",
                    "objects.object_1.radius",
                    "objects.object_1.initial_angle",
                    "environment.string_length",
                ],
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
                "release": "12.0.0",
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

    def test_current_case_materializes_minimal_caption_and_physics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = load_dataset(
                self._write_dataset(Path(temporary)),
                check_assets=False,
            )
        self.assertEqual(
            "A bob of mass m swings about a fixed pivot.",
            snapshot.cases[0]["text"]["prompt"],
        )
        self.assertEqual({"prompt"}, set(snapshot.cases[0]["text"]))
        self.assertNotIn("schema_version", snapshot.cases[0])
        self.assertEqual(
            "m",
            snapshot.cases[0]["physics"]["objects"]["object_1"]["mass"]["symbol"],
        )

    def test_reference_video_is_the_only_same_case_gt_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = self._write_dataset(Path(temporary))
            snapshot = load_dataset(descriptor, check_assets=False)
            case = snapshot.cases[0]
            request = CaseEvaluationRequest(
                job={},
                case=case,
                case_catalog={case["case_id"]: case},
                prediction=None,
                asset_root=snapshot.asset_root,
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config={},
            )
            path, mode, parent_id = resolve_physics_reference(request)
        self.assertEqual("frame.bin", path.name)
        self.assertEqual("same_case_reference", mode)
        self.assertIsNone(parent_id)

    def test_current_case_rejects_retired_asset_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case()
            case["assets"]["physics_reference_video"] = "frame.bin"
            descriptor = self._write_dataset(Path(temporary), case=case)
            with self.assertRaisesRegex(ValueError, "invalid current assets"):
                load_dataset(descriptor, check_assets=False)

    def test_schema_5_rejects_caption_bound_to_another_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = self._write_dataset(
                Path(temporary), caption_case_id="pendulum_case_2"
            )
            with self.assertRaisesRegex(ValueError, "caption Case mismatch"):
                load_dataset(descriptor, check_assets=False)

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
            mutate(case["physics"]["objects"]["object_1"]["mass"])
            with self.subTest(label=label), self.assertRaises(ValueError):
                _validate_case(case, {"pendulum"}, schema_version="5.0")

    def test_current_physics_document_rejects_obsolete_schema_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = self._write_dataset(
                Path(temporary), include_document_schema=True
            )
            with self.assertRaisesRegex(ValueError, "physics annotation fields"):
                load_dataset(descriptor, check_assets=False)

    def test_schema_5_accepts_grouped_three_field_quantities(self) -> None:
        case = self._case()
        try:
            _validate_case(case, {"pendulum"}, schema_version="5.0")
        except ValueError as exc:
            self.fail(f"grouped current physics was rejected: {exc}")

    def test_schema_5_rejects_obsolete_annotated_member(self) -> None:
        case = self._case()
        case["assets"]["caption"] = "caption.json"
        case["physics"]["objects"]["object_1"]["mass"]["annotated"] = True
        with self.assertRaisesRegex(ValueError, "annotated"):
            _validate_case(case, {"pendulum"}, schema_version="5.0")


if __name__ == "__main__":
    unittest.main()
