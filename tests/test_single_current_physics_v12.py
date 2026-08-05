from __future__ import annotations

import unittest
from pathlib import Path

from physbench.datasets import load_dataset
from physbench.io import load_json
from physbench.io import load_jsonl
from scripts import build_dataset_v12 as builder


class SingleCurrentPhysicsV12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v11_cases = load_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")

    def test_v11_physics_is_a_complete_corrected_replacement(self) -> None:
        v10_cases = load_jsonl(
            builder.DATASETS_ROOT / "releases" / "10.0.0" / "cases.jsonl"
        )
        self.assertEqual(799, len(v10_cases))
        self.assertEqual(
            [case["case_id"] for case in v10_cases],
            [case["case_id"] for case in self.v11_cases],
        )
        parameters = symbols = absolute_changes = flag_changes = 0
        for old, new in zip(v10_cases, self.v11_cases, strict=True):
            self.assertEqual(set(old["physics"]), set(new["physics"]))
            for name, old_quantity in old["physics"].items():
                quantity = new["physics"][name]
                parameters += 1
                self.assertEqual(old_quantity["unit"], quantity["unit"])
                self.assertTrue(quantity["symbol"])
                symbols += 1
                if old_quantity["value"] != quantity["value"]:
                    self.assertLess(old_quantity["value"], 0)
                    self.assertEqual(abs(old_quantity["value"]), quantity["value"])
                    absolute_changes += 1
                if old_quantity["annotated"] != quantity["annotated"]:
                    self.assertTrue(old_quantity["annotated"])
                    self.assertFalse(quantity["annotated"])
                    flag_changes += 1
        self.assertEqual(5286, parameters)
        self.assertEqual(5286, symbols)
        self.assertEqual(494, absolute_changes)
        self.assertEqual(715, flag_changes)

    def test_migration_changes_only_identity_and_physics_asset_path(self) -> None:
        for case in self.v11_cases:
            migrated = builder.migrate_case(case)
            expected = builder._case_directory(case) / "physics.json"
            self.assertEqual(expected.as_posix(), migrated["assets"]["physics_annotation"])
            self.assertEqual(case["physics"], migrated["physics"])
            self.assertEqual(case["text"], migrated["text"])
            restored = dict(migrated)
            restored["assets"] = dict(restored["assets"])
            restored["assets"]["physics_annotation"] = case["assets"][
                "physics_annotation"
            ]
            self.assertEqual(case, restored)

    def test_documents_are_deterministic_schema_2_payloads(self) -> None:
        writes, cases = builder.prepare_v12(self.v11_cases)
        self.assertEqual(799, len(writes))
        self.assertEqual(799, len(cases))
        self.assertEqual(799, len({write.relative_path for write in writes}))
        for write, case in zip(writes, cases, strict=True):
            self.assertTrue(write.relative_path.endswith("/physics.json"))
            self.assertEqual(builder._physics_payload(case), write.payload)
            self.assertTrue(write.payload.endswith(b"\n"))

    def test_published_release_is_minimal_and_single_file(self) -> None:
        self.assertEqual(
            {
                "README.md",
                "assets.lock.json",
                "cases.jsonl",
                "dataset.json",
                "release.json",
                "scenes",
                "views",
            },
            {path.name for path in builder.OUTPUT_RELEASE_ROOT.iterdir()},
        )
        snapshot = load_dataset(
            builder.OUTPUT_RELEASE_ROOT / "dataset.json",
            check_asset_hashes=True,
        )
        self.assertEqual(builder.OUTPUT_DATASET_ID, snapshot.dataset_id)
        self.assertEqual(799, len(snapshot.cases))
        self.assertEqual(6038, len(snapshot.asset_lock["files"]))
        self.assertTrue(all(
            case["assets"]["physics_annotation"].endswith("/physics.json")
            for case in snapshot.cases
        ))
        self.assertFalse(list(builder.DATASETS_ROOT.glob("assets/*/*/physics.v11.json")))
        self.assertEqual(
            799,
            len(list(builder.DATASETS_ROOT.glob("assets/*/*/physics.json"))),
        )

    def test_release_changes_only_identity_and_physics_path(self) -> None:
        output = load_jsonl(builder.OUTPUT_RELEASE_ROOT / "cases.jsonl")
        self.assertEqual(799, len(output))
        for old, new in zip(self.v11_cases, output, strict=True):
            restored = dict(new)
            restored["assets"] = dict(restored["assets"])
            restored["assets"]["physics_annotation"] = old["assets"][
                "physics_annotation"
            ]
            self.assertEqual(old, restored)
        for name in ("view_a.json", "view_b.json"):
            self.assertEqual(
                (builder.BASE_RELEASE_ROOT / "views" / name).read_bytes(),
                (builder.OUTPUT_RELEASE_ROOT / "views" / name).read_bytes(),
            )
        evidence = load_json(builder.PROVENANCE_ROOT / "migration.json")
        self.assertEqual(799, evidence["counts"]["physics_documents_renamed"])
        self.assertEqual(0, evidence["counts"]["media_changes"])


if __name__ == "__main__":
    unittest.main()
