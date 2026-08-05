from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
