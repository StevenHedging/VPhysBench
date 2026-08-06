from __future__ import annotations

import unittest


try:
    from scripts.build_dataset_v13 import select_spring_test_ids
except ImportError:
    select_spring_test_ids = None


class SpringSplitTests(unittest.TestCase):
    def test_id_selection_leaves_each_selected_stratum_in_training(self) -> None:
        records = [
            {
                "case_id": f"spring_above_40_{index}",
                "direction": "above",
                "signed_displacement_mm": -40.0,
                "source_group": f"source_{index}",
            }
            for index in range(2)
        ]

        selected = select_spring_test_ids(records, limit=20)

        self.assertEqual(1, len(selected))

    def test_id_test_selection_is_bounded_stratified_and_group_safe(self) -> None:
        if select_spring_test_ids is None:
            self.fail("Dataset 13 spring split selector is not implemented")

        records = []
        for direction in ("above", "below"):
            for magnitude in (10.0, 20.0, 30.0):
                for index in range(4):
                    records.append(
                        {
                            "case_id": (
                                f"spring_{direction}_{int(magnitude)}_{index}"
                            ),
                            "direction": direction,
                            "signed_displacement_mm": (
                                -magnitude if direction == "above" else magnitude
                            ),
                            "source_group": (
                                f"duplicate_{direction}_{int(magnitude)}"
                                if index < 2
                                else f"unique_{direction}_{int(magnitude)}_{index}"
                            ),
                        }
                    )

        selected = select_spring_test_ids(records, limit=12)
        selected_set = set(selected)

        self.assertEqual(12, len(selected))
        self.assertEqual(12, len(selected_set))
        selected_records = [
            record for record in records if record["case_id"] in selected_set
        ]
        self.assertEqual(
            {"above", "below"},
            {record["direction"] for record in selected_records},
        )
        self.assertEqual(
            {10.0, 20.0, 30.0},
            {abs(record["signed_displacement_mm"]) for record in selected_records},
        )
        for source_group in {record["source_group"] for record in records}:
            members = {
                record["case_id"]
                for record in records
                if record["source_group"] == source_group
            }
            self.assertIn(len(members & selected_set), {0, len(members)})


if __name__ == "__main__":
    unittest.main()
