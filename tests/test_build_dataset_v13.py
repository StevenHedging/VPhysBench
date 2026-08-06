from __future__ import annotations

from collections import Counter
import unittest


try:
    from scripts.build_dataset_v13 import select_spring_test_ids
except ImportError:
    select_spring_test_ids = None


class SpringSplitTests(unittest.TestCase):
    def test_id_selection_balances_directions_with_uneven_stratum_counts(self) -> None:
        records = []
        for direction, magnitudes, per_stratum in (
            ("above", (20, 40), 6),
            ("below", (10, 20, 30, 40, 50, 60), 2),
        ):
            for magnitude in magnitudes:
                for index in range(per_stratum):
                    records.append(
                        {
                            "case_id": f"{direction}_{magnitude}_{index}",
                            "direction": direction,
                            "signed_displacement_mm": (
                                -magnitude if direction == "above" else magnitude
                            ),
                            "source_group": f"{direction}_{magnitude}_{index}",
                        }
                    )

        selected = select_spring_test_ids(records, limit=8)

        self.assertEqual(
            {"above": 4, "below": 4},
            dict(Counter(case_id.split("_", 1)[0] for case_id in selected)),
        )

    def test_id_selection_balances_directions_when_limit_is_smaller_than_strata(self) -> None:
        records = []
        for direction in ("above", "below"):
            for magnitude in range(10, 160, 10):
                for index in range(2):
                    records.append(
                        {
                            "case_id": f"{direction}_{magnitude}_{index}",
                            "direction": direction,
                            "signed_displacement_mm": (
                                -magnitude if direction == "above" else magnitude
                            ),
                            "source_group": f"{direction}_{magnitude}_{index}",
                        }
                    )

        selected = set(select_spring_test_ids(records, limit=10))

        self.assertEqual(
            {"above": 5, "below": 5},
            {
                direction: sum(case_id.startswith(direction) for case_id in selected)
                for direction in ("above", "below")
            },
        )

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
