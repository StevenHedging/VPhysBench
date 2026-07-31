from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from physbench.data_layout import V4_DATASET, V5_DATASET
from physbench.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


class SixSceneDatasetV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v4 = load_dataset(V4_DATASET)
        cls.v5 = load_dataset(V5_DATASET, check_assets=True)

    def test_v5_has_the_frozen_release_identity(self) -> None:
        self.assertEqual(
            "physics_video_six_scene_v5",
            self.v5.descriptor["dataset_id"],
        )
        self.assertEqual("5.0.0", self.v5.descriptor["release"])
        self.assertEqual(609, len(self.v5.cases))
        self.assertEqual(1655, len(self.v5.asset_lock["files"]))

    def test_case_scene_counts_and_catalog_are_exact(self) -> None:
        expected = {
            "collision_1d": 330,
            "free_fall": 11,
            "inclined_plane_slide": 95,
            "parabolic_motion": 97,
            "pendulum": 40,
            "uniform_circular_motion": 36,
        }
        self.assertEqual(expected, dict(Counter(
            case["scene_id"] for case in self.v5.cases
        )))
        self.assertEqual(set(expected), set(self.v5.scene_configs))

    def test_v4_cases_are_preserved_byte_for_byte_at_case_level(self) -> None:
        v5_by_id = {case["case_id"]: case for case in self.v5.cases}
        self.assertEqual(214, len(self.v4.cases))
        for case in self.v4.cases:
            self.assertEqual(case, v5_by_id[case["case_id"]])

    def test_added_cases_are_reviewed_and_view_counts_are_frozen(self) -> None:
        added = [
            case
            for case in self.v5.cases
            if case["scene_id"] == "parabolic_motion"
            or case["case_id"].startswith("collision_supp_20260729_")
        ]
        self.assertEqual(395, len(added))
        self.assertTrue(all(
            case["alignment"]["review_status"] == "visually_verified"
            for case in added
        ))

        expected_view_a = {
            "collision_1d": {
                "train": 95,
                "test_id": 32,
                "test_ood1": 63,
            },
            "free_fall": {"train": 7, "test_id": 4, "test_ood1": 0},
            "inclined_plane_slide": {
                "train": 58,
                "test_id": 6,
                "test_ood1": 16,
            },
            "parabolic_motion": {
                "train": 70,
                "test_id": 27,
                "test_ood1": 0,
            },
            "pendulum": {"train": 27, "test_id": 8, "test_ood1": 5},
            "uniform_circular_motion": {
                "train": 18,
                "test_id": 2,
                "test_ood1": 4,
            },
        }
        actual = {
            scene_id: {
                partition: len(case_ids)
                for partition, case_ids in groups.items()
            }
            for scene_id, groups in self.v5.views["view_a"]["scenes"].items()
        }
        self.assertEqual(expected_view_a, actual)

        view_b_ids = {
            case_id
            for groups in self.v5.views["view_b"]["scenes"].values()
            for case_ids in groups.values()
            for case_id in case_ids
        }
        self.assertEqual(
            {case["case_id"] for case in self.v5.cases},
            view_b_ids,
        )

    def test_exclusion_and_duplicate_audits_match_import_contract(self) -> None:
        audit = json.loads(
            (
                ROOT
                / "datasets/physics_video/releases/5.0.0/"
                "expansion_audit.json"
            ).read_text(encoding="utf-8")
        )
        excluded = audit["excluded_source_counts"]
        self.assertEqual(
            31,
            excluded["parabolic_video_without_independent_annotation"],
        )
        self.assertEqual(
            107,
            excluded["collision_annotation_row_without_video"],
        )
        self.assertEqual(
            0,
            excluded["collision_video_without_annotation"],
        )
        self.assertEqual(
            0,
            excluded["collision_duplicate_of_existing_benchmark"],
        )
        self.assertEqual(
            0,
            audit["collision_existing_benchmark_duplicate_count"],
        )


if __name__ == "__main__":
    unittest.main()
