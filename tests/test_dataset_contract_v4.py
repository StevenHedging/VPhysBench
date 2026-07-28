from __future__ import annotations

import copy
import unittest

from physbench.data_layout import V4_DATASET
from physbench.datasets.loader import (
    _validate_case,
    _validate_views,
    load_dataset,
)


class DatasetContractV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(V4_DATASET)
        cls.known_scenes = set(cls.dataset.scene_configs)

    def test_physics_quantity_types_are_strict(self) -> None:
        case = copy.deepcopy(self.dataset.cases[0])
        quantity = next(iter(case["physics"].values()))
        quantity["annotated"] = "true"
        with self.assertRaisesRegex(ValueError, "annotated must be boolean"):
            _validate_case(case, self.known_scenes)

        case = copy.deepcopy(self.dataset.cases[0])
        quantity = next(iter(case["physics"].values()))
        quantity["value"] = True
        with self.assertRaisesRegex(ValueError, "value must be a finite number"):
            _validate_case(case, self.known_scenes)

    def test_case_asset_paths_cannot_escape_asset_root(self) -> None:
        case = copy.deepcopy(self.dataset.cases[0])
        case["assets"]["first_frame"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "relative path inside asset_root"):
            _validate_case(case, self.known_scenes)

    def test_view_scene_bucket_must_match_case_scene(self) -> None:
        view = copy.deepcopy(self.dataset.views["view_b"])
        pendulum_groups = view["scenes"]["pendulum"]
        source_group = next(
            name for name, members in pendulum_groups.items() if members
        )
        case_id = pendulum_groups[source_group].pop()
        free_fall_groups = view["scenes"]["free_fall"]
        destination_group = next(iter(free_fall_groups))
        free_fall_groups[destination_group].append(case_id)

        with self.assertRaisesRegex(ValueError, "scene bucket"):
            _validate_views(
                self.dataset.cases,
                {"view_b": view},
            )

    def test_view_case_set_digest_is_verified(self) -> None:
        view = copy.deepcopy(self.dataset.views["view_a"])
        view["case_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "case_set_sha256 mismatch"):
            _validate_views(
                self.dataset.cases,
                {"view_a": view},
            )


if __name__ == "__main__":
    unittest.main()
