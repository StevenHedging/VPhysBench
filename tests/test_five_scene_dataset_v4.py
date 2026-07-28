from __future__ import annotations

import unittest

from _paths import ROOT
from physbench.data_layout import V4_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task, plan_atomic_task


class FiveSceneDatasetV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(V4_DATASET, check_assets=True)
        cls.by_id = {
            case["case_id"]: case for case in cls.dataset.cases
        }

    def _cases(self, identifiers: list[str]):
        return [self.by_id[identifier] for identifier in identifiers]

    def test_release_contains_all_annotation_backed_cases(self) -> None:
        counts: dict[str, int] = {}
        for case in self.dataset.cases:
            counts[case["scene_id"]] = counts.get(
                case["scene_id"], 0
            ) + 1
            self.assertIn("text", case)
            self.assertTrue(case["text"]["prompt"].strip())
        self.assertEqual(counts["inclined_plane_slide"], 95)
        self.assertEqual(counts["uniform_circular_motion"], 36)
        self.assertEqual(len(self.dataset.cases), 214)

    def test_view_a_is_training_dominant_and_factor_pure(self) -> None:
        view = self.dataset.views["view_a"]
        self.assertEqual(view["coverage"], "subset")

        incline = view["scenes"]["inclined_plane_slide"]
        self.assertEqual(
            {
                partition: len(values)
                for partition, values in incline.items()
            },
            {"train": 58, "test_id": 6, "test_ood1": 16},
        )
        self.assertGreaterEqual(58 / 80, 0.70)
        incline_train = self._cases(incline["train"])
        incline_id = self._cases(incline["test_id"])
        incline_ood = self._cases(incline["test_ood1"])
        train_angles = {
            case["physics"]["incline_angle"]["value"]
            for case in incline_train
        }
        train_backgrounds = {
            case["appearance"]["background"] for case in incline_train
        }
        self.assertEqual(train_angles, {32.0, 35.0, 41.0, 44.0})
        self.assertEqual(
            train_backgrounds,
            {"default_white", "oil_painting", "green_cardstock"},
        )
        self.assertEqual(
            {
                case["physics"]["incline_angle"]["value"]
                for case in incline_id
            },
            {38.0},
        )
        self.assertTrue(
            {
                case["appearance"]["background"]
                for case in incline_id
            }
            <= train_backgrounds
        )
        self.assertEqual(
            {
                case["appearance"]["background"]
                for case in incline_ood
            },
            {"black_foam_board"},
        )
        self.assertTrue(
            {
                case["physics"]["incline_angle"]["value"]
                for case in incline_ood
            }
            <= train_angles
        )

        circular = view["scenes"]["uniform_circular_motion"]
        self.assertEqual(
            {
                partition: len(values)
                for partition, values in circular.items()
            },
            {"train": 18, "test_id": 2, "test_ood1": 4},
        )
        self.assertEqual(18 / 24, 0.75)
        circular_train = self._cases(circular["train"])
        circular_id = self._cases(circular["test_id"])
        circular_ood = self._cases(circular["test_ood1"])
        train_radii = {
            case["physics"]["object_1_orbit_radius"]["value"]
            for case in circular_train
        }
        train_objects = {
            tuple(case["appearance"]["moving_objects"])
            for case in circular_train
        }
        self.assertEqual(train_radii, {0.02, 0.06, 0.08})
        self.assertEqual(
            train_objects,
            {
                ("silver_metal_block",),
                ("rectangular_wood_block",),
            },
        )
        self.assertEqual(
            {
                case["physics"]["object_1_orbit_radius"]["value"]
                for case in circular_id
            },
            {0.04},
        )
        self.assertEqual(
            {
                tuple(case["appearance"]["moving_objects"])
                for case in circular_id
            },
            train_objects,
        )
        for case in circular_ood:
            self.assertEqual(case["appearance"]["object_count"], 2)
            self.assertIn(
                case["physics"]["object_1_orbit_radius"]["value"],
                train_radii,
            )
            self.assertIn(
                case["physics"]["object_2_orbit_radius"]["value"],
                train_radii,
            )

    def test_all_incline_cases_have_reviewed_canonical_start(self) -> None:
        incline = [
            case
            for case in self.dataset.cases
            if case["scene_id"] == "inclined_plane_slide"
        ]
        self.assertTrue(incline)
        for case in incline:
            alignment = case["alignment"]
            self.assertEqual(
                alignment["review_status"], "visually_verified"
            )
            self.assertGreaterEqual(
                alignment["source_start_frame"], 0
            )
            self.assertNotEqual(
                case["assets"]["reference_video"],
                case["assets"]["source_archive"],
            )

    def test_view_b_covers_cases_excluded_from_view_a(self) -> None:
        view_a_ids = {
            identifier
            for groups in self.dataset.views["view_a"]["scenes"].values()
            for values in groups.values()
            for identifier in values
        }
        view_b_ids = {
            identifier
            for groups in self.dataset.views["view_b"]["scenes"].values()
            for values in groups.values()
            for identifier in values
        }
        all_ids = set(self.by_id)
        self.assertEqual(view_b_ids, all_ids)
        self.assertLess(view_a_ids, all_ids)

    def test_unified_tasks_plan_dataset_owned_splits(self) -> None:
        finetune = load_task(
            ROOT
            / "tasks"
            / "official"
            / "five_scene_finetune_eval.json"
        )
        finetune_plan = plan_atomic_task(finetune, self.dataset).value
        expected_train = sorted({
            case_id
            for groups in self.dataset.views["view_a"]["scenes"].values()
            for case_id in groups["train"]
        })
        expected_eval = sorted({
            (case_id, partition)
            for groups in self.dataset.views["view_a"]["scenes"].values()
            for partition in ("test_id", "test_ood1")
            for case_id in groups[partition]
        })
        self.assertEqual(
            expected_train, finetune_plan["train_case_ids"]
        )
        self.assertEqual(
            expected_eval,
            sorted(
                (job["case_id"], job["evaluation_partition"])
                for job in finetune_plan["jobs"]
            ),
        )

        direct = load_task(
            ROOT
            / "tasks"
            / "official"
            / "five_scene_direct_eval.json"
        )
        direct_plan = plan_atomic_task(direct, self.dataset).value
        self.assertEqual([], direct_plan["train_case_ids"])
        self.assertEqual(
            set(self.by_id),
            {job["case_id"] for job in direct_plan["jobs"]},
        )
        self.assertNotIn("conditioning", finetune.value)
        self.assertNotIn("conditioning", direct.value)


if __name__ == "__main__":
    unittest.main()
