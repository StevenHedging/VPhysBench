from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from physbench.data_layout import LATEST_DATASET, V8_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/five_scene_finetune_eval.json"
DIRECT_TASK = ROOT / "tasks/official/five_scene_direct_eval.json"
PENDULUM_PROMPT = (
    "A pendulum bob is released from rest at the first frame and swings back "
    "and forth about the fixed pivot."
)


class SixSceneDatasetV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(V8_DATASET, check_assets=True)
        cls.view = cls.dataset.views["view_a"]

    def test_v8_is_current_and_complete(self) -> None:
        self.assertEqual(V8_DATASET, LATEST_DATASET)
        self.assertEqual("physics_video_six_scene_v8", self.dataset.dataset_id)
        self.assertEqual("8.0.0", self.dataset.descriptor["release"])
        self.assertEqual(799, len(self.dataset.cases))
        self.assertEqual(2032, len(self.dataset.asset_lock["files"]))
        self.assertEqual(
            {
                "collision_1d",
                "inclined_plane_slide",
                "parabolic_motion",
                "pendulum",
                "push_bottle",
                "uniform_circular_motion",
            },
            set(self.dataset.scene_configs),
        )

    def test_view_a_is_train_plus_id_test_only(self) -> None:
        expected = {
            "collision_1d": (310, 20),
            "inclined_plane_slide": (80, 15),
            "parabolic_motion": (82, 15),
            "pendulum": (80, 20),
            "push_bottle": (127, 14),
            "uniform_circular_motion": (30, 6),
        }
        all_ids = []
        test_ids = []
        for scene_id, (train_count, test_count) in expected.items():
            groups = self.view["scenes"][scene_id]
            self.assertEqual(train_count, len(groups["train"]))
            self.assertEqual(test_count, len(groups["test"]))
            self.assertLessEqual(test_count, 20)
            all_ids.extend(groups["train"])
            all_ids.extend(groups["test"])
            test_ids.extend(groups["test"])
        self.assertEqual(799, len(all_ids))
        self.assertEqual(799, len(set(all_ids)))
        self.assertEqual(set(test_ids), set(self.view["test_annotations"]))
        self.assertEqual(
            {"id": 90},
            dict(Counter(
                value["generalization_regime"]
                for value in self.view["test_annotations"].values()
            )),
        )
        for annotation in self.view["test_annotations"].values():
            self.assertEqual([], annotation["ood_factors"])
            self.assertEqual([], annotation["co_varying_factors"])

    def test_all_pendulum_cases_use_initial_release_semantics(self) -> None:
        pendulum = [
            case for case in self.dataset.cases if case["scene_id"] == "pendulum"
        ]
        self.assertEqual(100, len(pendulum))
        self.assertTrue(all(
            case["text"]["prompt"] == PENDULUM_PROMPT for case in pendulum
        ))
        self.assertTrue(all(
            case["alignment"]["canonical_first_frame_event"]
            == "initial release point"
            for case in pendulum
        ))
        supplement = [
            case for case in pendulum
            if case["provenance"].get("import_id")
            == "pendulum_supplement_20260804"
        ]
        self.assertEqual(65, len(supplement))
        self.assertTrue(all(
            case["alignment"]["trim_purpose"] == "remove_person_hand_only"
            for case in supplement
        ))

    def test_push_bottle_import_is_annotated_and_byte_preserving(self) -> None:
        cases = [
            case for case in self.dataset.cases
            if case["scene_id"] == "push_bottle"
        ]
        self.assertEqual(141, len(cases))
        expected_physics = {
            "bottle_mass",
            "bottle_height",
            "peak_applied_force",
            "mean_applied_force",
        }
        self.assertTrue(all(set(case["physics"]) == expected_physics for case in cases))
        audit_path = (
            ROOT
            / "datasets/provenance/imports/"
            "push_bottle_20260804_import_audit.jsonl"
        )
        audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
        self.assertEqual(141, len(audits))
        self.assertTrue(all(
            item["source_sha256"] == item["canonical_reference_sha256"]
            for item in audits
        ))
        exclusions = json.loads((
            ROOT
            / "datasets/provenance/imports/"
            "push_bottle_20260804_exclusions.json"
        ).read_text())
        self.assertEqual(
            ["IMG_0076"],
            [item["source_stem"] for item in exclusions["excluded_videos"]],
        )

    def test_official_five_evaluator_scene_tasks_use_v8(self) -> None:
        finetune = plan_atomic_task(
            load_task(FINETUNE_TASK), self.dataset
        ).value
        self.assertEqual(582, len(finetune["train_case_ids"]))
        self.assertEqual(76, len(finetune["jobs"]))
        self.assertTrue(all(
            annotation["generalization_regime"] == "id"
            for annotation in finetune["evaluation_annotations"].values()
        ))
        direct = plan_atomic_task(load_task(DIRECT_TASK), self.dataset).value
        self.assertEqual(658, len(direct["jobs"]))
        self.assertNotIn("push_bottle", direct["scene_ids"])

    def test_collision_replicate_components_do_not_cross_split(self) -> None:
        audit = json.loads((self.dataset.root / "split_audit.json").read_text())
        collision = audit["collision_replicate_audit"]
        self.assertEqual(0, collision["replicate_component_overlap"])
        self.assertEqual(13, len(collision["strata"]))


if __name__ == "__main__":
    unittest.main()
