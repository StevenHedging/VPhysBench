from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import unittest

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.io import load_jsonl
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/five_scene_finetune_eval.json"
DIRECT_TASK = ROOT / "tasks/official/five_scene_direct_eval.json"
PENDULUM_PROMPTS = (
    "A pendulum bob of radius r is released from rest at initial angle θ_0 on "
    "a string of length l_s, then swings back and forth about the fixed pivot.",
    "A pendulum bob of mass m and radius r is released from rest at initial "
    "angle θ_0 on a string of length l_s, then swings back and forth about the "
    "fixed pivot.",
)


class CurrentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=True)
        cls.view = cls.dataset.views["view_a"]
        provenance_path = (
            ROOT / "datasets/provenance/releases/13.0.0/cases.jsonl"
        )
        cls.provenance = (
            {
                case["case_id"]: case
                for case in load_jsonl(provenance_path)
            }
            if provenance_path.is_file()
            else {}
        )

    def test_v13_is_current_and_complete(self) -> None:
        self.assertEqual(
            ROOT / "datasets/releases/13.0.0/dataset.json",
            LATEST_DATASET,
        )
        self.assertEqual("physics_video_seven_scene_v13", self.dataset.dataset_id)
        self.assertEqual("13.0.0", self.dataset.descriptor["release"])
        self.assertEqual(916, len(self.dataset.cases))
        self.assertIsNone(self.dataset.asset_lock)
        self.assertEqual(
            {
                "collision_1d",
                "inclined_plane_slide",
                "parabolic_motion",
                "pendulum",
                "push_bottle",
                "uniform_circular_motion",
                "vertical_spring_oscillator",
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
            "vertical_spring_oscillator": (97, 20),
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
        self.assertEqual(916, len(all_ids))
        self.assertEqual(916, len(set(all_ids)))
        self.assertEqual(set(test_ids), set(self.view["test_annotations"]))
        self.assertEqual(
            {"id": 110},
            dict(Counter(
                value["generalization_regime"]
                for value in self.view["test_annotations"].values()
            )),
        )
        for annotation in self.view["test_annotations"].values():
            self.assertEqual([], annotation["ood_factors"])
            self.assertEqual([], annotation["co_varying_factors"])

    def test_vertical_spring_cases_match_reviewed_import_contract(self) -> None:
        spring = [
            case
            for case in self.dataset.cases
            if case["scene_id"] == "vertical_spring_oscillator"
        ]
        self.assertEqual(117, len(spring))
        for case in spring:
            case_id = case["case_id"]
            provenance = self.provenance[case_id]
            displacement = case["physics"]["objects"]["object_1"][
                "initial_displacement"
            ]["value"]
            self.assertGreater(displacement, 0)
            self.assertEqual("approved", provenance["review"]["status"])
            self.assertEqual(
                "first return to the release-side turning point after one complete oscillation",
                provenance["alignment"]["canonical_first_frame_event"],
            )
            caption = case["text"]["prompt"]
            direction = case["appearance"]["release_side"]
            self.assertIn(f" {direction} equilibrium", caption)
            manifest_path = ROOT / "datasets" / case["assets"][
                "first_frame_mask_manifest"
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(manifest["instances"]))

    def test_independent_v13_validator_accepts_current_release(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/validate_dataset_v13.py"],
            cwd=ROOT,
            env={"PYTHONPATH": "src"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("dataset_id=physics_video_seven_scene_v13", completed.stdout)
        self.assertIn("cases=916", completed.stdout)

    def test_all_pendulum_cases_use_initial_release_semantics(self) -> None:
        pendulum = [
            case for case in self.dataset.cases if case["scene_id"] == "pendulum"
        ]
        self.assertEqual(100, len(pendulum))
        self.assertEqual(
            {35, 65},
            {
                sum(case["text"]["prompt"] == prompt for case in pendulum)
                for prompt in PENDULUM_PROMPTS
            },
        )
        self.assertTrue(all(
            self.provenance[case["case_id"]]["alignment"][
                "canonical_first_frame_event"
            ]
            == "initial release point"
            for case in pendulum
        ))
        supplement = [
            case for case in pendulum
            if self.provenance[case["case_id"]].get("import_id")
            == "pendulum_supplement_20260804"
        ]
        self.assertEqual(65, len(supplement))
        self.assertTrue(all(
            self.provenance[case["case_id"]]["alignment"]["trim_purpose"]
            == "remove_person_hand_only"
            for case in supplement
        ))

    def test_push_bottle_import_has_complete_force_series_and_expected_exclusion(self) -> None:
        cases = [
            case for case in self.dataset.cases
            if case["scene_id"] == "push_bottle"
        ]
        self.assertEqual(141, len(cases))
        self.assertTrue(all(
            set(case["physics"]) == {"objects", "environment"}
            and case["physics"]["environment"] == {}
            and set(case["physics"]["objects"]["object_1"])
            == {"mass", "height", "applied_force"}
            and case["physics"]["objects"]["object_1"]["applied_force"]["samples"]
            for case in cases
        ))
        audit_path = (
            ROOT
            / "datasets/provenance/imports/"
            "push_bottle_20260804_import_audit.jsonl"
        )
        audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
        self.assertEqual(141, len(audits))
        exclusions = json.loads((
            ROOT
            / "datasets/provenance/imports/"
            "push_bottle_20260804_exclusions.json"
        ).read_text())
        self.assertEqual(
            ["IMG_0076"],
            [item["source_stem"] for item in exclusions["excluded_videos"]],
        )

    def test_official_five_evaluator_scene_tasks_use_v13(self) -> None:
        finetune_task = load_task(FINETUNE_TASK)
        direct_task = load_task(DIRECT_TASK)
        self.assertEqual("five_scene_finetune_eval_v13", finetune_task.task_id)
        self.assertEqual("five_scene_direct_eval_v13", direct_task.task_id)
        finetune = plan_atomic_task(finetune_task, self.dataset).value
        self.assertEqual(582, len(finetune["train_case_ids"]))
        self.assertEqual(76, len(finetune["jobs"]))
        self.assertTrue(all(
            annotation["generalization_regime"] == "id"
            for annotation in finetune["evaluation_annotations"].values()
        ))
        direct = plan_atomic_task(direct_task, self.dataset).value
        self.assertEqual(658, len(direct["jobs"]))
        self.assertNotIn("push_bottle", direct["scene_ids"])

if __name__ == "__main__":
    unittest.main()
