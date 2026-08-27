from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unittest

import numpy as np

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_TASK = ROOT / "tasks/official/six_scene_train_six_scene_eval_v1.json"
DIRECT_TASK = ROOT / "tasks/official/six_scene_direct_eval_v1.json"


class CurrentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cls.view = cls.dataset.views["view_a"]

    def test_v14_is_current_and_complete(self) -> None:
        self.assertEqual(
            {"14.0.0"},
            {
                path.name
                for path in (ROOT / "datasets" / "releases").iterdir()
                if path.is_dir()
            },
        )
        self.assertEqual(
            ROOT / "datasets/releases/14.0.0/dataset.json",
            LATEST_DATASET,
        )
        self.assertEqual("physics_video_seven_scene_v14", self.dataset.dataset_id)
        self.assertEqual("14.0.0", self.dataset.descriptor["release"])
        self.assertEqual(916, len(self.dataset.cases))
        self.assertIsNotNone(self.dataset.asset_lock)
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
        all_ids: list[str] = []
        test_ids: list[str] = []
        for scene_id, (train_count, test_count) in expected.items():
            groups = self.view["scenes"][scene_id]
            self.assertEqual(train_count, len(groups["train"]))
            self.assertEqual(test_count, len(groups["test"]))
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
        self.assertEqual(127, len(self.view["scenes"]["push_bottle"]["train"]))
        self.assertEqual(14, len(self.view["scenes"]["push_bottle"]["test"]))

    def test_official_v1_tasks_score_six_scenes_and_exclude_push_bottle(self) -> None:
        finetune_task = load_task(FINETUNE_TASK)
        direct_task = load_task(DIRECT_TASK)
        self.assertEqual(
            "six_scene_train_six_scene_eval_v1",
            finetune_task.task_id,
        )
        self.assertEqual("six_scene_direct_eval_v1", direct_task.task_id)
        finetune = plan_atomic_task(finetune_task, self.dataset).value
        direct = plan_atomic_task(direct_task, self.dataset).value
        self.assertEqual(6, len(finetune["scene_ids"]))
        self.assertEqual(6, len(direct["scene_ids"]))
        self.assertEqual(6, len(finetune["training_scene_ids"]))
        self.assertEqual(679, len(finetune["train_case_ids"]))
        self.assertEqual(96, len(finetune["jobs"]))
        self.assertEqual(775, len(direct["jobs"]))
        selected_training_scenes = {
            case["scene_id"]
            for case in self.dataset.cases
            if case["case_id"] in set(finetune["train_case_ids"])
        }
        self.assertNotIn("push_bottle", finetune["training_scene_ids"])
        self.assertNotIn("push_bottle", selected_training_scenes)
        self.assertNotIn("push_bottle", finetune["scene_ids"])
        self.assertNotIn(
            "push_bottle",
            {job["scene_id"] for job in finetune["jobs"]},
        )
        self.assertNotIn("push_bottle", direct["scene_ids"])
        self.assertIn("vertical_spring_oscillator", direct["scene_ids"])
        self.assertEqual(
            117,
            sum(
                job["scene_id"] == "vertical_spring_oscillator"
                for job in direct["jobs"]
            ),
        )
        self.assertEqual(
            20,
            sum(
                job["scene_id"] == "vertical_spring_oscillator"
                for job in finetune["jobs"]
            ),
        )

    def test_collision_test_audit_covers_every_official_case_once(self) -> None:
        plan = plan_atomic_task(load_task(FINETUNE_TASK), self.dataset).value
        expected = {
            job["case_id"]
            for job in plan["jobs"]
            if job["scene_id"] == "collision_1d"
        }
        audit_path = (
            ROOT
            / "docs/audits/collision_v14_official_test_audit_20260818.md"
        )
        audited = re.findall(
            r"^\| `([^`]+)` \|",
            audit_path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        self.assertEqual(20, len(audited))
        self.assertEqual(expected, set(audited))

    def test_every_official_collision_case_has_current_full_video_review(self) -> None:
        plan = plan_atomic_task(load_task(FINETUNE_TASK), self.dataset).value
        cases_by_id = {case["case_id"]: case for case in self.dataset.cases}
        collision_ids = {
            job["case_id"]
            for job in plan["jobs"]
            if job["scene_id"] == "collision_1d"
        }
        for case_id in sorted(collision_ids):
            manifest_path = ROOT / "datasets" / cases_by_id[case_id]["assets"][
                "reference_observation_manifest"
            ]
            review = json.loads(
                (manifest_path.parent / "review.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("approved", review["decision"], case_id)
            self.assertEqual("full_video", review["scope"], case_id)
            self.assertEqual(
                "codex_sam31_collision_gt_full330_audit_20260827",
                review["reviewer"],
                case_id,
            )

    def test_all_v14_curated_anchor_ids_follow_the_frozen_scene_contract(self) -> None:
        expected_counts = {
            ("collision_1d", "sam31_collision_gt_curation_v1"): 330,
            ("inclined_plane_slide", "v14_full_reference_observation_audit"): 5,
            ("parabolic_motion", "v14_full_reference_observation_audit"): 69,
            ("pendulum", "v14_full_reference_observation_audit"): 20,
            ("push_bottle", "v14_full_reference_observation_audit"): 16,
            ("uniform_circular_motion", "v14_full_reference_observation_audit"): 1,
            ("vertical_spring_oscillator", "v14_full_reference_observation_audit"): 5,
        }
        fixed_ids = {
            "inclined_plane_slide": ("sliding_block",),
            "parabolic_motion": ("projectile_ball",),
            "pendulum": ("bob",),
            "push_bottle": ("bottle",),
            "vertical_spring_oscillator": ("object_1",),
        }
        observed = Counter()
        for case in self.dataset.cases:
            manifest_path = ROOT / "datasets" / case["assets"][
                "first_frame_mask_manifest"
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generator_id = manifest.get("generator", {}).get("id")
            if generator_id not in {
                "v14_full_reference_observation_audit",
                "sam31_collision_gt_curation_v1",
            }:
                continue
            observed[(case["scene_id"], generator_id)] += 1
            instance_count = len(manifest["instances"])
            if case["scene_id"] == "collision_1d":
                expected_ids = tuple(
                    f"ball_{index}" for index in range(1, instance_count + 1)
                )
            elif case["scene_id"] == "uniform_circular_motion":
                expected_ids = tuple(
                    f"object_{index}" for index in range(1, instance_count + 1)
                )
            else:
                expected_ids = fixed_ids[case["scene_id"]]
            actual_ids = []
            for instance, expected_id in zip(
                manifest["instances"], expected_ids, strict=True
            ):
                npz_path = ROOT / "datasets" / instance["npz_asset"]
                with np.load(npz_path, allow_pickle=False) as payload:
                    actual_ids.extend(str(value) for value in payload["object_ids"])
                    mask = payload["masks"][0]
                ys, xs = np.where(mask > 0)
                self.assertEqual(expected_id, actual_ids[-1], case["case_id"])
                self.assertEqual(int(xs.size), instance["area_pixels"], case["case_id"])
                self.assertEqual(
                    [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ],
                    instance["bbox_xyxy"],
                    case["case_id"],
                )
                np.testing.assert_allclose(
                    np.asarray(instance["centroid_xy"], dtype=np.float64),
                    np.asarray([xs.mean(), ys.mean()], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-6,
                    err_msg=case["case_id"],
                )
            self.assertEqual(expected_ids, tuple(actual_ids), case["case_id"])

            observation_path = ROOT / "datasets" / case["assets"][
                "reference_observation_manifest"
            ]
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
            source_records = [
                observation["source"]["first_frame_mask_manifest"],
                *(
                    item["asset"]
                    for item in observation["source"]["anchor_masks"]
                ),
            ]
            for record in source_records:
                recorded_path = ROOT / "datasets" / record["path"]
                self.assertEqual(recorded_path.stat().st_size, record["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(recorded_path.read_bytes()).hexdigest(),
                    record["sha256"],
                    case["case_id"],
                )
            visualization_path = ROOT / "datasets" / case["assets"][
                "reference_observation_visualization_manifest"
            ]
            visualization = json.loads(
                visualization_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                hashlib.sha256(observation_path.read_bytes()).hexdigest(),
                visualization["observation_manifest_sha256"],
                case["case_id"],
            )
        self.assertEqual(expected_counts, dict(observed))
        tracking = json.loads(
            (
                ROOT
                / "datasets/releases/14.0.0/reference_observation_curation.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(446, tracking["curated_case_count"])
        self.assertEqual(842, tracking["anchor_file_count"])
        self.assertEqual(2180, tracking["hash_closed_file_count"])
        self.assertEqual(
            self.dataset.asset_lock["files_digest"],
            tracking["asset_lock_files_digest"],
        )


if __name__ == "__main__":
    unittest.main()
