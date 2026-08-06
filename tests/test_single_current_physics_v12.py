from __future__ import annotations

import unittest
from pathlib import Path
import re

from physbench.datasets import load_dataset
from physbench import data_layout
from physbench.io import load_json
from physbench.io import load_jsonl
from physbench.tasks import load_task, plan_atomic_task


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
RELEASE_ROOT = DATASETS_ROOT / "releases" / "12.0.0"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "12.0.0"
DATASET_ID = "physics_video_six_scene_v12"


class SingleCurrentPhysicsV12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(RELEASE_ROOT / "cases.jsonl")
        cls.provenance_cases = load_jsonl(PROVENANCE_ROOT / "cases.jsonl")
        cls.loaded_cases = {
            case["case_id"]: case
            for case in load_dataset(RELEASE_ROOT / "dataset.json").cases
        }

    def test_v11_content_coverage_is_preserved_as_migration_evidence(self) -> None:
        evidence = load_json(PROVENANCE_ROOT / "migration.json")
        self.assertEqual(
            {
                "quantities": 5286,
                "negative_values_normalized": 494,
                "annotated_flags_corrected": 715,
                "symbols_added": 5286,
            },
            evidence["legacy_coverage"],
        )

    def test_every_case_has_one_minimal_physics_document(self) -> None:
        self.assertEqual(799, len(self.cases))
        paths = []
        for case in self.cases:
            relative = case["assets"]["physics_annotation"]
            self.assertTrue(relative.endswith("/physics.json"))
            paths.append(relative)
            document = load_json(DATASETS_ROOT / relative)
            self.assertEqual(
                {"case_id", "scene_id", "physics"},
                set(document),
            )
            self.assertEqual(case["case_id"], document["case_id"])
            self.assertEqual(case["scene_id"], document["scene_id"])
            self.assertEqual(
                self.loaded_cases[case["case_id"]]["physics"],
                document["physics"],
            )
        self.assertEqual(799, len(set(paths)))

    def test_every_indexed_case_owns_one_caption_document(self) -> None:
        caption_paths = []
        for case in self.cases:
            self.assertNotIn("text", case)
            self.assertNotIn("physics", case)
            relative = case["assets"]["caption"]
            self.assertTrue(relative.endswith("/caption.json"))
            caption_paths.append(relative)
            document = load_json(DATASETS_ROOT / relative)
            self.assertEqual(
                {"caption", "case_id", "scene_id"},
                set(document),
            )
            self.assertEqual(case["case_id"], document["case_id"])
            self.assertEqual(case["scene_id"], document["scene_id"])
            self.assertTrue(document["caption"].strip())
            self.assertEqual(
                document["caption"],
                self.loaded_cases[case["case_id"]]["text"]["prompt"],
            )
        self.assertEqual(799, len(set(caption_paths)))

    def test_case_index_contains_only_runtime_fields_and_assets(self) -> None:
        required_assets = {
            "caption",
            "first_frame",
            "physics_annotation",
            "reference_video",
        }
        allowed_assets = required_assets | {"first_frame_mask_manifest"}
        for case in self.cases:
            self.assertEqual(
                {"case_id", "scene_id", "assets", "appearance", "temporal"},
                set(case),
            )
            self.assertEqual(
                {"encoded_to_physical_speed"},
                set(case["temporal"]),
            )
            self.assertTrue(required_assets <= set(case["assets"]))
            self.assertTrue(set(case["assets"]) <= allowed_assets)
            self.assertTrue(all(value for value in case["assets"].values()))

            loaded = self.loaded_cases[case["case_id"]]
            self.assertEqual(set(case) | {"text", "physics"}, set(loaded))
            self.assertEqual({"prompt"}, set(loaded["text"]))

    def test_case_audit_metadata_lives_only_in_provenance(self) -> None:
        self.assertEqual(799, len(self.provenance_cases))
        self.assertEqual(
            {case["case_id"] for case in self.cases},
            {case["case_id"] for case in self.provenance_cases},
        )
        self.assertEqual(
            763,
            sum("alignment" in case for case in self.provenance_cases),
        )
        self.assertEqual(
            568,
            sum(
                "source_video" in case.get("source_assets", {})
                for case in self.provenance_cases
            ),
        )
        self.assertEqual(
            732,
            sum(
                "archive" in case.get("source_locator", {})
                for case in self.provenance_cases
            ),
        )
        self.assertEqual(
            141,
            sum(
                "annotation_archive" in case.get("source_locator", {})
                for case in self.provenance_cases
            ),
        )
        self.assertTrue(all(
            set(case.get("source_assets", {})) <= {"source_video"}
            for case in self.provenance_cases
        ))
        self.assertTrue(all(
            "temporal_metadata" in case
            for case in self.provenance_cases
        ))
        self.assertTrue(all(
            "source_member" not in case and "source_workbook_row" not in case
            for case in self.provenance_cases
        ))

        def forbidden_keys(value: object):
            if isinstance(value, list):
                return [
                    key
                    for child in value
                    for key in forbidden_keys(child)
                ]
            if not isinstance(value, dict):
                return []
            return [
                key
                for key in value
                if "sha256" in key or "digest" in key
            ] + [
                key
                for child in value.values()
                for key in forbidden_keys(child)
            ]

        self.assertEqual([], forbidden_keys(self.provenance_cases))

    def test_active_mask_manifests_do_not_record_checkpoint_hashes(self) -> None:
        for case in self.cases:
            relative = case["assets"].get("first_frame_mask_manifest")
            if relative is None:
                continue
            manifest = load_json(DATASETS_ROOT / relative)
            self.assertNotIn("checkpoint_sha256", manifest.get("generator", {}))

    def test_published_release_is_minimal_with_case_local_members(self) -> None:
        self.assertEqual(
            {
                "cases.jsonl",
                "dataset.json",
                "scenes",
                "views",
            },
            {path.name for path in RELEASE_ROOT.iterdir()},
        )
        snapshot = load_dataset(RELEASE_ROOT / "dataset.json")
        self.assertEqual(DATASET_ID, snapshot.dataset_id)
        self.assertEqual(799, len(snapshot.cases))
        self.assertIsNone(snapshot.asset_lock)
        self.assertNotIn("asset_lock", snapshot.descriptor)
        self.assertNotIn("release_manifest", snapshot.descriptor)
        self.assertTrue(all(
            case["assets"]["physics_annotation"].endswith("/physics.json")
            for case in snapshot.cases
        ))
        self.assertFalse(list(DATASETS_ROOT.glob("assets/*/*/physics.v11.json")))
        self.assertEqual(
            799,
            len(list(DATASETS_ROOT.glob("assets/*/*/physics.json"))),
        )
        self.assertEqual(
            799,
            len(list(DATASETS_ROOT.glob("assets/*/*/caption.json"))),
        )

    def test_release_records_zero_media_changes(self) -> None:
        evidence = load_json(PROVENANCE_ROOT / "migration.json")
        self.assertEqual(799, evidence["counts"]["physics_documents_renamed"])
        self.assertEqual(0, evidence["counts"]["media_changes"])

        def forbidden_keys(value: object):
            if not isinstance(value, dict):
                return []
            return [
                key
                for key, child in value.items()
                if "sha256" in key or "digest" in key
                for _ in [None]
            ] + [
                key
                for child in value.values()
                for key in forbidden_keys(child)
            ]

        self.assertEqual([], forbidden_keys(evidence))

    def test_current_cases_do_not_reference_retired_runtime_releases(self) -> None:
        retired = re.compile(r"^releases/(?:[1-9]|10|11)\.0\.0(?:/|$)")

        def strings(value: object):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)

        offenders = [
            (case["case_id"], value)
            for case in self.cases
            for value in strings(case)
            if retired.search(value)
        ]
        self.assertEqual([], offenders)

    def test_v12_is_the_only_active_release_and_official_task_target(self) -> None:
        self.assertEqual(
            RELEASE_ROOT / "dataset.json",
            data_layout.LATEST_DATASET,
        )
        release_dirs = {
            path.name
            for path in data_layout.RELEASES_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        self.assertEqual({"12.0.0"}, release_dirs)
        snapshot = load_dataset(data_layout.LATEST_DATASET)
        for filename, task_id, jobs in (
            ("five_scene_finetune_eval.json", "five_scene_finetune_eval_v12", 76),
            ("five_scene_direct_eval.json", "five_scene_direct_eval_v12", 658),
        ):
            task = load_task(ROOT / "tasks" / "official" / filename)
            self.assertEqual(task_id, task.task_id)
            plan = plan_atomic_task(task, snapshot).value
            self.assertEqual(jobs, len(plan["jobs"]))
            if task.value["family"] == "finetune_eval":
                self.assertEqual(582, len(plan["train_case_ids"]))


if __name__ == "__main__":
    unittest.main()
