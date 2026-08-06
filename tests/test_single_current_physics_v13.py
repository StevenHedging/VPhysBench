from __future__ import annotations

import unittest
from pathlib import Path
import re

from physbench.datasets import load_dataset
from physbench import data_layout
from physbench.io import load_json
from physbench.io import load_jsonl
from physbench.tasks import load_task, plan_atomic_task
from scripts.validate_dataset_v13 import validate_v13


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
RELEASE_ROOT = DATASETS_ROOT / "releases" / "13.0.0"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "13.0.0"
DATASET_ID = "physics_video_seven_scene_v13"


class SingleCurrentPhysicsV13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(RELEASE_ROOT / "cases.jsonl")
        cls.provenance_cases = load_jsonl(PROVENANCE_ROOT / "cases.jsonl")
        cls.loaded_cases = {
            case["case_id"]: case
            for case in load_dataset(RELEASE_ROOT / "dataset.json").cases
        }

    def test_v12_content_and_spring_addition_are_migration_evidence(self) -> None:
        evidence = load_json(PROVENANCE_ROOT / "migration.json")
        self.assertEqual(
            {
                "base_cases_preserved": 799,
                "cases": 916,
                "media_changes": 0,
                "spring_cases_added": 117,
            },
            evidence["counts"],
        )

    def test_every_case_has_one_minimal_physics_document(self) -> None:
        self.assertEqual(916, len(self.cases))
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
        self.assertEqual(916, len(set(paths)))

    def test_formal_physics_is_grouped_without_annotation_flags(self) -> None:
        grouped_scenes = {
            "collision_1d",
            "inclined_plane_slide",
            "parabolic_motion",
            "pendulum",
            "push_bottle",
            "uniform_circular_motion",
            "vertical_spring_oscillator",
        }
        leaf_count = 0
        for case in self.cases:
            document = load_json(
                DATASETS_ROOT / case["assets"]["physics_annotation"]
            )
            physics = document["physics"]
            if case["scene_id"] in grouped_scenes:
                self.assertEqual({"objects", "environment"}, set(physics))
                object_ids = sorted(physics["objects"])
                self.assertEqual(
                    [f"object_{index}" for index in range(1, len(object_ids) + 1)],
                    object_ids,
                )
                quantities = [
                    quantity
                    for values in physics["objects"].values()
                    for quantity in values.values()
                ] + list(physics["environment"].values())
            leaf_count += len(quantities)
            for quantity in quantities:
                self.assertIn(
                    set(quantity),
                    (
                        {"value", "unit", "symbol"},
                        {"samples", "time_unit", "unit", "symbol"},
                    ),
                )
        self.assertEqual(4520, leaf_count)

    def test_v13_validator_audits_push_force_series_against_source(self) -> None:
        try:
            report = validate_v13()
        except (KeyError, ValueError) as error:
            self.fail(f"V13 force-series validation failed: {error}")
        self.assertEqual(4520, report["quantities"])
        self.assertEqual(4379, report["scalar_quantities"])
        self.assertEqual(141, report["time_series_quantities"])
        self.assertEqual(141, report.get("source_verified_force_series"))

    def test_scene_classification_has_no_auxiliary_physics(self) -> None:
        for case in self.cases:
            document = load_json(
                DATASETS_ROOT / case["assets"]["physics_annotation"]
            )
            physics = document["physics"]
            scene_id = case["scene_id"]
            if set(physics) != {
                "objects",
                "environment",
            }:
                self.fail(f"{case['case_id']} does not use grouped physics")
            if scene_id == "collision_1d":
                self.assertEqual({}, physics["environment"])
                self.assertTrue(all(
                    set(values) == {"mass", "radius", "initial_velocity"}
                    for values in physics["objects"].values()
                ))
            elif scene_id == "inclined_plane_slide":
                self.assertEqual(
                    {"incline_angle", "gravity_acceleration"},
                    set(physics["environment"]),
                )
                self.assertEqual(
                    {"mass", "length"},
                    set(physics["objects"]["object_1"]),
                )
            elif scene_id == "parabolic_motion":
                self.assertEqual({}, physics["environment"])
                self.assertEqual(
                    {"mass", "radius", "initial_horizontal_velocity", "launch_height"},
                    set(physics["objects"]["object_1"]),
                )
            elif scene_id == "pendulum":
                self.assertEqual({"string_length"}, set(physics["environment"]))
                self.assertTrue(
                    set(physics["objects"]["object_1"])
                    in ({"radius", "initial_angle"}, {"mass", "radius", "initial_angle"})
                )
            elif scene_id == "uniform_circular_motion":
                self.assertEqual({"angular_velocity"}, set(physics["environment"]))
                self.assertTrue(all(
                    set(values) == {"orbit_radius"}
                    for values in physics["objects"].values()
                ))
            elif scene_id == "push_bottle":
                self.assertEqual({}, physics["environment"])
                self.assertEqual(
                    {"mass", "height", "applied_force"},
                    set(physics["objects"]["object_1"]),
                )
                series = physics["objects"]["object_1"]["applied_force"]
                self.assertEqual(
                    {"samples", "time_unit", "unit", "symbol"},
                    set(series),
                )
                self.assertTrue(series["samples"])
            elif scene_id == "vertical_spring_oscillator":
                self.assertEqual(
                    {
                        "gravity_acceleration",
                        "natural_spring_length",
                        "spring_stiffness",
                    },
                    set(physics["environment"]),
                )
                self.assertEqual(
                    {"initial_displacement", "mass", "radius"},
                    set(physics["objects"]["object_1"]),
                )

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
        self.assertEqual(916, len(set(caption_paths)))

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
        self.assertEqual(916, len(self.provenance_cases))
        self.assertEqual(
            {case["case_id"] for case in self.cases},
            {case["case_id"] for case in self.provenance_cases},
        )
        self.assertEqual(
            880,
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
            849,
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
        self.assertEqual(916, len(snapshot.cases))
        self.assertIsNone(snapshot.asset_lock)
        self.assertNotIn("asset_lock", snapshot.descriptor)
        self.assertNotIn("release_manifest", snapshot.descriptor)
        self.assertTrue(all(
            case["assets"]["physics_annotation"].endswith("/physics.json")
            for case in snapshot.cases
        ))
        self.assertFalse(list(DATASETS_ROOT.glob("assets/*/*/physics.v11.json")))
        self.assertEqual(
            916,
            len(list(DATASETS_ROOT.glob("assets/*/*/physics.json"))),
        )
        self.assertEqual(
            916,
            len(list(DATASETS_ROOT.glob("assets/*/*/caption.json"))),
        )

    def test_release_records_zero_media_changes(self) -> None:
        evidence = load_json(PROVENANCE_ROOT / "migration.json")
        self.assertEqual(117, evidence["counts"]["spring_cases_added"])
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
        retired = re.compile(r"^releases/(?:[1-9]|10|11|12)\.0\.0(?:/|$)")

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

    def test_v13_is_the_only_active_release_and_official_task_target(self) -> None:
        self.assertEqual(
            RELEASE_ROOT / "dataset.json",
            data_layout.LATEST_DATASET,
        )
        release_dirs = {
            path.name
            for path in data_layout.RELEASES_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        self.assertEqual({"13.0.0"}, release_dirs)
        snapshot = load_dataset(data_layout.LATEST_DATASET)
        for filename, task_id, jobs in (
            ("five_scene_finetune_eval.json", "five_scene_finetune_eval_v13", 76),
            ("five_scene_direct_eval.json", "five_scene_direct_eval_v13", 658),
        ):
            task = load_task(ROOT / "tasks" / "official" / filename)
            self.assertEqual(task_id, task.task_id)
            plan = plan_atomic_task(task, snapshot).value
            self.assertEqual(jobs, len(plan["jobs"]))
            if task.value["family"] == "finetune_eval":
                self.assertEqual(582, len(plan["train_case_ids"]))


if __name__ == "__main__":
    unittest.main()
