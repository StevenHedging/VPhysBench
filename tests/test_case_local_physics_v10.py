from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from physbench.io import canonical_sha256, load_json, load_jsonl
from scripts import build_dataset_v10 as builder


EXPECTED_RELEASE_ENTRIES = {
    "README.md",
    "dataset.json",
    "release.json",
    "cases.jsonl",
    "assets.lock.json",
    "scenes",
    "views",
}
EXPECTED_PHYSICS_FINGERPRINT = (
    "ca8c593007ce4e3d6d7437656f9aa0c3f32b5ff245a4592c570297ba9f957d1a"
)


class CaseLocalPhysicsPreparationTests(unittest.TestCase):
    def test_v9_has_799_unique_case_directories(self) -> None:
        cases = builder.read_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")

        writes, output_cases = builder.prepare_v10_cases(
            cases, builder.DATASETS_ROOT
        )

        self.assertEqual(799, len(writes))
        self.assertEqual(799, len({item.relative_path for item in writes}))
        self.assertTrue(
            all(
                case["assets"]["physics_annotation"].endswith(
                    "/physics.json"
                )
                for case in output_cases
            )
        )

    def test_case_assets_must_resolve_to_one_case_directory(self) -> None:
        case = self._case(
            first_frame="assets/pendulum/case_a/canonical/first_frame.png",
            reference_video="assets/pendulum/case_b/canonical/reference.mp4",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "does not resolve to one Case directory"
            ):
                builder.prepare_v10_cases([case], Path(temporary))

    def test_physics_document_bytes_are_exact_and_deterministic(self) -> None:
        case = self._case()
        with tempfile.TemporaryDirectory() as temporary:
            first, _ = builder.prepare_v10_cases([case], Path(temporary))
            second, _ = builder.prepare_v10_cases([copy.deepcopy(case)], Path(temporary))

        self.assertEqual(first[0].payload, second[0].payload)
        self.assertTrue(first[0].payload.endswith(b"\n"))
        self.assertFalse(first[0].payload.endswith(b"\n\n"))
        self.assertEqual(
            {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "scene_id": "pendulum",
                "physics": {
                    "gravity": {
                        "value": 9.8,
                        "unit": "m/s^2",
                        "annotated": True,
                    }
                },
            },
            json.loads(first[0].payload),
        )

    def test_conflict_aborts_before_any_missing_document_is_written(self) -> None:
        cases = [
            self._case(case_id="pendulum_case_1", case_directory="case_a"),
            self._case(case_id="pendulum_case_2", case_directory="case_b"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writes, _ = builder.prepare_v10_cases(cases, root)
            writes[1].absolute_path.parent.mkdir(parents=True)
            writes[1].absolute_path.write_bytes(b"conflicting bytes\n")

            with self.assertRaisesRegex(ValueError, "conflicting physics document"):
                builder.materialize_physics_documents(writes)

            self.assertFalse(writes[0].absolute_path.exists())
            self.assertEqual(
                b"conflicting bytes\n", writes[1].absolute_path.read_bytes()
            )

    def test_real_legacy_cleanup_preflight_enumerates_exact_32_directories(self) -> None:
        cases = builder.read_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")
        candidates = list(
            (builder.DATASETS_ROOT / "assets" / "collision_1d").glob(
                "collision_r2_*"
            )
        )

        if candidates:
            removals = builder.preflight_legacy_cleanup(cases)
            records = [
                {
                    "case_id": item.case_id,
                    "directory": item.relative_directory,
                    "file": (
                        Path(item.relative_directory) / item.relative_file
                    ).as_posix(),
                }
                for item in removals
            ]
        else:
            migration = load_json(
                builder.PROVENANCE_ROOT / "migration.json"
            )
            records = migration["removed_legacy_directories"]

        self.assertEqual(32, len(records))
        self.assertEqual(32, len({item["directory"] for item in records}))
        for item in records:
            name = Path(item["directory"]).name
            self.assertTrue(name.startswith("collision_r2_"))
            self.assertTrue(item["file"].endswith("/source/first_frame_source.png"))
            active = next(case for case in cases if case["case_id"] == item["case_id"])
            self.assertNotIn(
                f"assets/collision_1d/{name}/canonical/",
                active["assets"]["first_frame"],
            )
            if not candidates:
                self.assertFalse(
                    (builder.DATASETS_ROOT / item["directory"]).exists()
                )

    def test_legacy_cleanup_preflight_rejects_an_extra_file(self) -> None:
        case_id = "collision_r2_fixture"
        case = self._case(
            case_id=case_id,
            scene_id="collision_1d",
            case_directory="collision_descriptive_fixture",
        )
        with tempfile.TemporaryDirectory() as temporary:
            datasets_root = Path(temporary)
            legacy = datasets_root / "assets" / "collision_1d" / case_id
            (legacy / "source").mkdir(parents=True)
            (legacy / "source" / "first_frame_source.png").write_bytes(b"png")
            (legacy / "unexpected.txt").write_text("no", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unexpected content"):
                builder.preflight_legacy_cleanup(
                    [case],
                    datasets_root=datasets_root,
                    historical_cases=[case],
                    expected_count=1,
                )

    def test_git_ignore_admits_only_case_root_physics_json(self) -> None:
        paths = {
            "datasets/assets/pendulum/example/physics.json": False,
            "datasets/assets/pendulum/example/canonical/reference.mp4": True,
            "datasets/assets/pendulum/example/canonical/masks/01.npz": True,
            "datasets/assets/pendulum/example/source/reference.mov": True,
        }
        for path, ignored in paths.items():
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path],
                    cwd=builder.ROOT,
                    check=False,
                )
                self.assertEqual(0 if ignored else 1, result.returncode)

    @staticmethod
    def _case(
        *,
        case_id: str = "pendulum_case_1",
        scene_id: str = "pendulum",
        case_directory: str = "case_a",
        first_frame: str | None = None,
        reference_video: str | None = None,
    ) -> dict:
        first_frame = first_frame or (
            f"assets/{scene_id}/{case_directory}/canonical/first_frame.png"
        )
        reference_video = reference_video or (
            f"assets/{scene_id}/{case_directory}/canonical/reference.mp4"
        )
        return {
            "schema_version": "4.0",
            "case_id": case_id,
            "scene_id": scene_id,
            "assets": {
                "first_frame": first_frame,
                "reference_video": reference_video,
                "physics_reference_video": reference_video,
            },
            "physics": {
                "gravity": {
                    "value": 9.8,
                    "unit": "m/s^2",
                    "annotated": True,
                }
            },
        }


class DatasetV10ReleaseTests(unittest.TestCase):
    def test_release_is_minimal_and_preserves_every_v9_case_fact(self) -> None:
        v9_cases = load_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")
        v10_cases = load_jsonl(builder.OUTPUT_RELEASE_ROOT / "cases.jsonl")

        self.assertEqual(
            EXPECTED_RELEASE_ENTRIES,
            {path.name for path in builder.OUTPUT_RELEASE_ROOT.iterdir()},
        )
        self.assertEqual(799, len(v10_cases))
        physics_paths = {
            case["assets"]["physics_annotation"] for case in v10_cases
        }
        self.assertEqual(799, len(physics_paths))
        for old, new in zip(v9_cases, v10_cases, strict=True):
            comparable = copy.deepcopy(new)
            comparable["assets"].pop("physics_annotation")
            self.assertEqual(old, comparable)

    def test_release_lock_adds_only_799_physics_documents(self) -> None:
        v9 = load_json(builder.BASE_RELEASE_ROOT / "assets.lock.json")
        v10 = load_json(builder.OUTPUT_RELEASE_ROOT / "assets.lock.json")
        old_by_path = {item["path"]: item for item in v9["files"]}
        new_by_path = {item["path"]: item for item in v10["files"]}

        self.assertEqual(5239, len(old_by_path))
        self.assertEqual(6038, len(new_by_path))
        for path, old in old_by_path.items():
            self.assertEqual(old, new_by_path[path])
        added = set(new_by_path) - set(old_by_path)
        self.assertEqual(799, len(added))
        self.assertTrue(all(path.endswith("/physics.json") for path in added))

    def test_v9_and_v10_physics_fingerprints_are_identical(self) -> None:
        fingerprints = []
        for release_root in (
            builder.BASE_RELEASE_ROOT,
            builder.OUTPUT_RELEASE_ROOT,
        ):
            cases = load_jsonl(release_root / "cases.jsonl")
            fingerprints.append(
                canonical_sha256(
                    [(case["case_id"], case["physics"]) for case in cases]
                )
            )

        self.assertEqual(
            [EXPECTED_PHYSICS_FINGERPRINT, EXPECTED_PHYSICS_FINGERPRINT],
            fingerprints,
        )


if __name__ == "__main__":
    unittest.main()
