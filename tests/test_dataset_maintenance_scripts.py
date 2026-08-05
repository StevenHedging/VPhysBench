from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.io import (
    canonical_sha256,
    load_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from scripts import build_dataset_asset_lock
from scripts import migrate_dataset_v4


class DatasetV4MigrationSafetyTests(unittest.TestCase):
    def _write_identity(
        self,
        target: Path,
        *,
        dataset_id: str = "target_dataset",
        release: str = "4.0.0",
    ) -> None:
        target.mkdir(parents=True)
        for filename, schema_version in {
            "dataset.json": "3.0",
            "assets.lock.json": "1.0",
            "release.json": "1.0",
        }.items():
            write_json(
                target / filename,
                {
                    "schema_version": schema_version,
                    "dataset_id": dataset_id,
                    "release": release,
                },
            )

    def test_force_refuses_release_outside_the_release_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            releases = root / "datasets" / "releases"
            outside = root / "unrelated" / "4.0.0"
            self._write_identity(outside)
            marker = outside / "keep.txt"
            marker.write_text("must survive", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "may only replace the expected release directory"
            ):
                migrate_dataset_v4._remove_verified_release(
                    outside,
                    root / "source" / "dataset.json",
                    releases_root=releases,
                    expected_dataset_id="target_dataset",
                    expected_release="4.0.0",
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "must survive")

    def test_force_refuses_a_release_with_the_wrong_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            releases = root / "datasets" / "releases"
            target = releases / "4.0.0"
            self._write_identity(target, dataset_id="different_dataset")
            marker = target / "keep.txt"
            marker.write_text("must survive", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "unexpected dataset.json dataset_id"
            ):
                migrate_dataset_v4._remove_verified_release(
                    target,
                    root / "source" / "dataset.json",
                    releases_root=releases,
                    expected_dataset_id="target_dataset",
                    expected_release="4.0.0",
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "must survive")

    def test_force_removes_only_a_verified_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            releases = root / "datasets" / "releases"
            target = releases / "4.0.0"
            self._write_identity(target)

            removed = migrate_dataset_v4._remove_verified_release(
                target,
                root / "source" / "dataset.json",
                releases_root=releases,
                expected_dataset_id="target_dataset",
                expected_release="4.0.0",
            )

            self.assertEqual(removed, target.resolve())
            self.assertFalse(target.exists())


class AssetLockBuilderSafetyTests(unittest.TestCase):
    def _write_dataset(
        self,
        root: Path,
        *,
        valid_case: bool,
        physics_document: dict | None = None,
    ) -> Path:
        (root / "assets").mkdir(parents=True)
        (root / "assets" / "frame.bin").write_bytes(b"immutable asset\n")
        (root / "scenes").mkdir()
        (root / "views").mkdir()

        case = {
            "schema_version": "3.0",
            "case_id": "pendulum_case_1",
            "scene_id": "pendulum",
            "assets": {
                "first_frame": "frame.bin",
                "reference_video": "frame.bin",
            },
            "text": {
                "schema_version": "1.0",
                "prompt": "A pendulum swings.",
                "language": "en",
                "annotation_source": "test",
            },
            "physics": {
                "gravity": {
                    "value": 9.8,
                    "unit": "m/s^2",
                    "annotated": True,
                }
            },
            "appearance": {},
            "temporal": {},
            "provenance": {},
            "ood": {},
            "has_real_reference_video": True,
        }
        if not valid_case:
            case["text"].pop("prompt")
        if physics_document is not None:
            write_json(root / "assets" / "physics.json", physics_document)
            case["assets"]["physics_annotation"] = "physics.json"
        write_jsonl(root / "cases.jsonl", [case])
        write_json(
            root / "scenes" / "pendulum.json",
            {"schema_version": "1.0", "scene_id": "pendulum"},
        )
        write_json(
            root / "views" / "view_a.json",
            {
                "schema_version": "2.0",
                "view_id": "view_a",
                "coverage": "complete",
                "case_set_sha256": canonical_sha256(
                    ["pendulum_case_1"]
                ),
                "scenes": {
                    "pendulum": {
                        "test_id": ["pendulum_case_1"],
                    }
                },
            },
        )
        descriptor = {
            "schema_version": "3.0",
            "dataset_id": "asset_lock_test",
            "release": "9.9.9",
            "cases": "cases.jsonl",
            "asset_root": "assets",
            "asset_lock": "assets.lock.json",
            "release_manifest": "release.json",
            "scene_catalog": "scenes",
            "views": {"view_a": "views/view_a.json"},
        }
        write_json(root / "dataset.json", descriptor)
        return root / "dataset.json"

    @staticmethod
    def _physics_document() -> dict:
        return {
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
        }

    @staticmethod
    def _remove_release_contract(descriptor: Path) -> None:
        value = load_json(descriptor)
        value.pop("asset_lock")
        value.pop("release_manifest")
        write_json(descriptor, value)

    def test_default_dataset_is_the_latest_release(self) -> None:
        self.assertEqual(
            build_dataset_asset_lock.LATEST_DATASET,
            LATEST_DATASET,
        )

    def test_validation_failure_does_not_overwrite_release_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "9.9.9"
            root.mkdir()
            descriptor = self._write_dataset(root, valid_case=False)
            lock_path = root / "assets.lock.json"
            release_path = root / "release.json"
            original_lock = b'{"sentinel":"lock"}\n'
            original_release = b'{"sentinel":"release"}\n'
            lock_path.write_bytes(original_lock)
            release_path.write_bytes(original_release)

            with self.assertRaisesRegex(
                ValueError, "text fields must be"
            ):
                build_dataset_asset_lock.rebuild_asset_lock(descriptor)

            self.assertEqual(lock_path.read_bytes(), original_lock)
            self.assertEqual(release_path.read_bytes(), original_release)

    def test_valid_candidate_replaces_lock_and_release_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "9.9.9"
            root.mkdir()
            descriptor = self._write_dataset(root, valid_case=True)
            (root / "assets.lock.json").write_text(
                '{"sentinel":"lock"}\n', encoding="utf-8"
            )
            (root / "release.json").write_text(
                '{"sentinel":"release"}\n', encoding="utf-8"
            )

            output, release_manifest = (
                build_dataset_asset_lock.rebuild_asset_lock(descriptor)
            )

            lock = load_json(output)
            self.assertEqual(lock["dataset_id"], "asset_lock_test")
            self.assertEqual(lock["release"], "9.9.9")
            self.assertEqual(len(lock["files"]), 1)
            self.assertEqual(lock["files"][0]["path"], "frame.bin")
            self.assertEqual(
                load_json(root / "release.json"),
                release_manifest,
            )
            snapshot = load_dataset(descriptor, check_assets=True)
            self.assertEqual(
                snapshot.digest,
                release_manifest["dataset_digest"],
            )

    def test_hash_validation_checks_each_unique_asset_path_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "9.9.9"
            root.mkdir()
            descriptor = self._write_dataset(root, valid_case=True)
            build_dataset_asset_lock.rebuild_asset_lock(descriptor)

            with mock.patch(
                "physbench.datasets.loader.sha256_file",
                wraps=sha256_file,
            ) as hash_file:
                load_dataset(descriptor, check_asset_hashes=True)

            hash_file.assert_called_once_with(
                (root / "assets" / "frame.bin").resolve()
            )

    def test_matching_case_local_physics_loads_and_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "9.9.9"
            root.mkdir()
            descriptor = self._write_dataset(
                root,
                valid_case=True,
                physics_document=self._physics_document(),
            )

            output, _ = build_dataset_asset_lock.rebuild_asset_lock(descriptor)
            snapshot = load_dataset(descriptor, check_asset_hashes=True)

            self.assertEqual(
                ["frame.bin", "physics.json"],
                [item["path"] for item in load_json(output)["files"]],
            )
            self.assertEqual(
                "physics.json",
                snapshot.cases[0]["assets"]["physics_annotation"],
            )

    def test_case_local_physics_mismatch_is_rejected_without_asset_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "9.9.9"
            root.mkdir()
            document = self._physics_document()
            document["physics"]["gravity"]["value"] = 9.81
            descriptor = self._write_dataset(
                root,
                valid_case=True,
                physics_document=document,
            )
            self._remove_release_contract(descriptor)

            with self.assertRaisesRegex(
                ValueError, "differs from inline case.physics"
            ):
                load_dataset(descriptor, check_assets=False)

    def test_missing_case_local_physics_is_rejected_without_asset_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "9.9.9"
            root.mkdir()
            descriptor = self._write_dataset(
                root,
                valid_case=True,
                physics_document=self._physics_document(),
            )
            self._remove_release_contract(descriptor)
            (root / "assets" / "physics.json").unlink()

            with self.assertRaisesRegex(
                FileNotFoundError, "missing assets.physics_annotation"
            ):
                load_dataset(descriptor, check_assets=False)

    def test_case_local_physics_identity_and_schema_are_exact(self) -> None:
        mutations = {
            "schema": ("schema_version", "2.0", "schema must be 1.0"),
            "case": ("case_id", "pendulum_case_2", "Case mismatch"),
            "scene": ("scene_id", "collision_1d", "Scene mismatch"),
            "fields": ("unexpected", True, "fields must be"),
        }
        for label, (field, value, message) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "9.9.9"
                root.mkdir()
                document = copy.deepcopy(self._physics_document())
                document[field] = value
                descriptor = self._write_dataset(
                    root,
                    valid_case=True,
                    physics_document=document,
                )
                self._remove_release_contract(descriptor)

                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(descriptor, check_assets=False)


if __name__ == "__main__":
    unittest.main()
