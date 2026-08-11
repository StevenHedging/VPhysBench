from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

from physbench.dataset_distribution import (
    load_distribution_manifest,
    verify_and_extract_shard,
)


DATASET_ID = "physics_video_seven_scene_v13"
RELEASE = "13.0.0"
DATASET_DIGEST = "d" * 64
SHARD_SHA256 = "a" * 64
FILE_SHA256 = "b" * 64


def valid_manifest_value() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "release": RELEASE,
        "dataset_digest": DATASET_DIGEST,
        "total_files": 1,
        "total_bytes": 7,
        "shards": [{
            "name": "assets-00000.zip",
            "path": "distribution/v1/shards/assets-00000.zip",
            "size_bytes": 149,
            "sha256": SHARD_SHA256,
        }],
        "files": [{
            "path": "assets/scene/case/frame.png",
            "shard": "assets-00000.zip",
            "size_bytes": 7,
            "sha256": FILE_SHA256,
        }],
    }


class DistributionManifestTests(unittest.TestCase):
    def write_manifest(self, value: object) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_loads_valid_literal_manifest_as_immutable_dataclasses(self) -> None:
        manifest = load_distribution_manifest(
            self.write_manifest(valid_manifest_value()),
            dataset_id=DATASET_ID,
            release=RELEASE,
        )

        self.assertEqual("1.0", manifest.schema_version)
        self.assertEqual(DATASET_DIGEST, manifest.dataset_digest)
        self.assertEqual("assets-00000.zip", manifest.shards[0].name)
        self.assertEqual(
            "assets/scene/case/frame.png",
            manifest.files[0].path,
        )
        with self.assertRaises(FrozenInstanceError):
            manifest.total_files = 2  # type: ignore[misc]

    def test_rejects_unknown_schema_version(self) -> None:
        value = valid_manifest_value()
        value["schema_version"] = "2.0"

        with self.assertRaisesRegex(ValueError, "schema_version"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_unknown_fields_at_every_schema_level(self) -> None:
        for level in ("manifest", "shard", "file"):
            with self.subTest(level=level):
                value = valid_manifest_value()
                if level == "manifest":
                    value["credential"] = "must-not-be-accepted"
                else:
                    items = value[f"{level}s"]
                    assert isinstance(items, list)
                    items[0]["extra"] = True

                with self.assertRaisesRegex(ValueError, "unknown fields"):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )


class VerifiedShardExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "assets-00000.zip"
        self.staging = self.root / "staging"

    def write_zip(
        self,
        members: list[tuple[str, bytes]],
        *,
        symlinks: set[str] | None = None,
        force_zip64: bool = False,
    ) -> None:
        with zipfile.ZipFile(
            self.archive,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name, content in members:
                    info = zipfile.ZipInfo(name)
                    if symlinks and name in symlinks:
                        info.create_system = 3
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    if force_zip64:
                        with archive.open(info, "w", force_zip64=True) as handle:
                            handle.write(content)
                    else:
                        archive.writestr(info, content)

    def load_manifest(
        self,
        declared: list[tuple[str, bytes]],
        *,
        shard_sha256: str | None = None,
        file_sha256: str | None = None,
        file_size: int | None = None,
    ):
        value = valid_manifest_value()
        shard = value["shards"]
        assert isinstance(shard, list)
        archive_bytes = self.archive.read_bytes()
        shard[0]["size_bytes"] = len(archive_bytes)
        shard[0]["sha256"] = shard_sha256 or hashlib.sha256(archive_bytes).hexdigest()
        files = []
        for index, (name, content) in enumerate(declared):
            files.append({
                "path": name,
                "shard": "assets-00000.zip",
                "size_bytes": (
                    file_size if index == 0 and file_size is not None else len(content)
                ),
                "sha256": (
                    file_sha256
                    if index == 0 and file_sha256 is not None
                    else hashlib.sha256(content).hexdigest()
                ),
            })
        value["files"] = files
        value["total_files"] = len(files)
        value["total_bytes"] = sum(int(item["size_bytes"]) for item in files)
        return load_distribution_manifest(
            self.write_manifest(value),
            dataset_id=DATASET_ID,
            release=RELEASE,
        )

    def write_manifest(self, value: object) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def extract(self, manifest) -> list[Path]:
        return verify_and_extract_shard(
            self.archive,
            manifest=manifest,
            shard_name="assets-00000.zip",
            staging_root=self.staging,
        )

    def test_extracts_valid_zip64_archive_to_staging(self) -> None:
        members = [
            ("assets/scene/a.bin", b"alpha"),
            ("assets/scene/b.bin", b"beta"),
        ]
        self.write_zip(members, force_zip64=True)
        manifest = self.load_manifest(members)

        extracted = self.extract(manifest)

        self.assertEqual(
            [
                self.staging / "assets/scene/a.bin",
                self.staging / "assets/scene/b.bin",
            ],
            extracted,
        )
        self.assertEqual(b"alpha", extracted[0].read_bytes())
        self.assertEqual(b"beta", extracted[1].read_bytes())

    def test_rejects_undeclared_and_missing_members_before_writing(self) -> None:
        cases = (
            (
                "undeclared",
                [("assets/declared.bin", b"ok"), ("assets/extra.bin", b"no")],
                [("assets/declared.bin", b"ok")],
            ),
            (
                "missing",
                [("assets/present.bin", b"ok")],
                [("assets/present.bin", b"ok"), ("assets/missing.bin", b"no")],
            ),
        )
        for label, members, declared in cases:
            with self.subTest(label=label):
                self.write_zip(members)
                manifest = self.load_manifest(declared)
                with self.assertRaisesRegex(ValueError, label):
                    self.extract(manifest)
                self.assertFalse((self.staging / "assets").exists())

    def test_rejects_zip_symlinks_before_writing(self) -> None:
        members = [("assets/link", b"../../outside")]
        self.write_zip(members, symlinks={"assets/link"})
        manifest = self.load_manifest(members)

        with self.assertRaisesRegex(ValueError, "symlink"):
            self.extract(manifest)
        self.assertFalse((self.staging / "assets/link").exists())

    def test_rejects_absolute_and_traversing_zip_members(self) -> None:
        for unsafe in ("/assets/file.bin", "assets/../outside.bin"):
            with self.subTest(unsafe=unsafe):
                members = [(unsafe, b"escape")]
                self.write_zip(members)
                value = valid_manifest_value()
                archive_bytes = self.archive.read_bytes()
                shards = value["shards"]
                assert isinstance(shards, list)
                shards[0]["size_bytes"] = len(archive_bytes)
                shards[0]["sha256"] = hashlib.sha256(archive_bytes).hexdigest()
                # The archive member is intentionally not declared because unsafe
                # paths cannot pass manifest validation.
                value["files"] = []
                value["total_files"] = 0
                value["total_bytes"] = 0
                manifest = load_distribution_manifest(
                    self.write_manifest(value),
                    dataset_id=DATASET_ID,
                    release=RELEASE,
                )
                with self.assertRaisesRegex(ValueError, "POSIX relative"):
                    self.extract(manifest)
                self.assertFalse((self.root / "outside.bin").exists())

    def test_rejects_duplicate_zip_member_names(self) -> None:
        members = [
            ("assets/duplicate.bin", b"first"),
            ("assets/duplicate.bin", b"second"),
        ]
        self.write_zip(members)
        manifest = self.load_manifest([members[0]])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.extract(manifest)
        self.assertFalse((self.staging / "assets/duplicate.bin").exists())

    def test_rejects_shard_hash_mismatch_before_opening(self) -> None:
        members = [("assets/file.bin", b"payload")]
        self.write_zip(members)
        manifest = self.load_manifest(members, shard_sha256="0" * 64)

        with self.assertRaisesRegex(ValueError, "shard sha256"):
            self.extract(manifest)
        self.assertFalse(self.staging.exists())

    def test_rejects_extracted_file_hash_and_size_mismatches(self) -> None:
        members = [("assets/file.bin", b"payload")]
        for label, overrides in (
            ("sha256", {"file_sha256": "0" * 64}),
            ("size", {"file_size": 8}),
        ):
            with self.subTest(label=label):
                self.write_zip(members)
                manifest = self.load_manifest(members, **overrides)
                with self.assertRaisesRegex(ValueError, label):
                    self.extract(manifest)
                self.assertFalse((self.staging / "assets/file.bin").exists())

    def test_rejects_truncated_archive_even_when_download_hash_matches(self) -> None:
        members = [("assets/file.bin", b"payload")]
        self.write_zip(members)
        self.archive.write_bytes(self.archive.read_bytes()[:-20])
        manifest = self.load_manifest(members)

        with self.assertRaisesRegex(ValueError, "invalid ZIP"):
            self.extract(manifest)
        self.assertFalse(self.staging.exists())


class DistributionManifestValidationTests(unittest.TestCase):
    def write_manifest(self, value: object) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_rejects_duplicate_file_paths(self) -> None:
        value = valid_manifest_value()
        files = value["files"]
        assert isinstance(files, list)
        files.append({
            "path": "assets/scene/case/frame.png",
            "shard": "assets-00000.zip",
            "size_bytes": 11,
            "sha256": "c" * 64,
        })
        value["total_files"] = 2
        value["total_bytes"] = 18

        with self.assertRaisesRegex(ValueError, "duplicate file path"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_duplicate_shard_names(self) -> None:
        value = valid_manifest_value()
        shards = value["shards"]
        assert isinstance(shards, list)
        shards.append({
            "name": "assets-00000.zip",
            "path": "distribution/v1/shards/assets-copy.zip",
            "size_bytes": 151,
            "sha256": "c" * 64,
        })

        with self.assertRaisesRegex(ValueError, "duplicate shard name"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_non_sha256_digests(self) -> None:
        for target in ("dataset", "shard", "file"):
            with self.subTest(target=target):
                value = valid_manifest_value()
                if target == "dataset":
                    value["dataset_digest"] = "not-a-digest"
                else:
                    items = value[f"{target}s"]
                    assert isinstance(items, list)
                    items[0]["sha256"] = "ABC123"

                with self.assertRaisesRegex(ValueError, "64 lowercase hex"):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_absolute_and_parent_traversing_paths(self) -> None:
        cases = (
            ("file-absolute", "files", "/assets/scene/frame.png"),
            ("file-parent", "files", "assets/../secrets.txt"),
            (
                "shard-absolute",
                "shards",
                "/distribution/v1/shards/assets-00000.zip",
            ),
            (
                "shard-parent",
                "shards",
                "distribution/v1/shards/../assets-00000.zip",
            ),
            ("file-dot", "files", "assets/./scene/frame.png"),
            ("file-empty-component", "files", "assets//scene/frame.png"),
            ("file-backslash", "files", "assets\\scene\\frame.png"),
        )
        for label, collection, unsafe_path in cases:
            with self.subTest(label=label):
                value = valid_manifest_value()
                items = value[collection]
                assert isinstance(items, list)
                items[0]["path"] = unsafe_path

                with self.assertRaisesRegex(ValueError, "POSIX relative"):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_files_outside_assets(self) -> None:
        value = valid_manifest_value()
        files = value["files"]
        assert isinstance(files, list)
        files[0]["path"] = "metadata/private.txt"

        with self.assertRaisesRegex(ValueError, "under assets/"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_archive_paths_outside_distribution_shards(self) -> None:
        value = valid_manifest_value()
        shards = value["shards"]
        assert isinstance(shards, list)
        shards[0]["path"] = "uploads/assets-00000.zip"

        with self.assertRaisesRegex(ValueError, "distribution/v1/shards/"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_inconsistent_totals(self) -> None:
        for total_field, inconsistent in (
            ("total_files", 2),
            ("total_bytes", 8),
        ):
            with self.subTest(total_field=total_field):
                value = valid_manifest_value()
                value[total_field] = inconsistent

                with self.assertRaisesRegex(ValueError, total_field):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_mismatched_dataset_identity(self) -> None:
        for field in (
            "dataset_id",
            "release",
        ):
            with self.subTest(field=field):
                value = valid_manifest_value()
                value[field] = "different"

                with self.assertRaisesRegex(ValueError, field):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_file_references_to_unknown_shards(self) -> None:
        value = valid_manifest_value()
        files = value["files"]
        assert isinstance(files, list)
        files[0]["shard"] = "missing.zip"

        with self.assertRaisesRegex(ValueError, "unknown shard"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_invalid_byte_counts(self) -> None:
        for collection, invalid in (("shards", True), ("files", -1)):
            with self.subTest(collection=collection):
                value = valid_manifest_value()
                items = value[collection]
                assert isinstance(items, list)
                items[0]["size_bytes"] = invalid

                with self.assertRaisesRegex(ValueError, "size_bytes"):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_malformed_schema_shapes(self) -> None:
        malformed_values: tuple[object, ...] = (
            [],
            {key: item for key, item in valid_manifest_value().items() if key != "release"},
            {**valid_manifest_value(), "shards": {}},
            {**valid_manifest_value(), "files": [7]},
        )
        for value in malformed_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_malformed_path_and_shard_reference_fields(self) -> None:
        for collection, field, invalid in (
            ("shards", "name", []),
            ("shards", "path", []),
            ("files", "path", []),
            ("files", "shard", {}),
        ):
            with self.subTest(collection=collection, field=field):
                value = valid_manifest_value()
                items = value[collection]
                assert isinstance(items, list)
                items[0][field] = invalid
                with self.assertRaises(ValueError):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )

    def test_rejects_duplicate_or_mismatched_shard_paths(self) -> None:
        for mutation in ("duplicate", "name-mismatch"):
            with self.subTest(mutation=mutation):
                value = valid_manifest_value()
                shards = value["shards"]
                assert isinstance(shards, list)
                if mutation == "duplicate":
                    shards.append({
                        "name": "assets-copy.zip",
                        "path": "distribution/v1/shards/assets-00000.zip",
                        "size_bytes": 151,
                        "sha256": "c" * 64,
                    })
                else:
                    shards[0]["name"] = "different.zip"

                with self.assertRaisesRegex(ValueError, "shard path"):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )


if __name__ == "__main__":
    unittest.main()
