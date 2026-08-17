from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
import zipfile
from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from physbench.dataset_distribution import (
    load_distribution_manifest,
    verify_and_extract_shard,
)
from physbench.datasets import load_dataset
from physbench.io import canonical_sha256
from scripts import build_dataset_distribution as distribution_builder


DATASET_ID = "physics_video_seven_scene_v13"
RELEASE = "13.0.0"
DATASET_DIGEST = "d" * 64
SHARD_SHA256 = "a" * 64
FILE_SHA256 = "b" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
        compression: int = zipfile.ZIP_STORED,
    ) -> None:
        self.write_zip_to(
            self.archive,
            members,
            symlinks=symlinks,
            force_zip64=force_zip64,
            compression=compression,
        )

    def write_zip_to(
        self,
        path: Path,
        members: list[tuple[str, bytes]],
        *,
        symlinks: set[str] | None = None,
        force_zip64: bool = False,
        compression: int = zipfile.ZIP_STORED,
    ) -> None:
        with zipfile.ZipFile(
            path,
            "w",
            compression=compression,
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

    def extract(self, manifest, **kwargs: object) -> list[Path]:
        return verify_and_extract_shard(
            self.archive,
            manifest=manifest,
            shard_name="assets-00000.zip",
            staging_root=self.staging,
            **kwargs,
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

    def test_rejects_symlink_staging_root_without_external_write(self) -> None:
        members = [("assets/scene/file.bin", b"payload")]
        self.write_zip(members)
        manifest = self.load_manifest(members)
        external = self.root / "external"
        external.mkdir()
        self.staging.symlink_to(external, target_is_directory=True)

        with self.assertRaisesRegex((ValueError, OSError), "symlink|staging"):
            self.extract(manifest)

        self.assertFalse((external / "assets/scene/file.bin").exists())

    def test_directory_fd_anchor_detects_parent_swap_without_external_write(self) -> None:
        members = [("assets/scene/file.bin", b"payload")]
        self.write_zip(members)
        manifest = self.load_manifest(members)
        external = self.root / "external"
        external.mkdir()
        held = self.root / "held-assets"
        real_open = os.open
        swapped = False

        def racing_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal swapped
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if path == "assets" and dir_fd is not None and not swapped:
                swapped = True
                (self.staging / "assets").rename(held)
                (self.staging / "assets").symlink_to(
                    external,
                    target_is_directory=True,
                )
            return descriptor

        with patch("physbench.dataset_distribution.os.open", side_effect=racing_open):
            with self.assertRaisesRegex((ValueError, OSError), "changed|symlink"):
                self.extract(manifest)

        self.assertTrue(swapped)
        self.assertFalse((external / "scene/file.bin").exists())

    def test_archive_is_opened_once_with_nofollow(self) -> None:
        members = [("assets/file.bin", b"payload")]
        self.write_zip(members)
        manifest = self.load_manifest(members)
        real_open = os.open
        archive_flags: list[int] = []

        def recording_open(path, flags, mode=0o777, *, dir_fd=None):
            if os.fspath(path) == os.fspath(self.archive) and dir_fd is None:
                archive_flags.append(flags)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with patch("physbench.dataset_distribution.os.open", side_effect=recording_open):
            self.extract(manifest)

        self.assertEqual(1, len(archive_flags))
        self.assertTrue(archive_flags[0] & os.O_NOFOLLOW)

    def test_archive_path_replacement_cannot_change_opened_inode(self) -> None:
        members = [("assets/original.bin", b"trusted")]
        self.write_zip(members)
        manifest = self.load_manifest(members)
        replacement = self.root / "replacement.zip"
        self.write_zip_to(replacement, [("assets/original.bin", b"evil")])
        real_zip_file = zipfile.ZipFile
        opened_values: list[object] = []

        def replacing_zip_file(opened, *args, **kwargs):
            opened_values.append(opened)
            os.replace(replacement, self.archive)
            return real_zip_file(opened, *args, **kwargs)

        with patch(
            "physbench.dataset_distribution.zipfile.ZipFile",
            side_effect=replacing_zip_file,
        ):
            extracted = self.extract(manifest)

        self.assertEqual(b"trusted", extracted[0].read_bytes())
        self.assertTrue(hasattr(opened_values[0], "read"))

    def test_rejects_archive_metadata_change_during_hash(self) -> None:
        members = [("assets/file.bin", b"payload")]
        self.write_zip(members)
        manifest = self.load_manifest(members)
        real_fstat = os.fstat
        calls = 0

        def changing_fstat(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                current = self.archive.stat()
                os.utime(
                    self.archive,
                    ns=(current.st_atime_ns, current.st_mtime_ns + 1),
                )
            return real_fstat(descriptor)

        with patch("physbench.dataset_distribution.os.fstat", side_effect=changing_fstat):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                self.extract(manifest)

    def test_rejects_archive_symlink(self) -> None:
        members = [("assets/file.bin", b"payload")]
        self.write_zip(members)
        target = self.root / "real.zip"
        self.archive.rename(target)
        self.archive.symlink_to(target.name)
        manifest = self.load_manifest(members)

        with self.assertRaisesRegex((ValueError, OSError), "symlink|loop"):
            self.extract(manifest)

    def test_rejects_member_size_metadata_before_creating_assets(self) -> None:
        content = b"0" * (128 * 1024)
        members = [("assets/bomb.bin", content)]
        self.write_zip(members, compression=zipfile.ZIP_DEFLATED)
        manifest = self.load_manifest(members, file_size=1)

        with self.assertRaisesRegex(ValueError, "file_size|size"):
            self.extract(manifest)

        self.assertFalse((self.staging / "assets").exists())

    def test_rejects_cumulative_extraction_budget_before_writing(self) -> None:
        members = [
            ("assets/a.bin", b"a" * 8),
            ("assets/b.bin", b"b" * 8),
        ]
        self.write_zip(members, compression=zipfile.ZIP_DEFLATED)
        manifest = self.load_manifest(members)

        with self.assertRaisesRegex(ValueError, "budget"):
            self.extract(manifest, max_extracted_bytes=15)

        self.assertFalse((self.staging / "assets").exists())

    def test_rejects_differently_normalized_zip_member(self) -> None:
        nfd_member = "assets/cafe\u0301.bin"
        nfc_member = "assets/caf\u00e9.bin"
        self.write_zip([(nfd_member, b"payload")])
        manifest = self.load_manifest([(nfc_member, b"payload")])

        with self.assertRaisesRegex(ValueError, "NFC"):
            self.extract(manifest)

        self.assertFalse((self.staging / "assets").exists())


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

    def test_rejects_strict_component_prefix_collision_across_shards(self) -> None:
        value = valid_manifest_value()
        shards = value["shards"]
        assert isinstance(shards, list)
        shards.append({
            "name": "assets-00001.zip",
            "path": "distribution/v1/shards/assets-00001.zip",
            "size_bytes": 149,
            "sha256": "c" * 64,
        })
        value["files"] = [
            {
                "path": "assets/a",
                "shard": "assets-00000.zip",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
            {
                "path": "assets/a/b",
                "shard": "assets-00001.zip",
                "size_bytes": 1,
                "sha256": "b" * 64,
            },
        ]
        value["total_files"] = 2
        value["total_bytes"] = 2

        with self.assertRaisesRegex(ValueError, "prefix collision"):
            load_distribution_manifest(
                self.write_manifest(value),
                dataset_id=DATASET_ID,
                release=RELEASE,
            )

    def test_rejects_non_nfc_manifest_paths_and_normalized_collisions(self) -> None:
        nfd_path = "assets/cafe\u0301.bin"
        for include_nfc in (False, True):
            with self.subTest(include_nfc=include_nfc):
                value = valid_manifest_value()
                files = [{
                    "path": nfd_path,
                    "shard": "assets-00000.zip",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                }]
                if include_nfc:
                    files.append({
                        "path": "assets/caf\u00e9.bin",
                        "shard": "assets-00000.zip",
                        "size_bytes": 1,
                        "sha256": "b" * 64,
                    })
                value["files"] = files
                value["total_files"] = len(files)
                value["total_bytes"] = len(files)

                with self.assertRaisesRegex(ValueError, "NFC|duplicate"):
                    load_distribution_manifest(
                        self.write_manifest(value),
                        dataset_id=DATASET_ID,
                        release=RELEASE,
                    )


class DatasetDistributionBuilderTests(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_dataset(self, root: Path) -> Path:
        release = root / "releases" / "1.0.0"
        assets = root / "assets"
        release.mkdir(parents=True)
        assets.mkdir()
        contents = {
            "00-caption.json": (
                b'{"case_id":"fixture_case","scene_id":"pendulum",'
                b'"caption":"fixture prompt"}'
            ),
            "01-frame.bin": b"ffffffffff",
            "02-mask.bin": b"mmmmmmmm",
            "03-physics.json": (
                b'{"case_id":"fixture_case","scene_id":"pendulum",'
                b'"physics":{"objects":{"object_1":{"radius":{"value":1,'
                b'"unit":"m","symbol":"r"},"initial_angle":{"value":1,'
                b'"unit":"rad","symbol":"theta"}}},"environment":{'
                b'"string_length":{"value":1,"unit":"m","symbol":"L"}}}}'
            ),
        }
        for name, content in contents.items():
            (assets / name).write_bytes(content)
        (assets / "unreferenced.bin").write_bytes(b"must not be distributed")
        (root / "provenance").mkdir()
        (root / "provenance" / "source.mov").write_bytes(
            b"must not be distributed"
        )

        case = {
            "case_id": "fixture_case",
            "scene_id": "pendulum",
            "assets": {
                "caption": "assets/00-caption.json",
                "first_frame": "assets/01-frame.bin",
                "first_frame_mask_manifest": "assets/02-mask.bin",
                "physics_annotation": "assets/03-physics.json",
                "reference_video": "assets/01-frame.bin",
            },
            "appearance": {},
            "temporal": {},
        }
        (release / "cases.jsonl").write_text(
            json.dumps(case, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_json(
            release / "scenes" / "pendulum.json",
            {
                "schema_version": "2.0",
                "scene_id": "pendulum",
                "display_name": "Pendulum",
                "structured_physics_parameters": [
                    "objects.object_1.radius",
                    "objects.object_1.initial_angle",
                    "environment.string_length",
                ],
                "generalization_factors": [],
                "constraints": [],
                "metric_spec": {},
            },
        )
        case_set_digest = canonical_sha256(["fixture_case"])
        self.write_json(
            release / "views" / "view_a.json",
            {
                "schema_version": "3.0",
                "view_id": "view_a",
                "coverage": "complete",
                "case_set_sha256": case_set_digest,
                "split_semantics": {
                    "primary_partitions": ["train", "test"],
                    "generalization_regimes": ["id", "ood", "mixed"],
                    "regime_is_relative_to": "view_a.train",
                },
                "scenes": {
                    "pendulum": {"train": ["fixture_case"], "test": []}
                },
                "test_annotations": {},
            },
        )
        self.write_json(
            release / "views" / "view_b.json",
            {
                "schema_version": "2.0",
                "view_id": "view_b",
                "coverage": "complete",
                "case_set_sha256": case_set_digest,
                "scenes": {"pendulum": {"group_1": ["fixture_case"]}},
            },
        )
        descriptor = release / "dataset.json"
        self.write_json(
            descriptor,
            {
                "schema_version": "5.0",
                "dataset_id": "fixture_distribution",
                "release": "1.0.0",
                "cases": "cases.jsonl",
                "asset_root": "../..",
                "scene_catalog": "scenes",
                "views": {
                    "view_a": "views/view_a.json",
                    "view_b": "views/view_b.json",
                },
            },
        )
        return descriptor

    def invoke_builder(
        self,
        descriptor: Path,
        output_root: Path,
        *,
        max_shard_bytes: int = 800,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "build_dataset_distribution.py"
                ),
                "--dataset",
                str(descriptor),
                "--output-root",
                str(output_root),
                "--max-shard-bytes",
                str(max_shard_bytes),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_builder(self, descriptor: Path, output_root: Path) -> None:
        result = self.invoke_builder(descriptor, output_root)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_builds_stable_stored_shards_from_only_indexed_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            first_output = root / "first"
            second_output = root / "second"

            self.run_builder(descriptor, first_output)
            self.run_builder(descriptor, second_output)

            first_distribution = first_output / "distribution" / "v1"
            second_distribution = second_output / "distribution" / "v1"
            first_files = {
                path.relative_to(first_distribution).as_posix(): path.read_bytes()
                for path in sorted(first_distribution.rglob("*"))
                if path.is_file()
            }
            second_files = {
                path.relative_to(second_distribution).as_posix(): path.read_bytes()
                for path in sorted(second_distribution.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual(
                {
                    "manifest.json",
                    "shards/shard-00001.zip",
                    "shards/shard-00002.zip",
                },
                set(first_files),
            )

            manifest_path = first_distribution / "manifest.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n",
                manifest_path.read_bytes(),
            )
            snapshot = load_dataset(descriptor, check_assets=True)
            self.assertEqual(snapshot.digest, value["dataset_digest"])
            self.assertEqual(4, value["total_files"])
            self.assertEqual(349, value["total_bytes"])
            self.assertEqual(
                [
                    {
                        "path": "assets/00-caption.json",
                        "shard": "shard-00001.zip",
                        "size_bytes": 75,
                        "sha256": (
                            "6995117aac8541e1d12c78704b3a312c41790dfc578b8e813"
                            "eda9df041fbd8c6"
                        ),
                    },
                    {
                        "path": "assets/01-frame.bin",
                        "shard": "shard-00001.zip",
                        "size_bytes": 10,
                        "sha256": (
                            "d429d65fab713c3e8d9984b8f0a93fd1639eee4e7ad8ffca6"
                            "7e0ce68e4fbf903"
                        ),
                    },
                    {
                        "path": "assets/02-mask.bin",
                        "shard": "shard-00001.zip",
                        "size_bytes": 8,
                        "sha256": (
                            "4c67aa086d2b2c9debcb8ea2571d00f9d4f00873508828e9f"
                            "8089079d80af8b2"
                        ),
                    },
                    {
                        "path": "assets/03-physics.json",
                        "shard": "shard-00002.zip",
                        "size_bytes": 256,
                        "sha256": (
                            "e640ebbd9974003591c06b4f600c0a29ecf71f3a53804f85"
                            "e4e8541414eea9d1"
                        ),
                    },
                ],
                value["files"],
            )

            manifest = load_distribution_manifest(
                manifest_path,
                dataset_id="fixture_distribution",
                release="1.0.0",
            )
            self.assertEqual(2, len(manifest.shards))
            expected_members = [
                [
                    "assets/00-caption.json",
                    "assets/01-frame.bin",
                    "assets/02-mask.bin",
                ],
                ["assets/03-physics.json"],
            ]
            for shard, members in zip(manifest.shards, expected_members, strict=True):
                archive_path = first_output / shard.path
                archive_bytes = archive_path.read_bytes()
                self.assertLessEqual(len(archive_bytes), 800)
                self.assertEqual(len(archive_bytes), shard.size_bytes)
                self.assertEqual(
                    hashlib.sha256(archive_bytes).hexdigest(),
                    shard.sha256,
                )
                with zipfile.ZipFile(archive_path) as archive:
                    infos = archive.infolist()
                    self.assertEqual(members, [info.filename for info in infos])
                    for info in infos:
                        self.assertEqual(zipfile.ZIP_STORED, info.compress_type)
                        self.assertEqual((1980, 1, 1, 0, 0, 0), info.date_time)
                        self.assertEqual(3, info.create_system)
                        self.assertEqual(
                            stat.S_IFREG | 0o644,
                            info.external_attr >> 16,
                        )

    def test_asset_lock_defines_complete_distribution_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            release = descriptor.parent
            descriptor_value = json.loads(descriptor.read_text(encoding="utf-8"))
            paths = [
                "assets/00-caption.json",
                "assets/01-frame.bin",
                "assets/02-mask.bin",
                "assets/03-physics.json",
                "assets/unreferenced.bin",
            ]
            files = []
            for relative in paths:
                payload = (root / "source" / relative).read_bytes()
                files.append({
                    "path": relative,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                })
            self.write_json(
                release / "assets.lock.json",
                {
                    "schema_version": "1.0",
                    "dataset_id": "fixture_distribution",
                    "release": "1.0.0",
                    "files": files,
                    "files_digest": canonical_sha256(files),
                },
            )
            descriptor_value["asset_lock"] = "assets.lock.json"
            self.write_json(descriptor, descriptor_value)

            output = root / "output"
            self.run_builder(descriptor, output)

            manifest = json.loads(
                (output / "distribution/v1/manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(paths, [record["path"] for record in manifest["files"]])

    def test_rejects_a_single_asset_larger_than_the_shard_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")

            result = self.invoke_builder(
                descriptor,
                root / "output",
                max_shard_bytes=200,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("exceeds max shard bytes", result.stderr)

    def test_rejects_when_a_single_stored_member_exceeds_the_zip_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")

            result = self.invoke_builder(
                descriptor,
                root / "output",
                max_shard_bytes=300,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("ZIP metadata", result.stderr)

    def test_rejects_a_referenced_asset_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            frame = root / "source" / "assets" / "01-frame.bin"
            target = root / "source" / "frame-target.bin"
            frame.rename(target)
            frame.symlink_to(target)

            result = self.invoke_builder(descriptor, root / "output")

            self.assertNotEqual(0, result.returncode)
            self.assertIn("symlink", result.stderr)

    def test_uses_the_validated_case_snapshot_if_the_index_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            real_load_dataset = distribution_builder.load_dataset

            def load_then_replace(*args, **kwargs):
                snapshot = real_load_dataset(*args, **kwargs)
                indexed_case = json.loads(
                    (snapshot.root / "cases.jsonl").read_text(encoding="utf-8")
                )
                indexed_case["assets"] = {
                    key: "assets/unreferenced.bin"
                    for key in indexed_case["assets"]
                }
                (snapshot.root / "cases.jsonl").write_text(
                    json.dumps(indexed_case) + "\n",
                    encoding="utf-8",
                )
                return snapshot

            with patch.object(
                distribution_builder,
                "load_dataset",
                side_effect=load_then_replace,
            ):
                manifest_path = distribution_builder.build_distribution(
                    descriptor,
                    output_root=root / "output",
                    max_shard_bytes=800,
                )

            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    "assets/00-caption.json",
                    "assets/01-frame.bin",
                    "assets/02-mask.bin",
                    "assets/03-physics.json",
                ],
                [record["path"] for record in value["files"]],
            )

    def test_rejects_source_path_replacement_after_inventory(self) -> None:
        for mutation in ("parent-symlink", "leaf-replacement"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                descriptor = self.make_dataset(root / "source")
                release = descriptor.parent
                indexed_case = json.loads(
                    (release / "cases.jsonl").read_text(encoding="utf-8")
                )
                indexed_case["assets"]["first_frame"] = (
                    "assets/sub/01-frame.bin"
                )
                indexed_case["assets"]["reference_video"] = (
                    "assets/sub/01-frame.bin"
                )
                (release / "cases.jsonl").write_text(
                    json.dumps(indexed_case) + "\n",
                    encoding="utf-8",
                )
                assets = root / "source" / "assets"
                nested = assets / "sub"
                nested.mkdir()
                (assets / "01-frame.bin").rename(nested / "01-frame.bin")
                real_write_shard = distribution_builder._write_shard
                swapped = False

                def write_after_swap(*args, **kwargs):
                    nonlocal swapped
                    if not swapped:
                        swapped = True
                        if mutation == "parent-symlink":
                            held = assets / "held-sub"
                            external = root / "external"
                            external.mkdir()
                            (external / "01-frame.bin").write_bytes(b"EEEEEEEEEE")
                            nested.rename(held)
                            nested.symlink_to(external, target_is_directory=True)
                        else:
                            frame = nested / "01-frame.bin"
                            frame.unlink()
                            frame.write_bytes(b"EEEEEEEEEE")
                    return real_write_shard(*args, **kwargs)

                with patch.object(
                    distribution_builder,
                    "_write_shard",
                    side_effect=write_after_swap,
                ), self.assertRaisesRegex(ValueError, "changed|symlink|directory"):
                    distribution_builder.build_distribution(
                        descriptor,
                        output_root=root / "output",
                        max_shard_bytes=800,
                    )

    def test_rejects_same_inode_rewrite_with_restored_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            frame = root / "source" / "assets" / "01-frame.bin"
            real_write_shard = distribution_builder._write_shard
            rewritten = False

            def write_after_rewrite(*args, **kwargs):
                nonlocal rewritten
                if not rewritten:
                    rewritten = True
                    before = frame.stat()
                    with frame.open("r+b") as handle:
                        handle.write(b"EEEEEEEEEE")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.utime(
                        frame,
                        ns=(before.st_atime_ns, before.st_mtime_ns),
                    )
                    self.assertEqual(before.st_ino, frame.stat().st_ino)
                    self.assertEqual(before.st_mtime_ns, frame.stat().st_mtime_ns)
                    self.assertNotEqual(before.st_ctime_ns, frame.stat().st_ctime_ns)
                return real_write_shard(*args, **kwargs)

            with patch.object(
                distribution_builder,
                "_write_shard",
                side_effect=write_after_rewrite,
            ), self.assertRaisesRegex(ValueError, "changed"):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=root / "output",
                    max_shard_bytes=800,
                )

    def test_rejects_symlinked_or_swapped_asset_root_components(self) -> None:
        for mutation in ("configured-symlink", "parent-swap"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                work = base / "work"
                descriptor = self.make_dataset(work / "source")
                if mutation == "configured-symlink":
                    link = work / "source-link"
                    link.symlink_to(work / "source", target_is_directory=True)
                    value = json.loads(descriptor.read_text(encoding="utf-8"))
                    value["asset_root"] = "../../../source-link"
                    descriptor.write_text(json.dumps(value), encoding="utf-8")
                    context = patch.object(
                        distribution_builder,
                        "_open_asset_root",
                        wraps=distribution_builder._open_asset_root,
                    )
                else:
                    real_open_asset_root = distribution_builder._open_asset_root
                    swapped = False

                    def open_after_parent_swap(*args, **kwargs):
                        nonlocal swapped
                        if not swapped:
                            swapped = True
                            held = base / "held-work"
                            external = base / "external"
                            work.rename(held)
                            shutil.copytree(held, external)
                            work.symlink_to(external, target_is_directory=True)
                        return real_open_asset_root(*args, **kwargs)

                    context = patch.object(
                        distribution_builder,
                        "_open_asset_root",
                        side_effect=open_after_parent_swap,
                    )

                with context, self.assertRaisesRegex(
                    ValueError,
                    "asset_root.*symlink|directory changed",
                ):
                    distribution_builder.build_distribution(
                        descriptor,
                        output_root=base / "output",
                        max_shard_bytes=800,
                    )

    def test_rejects_ordinary_asset_root_parent_swap_after_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            work = base / "work"
            descriptor = self.make_dataset(work / "source")
            real_load_dataset = distribution_builder.load_dataset
            swapped = False

            def load_then_swap(*args, **kwargs):
                nonlocal swapped
                snapshot = real_load_dataset(*args, **kwargs)
                if not swapped:
                    swapped = True
                    held = base / "held-work"
                    work.rename(held)
                    shutil.copytree(held, work)
                    replacement = work / "source" / "assets" / "01-frame.bin"
                    replacement.write_bytes(b"EEEEEEEEEE")
                return snapshot

            with patch.object(
                distribution_builder,
                "load_dataset",
                side_effect=load_then_swap,
            ), self.assertRaisesRegex(
                ValueError,
                "asset_root.*changed|directory changed",
            ):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=base / "output",
                    max_shard_bytes=800,
                )

    def test_allows_unrelated_sibling_changes_in_an_asset_root_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            real_write_shard = distribution_builder._write_shard
            changed = False

            def write_then_change_unrelated_sibling(*args, **kwargs):
                nonlocal changed
                records = real_write_shard(*args, **kwargs)
                if not changed:
                    changed = True
                    (root / "unrelated-sibling").mkdir()
                return records

            with patch.object(
                distribution_builder,
                "_write_shard",
                side_effect=write_then_change_unrelated_sibling,
            ):
                manifest = distribution_builder.build_distribution(
                    descriptor,
                    output_root=root / "output",
                    max_shard_bytes=800,
                )

            self.assertTrue(changed)
            self.assertTrue(manifest.is_file())

    def test_rejects_rebuilding_over_an_existing_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            output = root / "output"
            self.run_builder(descriptor, output)
            distribution = output / "distribution" / "v1"
            before = {
                path.relative_to(distribution).as_posix(): path.read_bytes()
                for path in sorted(distribution.rglob("*"))
                if path.is_file()
            }

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=output,
                    max_shard_bytes=400,
                )

            after = {
                path.relative_to(distribution).as_posix(): path.read_bytes()
                for path in sorted(distribution.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_failed_builds_do_not_leave_the_final_distribution(self) -> None:
        for failure in ("source", "self-verification"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                descriptor = self.make_dataset(root / "source")
                target = (
                    "_write_shard"
                    if failure == "source"
                    else "_verify_output"
                )
                with patch.object(
                    distribution_builder,
                    target,
                    side_effect=ValueError(f"forced {failure} failure"),
                ), self.assertRaisesRegex(ValueError, f"forced {failure}"):
                    distribution_builder.build_distribution(
                        descriptor,
                        output_root=root / "output",
                        max_shard_bytes=800,
                    )

                self.assertFalse(
                    (root / "output" / "distribution" / "v1").exists()
                )

    def test_concurrent_builders_publish_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            output = root / "output"
            barrier = threading.Barrier(2)
            real_open_asset_root = distribution_builder._open_asset_root

            def synchronized_open(*args, **kwargs):
                barrier.wait(timeout=5)
                return real_open_asset_root(*args, **kwargs)

            def build():
                try:
                    return distribution_builder.build_distribution(
                        descriptor,
                        output_root=output,
                        max_shard_bytes=800,
                    )
                except BaseException as exc:
                    return exc

            with patch.object(
                distribution_builder,
                "_open_asset_root",
                side_effect=synchronized_open,
            ), ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: build(), range(2)))

            self.assertEqual(1, sum(isinstance(item, Path) for item in results))
            self.assertEqual(
                1,
                sum(isinstance(item, FileExistsError) for item in results),
                results,
            )

    def test_rejects_pre_publish_temp_swap_without_deleting_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            output = root / "output"
            real_verify = distribution_builder._verify_output
            replacement: Path | None = None

            def verify_then_swap(*args, **kwargs):
                nonlocal replacement
                real_verify(*args, **kwargs)
                verified = Path(kwargs["version_root"])
                held = verified.parent / "held-verified-tree"
                verified.rename(held)
                verified.mkdir()
                replacement = verified
                (verified / "do-not-delete.txt").write_text(
                    "replacement",
                    encoding="utf-8",
                )

            with patch.object(
                distribution_builder,
                "_verify_output",
                side_effect=verify_then_swap,
            ), self.assertRaisesRegex(
                (ValueError, RuntimeError),
                "temporary.*changed|owned.*changed",
            ):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=output,
                    max_shard_bytes=800,
                )

            self.assertFalse((output / "distribution" / "v1").exists())
            self.assertIsNotNone(replacement)
            assert replacement is not None
            self.assertEqual(
                "replacement",
                (replacement / "do-not-delete.txt").read_text(encoding="utf-8"),
            )

    def test_withdraws_tree_swapped_after_the_final_temp_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            output = root / "output"
            real_verify = distribution_builder._verify_owned_temporary
            checks = 0
            swapped_marker: Path | None = None

            def verify_then_swap(parent_descriptor, owned_temporary):
                nonlocal checks, swapped_marker
                real_verify(parent_descriptor, owned_temporary)
                checks += 1
                if checks == 3:
                    public = output / "distribution" / owned_temporary.name
                    public.rename(output / "distribution" / "held-verified-tree")
                    public.mkdir()
                    swapped_marker = public / "unverified.txt"
                    swapped_marker.write_text("unverified", encoding="utf-8")

            with patch.object(
                distribution_builder,
                "_verify_owned_temporary",
                side_effect=verify_then_swap,
            ), self.assertRaisesRegex(
                (ValueError, RuntimeError),
                "published.*changed|temporary.*changed|owned.*changed",
            ):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=output,
                    max_shard_bytes=800,
                )

            self.assertEqual(3, checks)
            self.assertFalse((output / "distribution" / "v1").exists())
            self.assertIsNotNone(swapped_marker)

            rollback_output = root / "rollback-output"
            real_claim = distribution_builder._claim_owned_temporary

            def claim_then_swap(parent_descriptor, owned_temporary, *, prefix):
                claim = real_claim(
                    parent_descriptor,
                    owned_temporary,
                    prefix=prefix,
                )
                if prefix == "v1-claim":
                    public = rollback_output / "distribution" / claim
                    public.rename(
                        rollback_output / "distribution" / "held-claimed-tree"
                    )
                    public.mkdir()
                    (public / "unverified.txt").write_text(
                        "unverified",
                        encoding="utf-8",
                    )
                return claim

            with patch.object(
                distribution_builder,
                "_claim_owned_temporary",
                side_effect=claim_then_swap,
            ), self.assertRaisesRegex(
                (ValueError, RuntimeError),
                "withdrawn|temporary.*missing|owned.*changed",
            ):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=rollback_output,
                    max_shard_bytes=800,
                )

            self.assertFalse(
                (rollback_output / "distribution" / "v1").exists()
            )
            quarantined = list(
                (rollback_output / "distribution").glob(
                    ".v1-quarantine-*/unverified.txt"
                )
            )
            self.assertEqual(1, len(quarantined))
            self.assertEqual(
                "unverified",
                quarantined[0].read_text(encoding="utf-8"),
            )

    def test_cleanup_never_rmdirs_the_public_temp_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = self.make_dataset(root / "source")
            output = root / "output"
            real_rmdir = distribution_builder.os.rmdir
            real_cleanup = distribution_builder._cleanup_owned_temporary
            public_rmdir_attempts: list[str] = []

            def swap_before_public_rmdir(path, *args, **kwargs):
                name = os.fspath(path)
                if name.startswith(".v1-build-"):
                    public_rmdir_attempts.append(name)
                    public = output / "distribution" / name
                    public.rename(output / "distribution" / "held-cleaned-tree")
                    public.mkdir()
                return real_rmdir(path, *args, **kwargs)

            def cleanup_with_rmdir_race(*args, **kwargs):
                with patch.object(
                    distribution_builder.os,
                    "rmdir",
                    side_effect=swap_before_public_rmdir,
                ):
                    return real_cleanup(*args, **kwargs)

            with patch.object(
                distribution_builder,
                "_write_shard",
                side_effect=ValueError("forced cleanup failure"),
            ), patch.object(
                distribution_builder,
                "_cleanup_owned_temporary",
                side_effect=cleanup_with_rmdir_race,
            ), self.assertRaisesRegex(ValueError, "forced cleanup failure"):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=output,
                    max_shard_bytes=800,
                )

            self.assertEqual([], public_rmdir_attempts)
            self.assertFalse((output / "distribution" / "v1").exists())

            creation_output = root / "creation-output"
            real_create = distribution_builder._create_owned_temporary
            real_open = distribution_builder.os.open
            creation_rmdir_attempts: list[str] = []

            def fail_owned_open(path, *args, **kwargs):
                if os.fspath(path).startswith(".v1-build-"):
                    raise OSError("forced owned temporary open failure")
                return real_open(path, *args, **kwargs)

            def record_creation_rmdir(path, *args, **kwargs):
                name = os.fspath(path)
                if name.startswith(".v1-build-"):
                    creation_rmdir_attempts.append(name)
                return real_rmdir(path, *args, **kwargs)

            def create_with_open_failure(*args, **kwargs):
                with patch.object(
                    distribution_builder.os,
                    "open",
                    side_effect=fail_owned_open,
                ), patch.object(
                    distribution_builder.os,
                    "rmdir",
                    side_effect=record_creation_rmdir,
                ):
                    return real_create(*args, **kwargs)

            with patch.object(
                distribution_builder,
                "_create_owned_temporary",
                side_effect=create_with_open_failure,
            ), self.assertRaisesRegex(OSError, "forced owned temporary open failure"):
                distribution_builder.build_distribution(
                    descriptor,
                    output_root=creation_output,
                    max_shard_bytes=800,
                )

            self.assertEqual([], creation_rmdir_attempts)
            self.assertFalse(
                (creation_output / "distribution" / "v1").exists()
            )


if __name__ == "__main__":
    unittest.main()
