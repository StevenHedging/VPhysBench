from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _record(path: Path, target: str) -> dict[str, object]:
    return {
        "candidate_path": path.name,
        "target_path": target,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_candidate(root: Path, *, corrupt_area: bool = False, missing_review: bool = False) -> Path:
    root.mkdir(parents=True)
    masks = np.zeros((3, 8, 8), np.uint8)
    masks[:, 2:4, 3:6] = 1
    mask_file = root / "mask.npz"
    np.savez_compressed(
        mask_file,
        packed_masks=np.packbits(masks, axis=2, bitorder="little"),
        height=np.asarray(8, np.int64),
        width=np.asarray(8, np.int64),
        observation_index=np.arange(3, dtype=np.int64),
        state=np.zeros(3, np.uint8),
    )
    trajectory_file = root / "trajectory.npz"
    np.savez_compressed(
        trajectory_file,
        centroid_xy=np.asarray([[4.0, 2.5]] * 3, np.float32),
        bbox_xyxy=np.asarray([[3.0, 2.0, 5.0, 3.0]] * 3, np.float32),
        area_pixels=np.asarray([7 if corrupt_area else 6] * 3, np.int64),
        observation_index=np.arange(3, dtype=np.int64),
        state=np.zeros(3, np.uint8),
    )
    review_file = root / "review.json"
    review_file.write_text('{"decision":"accepted"}\n', encoding="utf-8")
    records = [
        _record(mask_file, "reference_observation/entities/object_1/mask_tube.npz"),
        _record(trajectory_file, "reference_observation/entities/object_1/trajectory.npz"),
    ]
    if not missing_review:
        records.append(_record(review_file, "reference_observation/review.json"))
    manifest = {
        "schema_version": "1.0",
        "case_id": "case_a",
        "required_targets": [
            "reference_observation/entities/object_1/mask_tube.npz",
            "reference_observation/entities/object_1/trajectory.npz",
            "reference_observation/review.json",
        ],
        "files": records,
    }
    manifest_path = root / "candidate_bundle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _tree_sha256(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ReferenceObservationCurationInstallTests(unittest.TestCase):
    @staticmethod
    def _bundle_api():
        from physbench.reference_observations.curation import bundle

        return bundle

    @staticmethod
    def _install_api():
        from physbench.reference_observations.curation import install

        return install

    def test_candidate_rejects_trajectory_not_derived_from_masks(self) -> None:
        bundle = self._bundle_api()
        with tempfile.TemporaryDirectory() as temporary:
            candidate = _write_candidate(Path(temporary) / "candidate", corrupt_area=True)
            with self.assertRaisesRegex(ValueError, "area_pixels"):
                bundle.validate_candidate_bundle(candidate)

    def test_candidate_requires_exact_declared_target_closure(self) -> None:
        bundle = self._bundle_api()
        with tempfile.TemporaryDirectory() as temporary:
            candidate = _write_candidate(Path(temporary) / "candidate", missing_review=True)
            with self.assertRaisesRegex(ValueError, "required targets"):
                bundle.validate_candidate_bundle(candidate)

    def test_failed_install_leaves_every_canonical_byte_unchanged(self) -> None:
        install = self._install_api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            target = canonical / "reference_observation" / "review.json"
            target.parent.mkdir(parents=True)
            target.write_text('{"decision":"old"}\n', encoding="utf-8")
            before = _tree_sha256(canonical)
            candidate = _write_candidate(root / "candidate", missing_review=True)
            with self.assertRaises(ValueError):
                install.install_candidate_bundle(candidate, canonical)
            self.assertEqual(before, _tree_sha256(canonical))

    def test_successful_install_replaces_declared_files_without_lock_residue(self) -> None:
        install = self._install_api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            canonical.mkdir()
            candidate = _write_candidate(root / "candidate")
            install.install_candidate_bundle(candidate, canonical)
            self.assertTrue(
                (canonical / "reference_observation" / "review.json").is_file()
            )
            self.assertFalse(
                (root / ".canonical.reference-observation-curation.lock").exists()
            )

    def test_refresh_locked_files_changes_only_requested_records(self) -> None:
        install = self._install_api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = root / "datasets"
            changed = asset_root / "assets" / "case" / "changed.bin"
            untouched = asset_root / "assets" / "case" / "untouched.bin"
            changed.parent.mkdir(parents=True)
            changed.write_bytes(b"new")
            untouched.write_bytes(b"same")
            lock_path = asset_root / "assets.lock.json"
            lock = {
                "schema_version": "1.0",
                "files": [
                    {"path": "assets/case/changed.bin", "size_bytes": 3, "sha256": "0" * 64},
                    {"path": "assets/case/untouched.bin", "size_bytes": 4, "sha256": "1" * 64},
                ],
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            install.refresh_locked_files(
                lock_path,
                asset_root,
                ["assets/case/changed.bin"],
            )
            refreshed = json.loads(lock_path.read_text())
            self.assertEqual("1" * 64, refreshed["files"][1]["sha256"])
            self.assertEqual(_sha256(changed), refreshed["files"][0]["sha256"])


if __name__ == "__main__":
    unittest.main()
