from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physbench.io import canonical_sha256


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
            lock["files_digest"] = canonical_sha256(lock["files"])
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            install.refresh_locked_files(
                lock_path,
                asset_root,
                ["assets/case/changed.bin"],
            )
            refreshed = json.loads(lock_path.read_text())
            self.assertEqual("1" * 64, refreshed["files"][1]["sha256"])
            self.assertEqual(_sha256(changed), refreshed["files"][0]["sha256"])
            self.assertEqual(
                canonical_sha256(refreshed["files"]), refreshed["files_digest"]
            )

    def test_finalize_derives_anchor_and_trajectory_from_tube_pixels(self) -> None:
        from physbench.reference_observations.curation import finalize

        masks = np.zeros((2, 8, 8), np.uint8)
        masks[0, 2:4, 3:6] = 1
        masks[1, 3:5, 4:7] = 1
        entity = finalize.entity_from_masks(
            "object_1", "01", masks, np.zeros(2, np.uint8)
        )
        self.assertEqual([6, 6], entity.area_pixels.tolist())
        self.assertEqual([4.0, 2.5], entity.centroid_xy[0].tolist())
        np.testing.assert_array_equal(entity.masks[0], masks[0])

    def test_finalize_remaps_physics_objects_and_symbol_suffixes(self) -> None:
        from physbench.reference_observations.curation import finalize

        physics = {
            "environment": {},
            "objects": {
                "object_1": {"radius": {"symbol": "r_1", "value": 0.02}},
                "object_2": {"radius": {"symbol": "r_2", "value": 0.01}},
            },
        }
        remapped = finalize.remap_physics_objects(
            physics, new_to_old={"object_1": "object_2", "object_2": "object_1"}
        )
        self.assertEqual(0.01, remapped["objects"]["object_1"]["radius"]["value"])
        self.assertEqual("r_1", remapped["objects"]["object_1"]["radius"]["symbol"])
        self.assertEqual("r_2", remapped["objects"]["object_2"]["radius"]["symbol"])

    def test_finalize_detects_whether_visual_size_order_already_matches_physics(self) -> None:
        from physbench.reference_observations.curation import finalize

        physics = {
            "environment": {},
            "objects": {
                "object_1": {"radius": {"symbol": "r_1", "value": 0.01}},
                "object_2": {"radius": {"symbol": "r_2", "value": 0.02}},
            },
        }
        self.assertTrue(
            finalize.visual_size_order_matches_physics(
                physics, {"object_1": 300, "object_2": 1200}
            )
        )
        self.assertFalse(
            finalize.visual_size_order_matches_physics(
                physics, {"object_1": 1200, "object_2": 300}
            )
        )

    def test_finalize_compares_circular_orbit_radius_when_object_radius_is_absent(self) -> None:
        from physbench.reference_observations.curation import finalize

        physics = {
            "environment": {},
            "objects": {
                "object_1": {"orbit_radius": {"symbol": "r_2", "value": 0.08}},
                "object_2": {"orbit_radius": {"symbol": "r_1", "value": 0.02}},
            },
        }
        self.assertFalse(
            finalize.visual_size_order_matches_physics(
                physics, {"object_1": 400, "object_2": 1600}
            )
        )

    def test_finalize_permutes_entity_arrays_and_rebinds_identity(self) -> None:
        from physbench.reference_observations import EntityObservation
        from physbench.reference_observations.curation import finalize

        first = EntityObservation(
            object_id="object_1",
            mask_id="01",
            masks=np.ones((1, 2, 2), np.uint8),
            centroid_xy=np.asarray([[0.5, 0.5]], np.float32),
            bbox_xyxy=np.asarray([[0, 0, 1, 1]], np.float32),
            area_pixels=np.asarray([4], np.int64),
            state=np.asarray([0], np.uint8),
        )
        second = EntityObservation(
            object_id="object_2",
            mask_id="02",
            masks=np.zeros((1, 2, 2), np.uint8),
            centroid_xy=np.asarray([[1.0, 1.0]], np.float32),
            bbox_xyxy=np.asarray([[1, 1, 1, 1]], np.float32),
            area_pixels=np.asarray([0], np.int64),
            state=np.asarray([3], np.uint8),
        )
        result = finalize.permute_entity_observations(
            {"object_1": first, "object_2": second},
            new_to_old={"object_1": "object_2", "object_2": "object_1"},
        )
        self.assertEqual("object_1", result["object_1"].object_id)
        self.assertEqual("01", result["object_1"].mask_id)
        self.assertEqual(0, int(result["object_1"].masks.sum()))
        self.assertEqual(4, int(result["object_2"].masks.sum()))

    def test_finalize_writes_anchor_from_observation_zero(self) -> None:
        from physbench.reference_observations.curation import finalize

        masks = np.zeros((2, 8, 8), np.uint8)
        masks[:, 2:4, 3:6] = 1
        entity = finalize.entity_from_masks(
            "object_1", "01", masks, np.zeros(2, np.uint8)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = finalize.write_anchor_assets(
                root,
                entity,
                semantic_object_id="ball_small",
            )
            with np.load(root / "01.npz", allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["masks"][0], masks[0])
                self.assertEqual("ball_small", str(payload["object_ids"][0]))
            self.assertEqual(6, record["area_pixels"])
            self.assertTrue((root / "01.png").is_file())

    def test_finalize_uses_the_frozen_scene_anchor_id_contract(self) -> None:
        from physbench.reference_observations.curation import finalize

        expected = {
            ("collision_1d", 3): ("ball_1", "ball_2", "ball_3"),
            ("inclined_plane_slide", 1): ("sliding_block",),
            ("parabolic_motion", 1): ("projectile_ball",),
            ("pendulum", 1): ("bob",),
            ("push_bottle", 1): ("bottle",),
            ("uniform_circular_motion", 2): ("object_1", "object_2"),
            ("vertical_spring_oscillator", 1): ("object_1",),
        }
        for (scene_id, object_count), object_ids in expected.items():
            with self.subTest(scene_id=scene_id):
                self.assertEqual(
                    object_ids,
                    finalize.anchor_object_ids(scene_id, object_count),
                )

        with self.assertRaisesRegex(ValueError, "unsupported scene"):
            finalize.anchor_object_ids("unknown_scene", 1)
        with self.assertRaisesRegex(ValueError, "object count"):
            finalize.anchor_object_ids("pendulum", 2)

    def test_finalize_rewrites_only_anchor_identity_and_is_idempotent(self) -> None:
        from physbench.reference_observations.curation import finalize

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "01.npz"
            masks = np.zeros((1, 4, 5), np.uint8)
            masks[:, 1:3, 2:4] = 1
            np.savez_compressed(
                path,
                masks=masks,
                mask_ids=np.asarray(["01"]),
                object_ids=np.asarray(["appearance_label"]),
                frame_index=np.asarray(0, np.int64),
            )
            self.assertTrue(finalize.rewrite_anchor_object_id(path, "bob"))
            rewritten = path.read_bytes()
            with np.load(path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["masks"], masks)
                self.assertEqual(["01"], payload["mask_ids"].tolist())
                self.assertEqual(["bob"], payload["object_ids"].tolist())
                self.assertEqual(0, int(payload["frame_index"]))
            self.assertFalse(finalize.rewrite_anchor_object_id(path, "bob"))
            self.assertEqual(rewritten, path.read_bytes())

    def test_finalize_remaps_ordered_appearance_and_striker_index(self) -> None:
        from physbench.reference_observations.curation import finalize

        appearance = {
            "ball_sequence": ["large", "small"],
            "ball_materials": ["steel", "glass"],
            "striker_ball_index": 2,
        }
        remapped = finalize.remap_ordered_appearance(
            appearance, new_to_old={"object_1": "object_2", "object_2": "object_1"}
        )
        self.assertEqual(["small", "large"], remapped["ball_sequence"])
        self.assertEqual(["glass", "steel"], remapped["ball_materials"])
        self.assertEqual(1, remapped["striker_ball_index"])


if __name__ == "__main__":
    unittest.main()
