from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physbench.io import load_json, sha256_file, write_json


def observation_api():
    return importlib.import_module("physbench.reference_observations")


class ReferenceObservationTimelineTests(unittest.TestCase):
    def test_timeline_maps_24hz_physical_grid_to_source_frames(self) -> None:
        timeline = observation_api().build_timeline(
            frame_count=241,
            source_fps=240.0,
            encoded_to_physical_speed=1.0,
            sampling_rate_hz=24.0,
        )

        self.assertEqual(25, len(timeline.samples))
        self.assertEqual([0, 10, 20], [x.source_frame_index for x in timeline.samples[:3]])
        self.assertEqual(240, timeline.samples[-1].source_frame_index)
        self.assertAlmostEqual(1.0, timeline.samples[-1].physical_time_seconds)

    def test_timeline_applies_encoded_to_physical_speed(self) -> None:
        timeline = observation_api().build_timeline(
            frame_count=121,
            source_fps=120.0,
            encoded_to_physical_speed=0.5,
            sampling_rate_hz=20.0,
        )

        self.assertAlmostEqual(2.0, timeline.samples[-1].physical_time_seconds)
        self.assertEqual(120, timeline.samples[-1].source_frame_index)
        self.assertEqual(3, timeline.samples[1].source_frame_index)

    def test_timeline_rejects_non_positive_video_metadata(self) -> None:
        for mutation in (
            {"frame_count": 0},
            {"source_fps": 0.0},
            {"encoded_to_physical_speed": 0.0},
            {"sampling_rate_hz": 0.0},
        ):
            with self.subTest(mutation=mutation):
                values = {
                    "frame_count": 10,
                    "source_fps": 24.0,
                    "encoded_to_physical_speed": 1.0,
                    "sampling_rate_hz": 24.0,
                    **mutation,
                }
                with self.assertRaisesRegex(ValueError, "positive"):
                    observation_api().build_timeline(**values)


class ReferenceObservationPackedMaskTests(unittest.TestCase):
    def test_packed_mask_round_trip_is_bit_exact_for_unaligned_width(self) -> None:
        masks = np.zeros((3, 5, 11), dtype=np.uint8)
        masks[0, 1:3, 4:8] = 1
        masks[1, 0:5, 10] = 1

        packed = observation_api().pack_mask_tube(masks)
        restored = observation_api().unpack_mask_tube(packed, width=11)

        self.assertEqual((3, 5, 2), packed.shape)
        np.testing.assert_array_equal(masks, restored)

    def test_packed_mask_rejects_wrong_dtype_and_non_binary_values(self) -> None:
        invalid = (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.full((2, 3, 4), 2, dtype=np.uint8),
            np.zeros((3, 4), dtype=np.uint8),
        )
        for masks in invalid:
            with self.subTest(dtype=str(masks.dtype), shape=masks.shape):
                with self.assertRaises(ValueError):
                    observation_api().pack_mask_tube(masks)

    def test_entity_npz_round_trip_uses_pickle_free_arrays(self) -> None:
        api = observation_api()
        masks = np.zeros((2, 4, 9), dtype=np.uint8)
        masks[0, 1:3, 2:5] = 1
        masks[1, 1:3, 3:6] = 1
        entity = api.EntityObservation(
            object_id="object_1",
            mask_id="01",
            masks=masks,
            centroid_xy=np.asarray([[3.0, 1.5], [4.0, 1.5]], dtype=np.float32),
            bbox_xyxy=np.asarray([[2, 1, 4, 2], [3, 1, 5, 2]], dtype=np.float32),
            area_pixels=np.asarray([6, 6], dtype=np.int64),
            state=np.asarray([api.ObservationState.VISIBLE] * 2, dtype=np.uint8),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "entity"
            api.write_entity_observation(directory, entity)
            loaded = api.load_entity_observation(
                directory / "mask_tube.npz",
                directory / "trajectory.npz",
                object_id="object_1",
                mask_id="01",
                expected_samples=2,
            )
            with np.load(directory / "mask_tube.npz", allow_pickle=False) as payload:
                self.assertEqual(
                    {"packed_masks", "height", "width", "observation_index", "state"},
                    set(payload.files),
                )

        np.testing.assert_array_equal(masks, loaded.masks)
        np.testing.assert_array_equal(entity.centroid_xy, loaded.centroid_xy)


class ReferenceObservationManifestLoaderTests(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        *,
        staged: bool = False,
    ) -> tuple[Path, Path, Path]:
        api = observation_api()
        asset_root = root / "assets"
        bundle_root = root / "staging" if staged else asset_root
        bundle = (
            bundle_root
            / "case"
            / "canonical"
            / "reference_observation"
        )
        entity_dir = bundle / "entities" / "object_1"
        bundle.mkdir(parents=True)
        source_video = asset_root / "case" / "canonical" / "reference.mp4"
        source_video.parent.mkdir(parents=True, exist_ok=True)
        source_video.write_bytes(b"fixture-video")
        anchor = asset_root / "case" / "canonical" / "masks" / "manifest.json"
        anchor.parent.mkdir()
        write_json(anchor, {"case_id": "case_1"})

        timeline = api.build_timeline(
            frame_count=2,
            source_fps=1.0,
            encoded_to_physical_speed=1.0,
            sampling_rate_hz=1.0,
            source_width=9,
            source_height=4,
        )
        api.write_timeline(bundle / "timeline.json", "case_1", timeline)
        quality = {
            "schema_version": "1.0",
            "case_id": "case_1",
            "status": "pass",
            "config_fingerprint": "a" * 64,
            "checks": [],
            "entities": {},
            "failures": [],
            "warnings": [],
        }
        write_json(bundle / "quality.json", quality)
        masks = np.zeros((2, 4, 9), dtype=np.uint8)
        masks[:, 1:3, 2:5] = 1
        entity = api.EntityObservation(
            object_id="object_1",
            mask_id="01",
            masks=masks,
            centroid_xy=np.asarray([[3.0, 1.5], [3.0, 1.5]], dtype=np.float32),
            bbox_xyxy=np.asarray([[2, 1, 4, 2], [2, 1, 4, 2]], dtype=np.float32),
            area_pixels=np.asarray([6, 6], dtype=np.int64),
            state=np.asarray([api.ObservationState.VISIBLE] * 2, dtype=np.uint8),
        )
        api.write_entity_observation(entity_dir, entity)

        def file_record(path: Path, *, base: Path) -> dict:
            return {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": "1.0",
            "case_id": "case_1",
            "scene_id": "pendulum",
            "source": {
                "reference_video": file_record(source_video, base=asset_root),
                "first_frame_mask_manifest": file_record(anchor, base=asset_root),
            },
            "generator": {
                "id": "fixture_generator",
                "code_revision": "fixture",
                "model_id": "fixture_model",
                "config_fingerprint": "b" * 64,
            },
            "timeline": file_record(bundle / "timeline.json", base=bundle),
            "quality": file_record(bundle / "quality.json", base=bundle),
            "entities": [
                {
                    "object_id": "object_1",
                    "mask_id": "01",
                    "mask_tube": file_record(
                        entity_dir / "mask_tube.npz",
                        base=bundle,
                    ),
                    "trajectory": file_record(
                        entity_dir / "trajectory.npz",
                        base=bundle,
                    ),
                }
            ],
        }
        manifest_path = bundle / "manifest.json"
        write_json(manifest_path, manifest)
        return asset_root, bundle_root, manifest_path

    @staticmethod
    def _refresh_record(manifest_path: Path, role: str, path: Path) -> None:
        manifest = load_json(manifest_path)
        if role.startswith("source."):
            record = manifest["source"][role.split(".", 1)[1]]
        elif role.startswith("entity."):
            record = manifest["entities"][0][role.split(".", 1)[1]]
        else:
            record = manifest[role]
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
        write_json(manifest_path, manifest)

    def test_manifest_loader_verifies_and_materializes_all_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = self._bundle(Path(temporary))
            loaded = observation_api().load_reference_observation(
                asset_root,
                manifest_path,
            )

        self.assertEqual("case_1", loaded.case_id)
        self.assertEqual("pendulum", loaded.scene_id)
        self.assertEqual(2, len(loaded.timeline.samples))
        self.assertEqual({"object_1"}, set(loaded.entities))

    def test_manifest_loader_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = self._bundle(Path(temporary))
            timeline = manifest_path.parent / "timeline.json"
            original = timeline.read_bytes()
            timeline.write_bytes(original.replace(b"case_1", b"case_2", 1))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                observation_api().load_reference_observation(
                    asset_root,
                    manifest_path,
                )

    def test_manifest_loader_rejects_symlinked_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root, _, manifest_path = self._bundle(root)
            timeline = manifest_path.parent / "timeline.json"
            external = root / "external.json"
            external.write_bytes(timeline.read_bytes())
            timeline.unlink()
            timeline.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symlink"):
                observation_api().load_reference_observation(
                    asset_root,
                    manifest_path,
                )

    def test_manifest_loader_rejects_symlinked_container(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root, _, manifest_path = self._bundle(root)
            entities = manifest_path.parent / "entities"
            relocated = root / "relocated_entities"
            entities.rename(relocated)
            entities.symlink_to(relocated, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                observation_api().load_reference_observation(
                    asset_root,
                    manifest_path,
                )

    def test_manifest_loader_rejects_anchor_bound_to_another_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = self._bundle(Path(temporary))
            anchor = asset_root / "case" / "canonical" / "masks" / "manifest.json"
            write_json(anchor, {"case_id": "case_2"})
            self._refresh_record(
                manifest_path,
                "source.first_frame_mask_manifest",
                anchor,
            )
            with self.assertRaisesRegex(ValueError, "mask manifest Case mismatch"):
                observation_api().load_reference_observation(
                    asset_root,
                    manifest_path,
                )

    def test_entity_loader_rejects_non_int64_observation_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            asset_root, _, manifest_path = self._bundle(Path(temporary))
            mask_path = (
                manifest_path.parent
                / "entities"
                / "object_1"
                / "mask_tube.npz"
            )
            with np.load(mask_path, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in payload.files}
            arrays["observation_index"] = arrays["observation_index"].astype(
                np.float64
            )
            with mask_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            self._refresh_record(manifest_path, "entity.mask_tube", mask_path)
            with self.assertRaisesRegex(ValueError, "indices.*int64"):
                observation_api().load_reference_observation(
                    asset_root,
                    manifest_path,
                )

    def test_staged_bundle_uses_portable_manifest_relative_child_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root, staging_root, manifest_path = self._bundle(
                root,
                staged=True,
            )
            loaded = observation_api().load_reference_observation(
                asset_root,
                manifest_path,
                bundle_root=staging_root,
            )

        self.assertEqual("case_1", loaded.case_id)
        self.assertEqual({"object_1"}, set(loaded.entities))


if __name__ == "__main__":
    unittest.main()
