from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "datasets"
    / "physics_video"
    / "releases"
    / "9.0.0"
    / "mask_storage.py"
)
UPGRADE_PATH = MODULE_PATH.with_name("upgrade_mask_storage.py")
GENERATOR_PATH = MODULE_PATH.with_name("generate_first_frame_masks.py")
BUILDER_PATH = MODULE_PATH.with_name("build_release.py")


def load_mask_storage():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"missing production module: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("mask_storage_v9", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import production module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upgrade_module():
    if not UPGRADE_PATH.is_file():
        raise AssertionError(f"missing production module: {UPGRADE_PATH}")
    spec = importlib.util.spec_from_file_location("upgrade_mask_storage_v9", UPGRADE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import production module: {UPGRADE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_first_frame_masks_v9",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import production module: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_mask_release_v9", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import production module: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BinaryMaskNormalizationTests(unittest.TestCase):
    def test_zero_one_and_zero_255_encode_the_same_binary_mask(self) -> None:
        storage = load_mask_storage()
        expected = np.array(
            [
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.uint8,
        )

        zero_one = storage.normalize_binary_mask(expected.copy())
        zero_255 = storage.normalize_binary_mask(expected * 255)

        np.testing.assert_array_equal(zero_one, expected)
        np.testing.assert_array_equal(zero_255, expected)
        self.assertEqual(np.uint8, zero_one.dtype)
        self.assertTrue(zero_one.flags.c_contiguous)

    def test_invalid_mask_shapes_values_and_dtype_are_rejected(self) -> None:
        storage = load_mask_storage()
        invalid = {
            "RGB": np.zeros((2, 3, 3), dtype=np.uint8),
            "empty": np.zeros((0, 3), dtype=np.uint8),
            "dtype": np.array([[0.0, 1.0]], dtype=np.float32),
            "values": np.array([[0, 2]], dtype=np.uint8),
            "no foreground": np.zeros((2, 3), dtype=np.uint8),
        }

        for label, mask in invalid.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    storage.normalize_binary_mask(mask)


class MaskBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = load_mask_storage()
        required = {
            "load_mask_npz",
            "upgrade_manifest_storage",
            "write_mask_bundle",
        }
        missing = sorted(name for name in required if not hasattr(self.storage, name))
        if missing:
            raise AssertionError(f"missing bundle API: {missing}")
        self.masks = [
            np.array(
                [
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ]
        base = "assets/collision_1d/case_a/canonical/masks"
        self.npz_asset = f"{base}/masks.npz"
        self.instances = [
            {
                "mask_id": "01",
                "object_id": "ball_1",
                "asset": f"{base}/01.png",
                "area_pixels": 3,
            },
            {
                "mask_id": "02",
                "object_id": "ball_2",
                "asset": f"{base}/02.png",
                "area_pixels": 3,
            },
        ]

    def test_bundle_is_model_readable_and_pngs_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "masks"

            storage_metadata = self.storage.write_mask_bundle(
                directory,
                self.masks,
                self.instances,
                self.npz_asset,
            )

            archive = self.storage.load_mask_npz(directory / "masks.npz")
            self.assertEqual(
                {"masks", "mask_ids", "object_ids", "frame_index"},
                set(archive),
            )
            self.assertEqual(np.uint8, archive["masks"].dtype)
            self.assertEqual((2, 4, 5), archive["masks"].shape)
            np.testing.assert_array_equal(archive["masks"], np.stack(self.masks))
            self.assertEqual("U", archive["mask_ids"].dtype.kind)
            self.assertEqual(["01", "02"], archive["mask_ids"].tolist())
            self.assertEqual("U", archive["object_ids"].dtype.kind)
            self.assertEqual(["ball_1", "ball_2"], archive["object_ids"].tolist())
            self.assertEqual(np.dtype(np.int64), archive["frame_index"].dtype)
            self.assertEqual((), archive["frame_index"].shape)
            self.assertEqual(0, int(archive["frame_index"]))

            for index, mask_id in enumerate(("01", "02")):
                png = cv2.imread(
                    str(directory / f"{mask_id}.png"),
                    cv2.IMREAD_UNCHANGED,
                )
                self.assertEqual({0, 255}, set(int(value) for value in np.unique(png)))
                np.testing.assert_array_equal(png > 0, archive["masks"][index] > 0)

            self.assertEqual(
                {
                    "model": {
                        "asset": self.npz_asset,
                        "array_key": "masks",
                        "layout": "OHW",
                        "dtype": "uint8",
                        "values": [0, 1],
                    },
                    "visualization": {
                        "asset_pattern": (
                            "assets/collision_1d/case_a/canonical/masks/"
                            "{mask_id}.png"
                        ),
                        "dtype": "uint8",
                        "values": [0, 255],
                    },
                },
                storage_metadata,
            )

    def test_bundle_rewrite_is_byte_stable_and_manifest_is_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "masks"
            metadata = self.storage.write_mask_bundle(
                directory,
                self.masks,
                self.instances,
                self.npz_asset,
            )
            first_bytes = (directory / "masks.npz").read_bytes()
            self.storage.write_mask_bundle(
                directory,
                [mask * 255 for mask in self.masks],
                self.instances,
                self.npz_asset,
            )
            self.assertEqual(first_bytes, (directory / "masks.npz").read_bytes())

            original = {
                "schema_version": "1.0",
                "case_id": "case_a",
                "dtype": "uint8",
                "values": [0, 1],
                "instances": json.loads(json.dumps(self.instances)),
            }
            upgraded = self.storage.upgrade_manifest_storage(
                original,
                self.npz_asset,
            )
            self.assertEqual("1.1", upgraded["schema_version"])
            self.assertNotIn("dtype", upgraded)
            self.assertNotIn("values", upgraded)
            self.assertEqual(metadata, upgraded["storage"])
            self.assertEqual([0, 1], [item["npz_index"] for item in upgraded["instances"]])
            self.assertNotIn("npz_index", original["instances"][0])

    def test_bundle_rejects_non_contiguous_ids_and_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "masks"
            invalid_instances = json.loads(json.dumps(self.instances))
            invalid_instances[1]["mask_id"] = "03"
            with self.assertRaises(ValueError):
                self.storage.write_mask_bundle(
                    directory,
                    self.masks,
                    invalid_instances,
                    self.npz_asset,
                )
            with self.assertRaises(ValueError):
                self.storage.write_mask_bundle(
                    directory,
                    [self.masks[0], np.ones((3, 5), dtype=np.uint8)],
                    self.instances,
                    self.npz_asset,
                )
            self.assertFalse(directory.exists())


class ReleaseUpgradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upgrade = load_upgrade_module()
        required = {"preflight_release", "upgrade_release"}
        missing = sorted(name for name in required if not hasattr(self.upgrade, name))
        if missing:
            raise AssertionError(f"missing release upgrade API: {missing}")

    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        physics_root = root / "physics_video"
        release_root = physics_root / "releases" / "9.0.0"
        release_root.mkdir(parents=True)
        case_relative = Path("assets/collision_1d/case_a/canonical")
        canonical = physics_root / case_relative
        masks_directory = canonical / "masks"
        masks_directory.mkdir(parents=True)
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(canonical / "first_frame.png"), frame))
        masks = [
            np.array(
                [
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ]
        instances = []
        for index, mask in enumerate(masks, start=1):
            mask_id = f"{index:02d}"
            asset = (case_relative / "masks" / f"{mask_id}.png").as_posix()
            self.assertTrue(cv2.imwrite(str(masks_directory / f"{mask_id}.png"), mask))
            ys, xs = np.where(mask > 0)
            instances.append(
                {
                    "mask_id": mask_id,
                    "object_id": f"ball_{index}",
                    "entity_class": "ball",
                    "physics_keys": [f"ball_{index}_mass"],
                    "asset": asset,
                    "area_pixels": int(np.count_nonzero(mask)),
                    "bbox_xyxy": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ],
                    "centroid_xy": [float(xs.mean()), float(ys.mean())],
                    "segmentation": {"sam_predicted_iou": 0.9},
                }
            )
        manifest_relative = case_relative / "masks" / "manifest.json"
        manifest = {
            "schema_version": "1.0",
            "case_id": "case_a",
            "scene_id": "collision_1d",
            "source_first_frame": (case_relative / "first_frame.png").as_posix(),
            "frame_index": 0,
            "frame_scope": "first_frame_only",
            "image_shape_hw": [4, 5],
            "dtype": "uint8",
            "values": [0, 1],
            "ordering": "row_major_top_to_bottom_then_left_to_right",
            "generator": {"id": "fixture"},
            "localization": {"localizer": "fixture"},
            "instances": instances,
        }
        (masks_directory / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        complete_case = {
            "case_id": "case_a",
            "scene_id": "collision_1d",
            "assets": {
                "first_frame": (case_relative / "first_frame.png").as_posix(),
                "first_frame_mask_manifest": manifest_relative.as_posix(),
                "first_frame_subject_mask_01": instances[0]["asset"],
                "first_frame_subject_mask_02": instances[1]["asset"],
            },
        }
        skipped_case = {
            "case_id": "case_skipped",
            "scene_id": "collision_1d",
            "assets": {
                "first_frame": "assets/collision_1d/case_skipped/canonical/first_frame.png",
                "first_frame_mask_manifest": None,
                "first_frame_subject_mask_01": None,
            },
        }
        records = [
            {
                "schema_version": "1.0",
                "case_id": "case_a",
                "scene_id": "collision_1d",
                "status": "complete",
                "frame_index": 0,
                "frame_scope": "first_frame_only",
                "source_first_frame": complete_case["assets"]["first_frame"],
                "ordering": "row_major_top_to_bottom_then_left_to_right",
                "expected_subject_count": 2,
                "manifest_asset": manifest_relative.as_posix(),
                "instances": instances,
            },
            {
                "schema_version": "1.0",
                "case_id": "case_skipped",
                "scene_id": "collision_1d",
                "status": "skipped",
                "frame_index": 0,
                "frame_scope": "first_frame_only",
                "source_first_frame": skipped_case["assets"]["first_frame"],
                "ordering": "row_major_top_to_bottom_then_left_to_right",
                "expected_subject_count": 1,
                "manifest_asset": None,
                "instances": [],
                "error": "fixture uncertainty",
            },
        ]
        (release_root / "cases.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in (complete_case, skipped_case)),
            encoding="utf-8",
        )
        (release_root / "masks.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in records),
            encoding="utf-8",
        )
        return physics_root, release_root, masks_directory

    def test_preflight_is_read_only_and_materialization_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            physics_root, release_root, masks_directory = self._write_fixture(
                Path(temporary)
            )
            original_png = (masks_directory / "01.png").read_bytes()

            operations = self.upgrade.preflight_release(release_root, physics_root)

            self.assertEqual(1, len(operations))
            self.assertFalse((masks_directory / "masks.npz").exists())
            self.assertEqual(original_png, (masks_directory / "01.png").read_bytes())
            report_path = release_root / "mask_storage_upgrade_report.json"
            report = self.upgrade.upgrade_release(
                release_root=release_root,
                physics_video_root=physics_root,
                materialize=True,
                report_path=report_path,
            )
            self.assertEqual(2, report["selected_cases"])
            self.assertEqual(1, report["complete_cases"])
            self.assertEqual(1, report["skipped_cases"])
            self.assertEqual(2, report["mask_files"])
            self.assertEqual(1, report["npz_files"])
            self.assertEqual(["case_skipped"], report["skipped_case_ids"])
            self.assertTrue(report_path.is_file())
            archive = np.load(masks_directory / "masks.npz", allow_pickle=False)
            try:
                self.assertEqual((2, 4, 5), archive["masks"].shape)
                self.assertEqual(["ball_1", "ball_2"], archive["object_ids"].tolist())
            finally:
                archive.close()
            png = cv2.imread(str(masks_directory / "01.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual({0, 255}, set(int(value) for value in np.unique(png)))
            manifest = json.loads((masks_directory / "manifest.json").read_text())
            self.assertEqual("1.1", manifest["schema_version"])
            self.assertEqual([0, 1], [item["npz_index"] for item in manifest["instances"]])
            first_archive = (masks_directory / "masks.npz").read_bytes()

            self.upgrade.upgrade_release(
                release_root=release_root,
                physics_video_root=physics_root,
                materialize=True,
                report_path=report_path,
            )

            self.assertEqual(first_archive, (masks_directory / "masks.npz").read_bytes())

    def test_bad_png_fails_before_any_bundle_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            physics_root, release_root, masks_directory = self._write_fixture(
                Path(temporary)
            )
            invalid = np.array(
                [
                    [0, 2, 2, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
            self.assertTrue(cv2.imwrite(str(masks_directory / "01.png"), invalid))
            second_before = (masks_directory / "02.png").read_bytes()

            with self.assertRaises(ValueError):
                self.upgrade.upgrade_release(
                    release_root=release_root,
                    physics_video_root=physics_root,
                    materialize=True,
                    report_path=None,
                )

            self.assertFalse((masks_directory / "masks.npz").exists())
            self.assertEqual(second_before, (masks_directory / "02.png").read_bytes())


class GeneratorIntegrationTests(unittest.TestCase):
    def test_generator_can_import_its_release_local_storage_dependency(self) -> None:
        try:
            generator = load_generator_module()
        except ModuleNotFoundError as exc:
            self.fail(f"generator release-local import failed: {exc}")

        self.assertEqual(
            "sam2.1_hiera_tiny_first_frame_physical_subject_v1",
            generator.GENERATOR_ID,
        )
        self.assertTrue(callable(generator.write_mask_bundle))


class ReleaseBuilderTests(unittest.TestCase):
    def test_builder_indexes_complete_npz_and_keeps_skipped_case_null(self) -> None:
        builder = load_builder_module()
        if not hasattr(builder, "build_mask_records"):
            self.fail("release builder lacks build_mask_records")
        base_cases = [
            {
                "case_id": "case_a",
                "scene_id": "collision_1d",
                "physics": {
                    "ball_1_mass": {"value": 1.0},
                    "ball_2_mass": {"value": 2.0},
                },
                "assets": {"first_frame": "assets/case_a/first_frame.png"},
            },
            {
                "case_id": "case_skipped",
                "scene_id": "collision_1d",
                "physics": {"ball_1_mass": {"value": 1.0}},
                "assets": {"first_frame": "assets/case_skipped/first_frame.png"},
            },
        ]
        instances = [
            {
                "mask_id": "01",
                "object_id": "ball_1",
                "asset": "assets/case_a/canonical/masks/01.png",
            },
            {
                "mask_id": "02",
                "object_id": "ball_2",
                "asset": "assets/case_a/canonical/masks/02.png",
            },
        ]
        report = {
            "selected_cases": 2,
            "results": [
                {
                    "case_id": "case_a",
                    "scene_id": "collision_1d",
                    "status": "complete",
                    "mask_directory": "assets/case_a/canonical/masks",
                    "instances": instances,
                },
                {
                    "case_id": "case_skipped",
                    "scene_id": "collision_1d",
                    "status": "skipped",
                    "error": "fixture uncertainty",
                },
            ],
        }
        original = json.loads(json.dumps(base_cases))

        cases, records, summary = builder.build_mask_records(base_cases, report)

        self.assertEqual(original, base_cases)
        complete_assets = cases[0]["assets"]
        self.assertEqual(
            "assets/case_a/canonical/masks/masks.npz",
            complete_assets["first_frame_masks_npz"],
        )
        self.assertEqual(
            "assets/case_a/canonical/masks/manifest.json",
            complete_assets["first_frame_mask_manifest"],
        )
        self.assertIsNone(cases[1]["assets"]["first_frame_masks_npz"])
        self.assertIsNone(cases[1]["assets"]["first_frame_mask_manifest"])
        self.assertEqual("1.1", records[0]["schema_version"])
        self.assertEqual(
            "assets/case_a/canonical/masks/masks.npz",
            records[0]["npz_asset"],
        )
        self.assertEqual([0, 1], [item["npz_index"] for item in records[0]["instances"]])
        self.assertEqual("1.1", records[1]["schema_version"])
        self.assertIsNone(records[1]["npz_asset"])
        self.assertEqual([], records[1]["instances"])
        self.assertEqual(
            {
                "completed": 1,
                "skipped": [
                    {
                        "case_id": "case_skipped",
                        "scene_id": "collision_1d",
                        "reason": "fixture uncertainty",
                    }
                ],
                "total_masks": 2,
                "total_npz": 1,
            },
            summary,
        )


if __name__ == "__main__":
    unittest.main()
