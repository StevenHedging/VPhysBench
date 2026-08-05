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


def load_mask_storage():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"missing production module: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("mask_storage_v9", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import production module: {MODULE_PATH}")
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


if __name__ == "__main__":
    unittest.main()
