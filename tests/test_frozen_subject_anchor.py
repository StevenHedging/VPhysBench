from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physbench.evaluation.common.errors import ReferenceAnalysisError
from physbench.evaluation.common.frozen_subject import (
    load_frozen_subject_anchor,
    transform_frozen_subject_mask,
)
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.io import write_json


class FrozenSubjectAnchorTests(unittest.TestCase):
    @staticmethod
    def _transform() -> dict[str, object]:
        return {
            "policy": "reference_content_crop_resize_no_pad",
            "crop_xywh": [2, 1, 4, 4],
            "scale": 2.0,
            "source_size": [8, 6],
            "target_size": [8, 8],
            "padding": None,
        }

    def _fixture(
        self,
        root: Path,
    ) -> tuple[CaseEvaluationRequest, Path, Path]:
        manifest_relative = "case/canonical/masks/manifest.json"
        npz_relative = "case/canonical/masks/01.npz"
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 1
        npz_path = root / npz_relative
        npz_path.parent.mkdir(parents=True)
        np.savez_compressed(
            npz_path,
            masks=mask[None, ...],
            mask_ids=np.asarray(["01"]),
            object_ids=np.asarray(["projectile_ball"]),
            frame_index=np.asarray(0, dtype=np.int64),
        )
        manifest_path = root / manifest_relative
        write_json(
            manifest_path,
            {
                "schema_version": "1.2",
                "case_id": "parabolic_case",
                "scene_id": "parabolic_motion",
                "frame_index": 0,
                "frame_scope": "first_frame_only",
                "source_first_frame": "case/canonical/first_frame.png",
                "image_shape_hw": [6, 8],
                "instances": [
                    {
                        "mask_id": "01",
                        "object_id": "object_1",
                        "entity_class": "ball",
                        "npz_asset": npz_relative,
                        "area_pixels": 4,
                        "bbox_xyxy": [3, 2, 5, 4],
                        "centroid_xy": [3.5, 2.5],
                    }
                ],
                "storage": {
                    "model": {
                        "array_key": "masks",
                        "layout": "1HW",
                        "dtype": "uint8",
                        "values": [0, 1],
                    }
                },
            },
        )
        case = {
            "case_id": "parabolic_case",
            "scene_id": "parabolic_motion",
            "assets": {
                "first_frame": "case/canonical/first_frame.png",
                "first_frame_mask_manifest": manifest_relative,
            },
        }
        request = CaseEvaluationRequest(
            job={"job_id": "anchor_test"},
            case=case,
            case_catalog={case["case_id"]: case},
            prediction={"status": "complete", "video_path": "mock.mp4"},
            asset_root=root,
            artifact_dir=root / "artifacts",
            evaluator_config={"type": "fixture"},
        )
        return request, manifest_path, npz_path

    def test_loads_logical_entity_from_distinct_dataset_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, manifest_path, npz_path = self._fixture(Path(temporary))

            anchor = load_frozen_subject_anchor(
                request,
                logical_entity_id="projectile_ball",
                entity_class="ball",
                spatial_transform=self._transform(),
                error_namespace="reference_projectile_subject",
            )

        self.assertEqual("parabolic_case", anchor.case_id)
        self.assertEqual("projectile_ball", anchor.logical_entity_id)
        self.assertEqual("object_1", anchor.dataset_object_id)
        self.assertEqual("ball", anchor.entity_class)
        self.assertEqual((8, 8), anchor.mask.shape)
        self.assertEqual(16.0, anchor.area_px2)
        np.testing.assert_allclose([3.5, 3.5], anchor.centroid_xy)
        self.assertFalse(anchor.mask.flags.writeable)
        self.assertFalse(anchor.source_mask.flags.writeable)
        self.assertEqual(str(manifest_path), anchor.provenance["manifest"])
        self.assertEqual(str(npz_path), anchor.provenance["npz"])
        self.assertEqual(64, len(anchor.provenance["manifest_sha256"]))
        self.assertEqual(64, len(anchor.provenance["npz_sha256"]))

    def test_loads_npz_bound_to_selected_dataset_object_identity(self) -> None:
        """Would fail if a reviewed Dataset object id cannot bind the logical role."""
        with tempfile.TemporaryDirectory() as temporary:
            request, _, npz_path = self._fixture(Path(temporary))
            with np.load(npz_path, allow_pickle=False) as payload:
                masks = np.array(payload["masks"], copy=True)
                mask_ids = np.array(payload["mask_ids"], copy=True)
                frame_index = np.array(payload["frame_index"], copy=True)
            np.savez_compressed(
                npz_path,
                masks=masks,
                mask_ids=mask_ids,
                object_ids=np.asarray(["object_1"]),
                frame_index=frame_index,
            )

            anchor = load_frozen_subject_anchor(
                request,
                logical_entity_id="projectile_ball",
                entity_class="ball",
                spatial_transform=self._transform(),
                dataset_object_id="object_1",
                error_namespace="reference_projectile_subject",
            )

        self.assertEqual("projectile_ball", anchor.logical_entity_id)
        self.assertEqual("object_1", anchor.dataset_object_id)
        self.assertEqual("object_1", anchor.provenance["npz_object_id"])

    def test_loads_reviewed_inclusive_maximum_bbox(self) -> None:
        """Would fail if a mask-exact legacy bbox convention is rejected."""
        with tempfile.TemporaryDirectory() as temporary:
            request, manifest_path, _ = self._fixture(Path(temporary))
            manifest = __import__("json").loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["instances"][0]["bbox_xyxy"] = [3, 2, 4, 3]
            write_json(manifest_path, manifest)

            anchor = load_frozen_subject_anchor(
                request,
                logical_entity_id="projectile_ball",
                entity_class="ball",
                spatial_transform=self._transform(),
                error_namespace="reference_projectile_subject",
            )

        self.assertEqual(
            "xyxy_inclusive_max_legacy",
            anchor.provenance["bbox_policy"],
        )

    def test_rejects_ambiguous_entity_class_without_object_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, manifest_path, _ = self._fixture(Path(temporary))
            manifest = __import__("json").loads(
                manifest_path.read_text(encoding="utf-8")
            )
            second = copy.deepcopy(manifest["instances"][0])
            second["mask_id"] = "02"
            second["object_id"] = "object_2"
            manifest["instances"].append(second)
            write_json(manifest_path, manifest)

            with self.assertRaises(ReferenceAnalysisError) as captured:
                load_frozen_subject_anchor(
                    request,
                    logical_entity_id="projectile_ball",
                    entity_class="ball",
                    spatial_transform=self._transform(),
                    error_namespace="reference_projectile_subject",
                )

        self.assertEqual(
            "reference_projectile_subject_manifest_invalid",
            captured.exception.code,
        )

    def test_rejects_npz_object_identity_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, _, npz_path = self._fixture(Path(temporary))
            with np.load(npz_path, allow_pickle=False) as payload:
                mask = np.array(payload["masks"], copy=True)
            np.savez_compressed(
                npz_path,
                masks=mask,
                mask_ids=np.asarray(["01"]),
                object_ids=np.asarray(["wrong_object"]),
                frame_index=np.asarray(0, dtype=np.int64),
            )

            with self.assertRaises(ReferenceAnalysisError) as captured:
                load_frozen_subject_anchor(
                    request,
                    logical_entity_id="projectile_ball",
                    entity_class="ball",
                    spatial_transform=self._transform(),
                    error_namespace="reference_projectile_subject",
                )

        self.assertEqual(
            "reference_projectile_subject_mask_invalid",
            captured.exception.code,
        )

    def test_rejects_manifest_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, _, _ = self._fixture(Path(temporary))
            request.case["assets"]["first_frame_mask_manifest"] = (
                "../manifest.json"
            )

            with self.assertRaises(ReferenceAnalysisError) as captured:
                load_frozen_subject_anchor(
                    request,
                    logical_entity_id="projectile_ball",
                    entity_class="ball",
                    spatial_transform=self._transform(),
                    error_namespace="reference_projectile_subject",
                )

        self.assertEqual(
            "reference_projectile_subject_manifest_path_escape",
            captured.exception.code,
        )

    def test_transform_rejects_source_size_mismatch(self) -> None:
        mask = np.ones((3, 4), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "source_size"):
            transform_frozen_subject_mask(mask, self._transform())


if __name__ == "__main__":
    unittest.main()
