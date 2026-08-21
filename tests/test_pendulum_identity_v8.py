from __future__ import annotations

import copy
from contextlib import ExitStack
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.entities import (
    EntityMatch,
    EntitySpec,
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
    ReferenceCapability,
)
from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.errors import ReferenceAnalysisError
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.scenes.pendulum.v8_identity import (
    PendulumSubjectAnchor,
    SubjectIdentityDecision,
    SubjectIdentityState,
    decide_pendulum_identity_v8,
    identity_decision_to_frozen_assignment,
    load_pendulum_subject_anchor,
    transform_evaluator_mask,
)
from physbench.evaluation.scenes.pendulum.v8_evaluator import (
    PendulumOpenWorldCaseEvaluatorV8,
    _write_subject_identity_artifacts,
)
from physbench.evaluation.scenes.pendulum.v8_open_world import (
    AnnotatedConditionDecision,
    infer_anchor_guided_pivot_v8,
    select_annotated_condition_structure_v8,
)
from physbench.evaluation.scenes.pendulum.open_world import (
    PendulumStructureSpec,
    letterbox_condition_image,
)
from physbench.evaluation.scenes.pendulum.segmentation import (
    MotionPrompt,
    Sam2PendulumSegmenter,
)
from physbench.evaluation.scenes.pendulum.v7_open_world import (
    ConditionStructureDecision,
    _residual_detections,
    detect_condition_structure_v7,
)
from physbench.io import write_json


class PendulumIdentityV8APITests(unittest.TestCase):
    def test_segmenter_detaches_read_only_backend_masks_before_cropping(self) -> None:
        source = np.ones((6, 5), dtype=np.uint8)
        source.setflags(write=False)

        class Backend:
            def segment(self, frames, *, prompt, temporary_prefix):
                return [source], {"backend": "fake"}

        segmenter = object.__new__(Sam2PendulumSegmenter)
        segmenter._backend = Backend()
        prompt = MotionPrompt(
            frame_index=0,
            box_xyxy=np.asarray([0, 0, 4, 5], dtype=np.float32),
            motion_box_xyxy=np.asarray([0, 2, 4, 5], dtype=np.float32),
            points_xy=np.asarray([[2, 3]], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            proposal_score=1.0,
        )

        masks, _, _ = segmenter.segment(
            [np.zeros((6, 5, 3), dtype=np.uint8)],
            proposal_config={},
            prompt=prompt,
        )

        self.assertTrue(masks[0].flags.writeable)
        self.assertEqual(0, int(masks[0][:2].sum()))
        self.assertEqual(20, int(masks[0][2:].sum()))
        self.assertEqual(30, int(source.sum()))

    def test_identity_module_exposes_anchor_api(self) -> None:
        try:
            module = importlib.import_module(
                "physbench.evaluation.scenes.pendulum.v8_identity"
            )
        except ModuleNotFoundError:
            module = None

        self.assertIsNotNone(module)
        assert module is not None
        self.assertTrue(hasattr(module, "PendulumSubjectAnchor"))
        self.assertTrue(hasattr(module, "transform_evaluator_mask"))
        self.assertTrue(hasattr(module, "load_pendulum_subject_anchor"))


class PendulumSubjectAnchorTests(unittest.TestCase):
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

    def _write_fixture(self, root: Path) -> CaseEvaluationRequest:
        first_frame = "case/canonical/first_frame.png"
        manifest_relative = "case/canonical/masks/manifest.json"
        npz_relative = "case/canonical/masks/01.npz"
        png_relative = "case/canonical/masks/01.png"
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 1
        npz_path = root / npz_relative
        npz_path.parent.mkdir(parents=True)
        np.savez_compressed(
            npz_path,
            masks=mask[None, ...],
            mask_ids=np.asarray(["01"]),
            object_ids=np.asarray(["bob"]),
            frame_index=np.asarray(0, dtype=np.int64),
        )
        write_json(
            root / manifest_relative,
            {
                "schema_version": "1.2",
                "case_id": "pendulum_case",
                "scene_id": "pendulum",
                "frame_index": 0,
                "frame_scope": "first_frame_only",
                "source_first_frame": first_frame,
                "image_shape_hw": [6, 8],
                "ordering": "fixture",
                "generator": {"id": "fixture"},
                "localization": {"localizer": "fixture"},
                "instances": [
                    {
                        "mask_id": "01",
                        "object_id": "object_1",
                        "entity_class": "pendulum_bob",
                        "physics_keys": ["objects.object_1.radius"],
                        "asset": png_relative,
                        "npz_asset": npz_relative,
                        "area_pixels": 4,
                        "bbox_xyxy": [3, 2, 5, 4],
                        "centroid_xy": [3.5, 2.5],
                        "segmentation": {"score": 1.0},
                    }
                ],
                "storage": {
                    "model": {
                        "array_key": "masks",
                        "asset_pattern": "case/canonical/masks/{mask_id}.npz",
                        "dtype": "uint8",
                        "layout": "1HW",
                        "values": [0, 1],
                    },
                    "visualization": {
                        "asset_pattern": "case/canonical/masks/{mask_id}.png",
                        "dtype": "uint8",
                        "values": [0, 255],
                    },
                },
            },
        )
        return CaseEvaluationRequest(
            job={"job_id": "pendulum_anchor_fixture"},
            case={
                "case_id": "pendulum_case",
                "scene_id": "pendulum",
                "assets": {
                    "first_frame": first_frame,
                    "first_frame_mask_manifest": manifest_relative,
                },
            },
            case_catalog={},
            prediction=None,
            asset_root=root,
            artifact_dir=root / "artifacts",
            evaluator_config={},
        )

    def test_crop_resize_maps_binary_mask_to_evaluation_canvas(self) -> None:
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 1

        transformed = transform_evaluator_mask(
            mask,
            {
                "policy": "reference_content_crop_resize_no_pad",
                "crop_xywh": [2, 1, 4, 4],
                "scale": 2.0,
                "source_size": [8, 6],
                "target_size": [8, 8],
                "padding": None,
            },
        )

        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[2:6, 2:6] = 255
        np.testing.assert_array_equal(expected, transformed)
        self.assertEqual(np.uint8, transformed.dtype)

    def test_letterbox_maps_mask_with_declared_offset(self) -> None:
        mask = np.zeros((2, 4), dtype=np.uint8)
        mask[0, 0] = 1

        transformed = transform_evaluator_mask(
            mask,
            {
                "policy": "preserve_aspect_ratio_letterbox",
                "scale": 2.0,
                "offset_xy": [0, 2],
                "source_size": [4, 2],
                "target_size": [8, 8],
            },
        )

        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[2:4, 0:2] = 255
        np.testing.assert_array_equal(expected, transformed)

    def test_valid_manifest_and_npz_load_frozen_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = self._write_fixture(Path(temporary))

            anchor = load_pendulum_subject_anchor(
                request,
                entity_id="bob",
                spatial_transform=self._transform(),
            )

            self.assertEqual("pendulum_case", anchor.case_id)
            self.assertEqual("bob", anchor.entity_id)
            self.assertEqual("pendulum_bob", anchor.entity_class)
            self.assertEqual(16.0, anchor.area_px2)
            np.testing.assert_allclose([3.5, 3.5], anchor.centroid_xy)
            self.assertFalse(anchor.mask.flags.writeable)
            self.assertFalse(anchor.source_mask.flags.writeable)
            self.assertRegex(
                str(anchor.provenance["manifest_sha256"]),
                "^[0-9a-f]{64}$",
            )
            self.assertRegex(
                str(anchor.provenance["npz_sha256"]),
                "^[0-9a-f]{64}$",
            )

    def test_npz_path_escape_fails_with_reference_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._write_fixture(root)
            manifest_path = root / request.case["assets"][
                "first_frame_mask_manifest"
            ]
            value = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            value["instances"][0]["npz_asset"] = "../outside.npz"
            write_json(manifest_path, value)

            with self.assertRaises(ReferenceAnalysisError) as raised:
                load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self._transform(),
                )

            self.assertEqual(
                "reference_pendulum_subject_mask_path_escape_v8",
                raised.exception.code,
            )

    def test_manifest_path_escape_fails_with_reference_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = self._write_fixture(Path(temporary))
            request.case["assets"]["first_frame_mask_manifest"] = (
                "../manifest.json"
            )

            with self.assertRaises(ReferenceAnalysisError) as raised:
                load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self._transform(),
                )

            self.assertEqual(
                "reference_pendulum_subject_mask_manifest_path_escape_v8",
                raised.exception.code,
            )

    def test_missing_manifest_role_uses_reference_missing_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = self._write_fixture(Path(temporary))
            request.case["assets"].pop("first_frame_mask_manifest")

            with self.assertRaises(ReferenceAnalysisError) as raised:
                load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self._transform(),
                )

            self.assertEqual(
                "reference_pendulum_subject_mask_manifest_missing_v8",
                raised.exception.code,
            )

    def test_letterbox_rejects_scale_or_offset_not_declared_by_geometry(
        self,
    ) -> None:
        mask = np.zeros((2, 4), dtype=np.uint8)
        mask[0, 0] = 1
        invalid_transforms = (
            {
                "policy": "preserve_aspect_ratio_letterbox",
                "scale": 1.0,
                "offset_xy": [0, 2],
                "source_size": [4, 2],
                "target_size": [8, 8],
            },
            {
                "policy": "preserve_aspect_ratio_letterbox",
                "scale": 2.0,
                "offset_xy": [1, 2],
                "source_size": [4, 2],
                "target_size": [8, 8],
            },
        )

        for transform in invalid_transforms:
            with self.subTest(transform=transform):
                with self.assertRaises(ValueError):
                    transform_evaluator_mask(mask, transform)

    def test_manifest_identity_and_geometry_mismatch_fail_closed(self) -> None:
        mutations = (
            ("schema", lambda value: value.update(schema_version="1.1")),
            ("case", lambda value: value.update(case_id="other_case")),
            ("scene", lambda value: value.update(scene_id="collision_1d")),
            ("frame", lambda value: value.update(frame_index=1)),
            ("scope", lambda value: value.update(frame_scope="all_frames")),
            (
                "source",
                lambda value: value.update(
                    source_first_frame="case/other.png"
                ),
            ),
            (
                "multiple_bobs",
                lambda value: value["instances"].append(
                    copy.deepcopy(value["instances"][0])
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request = self._write_fixture(root)
                path = root / request.case["assets"][
                    "first_frame_mask_manifest"
                ]
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                write_json(path, value)

                with self.assertRaises(ReferenceAnalysisError) as raised:
                    load_pendulum_subject_anchor(
                        request,
                        entity_id="bob",
                        spatial_transform=self._transform(),
                    )

                self.assertEqual(
                    "reference_pendulum_subject_mask_manifest_invalid_v8",
                    raised.exception.code,
                )

    def test_npz_contract_and_manifest_geometry_mismatch_fail_closed(
        self,
    ) -> None:
        variants = (
            ("wrong_mask_id", {"mask_ids": np.asarray(["02"])}),
            ("wrong_entity", {"object_ids": np.asarray(["object_1"])}),
            (
                "wrong_frame_dtype",
                {"frame_index": np.asarray(0, dtype=np.int32)},
            ),
            (
                "nonbinary",
                {
                    "masks": np.asarray(
                        [[[0, 2], [0, 0]]],
                        dtype=np.uint8,
                    )
                },
            ),
            (
                "empty",
                {"masks": np.zeros((1, 6, 8), dtype=np.uint8)},
            ),
        )
        for label, overrides in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request = self._write_fixture(root)
                path = root / "case/canonical/masks/01.npz"
                mask = np.zeros((6, 8), dtype=np.uint8)
                mask[2:4, 3:5] = 1
                values = {
                    "masks": mask[None, ...],
                    "mask_ids": np.asarray(["01"]),
                    "object_ids": np.asarray(["bob"]),
                    "frame_index": np.asarray(0, dtype=np.int64),
                }
                values.update(overrides)
                np.savez_compressed(path, **values)

                with self.assertRaises(ReferenceAnalysisError) as raised:
                    load_pendulum_subject_anchor(
                        request,
                        entity_id="bob",
                        spatial_transform=self._transform(),
                    )

                self.assertEqual(
                    "reference_pendulum_subject_mask_invalid_v8",
                    raised.exception.code,
                )

        geometry_mutations = (
            ("area", lambda instance: instance.update(area_pixels=5)),
            ("bbox", lambda instance: instance.update(bbox_xyxy=[0, 0, 1, 1])),
            (
                "centroid",
                lambda instance: instance.update(centroid_xy=[0.0, 0.0]),
            ),
        )
        for label, mutate in geometry_mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request = self._write_fixture(root)
                path = root / request.case["assets"][
                    "first_frame_mask_manifest"
                ]
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value["instances"][0])
                write_json(path, value)

                with self.assertRaises(ReferenceAnalysisError) as raised:
                    load_pendulum_subject_anchor(
                        request,
                        entity_id="bob",
                        spatial_transform=self._transform(),
                    )

                self.assertEqual(
                    "reference_pendulum_subject_mask_invalid_v8",
                    raised.exception.code,
                )

    def test_npz_key_set_and_missing_asset_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._write_fixture(root)
            path = root / "case/canonical/masks/01.npz"
            np.savez_compressed(
                path,
                masks=np.zeros((1, 6, 8), dtype=np.uint8),
            )

            with self.assertRaises(ReferenceAnalysisError) as raised:
                load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self._transform(),
                )

            self.assertEqual(
                "reference_pendulum_subject_mask_invalid_v8",
                raised.exception.code,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._write_fixture(root)
            (root / "case/canonical/masks/01.npz").unlink()

            with self.assertRaises(ReferenceAnalysisError) as raised:
                load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self._transform(),
                )

            self.assertEqual(
                "reference_pendulum_subject_mask_missing_v8",
                raised.exception.code,
            )

    def test_loader_wraps_invalid_spatial_transform_as_reference_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = self._write_fixture(Path(temporary))
            transform = self._transform()
            transform["source_size"] = [7, 6]

            with self.assertRaises(ReferenceAnalysisError) as raised:
                load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=transform,
                )

            self.assertEqual(
                "reference_pendulum_subject_mask_transform_invalid_v8",
                raised.exception.code,
            )


class PendulumFalseHighAnchorRegressionTests(unittest.TestCase):
    TRANSFORM = {
        "policy": "reference_content_crop_resize_no_pad",
        "crop_xywh": [0, 0, 1080, 1920],
        "scale": 13.0 / 30.0,
        "source_size": [1080, 1920],
        "target_size": [468, 832],
        "padding": None,
    }
    CASES = {
        "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a025deg_img1369": {
            "expected_xy": (273.7137, 449.3651),
            "false_track_xy": (196.97, 280.61),
        },
        "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a055deg_img1386": {
            "expected_xy": (299.6097, 410.2962),
            "false_track_xy": (222.49, 238.85),
        },
        "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a065deg_img1391": {
            "expected_xy": (311.1461538461538, 423.5701923076923),
            "false_track_xy": (226.15, 333.86),
        },
    }

    @classmethod
    def setUpClass(cls) -> None:
        try:
            dataset = load_dataset(LATEST_DATASET, check_assets=True)
        except FileNotFoundError as exc:
            raise unittest.SkipTest(
                "full pendulum media assets are not published"
            ) from exc
        cls.cases = {case["case_id"]: case for case in dataset.cases}

    def test_real_annotations_anchor_the_bob_not_the_old_support_track(
        self,
    ) -> None:
        asset_root = LATEST_DATASET.parents[2]
        for case_id, expected in self.CASES.items():
            with self.subTest(case_id=case_id):
                request = CaseEvaluationRequest(
                    job={"job_id": f"anchor_regression__{case_id}"},
                    case=self.cases[case_id],
                    case_catalog={},
                    prediction=None,
                    asset_root=asset_root,
                    artifact_dir=Path("/tmp") / case_id,
                    evaluator_config={},
                )

                anchor = load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self.TRANSFORM,
                )

                np.testing.assert_allclose(
                    expected["expected_xy"],
                    anchor.centroid_xy,
                    rtol=0.0,
                    atol=0.01,
                )
                false_distance = np.linalg.norm(
                    anchor.centroid_xy
                    - np.asarray(expected["false_track_xy"])
                )
                self.assertGreater(
                    false_distance,
                    5.0 * anchor.equivalent_radius_px,
                )

    def test_old_apparatus_tracks_fail_identity_anchor_gates(self) -> None:
        # false_track_xy comes from each v11 entity_tracks.json; the anchor is
        # recomputed from the frozen manifest/NPZ through the declared 13/30
        # reference transform so coordinate constants cannot hide asset drift.
        asset_root = LATEST_DATASET.parents[2]
        entity = EntitySpec(
            entity_id="bob",
            role_id="primary_subject",
            entity_class="pendulum_bob",
        )
        config = {
            "minimum_anchor_iou": 0.10,
            "maximum_center_distance_radii": 1.75,
            "minimum_area_ratio": 0.35,
            "maximum_area_ratio": 2.5,
            "minimum_length_ratio": 0.60,
            "maximum_length_ratio": 1.35,
            "minimum_color_similarity": 0.20,
            "minimum_score": 0.45,
            "ambiguity_margin": 0.05,
        }
        for case_id, expected in self.CASES.items():
            with self.subTest(case_id=case_id):
                request = CaseEvaluationRequest(
                    job={"job_id": f"apparatus_regression__{case_id}"},
                    case=self.cases[case_id],
                    case_catalog={},
                    prediction=None,
                    asset_root=asset_root,
                    artifact_dir=Path("/tmp") / case_id,
                    evaluator_config={},
                )
                anchor = load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=self.TRANSFORM,
                )
                false_xy = np.asarray(expected["false_track_xy"])
                false_mask = np.zeros(anchor.mask.shape, dtype=np.uint8)
                cv2.circle(
                    false_mask,
                    tuple(int(round(value)) for value in false_xy),
                    max(2, int(round(anchor.equivalent_radius_px))),
                    255,
                    -1,
                )
                detection = ObjectDetection(
                    frame_index=0,
                    detection_id="old_v11_apparatus_track",
                    xy=false_xy,
                    area_px2=float(np.count_nonzero(false_mask)),
                    entity_class="pendulum_bob",
                    mask=false_mask,
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("condition_directed_sam2",),
                    metadata={
                        "identity_anchor_valid": True,
                        "pivot_distance_ratio": 1.0,
                    },
                )
                observation = OpenWorldObservation(
                    tracks=(
                        OpenWorldTrack(
                            track_id="track_condition_bob",
                            detections=(detection,),
                            confirmed=True,
                            evidence_tier=EvidenceTier.PARTICIPANT,
                        ),
                    ),
                    overflow_counts=np.zeros(1, dtype=np.float64),
                )
                pivot = anchor.centroid_xy - np.asarray([0.0, 100.0])
                structure = PendulumStructureSpec(
                    pivot_xy=pivot,
                    bob_xy=anchor.centroid_xy,
                    bob_radius_px=anchor.equivalent_radius_px,
                    bob_mask=anchor.mask,
                    subject_mask=anchor.mask,
                    confidence=1.0,
                    source="real_anchor_regression",
                )
                frame = np.full((*anchor.mask.shape, 3), 127, dtype=np.uint8)

                decision = decide_pendulum_identity_v8(
                    entity=entity,
                    observation=observation,
                    anchor=anchor,
                    structure=structure,
                    anchor_frame=frame,
                    observed_frame=frame,
                    config=config,
                    reference=False,
                )

                self.assertEqual(SubjectIdentityState.ABSENT, decision.state)
                self.assertFalse(
                    bool(decision.candidates[0]["gates"]["anchor_iou"])
                )
                self.assertFalse(
                    bool(decision.candidates[0]["gates"]["center_distance"])
                )


class PendulumAnnotatedStructureSelectionTests(unittest.TestCase):
    CONFIG = {
        "v8_anchor_dilation_radius_ratio": 0.75,
        "v8_minimum_circle_anchor_containment": 0.45,
        "v8_maximum_anchor_center_distance_radii": 1.75,
        "v8_minimum_identity_source_agreement": 3,
        "v8_minimum_anchor_geometry_ratio_score": 0.45,
        "v8_condition_identity_ambiguity_margin": 0.05,
    }
    EXPECTED_RADIUS_LENGTH_RATIO = 0.16

    @staticmethod
    def _anchor(
        circles: tuple[tuple[int, int, int], ...] = ((90, 80, 8),),
    ) -> PendulumSubjectAnchor:
        mask = np.zeros((128, 128), dtype=np.uint8)
        for x, y, radius in circles:
            cv2.circle(mask, (x, y), radius, 255, -1)
        ys, xs = np.where(mask > 0)
        source = np.where(mask > 0, 1, 0).astype(np.uint8)
        source.setflags(write=False)
        mask.setflags(write=False)
        return PendulumSubjectAnchor(
            case_id="pendulum_selection_fixture",
            entity_id="bob",
            entity_class="pendulum_bob",
            manifest_path=Path("manifest.json"),
            npz_path=Path("01.npz"),
            source_mask=source,
            mask=mask,
            centroid_xy=np.asarray([xs.mean(), ys.mean()]),
            area_px2=float(xs.size),
            equivalent_radius_px=float(np.sqrt(xs.size / np.pi)),
            provenance={"fixture": True},
        )

    @staticmethod
    def _hypothesis(
        x: float,
        y: float,
        *,
        score: float,
        radius: float = 8.0,
        sources: int = 4,
        edge: float = 0.7,
        body: float = 0.7,
    ) -> dict[str, object]:
        return {
            "center_xy": [x, y],
            "pivot_xy": [x - 20.0, y - 45.0],
            "radius_px": radius,
            "length_px": float(np.hypot(20.0, 45.0)),
            "score": score,
            "source_agreement_count": sources,
            "edge_closure_score": edge,
            "body_contrast_score": body,
        }

    @classmethod
    def _decision(
        cls,
        hypotheses: tuple[dict[str, object], ...],
    ) -> ConditionStructureDecision:
        first = hypotheses[0]
        center = np.asarray(first["center_xy"], dtype=np.float64)
        pivot = np.asarray(first["pivot_xy"], dtype=np.float64)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[28:34, 28:34] = 255
        structure = PendulumStructureSpec(
            pivot_xy=pivot,
            bob_xy=center,
            bob_radius_px=float(first["radius_px"]),
            bob_mask=mask,
            subject_mask=mask,
            confidence=float(first["score"]),
            source="v7_fixture",
        )
        return ConditionStructureDecision(
            structure=structure,
            hypotheses=hypotheses,
            confidence_margin=0.1,
            source_agreement=1.0,
            rejection_counts={},
        )

    def test_annotation_selects_real_bob_over_higher_scoring_apparatus(
        self,
    ) -> None:
        decision = self._decision(
            (
                self._hypothesis(30.0, 30.0, score=0.90),
                self._hypothesis(90.0, 80.0, score=0.65),
            )
        )

        selected = select_annotated_condition_structure_v8(
            decision,
            anchor=self._anchor(),
            expected_radius_length_ratio=(
                self.EXPECTED_RADIUS_LENGTH_RATIO
            ),
            config=self.CONFIG,
        )

        np.testing.assert_allclose([90.0, 80.0], selected.structure.bob_xy)
        self.assertEqual(
            "condition_v8_frozen_subject_anchor_fused_v1",
            selected.structure.source,
        )
        self.assertEqual(2, len(selected.candidates))
        self.assertFalse(bool(selected.candidates[0]["eligible"]))
        self.assertTrue(bool(selected.candidates[1]["eligible"]))
        self.assertEqual(0.90, selected.candidates[0]["hypothesis"]["score"])

    def test_four_real_failed_conditions_use_anchor_owned_bob_geometry(
        self,
    ) -> None:
        dataset = load_dataset(LATEST_DATASET, check_assets=False)
        cases = {case["case_id"]: case for case in dataset.cases}
        protocol = json.loads(
            (
                Path(__file__).parents[1]
                / "configs/evaluation/protocols/scene_default_v1.json"
            ).read_text(encoding="utf-8")
        )
        config = protocol["scenes"]["pendulum"]["open_world_observation"]
        fixtures = (
            (
                "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_"
                "r010mm_a015deg_img1449",
                0.01 / 0.06,
                15.0,
            ),
            (
                "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_"
                "r010mm_a045deg_img1383",
                0.01 / 0.06,
                45.0,
            ),
            (
                "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_"
                "r010mm_a065deg_img1391",
                0.01 / 0.06,
                65.0,
            ),
            (
                "pendulum_s3_ltot0175mm_lrope0165mm_m031p5g_"
                "r010mm_a035deg_img1508",
                0.01 / 0.175,
                35.0,
            ),
        )
        for case_id, expected_ratio, angle_deg in fixtures:
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                frame, transform = letterbox_condition_image(
                    dataset.asset_root / case["assets"]["first_frame"],
                    width=480,
                    height=832,
                )
                request = CaseEvaluationRequest(
                    job={"job_id": case_id},
                    case=case,
                    case_catalog=cases,
                    prediction=None,
                    asset_root=dataset.asset_root,
                    artifact_dir=Path("/tmp") / case_id,
                    evaluator_config={},
                )
                anchor = load_pendulum_subject_anchor(
                    request,
                    entity_id="bob",
                    spatial_transform=transform,
                )
                v7 = detect_condition_structure_v7(
                    frame,
                    config=config,
                    expected_radius_length_ratio=expected_ratio,
                    expected_initial_angle_deg=angle_deg,
                )

                selected = select_annotated_condition_structure_v8(
                    v7,
                    anchor=anchor,
                    expected_radius_length_ratio=expected_ratio,
                    condition_frame=frame,
                    expected_initial_angle_deg=angle_deg,
                    config=config,
                )

                np.testing.assert_allclose(
                    anchor.centroid_xy,
                    selected.structure.bob_xy,
                )
                self.assertLess(
                    float(selected.structure.pivot_xy[1]),
                    float(selected.structure.bob_xy[1]),
                )

    def test_anchor_guided_pivot_fuses_fragmented_collinear_string(self) -> None:
        frame = np.full((128, 128, 3), 220, dtype=np.uint8)
        cv2.line(frame, (70, 18), (78, 49), (20, 20, 20), 2)
        cv2.line(frame, (81, 60), (88, 78), (20, 20, 20), 2)
        anchor = self._anchor(circles=((90, 88, 8),))

        evidence = infer_anchor_guided_pivot_v8(
            frame,
            anchor=anchor,
            expected_initial_angle_deg=15.0,
            config={
                **self.CONFIG,
                "line_canny_low": 20.0,
                "line_canny_high": 80.0,
                "line_hough_threshold": 8,
                "minimum_string_segment_length_px": 8,
                "maximum_string_line_gap_px": 4,
            },
        )

        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertLess(float(evidence["pivot_xy"][1]), 25.0)
        self.assertGreaterEqual(int(evidence["supporting_segments"]), 2)

    def test_no_anchor_containment_fails_closed(self) -> None:
        decision = self._decision(
            (self._hypothesis(30.0, 30.0, score=0.90),)
        )

        with self.assertRaises(ReferenceAnalysisError) as raised:
            select_annotated_condition_structure_v8(
                decision,
                anchor=self._anchor(),
                expected_radius_length_ratio=(
                    self.EXPECTED_RADIUS_LENGTH_RATIO
                ),
                config=self.CONFIG,
            )

        self.assertEqual(
            "reference_condition_pendulum_subject_missing_v8",
            raised.exception.code,
        )

    def test_anchor_center_distance_fails_closed(self) -> None:
        decision = self._decision(
            (
                self._hypothesis(
                    72.0,
                    80.0,
                    score=0.80,
                    radius=4.0,
                ),
            )
        )

        with self.assertRaises(ReferenceAnalysisError) as raised:
            select_annotated_condition_structure_v8(
                decision,
                anchor=self._anchor(circles=((72, 80, 4), (90, 80, 8))),
                expected_radius_length_ratio=(
                    self.EXPECTED_RADIUS_LENGTH_RATIO
                ),
                config=self.CONFIG,
            )

        self.assertEqual(
            "reference_condition_pendulum_subject_misaligned_v8",
            raised.exception.code,
        )

    def test_insufficient_source_agreement_fails_closed(self) -> None:
        decision = self._decision(
            (
                self._hypothesis(
                    90.0,
                    80.0,
                    score=0.85,
                    sources=2,
                ),
            )
        )

        with self.assertRaises(ReferenceAnalysisError) as raised:
            select_annotated_condition_structure_v8(
                decision,
                anchor=self._anchor(),
                expected_radius_length_ratio=(
                    self.EXPECTED_RADIUS_LENGTH_RATIO
                ),
                config=self.CONFIG,
            )

        self.assertEqual(
            "reference_condition_pendulum_subject_weak_v8",
            raised.exception.code,
        )

    def test_spatially_distinct_supported_candidates_fail_as_ambiguous(
        self,
    ) -> None:
        first = self._hypothesis(80.0, 80.0, score=0.80)
        second = self._hypothesis(100.0, 80.0, score=0.79)
        first["pivot_xy"] = [70.0, 35.0]
        second["pivot_xy"] = [110.0, 35.0]
        first["length_px"] = second["length_px"] = float(
            np.hypot(20.0, 45.0)
        )
        decision = self._decision(
            (first, second)
        )

        with self.assertRaises(ReferenceAnalysisError) as raised:
            select_annotated_condition_structure_v8(
                decision,
                anchor=self._anchor(circles=((80, 80, 8), (100, 80, 8))),
                expected_radius_length_ratio=(
                    self.EXPECTED_RADIUS_LENGTH_RATIO
                ),
                config=self.CONFIG,
            )

        self.assertEqual(
            "reference_condition_pendulum_subject_ambiguous_v8",
            raised.exception.code,
        )

    def test_anchor_geometry_rejects_a_local_highlight_pseudo_pivot(
        self,
    ) -> None:
        anchor = self._anchor()
        local_highlight = self._hypothesis(
            90.0,
            80.0,
            score=0.95,
            radius=4.0,
            sources=5,
        )
        local_highlight["pivot_xy"] = [90.0, 65.0]
        local_highlight["length_px"] = 15.0
        full_bob_edge = self._hypothesis(
            96.0,
            80.0,
            score=0.55,
            radius=8.0,
            sources=4,
        )
        full_bob_edge["pivot_xy"] = [60.0, 35.0]
        full_bob_edge["length_px"] = float(np.hypot(36.0, 45.0))

        selected = select_annotated_condition_structure_v8(
            self._decision((local_highlight, full_bob_edge)),
            anchor=anchor,
            expected_radius_length_ratio=(
                self.EXPECTED_RADIUS_LENGTH_RATIO
            ),
            config=self.CONFIG,
        )

        np.testing.assert_allclose(
            anchor.centroid_xy,
            selected.structure.bob_xy,
        )
        self.assertAlmostEqual(
            anchor.equivalent_radius_px,
            selected.structure.bob_radius_px,
        )
        np.testing.assert_array_equal(
            anchor.mask,
            selected.structure.bob_mask,
        )
        np.testing.assert_allclose(
            [60.0, 35.0],
            selected.structure.pivot_xy,
        )
        candidates = {
            candidate["hypothesis_index"]: candidate
            for candidate in selected.candidates
        }
        self.assertFalse(
            bool(candidates[0]["gates"]["anchor_geometry_ratio"])
        )
        self.assertTrue(
            bool(candidates[1]["gates"]["anchor_geometry_ratio"])
        )


class PendulumV8ResidualEvidenceTests(unittest.TestCase):
    @staticmethod
    def _observe(*, line_score: float, line_pivot: np.ndarray | None):
        condition = np.zeros((128, 128, 3), dtype=np.uint8)
        frame = np.full_like(condition, 220)
        empty = np.zeros(condition.shape[:2], dtype=np.uint8)
        bob_mask = np.zeros_like(empty)
        cv2.circle(bob_mask, (64, 100), 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray([64.0, 20.0]),
            bob_xy=np.asarray([64.0, 100.0]),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=bob_mask,
            confidence=1.0,
            source="residual_evidence_fixture",
        )
        with (
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_circle_candidates",
                return_value=[(90.0, 80.0, 8.0)],
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_circle_string_evidence",
                return_value=(line_score, line_pivot),
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_circle_edge_support",
                return_value=1.0,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_circle_body_contrast",
                return_value=1.0,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_histogram_intersection",
                return_value=0.8,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_weak_directed_string_circle",
                return_value=(False, {}),
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_terminal_residual_relation",
                return_value={"terminal_residual_repair": False},
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_condition_preexistence_score",
                return_value=0.0,
            ),
        ):
            return _residual_detections(
                [frame],
                [empty],
                directed_subject_masks=[empty],
                availability=np.asarray([True]),
                condition_frame=condition,
                structure=structure,
                config={
                    "v8_residual_requires_pivot_string": True,
                    "participant_string_score": 0.55,
                    "maximum_residual_pivot_error_ratio": 0.25,
                },
            )

    def test_changed_round_apparatus_without_pivot_string_is_rejected(
        self,
    ) -> None:
        detections, rejected = self._observe(
            line_score=0.1,
            line_pivot=None,
        )

        self.assertEqual([[]], detections)
        self.assertEqual(1, rejected["single_source"])

    def test_independent_pivot_string_still_admits_a_residual_bob(
        self,
    ) -> None:
        detections, _ = self._observe(
            line_score=0.9,
            line_pivot=np.asarray([64.0, 20.0]),
        )

        self.assertEqual(1, len(detections[0]))
        self.assertIn(
            "pivot_string_geometry",
            detections[0][0].metadata["proposal_sources"],
        )


class PendulumSubjectIdentityDecisionTests(unittest.TestCase):
    CONFIG = {
        "minimum_anchor_iou": 0.10,
        "maximum_center_distance_radii": 1.75,
        "minimum_area_ratio": 0.35,
        "maximum_area_ratio": 2.5,
        "minimum_length_ratio": 0.60,
        "maximum_length_ratio": 1.35,
        "minimum_color_similarity": 0.20,
        "minimum_score": 0.45,
        "ambiguity_margin": 0.05,
    }

    @staticmethod
    def _entity() -> EntitySpec:
        return EntitySpec(
            entity_id="bob",
            role_id="primary_subject",
            entity_class="pendulum_bob",
        )

    @staticmethod
    def _structure(
        anchor: PendulumSubjectAnchor,
    ) -> PendulumStructureSpec:
        return PendulumStructureSpec(
            pivot_xy=np.asarray([90.0, 30.0]),
            bob_xy=np.asarray([90.0, 80.0]),
            bob_radius_px=8.0,
            bob_mask=anchor.mask,
            subject_mask=anchor.mask,
            confidence=1.0,
            source="identity_fixture",
        )

    @staticmethod
    def _frame(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        frame[:, :] = (30, 170, 70)
        return frame

    @staticmethod
    def _detection(
        *,
        detection_id: str,
        center_xy: tuple[float, float],
        radius: int = 8,
        frame_index: int = 0,
        source: str = "condition_directed_sam2",
        identity_anchor_valid: bool = True,
        pivot_distance_ratio: float = 1.0,
        confidence: float = 0.95,
    ) -> ObjectDetection:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(
            mask,
            tuple(int(round(value)) for value in center_xy),
            radius,
            255,
            -1,
        )
        return ObjectDetection(
            frame_index=frame_index,
            detection_id=detection_id,
            xy=np.asarray(center_xy),
            area_px2=float(np.count_nonzero(mask)),
            entity_class="pendulum_bob",
            mask=mask,
            confidence=confidence,
            evidence_tier=EvidenceTier.PARTICIPANT,
            sources=(source,),
            metadata={
                "identity_anchor_valid": identity_anchor_valid,
                "pivot_distance_ratio": pivot_distance_ratio,
            },
        )

    @staticmethod
    def _observation(
        tracks: tuple[tuple[str, tuple[ObjectDetection, ...]], ...],
        *,
        frame_count: int = 3,
    ) -> OpenWorldObservation:
        return OpenWorldObservation(
            tracks=tuple(
                OpenWorldTrack(
                    track_id=track_id,
                    detections=detections,
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                )
                for track_id, detections in tracks
            ),
            overflow_counts=np.zeros(frame_count, dtype=np.float64),
        )

    def _decide(
        self,
        observation: OpenWorldObservation,
        *,
        anchor: PendulumSubjectAnchor | None = None,
        reference: bool = False,
        anchor_frame: np.ndarray | None = None,
        observed_frame: np.ndarray | None = None,
        config: dict[str, float] | None = None,
    ):
        selected_anchor = (
            PendulumAnnotatedStructureSelectionTests._anchor()
            if anchor is None
            else anchor
        )
        default_frame = self._frame(selected_anchor.mask.shape)
        return decide_pendulum_identity_v8(
            entity=self._entity(),
            observation=observation,
            anchor=selected_anchor,
            structure=self._structure(selected_anchor),
            anchor_frame=(default_frame if anchor_frame is None else anchor_frame),
            observed_frame=(
                default_frame if observed_frame is None else observed_frame
            ),
            config=self.CONFIG if config is None else config,
            reference=reference,
        )

    def test_unique_candidate_is_confirmed(self) -> None:
        detection = self._detection(
            detection_id="bob_frame_zero",
            center_xy=(90.0, 80.0),
        )
        observation = self._observation((("track_bob", (detection,)),))

        decision = self._decide(observation)

        self.assertEqual(SubjectIdentityState.CONFIRMED, decision.state)
        self.assertEqual("track_bob", decision.track_id)
        self.assertIsNone(decision.reason_code)
        self.assertGreater(float(decision.candidates[0]["score"]), 0.45)
        self.assertTrue(all(decision.candidates[0]["gates"].values()))

    def test_no_eligible_candidate_is_absent_or_reference_invalid(
        self,
    ) -> None:
        observation = self._observation(())

        prediction = self._decide(observation, reference=False)
        reference = self._decide(observation, reference=True)

        self.assertEqual(SubjectIdentityState.ABSENT, prediction.state)
        self.assertEqual(
            "prediction_pendulum_identity_absent_v8",
            prediction.reason_code,
        )
        self.assertEqual(
            SubjectIdentityState.REFERENCE_INVALID,
            reference.state,
        )
        self.assertEqual(
            "reference_pendulum_identity_unconfirmed_v8",
            reference.reason_code,
        )

    def test_two_supported_distinct_candidates_are_ambiguous(self) -> None:
        anchor = PendulumAnnotatedStructureSelectionTests._anchor(
            circles=((85, 80, 10), (95, 80, 10))
        )
        first = self._detection(
            detection_id="candidate_left",
            center_xy=(85.0, 80.0),
        )
        second = self._detection(
            detection_id="candidate_right",
            center_xy=(95.0, 80.0),
        )
        observation = self._observation(
            (
                ("track_left", (first,)),
                ("track_right", (second,)),
            )
        )

        decision = self._decide(observation, anchor=anchor)

        self.assertEqual(SubjectIdentityState.AMBIGUOUS, decision.state)
        self.assertEqual(
            "prediction_pendulum_identity_ambiguous_v8",
            decision.reason_code,
        )
        self.assertLess(float(decision.winning_margin), 0.05)

    def test_frozen_assignment_never_transfers_identity_after_gap(self) -> None:
        selected_frame_zero = self._detection(
            detection_id="selected_zero",
            center_xy=(90.0, 80.0),
            frame_index=0,
        )
        selected_after_gap = self._detection(
            detection_id="selected_two",
            center_xy=(88.0, 80.0),
            frame_index=2,
        )
        residual_during_gap = self._detection(
            detection_id="residual_one",
            center_xy=(90.0, 80.0),
            frame_index=1,
        )
        observation = self._observation(
            (
                (
                    "track_original",
                    (selected_frame_zero, selected_after_gap),
                ),
                ("track_replacement", (residual_during_gap,)),
            )
        )
        decision = self._decide(observation)

        assignment = identity_decision_to_frozen_assignment(
            decision,
            entity_id="bob",
            observation=observation,
        )

        self.assertIsNotNone(assignment)
        assert assignment is not None
        self.assertEqual(
            {"bob": "track_original"},
            assignment.entity_to_track,
        )
        self.assertEqual(
            ("track_replacement",),
            assignment.residual_track_ids,
        )
        selected = next(
            track
            for track in observation.tracks
            if track.track_id == assignment.entity_to_track["bob"]
        )
        self.assertEqual([0, 2], [item.frame_index for item in selected.detections])

    def test_wrong_source_or_explicitly_invalid_anchor_cannot_confirm(
        self,
    ) -> None:
        variants = (
            self._detection(
                detection_id="wrong_source",
                center_xy=(90.0, 80.0),
                source="residual_circle",
            ),
            self._detection(
                detection_id="invalid_directed",
                center_xy=(90.0, 80.0),
                identity_anchor_valid=False,
            ),
        )
        for detection in variants:
            with self.subTest(detection=detection.detection_id):
                decision = self._decide(
                    self._observation((("track", (detection,)),))
                )
                self.assertEqual(SubjectIdentityState.ABSENT, decision.state)

    def test_area_length_color_and_score_are_independent_hard_gates(
        self,
    ) -> None:
        anchor = PendulumAnnotatedStructureSelectionTests._anchor()
        default_frame = self._frame(anchor.mask.shape)
        different_frame = np.full_like(default_frame, (180, 30, 200))
        strict_score = dict(self.CONFIG)
        strict_score["minimum_score"] = 1.0
        variants = (
            (
                "area_ratio",
                self._detection(
                    detection_id="too_small",
                    center_xy=(90.0, 80.0),
                    radius=4,
                ),
                {},
            ),
            (
                "length_ratio",
                self._detection(
                    detection_id="wrong_length",
                    center_xy=(90.0, 80.0),
                    pivot_distance_ratio=1.8,
                ),
                {},
            ),
            (
                "color_similarity",
                self._detection(
                    detection_id="wrong_color",
                    center_xy=(90.0, 80.0),
                ),
                {"observed_frame": different_frame},
            ),
            (
                "minimum_score",
                self._detection(
                    detection_id="below_strict_score",
                    center_xy=(90.0, 80.0),
                ),
                {"config": strict_score},
            ),
        )
        for failed_gate, detection, overrides in variants:
            with self.subTest(failed_gate=failed_gate):
                decision = self._decide(
                    self._observation((("track", (detection,)),)),
                    anchor=anchor,
                    **overrides,
                )
                self.assertEqual(SubjectIdentityState.ABSENT, decision.state)
                self.assertFalse(
                    bool(decision.candidates[0]["gates"][failed_gate])
                )


class PendulumV8EvaluatorIntegrationTests(unittest.TestCase):
    CSTI_CONFIG = CSTIConfig.from_mapping(
        {
            "enabled": True,
            "algorithm": "exact_full_tube_edt",
            "spatial_tolerance_fraction": 0.005,
            "temporal_tolerance_s": 0.05,
            "condition_frame_policy": "exclude_initial_samples",
            "initial_frames_excluded": 1,
            "score_aggregation": "full_tube",
            "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
            "case_aggregation": "mean_gt_entities",
            "timeline_policy": "physical_overlap",
            "mask_resolution": "scene_analysis_native",
        }
    )

    @staticmethod
    def _annotated_decision() -> AnnotatedConditionDecision:
        anchor = PendulumAnnotatedStructureSelectionTests._anchor()
        v7 = PendulumAnnotatedStructureSelectionTests._decision(
            (
                PendulumAnnotatedStructureSelectionTests._hypothesis(
                    90.0,
                    80.0,
                    score=0.80,
                ),
            )
        )
        return select_annotated_condition_structure_v8(
            v7,
            anchor=anchor,
            expected_radius_length_ratio=(
                PendulumAnnotatedStructureSelectionTests.
                EXPECTED_RADIUS_LENGTH_RATIO
            ),
            config=PendulumAnnotatedStructureSelectionTests.CONFIG,
        )

    def test_reference_identity_failure_is_reference_unavailable_reason(
        self,
    ) -> None:
        annotated = self._annotated_decision()
        entity = PendulumSubjectIdentityDecisionTests._entity()
        declaration = SimpleNamespace(
            entity_class="pendulum_bob",
            to_entity_spec=lambda: entity,
        )
        manifest = SimpleNamespace(
            scene_id="pendulum",
            entities=(declaration,),
            reference_capability=ReferenceCapability.SAME_CASE_GT,
        )
        frame = PendulumSubjectIdentityDecisionTests._frame()
        masks = [np.array(annotated.structure.bob_mask, copy=True)] * 3
        evaluator = object.__new__(PendulumOpenWorldCaseEvaluatorV8)
        evaluator.config = {
            "motion_proposal": {},
            "open_world_observation": {},
            "subject_identity": PendulumSubjectIdentityDecisionTests.CONFIG,
        }
        evaluator._segmenter = SimpleNamespace(
            segment=lambda *args, **kwargs: (
                masks,
                {"status": "fixture"},
                None,
            )
        )
        empty_observation = OpenWorldObservation(
            tracks=(),
            overflow_counts=np.zeros(3),
        )
        request = CaseEvaluationRequest(
            job={"job_id": "reference_identity_failure"},
            case={"case_id": "pendulum_fixture", "scene_id": "pendulum"},
            case_catalog={},
            prediction=None,
            asset_root=Path("/tmp"),
            artifact_dir=Path("/tmp/pendulum_fixture"),
            evaluator_config={},
        )
        video = SimpleNamespace(
            frames=[frame, frame, frame],
            spatial_transform={},
        )

        with (
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "materialize_entity_manifest",
                return_value=manifest,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_load_condition_frame",
                return_value=(frame, Path("/tmp/condition.png"), {}),
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "load_pendulum_subject_anchor",
                return_value=annotated.anchor,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "flat_physics_quantities",
                return_value={"initial_angle": {"value": 25.0}},
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_pendulum_radius_length_ratio",
                return_value=0.1,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "detect_condition_structure_v7",
                return_value=SimpleNamespace(),
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "select_annotated_condition_structure_v8",
                return_value=annotated,
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "infer_reference_mode",
                return_value="same_case_reference",
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "extract_bob_masks",
                return_value=(masks, np.zeros((3, 2)), np.ones(3)),
            ),
            patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "discover_pendulum_objects_v7",
                return_value=empty_observation,
            ),
        ):
            with self.assertRaises(ReferenceAnalysisError) as raised:
                evaluator.analyze(
                    request,
                    times_s=[0.0, 0.1, 0.2],
                    reference_video=video,
                    prediction_video=video,
                )

        self.assertEqual(
            "reference_pendulum_identity_unconfirmed_v8",
            raised.exception.code,
        )

    def test_csti_uses_confirmed_tube_and_empty_tube_fails_closed(self) -> None:
        reference_masks = (
            PendulumSubjectIdentityDecisionTests._detection(
                detection_id="reference_zero",
                center_xy=(90.0, 80.0),
                frame_index=0,
            ).mask,
            PendulumSubjectIdentityDecisionTests._detection(
                detection_id="reference_one",
                center_xy=(88.0, 80.0),
                frame_index=1,
            ).mask,
        )
        bob_detections = tuple(
            PendulumSubjectIdentityDecisionTests._detection(
                detection_id=f"bob_{index}",
                center_xy=center,
                frame_index=index,
            )
            for index, center in enumerate(((90.0, 80.0), (88.0, 80.0)))
        )
        apparatus_mask = np.zeros((128, 128), dtype=np.uint8)
        apparatus_mask[10:110, 10:110] = 255
        apparatus = ObjectDetection(
            frame_index=0,
            detection_id="large_apparatus",
            xy=np.asarray([60.0, 60.0]),
            area_px2=float(np.count_nonzero(apparatus_mask)),
            entity_class="pendulum_bob",
            mask=apparatus_mask,
            sources=("residual_circle",),
        )
        observation = OpenWorldObservation(
            tracks=(
                OpenWorldTrack(
                    track_id="track_confirmed_bob",
                    detections=bob_detections,
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
                OpenWorldTrack(
                    track_id="track_large_apparatus",
                    detections=(apparatus,),
                    confirmed=True,
                    evidence_tier=EvidenceTier.INDEPENDENT_SALIENT,
                ),
            ),
            overflow_counts=np.zeros(2),
        )
        comparison = SimpleNamespace(
            matches=tuple(
                EntityMatch(
                    index,
                    "bob",
                    "track_confirmed_bob",
                    1.0,
                    0.0,
                )
                for index in range(2)
            )
        )
        evaluator = object.__new__(PendulumOpenWorldCaseEvaluatorV8)
        entity = SimpleNamespace(entity_id="bob", role_id="moving_bob")

        confirmed_input = evaluator._build_csti_input(
            capability=ReferenceCapability.SAME_CASE_GT,
            times_s=[0.0, 0.1],
            frame_shape=(128, 128),
            entity=entity,
            reference_bobs=reference_masks,
            prediction_observation=observation,
            comparison=comparison,
        )
        confirmed = evaluate_csti(
            confirmed_input,
            expected_entities=(("bob", "moving_bob"),),
            config=self.CSTI_CONFIG,
        )
        absent_input = evaluator._build_csti_input(
            capability=ReferenceCapability.SAME_CASE_GT,
            times_s=[0.0, 0.1],
            frame_shape=(128, 128),
            entity=entity,
            reference_bobs=reference_masks,
            prediction_observation=None,
            comparison=SimpleNamespace(matches=()),
        )
        absent = evaluate_csti(
            absent_input,
            expected_entities=(("bob", "moving_bob"),),
            config=self.CSTI_CONFIG,
        )

        self.assertEqual(1.0, confirmed["score"])
        self.assertEqual(
            ["track_confirmed_bob"],
            confirmed["objects"][0]["matched_prediction_track_ids"],
        )
        self.assertNotIn(
            "track_large_apparatus",
            confirmed["objects"][0]["matched_prediction_track_ids"],
        )
        self.assertEqual(0.0, absent["score"])
        self.assertFalse(absent["objects"][0]["matched"])

    def test_identity_artifact_records_decisions_and_overlay(self) -> None:
        annotated = self._annotated_decision()
        detection = PendulumSubjectIdentityDecisionTests._detection(
            detection_id="bob_zero",
            center_xy=(90.0, 80.0),
        )
        observation = PendulumSubjectIdentityDecisionTests._observation(
            (("track_bob", (detection,)),)
        )
        identity = PendulumSubjectIdentityDecisionTests()._decide(observation)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = CaseEvaluationRequest(
                job={"job_id": "identity_artifact"},
                case={"case_id": "pendulum_identity_artifact"},
                case_catalog={},
                prediction=None,
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config={},
            )

            artifacts = _write_subject_identity_artifacts(
                request,
                condition_frame=PendulumSubjectIdentityDecisionTests._frame(),
                anchor=annotated.anchor,
                condition_decision=annotated,
                reference_decision=identity,
                prediction_decision=identity,
                failure_reason=None,
            )

            record = json.loads(
                Path(artifacts["subject_identity"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("1.0", record["schema_version"])
            self.assertEqual("confirmed", record["prediction"]["state"])
            self.assertTrue(Path(artifacts["subject_identity_overlay"]).is_file())

    def test_exact_frame_reuse_still_runs_prediction_identity_gate(self) -> None:
        annotated = self._annotated_decision()
        entity = PendulumSubjectIdentityDecisionTests._entity()
        declaration = SimpleNamespace(
            entity_class="pendulum_bob",
            to_entity_spec=lambda: entity,
        )
        manifest = SimpleNamespace(
            scene_id="pendulum",
            entities=(declaration,),
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            materializer_id="fixture",
            digest="fixture",
            to_canonical_dict=lambda: {"fixture": True},
        )
        frame = PendulumSubjectIdentityDecisionTests._frame()
        detections = tuple(
            PendulumSubjectIdentityDecisionTests._detection(
                detection_id=f"bob_{index}",
                center_xy=(90.0 - index, 80.0),
                frame_index=index,
            )
            for index in range(3)
        )
        masks = [np.array(value.mask, copy=True) for value in detections]
        observation = OpenWorldObservation(
            tracks=(
                OpenWorldTrack(
                    track_id="track_bob",
                    detections=detections,
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
            ),
            overflow_counts=np.zeros(3),
        )
        confirmed = SubjectIdentityDecision(
            state=SubjectIdentityState.CONFIRMED,
            entity_id="bob",
            track_id="track_bob",
            reason_code=None,
            candidates=(),
            winning_margin=1.0,
        )
        absent = SubjectIdentityDecision(
            state=SubjectIdentityState.ABSENT,
            entity_id="bob",
            track_id=None,
            reason_code="prediction_pendulum_identity_absent_v8",
            candidates=(),
            winning_margin=None,
        )
        topology = {
            "per_frame": [
                {"string_intact_score": 1.0, "branch_string_detected": False}
                for _ in range(3)
            ]
        }
        subject = SimpleNamespace(
            components={"appearance": 1.0},
            score=1.0,
            per_frame=[{} for _ in range(3)],
            to_metric=lambda weights: {"score": 1.0},
        )
        trace = SimpleNamespace(valid_ratio=1.0)
        evaluator = object.__new__(PendulumOpenWorldCaseEvaluatorV8)
        evaluator.config = {
            "motion_proposal": {},
            "open_world_observation": {
                "v8_condition_low_margin_warning": 0.0,
                "v7_low_matched_coverage_warning": 0.0,
                "v7_residual_fragmentation_warning_tracks": 99,
            },
            "subject_identity": PendulumSubjectIdentityDecisionTests.CONFIG,
            "object_centric_scoring": {
                "assignment": {"minimum_match_position_similarity": 0.0}
            },
            "period": {},
            "scoring": {},
            "subject_scoring": {"weights": {"appearance": 1.0}},
            "topology": {},
            "visualization": {},
        }
        evaluator.csti_enabled = False
        evaluator._segmenter = SimpleNamespace(
            segment=lambda *args, **kwargs: (
                masks,
                {"status": "fixture"},
                None,
            )
        )
        request = CaseEvaluationRequest(
            job={"job_id": "exact_reuse_identity_gate"},
            case={"case_id": "pendulum_fixture", "scene_id": "pendulum"},
            case_catalog={},
            prediction=None,
            asset_root=Path("/tmp"),
            artifact_dir=Path("/tmp/pendulum_fixture"),
            evaluator_config={},
        )
        video = SimpleNamespace(
            frames=[np.array(frame, copy=True) for _ in range(3)],
            spatial_transform={},
            available=np.ones(3, dtype=bool),
        )

        with ExitStack() as stack:
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "materialize_entity_manifest",
                return_value=manifest,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_load_condition_frame",
                return_value=(frame, Path("/tmp/condition.png"), {}),
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "load_pendulum_subject_anchor",
                return_value=annotated.anchor,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "flat_physics_quantities",
                return_value={"initial_angle": {"value": 25.0}},
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_pendulum_radius_length_ratio",
                return_value=0.1,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "detect_condition_structure_v7",
                return_value=SimpleNamespace(),
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "select_annotated_condition_structure_v8",
                return_value=annotated,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "infer_reference_mode",
                return_value="same_case_reference",
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "extract_bob_masks",
                return_value=(masks, np.zeros((3, 2)), np.ones(3)),
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "discover_pendulum_objects_v7",
                return_value=observation,
            ))
            identity_gate = stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "decide_pendulum_identity_v8",
                side_effect=(confirmed, absent),
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "extract_bob_trace_v7",
                return_value=trace,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "observe_pendulum_topology_v7",
                return_value=topology,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "compare_pendulum_topology_v7",
                return_value={"score": 1.0, "uncertain": False},
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "score_traces",
                return_value={"score": 1.0},
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "compare_subjects",
                return_value=subject,
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_content_contract",
                return_value=({"physics": 1.0}, {"physics": 1.0}),
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_write_artifacts",
                return_value={},
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "write_open_world_v2_artifacts",
                return_value={},
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "_write_subject_identity_artifacts",
                return_value={},
            ))
            stack.enter_context(patch(
                "physbench.evaluation.scenes.pendulum.v8_evaluator."
                "sha256_file",
                return_value="0" * 64,
            ))
            analysis = evaluator.analyze(
                request,
                times_s=[0.0, 0.1, 0.2],
                reference_video=video,
                prediction_video=video,
            )

        self.assertEqual(2, identity_gate.call_count)
        self.assertEqual(0.0, analysis.score)
        self.assertEqual(
            0.0,
            analysis.metrics["object_centric_integrity"]["integrity_gate"],
        )
        self.assertIn(
            "prediction_pendulum_identity_absent_v8",
            analysis.quality["degradation_codes"],
        )
        self.assertTrue(
            analysis.provenance["exact_same_sampled_frames_reused"]
        )
        self.assertEqual(
            "absent",
            analysis.provenance["prediction_identity_decision"]["state"],
        )


@unittest.skipUnless(
    (
        Path(__file__).resolve().parents[1]
        / "scripts/regress_pendulum_identity_v8.py"
    ).is_file(),
    "clean release excludes the model-specific pendulum regression script",
)
class PendulumV8RealRegressionContractTests(unittest.TestCase):
    def test_existing_result_reuse_requires_exact_identity_and_hashes(
        self,
    ) -> None:
        module = importlib.import_module("scripts.regress_pendulum_identity_v8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path = root / "prediction.mp4"
            reference_path = root / "reference.mp4"
            prediction_path.write_bytes(b"prediction-v1")
            reference_path.write_bytes(b"reference-v1")
            artifact_dir = root / "cases" / "job"
            write_json(
                artifact_dir / "result.json",
                {
                    "job_id": "job",
                    "case_id": "case",
                    "scene_id": "pendulum",
                    "status": "evaluated",
                    "regression_variant": "false_high_negative",
                    "evaluator": {"fingerprint": "evaluator-v1"},
                    "provenance": {
                        "prediction_video_sha256": module.sha256_file(
                            prediction_path
                        ),
                        "reference_video_sha256": module.sha256_file(
                            reference_path
                        ),
                    },
                },
            )

            reusable = module._load_reusable_result(
                artifact_dir=artifact_dir,
                job_id="job",
                case_id="case",
                scene_id="pendulum",
                variant="false_high_negative",
                evaluator_fingerprint="evaluator-v1",
                prediction_path=prediction_path,
                reference_path=reference_path,
            )
            self.assertIsNotNone(reusable)

            self.assertIsNone(
                module._load_reusable_result(
                    artifact_dir=artifact_dir,
                    job_id="job",
                    case_id="case",
                    scene_id="pendulum",
                    variant="false_high_negative",
                    evaluator_fingerprint="evaluator-v2",
                    prediction_path=prediction_path,
                    reference_path=reference_path,
                )
            )
            prediction_path.write_bytes(b"prediction-v2")
            self.assertIsNone(
                module._load_reusable_result(
                    artifact_dir=artifact_dir,
                    job_id="job",
                    case_id="case",
                    scene_id="pendulum",
                    variant="false_high_negative",
                    evaluator_fingerprint="evaluator-v1",
                    prediction_path=prediction_path,
                    reference_path=reference_path,
                )
            )

    def test_script_freezes_three_false_high_cases_and_existing_assets(
        self,
    ) -> None:
        module = importlib.import_module("scripts.regress_pendulum_identity_v8")
        expected = {
            "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a025deg_img1369": (
                0.7963642275035674,
                "seven_scene_symbol_value_finetune_eval_v13__"
                "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_"
                "a025deg_img1369__seed000042.mp4",
            ),
            "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a055deg_img1386": (
                0.7392562637847697,
                "seven_scene_symbol_value_finetune_eval_v13__"
                "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_"
                "a055deg_img1386__seed000042.mp4",
            ),
            "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a065deg_img1391": (
                0.6675888161540786,
                "seven_scene_symbol_value_finetune_eval_v13__"
                "pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_"
                "a065deg_img1391__seed000042.mp4",
            ),
        }
        self.assertEqual(set(expected), set(module.REGRESSION_CASES))
        try:
            dataset = load_dataset(LATEST_DATASET, check_assets=True)
        except FileNotFoundError:
            self.skipTest("full pendulum media assets are not published")
        cases = {case["case_id"]: case for case in dataset.cases}
        for case_id, (old_score, filename) in expected.items():
            with self.subTest(case_id=case_id):
                frozen = module.REGRESSION_CASES[case_id]
                self.assertEqual(old_score, frozen["old_expert_score"])
                self.assertEqual(0.25, frozen["maximum_score"])
                self.assertEqual(filename, Path(frozen["prediction_path"]).name)
                self.assertTrue(Path(frozen["prediction_path"]).is_file())
                manifest = (
                    LATEST_DATASET.parents[2]
                    / cases[case_id]["assets"]["first_frame_mask_manifest"]
                )
                self.assertTrue(manifest.is_file())
        self.assertEqual(
            {
                "pendulum_r2_ltot0110mm_lrope0100mm_r010mm_a020deg"
            },
            set(module.REVIEWED_GOOD_CONTROLS),
        )
        reviewed = next(iter(module.REVIEWED_GOOD_CONTROLS.values()))
        self.assertEqual(0.40, reviewed["minimum_expert_score"])
        self.assertEqual(0.05, reviewed["minimum_csti_score"])


if __name__ == "__main__":
    unittest.main()
