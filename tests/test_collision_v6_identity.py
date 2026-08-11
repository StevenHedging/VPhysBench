from __future__ import annotations

import tempfile
import unittest
from math import gcd
from pathlib import Path

import cv2
import numpy as np

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.reference import resolve_physics_reference
from physbench.evaluation.common.errors import SceneAnalysisError
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.scenes.collision.v6_identity import (
    assign_collision_frame_zero_candidates,
    build_motion_validated_collision_reference_anchors,
    load_collision_identity_anchors,
    localize_collision_reference_candidates,
    reference_prompts_from_collision_anchors,
)
from physbench.io import write_json


_HEIGHT = 100
_WIDTH = 200


def _transform() -> dict[str, object]:
    return {
        "policy": "reference_content_crop_resize_no_pad",
        "crop_xywh": [0, 0, _WIDTH, _HEIGHT],
        "scale": 1.0,
        "source_size": [_WIDTH, _HEIGHT],
        "target_size": [_WIDTH, _HEIGHT],
        "padding": None,
    }


def _binding_config() -> dict[str, object]:
    return {
        "box_expand": 1.5,
        "minimum_box_side": 20,
        "maximum_center_distance_radii": 3.0,
        "minimum_position_similarity": 0.2,
        "minimum_scale_similarity": 0.45,
        "minimum_appearance_similarity": 0.15,
        "minimum_binding_score": 0.55,
        "minimum_assignment_margin": 0.05,
        "weights": {
            "position": 0.5,
            "scale": 0.2,
            "appearance": 0.3,
        },
    }


def _fixture(root: Path) -> tuple[CaseEvaluationRequest, np.ndarray]:
    frame = np.full((_HEIGHT, _WIDTH, 3), 210, dtype=np.uint8)
    circles = [
        ("ball_1", "object_1", "01", (48, 70), (25, 25, 25)),
        ("ball_2", "object_2", "02", (142, 70), (90, 90, 90)),
    ]
    instances = []
    for logical_id, object_id, mask_id, center, color in circles:
        mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
        cv2.circle(mask, center, 8, 1, -1)
        cv2.circle(frame, center, 8, color, -1)
        ys, xs = np.where(mask > 0)
        npz_relative = f"case/canonical/masks/{mask_id}.npz"
        npz_path = root / npz_relative
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            masks=mask[None, ...],
            mask_ids=np.asarray([mask_id]),
            object_ids=np.asarray([logical_id]),
            frame_index=np.asarray(0, dtype=np.int64),
        )
        instances.append(
            {
                "mask_id": mask_id,
                "object_id": object_id,
                "entity_class": "ball",
                "npz_asset": npz_relative,
                "area_pixels": int(xs.size),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
            }
        )
    manifest_relative = "case/canonical/masks/manifest.json"
    write_json(
        root / manifest_relative,
        {
            "schema_version": "1.2",
            "case_id": "collision_v6_identity_fixture",
            "scene_id": "collision_1d",
            "frame_index": 0,
            "frame_scope": "first_frame_only",
            "source_first_frame": "case/canonical/first_frame.png",
            "image_shape_hw": [_HEIGHT, _WIDTH],
            "instances": instances,
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
        "case_id": "collision_v6_identity_fixture",
        "scene_id": "collision_1d",
        "assets": {
            "first_frame": "case/canonical/first_frame.png",
            "first_frame_mask_manifest": manifest_relative,
        },
    }
    request = CaseEvaluationRequest(
        job={"job_id": "collision_v6_identity_fixture"},
        case=case,
        case_catalog={case["case_id"]: case},
        prediction={"status": "complete", "video_path": "mock.mp4"},
        asset_root=root,
        artifact_dir=root / "artifacts",
        evaluator_config={"type": "collision_1d_state_v6"},
    )
    return request, frame


class CollisionV6IdentityTests(unittest.TestCase):
    def test_loads_ordered_logical_and_dataset_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, reference_frame = _fixture(Path(temporary))
            anchors = load_collision_identity_anchors(
                request,
                entity_ids=("ball_1", "ball_2"),
                reference_frame=reference_frame,
                spatial_transform=_transform(),
            )

        self.assertEqual(
            ["ball_1", "ball_2"],
            [value.logical_entity_id for value in anchors],
        )
        self.assertEqual(
            ["object_1", "object_2"],
            [value.dataset_object_id for value in anchors],
        )
        self.assertTrue(all(not value.mask.flags.writeable for value in anchors))
        self.assertTrue(all(not value.histogram.flags.writeable for value in anchors))

    def test_reference_prompts_are_frozen_at_frame_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, reference_frame = _fixture(Path(temporary))
            anchors = load_collision_identity_anchors(
                request,
                entity_ids=("ball_1", "ball_2"),
                reference_frame=reference_frame,
                spatial_transform=_transform(),
            )
            prompts, metadata = reference_prompts_from_collision_anchors(
                anchors,
                config=_binding_config(),
            )

        self.assertEqual([0, 0], [prompt.frame_index for prompt in prompts])
        self.assertEqual(
            ["ball_1", "ball_2"],
            [prompt.metadata["entity_id"] for prompt in prompts],
        )
        self.assertEqual(0, metadata["seed_frame"])
        self.assertEqual(
            "frozen_dataset_subject_annotation_v1",
            metadata["seed_source"],
        )

    def test_reference_motion_rejects_static_false_mask_location(self) -> None:
        first = np.full((_HEIGHT, _WIDTH, 3), 210, dtype=np.uint8)
        cv2.circle(first, (48, 70), 8, (25, 25, 25), -1)
        cv2.circle(first, (142, 70), 8, (90, 90, 90), -1)
        cv2.circle(first, (95, 70), 8, (150, 150, 150), 1)
        frames = [first]
        for offset in (4, 8, 12, 16):
            frame = np.full_like(first, 210)
            cv2.circle(frame, (48 + offset, 70), 8, (25, 25, 25), -1)
            cv2.circle(frame, (142 - offset, 70), 8, (90, 90, 90), -1)
            cv2.circle(frame, (95, 70), 8, (150, 150, 150), 1)
            frames.append(frame)

        anchors, metadata = localize_collision_reference_candidates(
            entity_ids=("ball_1", "ball_2"),
            reference_frames=frames,
            candidates=(
                (48.0, 70.0, 8.0, 30.0),
                (95.0, 70.0, 8.0, 42.0),
                (142.0, 70.0, 8.0, 30.0),
            ),
            config={
                "minimum_motion_fraction": 0.20,
                "minimum_mean_change": 8.0,
                "maximum_entity_y_spread_px": 20.0,
                "maximum_radius_ratio": 2.0,
                "minimum_gap_radius_fraction": 0.45,
                "minimum_set_score_margin": 1.0,
            },
        )

        self.assertEqual(
            [[48.0, 70.0, 8.0], [142.0, 70.0, 8.0]],
            metadata["selected_circles_xyr"],
        )
        self.assertEqual(
            "reference_motion_validated_frame_zero_v1",
            metadata["seed_source"],
        )
        self.assertTrue(all(value.provenance["motion_validated"] for value in anchors))
        prompts, prompt_metadata = reference_prompts_from_collision_anchors(
            anchors,
            config=_binding_config(),
        )
        self.assertEqual([0, 0], [value.frame_index for value in prompts])
        self.assertEqual(
            "reference_motion_validated_frame_zero_v1",
            prompt_metadata["seed_source"],
        )

    def test_reference_motion_ambiguity_is_unavailable(self) -> None:
        first = np.full((_HEIGHT, _WIDTH, 3), 210, dtype=np.uint8)
        frames = [first.copy(), first.copy()]
        for x in (40, 100, 160):
            cv2.circle(frames[0], (x, 70), 8, (20, 20, 20), -1)
        frames[1] = np.full_like(first, 210)

        with self.assertRaises(SceneAnalysisError) as captured:
            localize_collision_reference_candidates(
                entity_ids=("ball_1", "ball_2"),
                reference_frames=frames,
                candidates=(
                    (40.0, 70.0, 8.0, 30.0),
                    (100.0, 70.0, 8.0, 30.0),
                    (160.0, 70.0, 8.0, 30.0),
                ),
                config={
                    "minimum_motion_fraction": 0.20,
                    "minimum_mean_change": 8.0,
                    "maximum_entity_y_spread_px": 20.0,
                    "maximum_radius_ratio": 2.0,
                    "minimum_gap_radius_fraction": 0.45,
                    "minimum_set_score_margin": 1.0,
                },
            )

        self.assertEqual(
            "reference_collision_identity_ambiguous", captured.exception.code
        )

    def test_prediction_candidates_bind_uniquely_without_gt_mask_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, reference_frame = _fixture(Path(temporary))
            anchors = load_collision_identity_anchors(
                request,
                entity_ids=("ball_1", "ball_2"),
                reference_frame=reference_frame,
                spatial_transform=_transform(),
            )
            prediction = np.full_like(reference_frame, 210)
            cv2.circle(prediction, (49, 70), 8, (28, 28, 28), -1)
            cv2.circle(prediction, (141, 70), 8, (92, 92, 92), -1)
            prompts, metadata = assign_collision_frame_zero_candidates(
                anchors,
                prediction_frame=prediction,
                candidates=((49.0, 70.0, 8.0), (141.0, 70.0, 8.0)),
                config=_binding_config(),
            )

        self.assertEqual([0, 0], [prompt.frame_index for prompt in prompts])
        self.assertTrue(metadata["accepted"])
        self.assertGreater(metadata["assignment_margin"], 0.05)
        self.assertEqual(
            [[49.0, 70.0, 8.0], [141.0, 70.0, 8.0]],
            metadata["assigned_circles_xyr"],
        )
        self.assertTrue(
            all(
                prompt.metadata["source"]
                == "prediction_frame_zero_unique_binding"
                for prompt in prompts
            )
        )

    def test_ambiguous_global_assignment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, reference_frame = _fixture(Path(temporary))
            anchors = load_collision_identity_anchors(
                request,
                entity_ids=("ball_1", "ball_2"),
                reference_frame=reference_frame,
                spatial_transform=_transform(),
            )
            prediction = np.full_like(reference_frame, 210)
            cv2.circle(prediction, (95, 70), 8, (55, 55, 55), -1)

            with self.assertRaises(SceneAnalysisError) as captured:
                assign_collision_frame_zero_candidates(
                    anchors,
                    prediction_frame=prediction,
                    candidates=((95.0, 70.0, 8.0), (95.0, 70.0, 8.0)),
                    config={
                        **_binding_config(),
                        "minimum_position_similarity": 0.0,
                        "minimum_appearance_similarity": 0.0,
                        "minimum_binding_score": 0.0,
                    },
                )

        self.assertEqual(
            "collision_prediction_identity_ambiguous", captured.exception.code
        )

    def test_missing_frame_zero_candidates_does_not_search_later_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, reference_frame = _fixture(Path(temporary))
            anchors = load_collision_identity_anchors(
                request,
                entity_ids=("ball_1", "ball_2"),
                reference_frame=reference_frame,
                spatial_transform=_transform(),
            )

            with self.assertRaises(SceneAnalysisError) as captured:
                assign_collision_frame_zero_candidates(
                    anchors,
                    prediction_frame=np.full_like(reference_frame, 210),
                    candidates=(),
                    config=_binding_config(),
                )

        self.assertEqual(
            "collision_prediction_frame_zero_entities_missing",
            captured.exception.code,
        )

    def test_real_img1073_ignores_static_frozen_mask_locations(self) -> None:
        try:
            dataset = load_dataset(LATEST_DATASET, check_assets=True)
        except FileNotFoundError:
            self.skipTest("full collision media assets are not published")
        catalog = {case["case_id"]: case for case in dataset.cases}
        case_id = (
            "collision_supp_20260729_img_1073_"
            "two_ball_single_incident"
        )
        case = catalog[case_id]
        request = CaseEvaluationRequest(
            job={"job_id": f"collision_identity__{case_id}"},
            case=case,
            case_catalog=catalog,
            prediction=None,
            asset_root=dataset.asset_root,
            artifact_dir=Path("/tmp") / case_id,
            evaluator_config={},
        )
        reference_path = resolve_physics_reference(request)[0]
        capture = cv2.VideoCapture(str(reference_path))
        self.assertTrue(capture.isOpened())
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        divisor = gcd(source_width, source_height)
        unit_width = source_width // divisor
        unit_height = source_height // divisor
        multiplier = min(960 // unit_width, 540 // unit_height)
        target_size = (unit_width * multiplier, unit_height * multiplier)
        selected_indices = set(np.rint(
            np.linspace(0, frame_count - 1, 16)
        ).astype(int).tolist())
        frames = []
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected_indices:
                frames.append(
                    cv2.resize(
                        frame,
                        target_size,
                        interpolation=cv2.INTER_AREA,
                    )
                )
        capture.release()
        self.assertEqual(len(selected_indices), len(frames))

        anchors, metadata = (
            build_motion_validated_collision_reference_anchors(
                entity_ids=("ball_1", "ball_2"),
                reference_frames=frames,
                observation_config={
                    "track_top_ratio": 0.55,
                    "track_bottom_ratio": 0.93,
                    "blur_kernel": 5,
                    "blur_sigma": 1.2,
                    "hough_dp": 1.2,
                    "edge_threshold": 100,
                    "hough_accumulator_thresholds": [30, 26, 22, 18, 14],
                    "minimum_circle_radius_px": 5,
                    "maximum_circle_radius_px": 28,
                    "minimum_circle_center_distance_px": 12,
                    "maximum_candidates": 18,
                },
                localization_config={
                    "minimum_motion_fraction": 0.20,
                    "minimum_mean_change": 8.0,
                    "maximum_entity_y_spread_px": 32.0,
                    "maximum_radius_ratio": 4.5,
                    "minimum_gap_radius_fraction": 0.45,
                    "minimum_two_body_gap_radius_fraction": 0.75,
                    "minimum_set_score_margin": 0.20,
                },
            )
        )

        centers_x = [float(anchor.centroid_xy[0]) for anchor in anchors]
        self.assertTrue(all(value > 800.0 for value in centers_x))
        self.assertTrue(all(value < 930.0 for value in centers_x))
        self.assertGreater(metadata["set_score_margin"], 0.20)
        # The structurally valid frozen masks cover static ruler texture at
        # approximately x=322 and x=393 on the evaluator canvas. They must
        # never seed v6.
        self.assertTrue(all(abs(value - 322.0) > 400.0 for value in centers_x))
        self.assertTrue(all(abs(value - 393.0) > 400.0 for value in centers_x))


if __name__ == "__main__":
    unittest.main()
