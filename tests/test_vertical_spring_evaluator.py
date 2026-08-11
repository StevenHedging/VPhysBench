from __future__ import annotations

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from physbench.evaluation.common.frozen_subject import FrozenSubjectAnchor
from physbench.evaluation.scenes.vertical_spring_oscillator.observation import (
    observe_spring_topology,
    prompt_from_anchor,
    validate_mask_tube,
    validate_prediction_identity,
)


IDENTITY = {
    "minimum_anchor_iou": 0.60,
    "maximum_centroid_distance_radii": 1.50,
    "minimum_anchor_area_ratio": 0.50,
    "maximum_anchor_area_ratio": 1.80,
}

MASK_QUALITY = {
    "minimum_mask_pixels": 40,
    "maximum_mask_area_ratio": 0.10,
    "minimum_anchor_area_ratio": 0.50,
    "maximum_anchor_area_ratio": 1.80,
}

TOPOLOGY = {
    "canny_low_threshold": 40,
    "canny_high_threshold": 120,
    "corridor_half_width_radius_ratio": 1.50,
    "minimum_edge_pixels_per_row": 2,
    "endpoint_height_radius_ratio": 1.00,
}


def circle_mask(center_x: int, center_y: int, *, radius: int = 6) -> np.ndarray:
    """Rasterize a hand-specified disk without observation helpers."""
    y, x = np.ogrid[:96, :96]
    return (
        (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    ).astype(np.uint8)


def frozen_circle_anchor(
    *, center_x: int = 48, center_y: int = 72, radius: int = 6
) -> FrozenSubjectAnchor:
    mask = circle_mask(center_x, center_y, radius=radius)
    area = float(np.count_nonzero(mask))
    source = mask.copy()
    mask.flags.writeable = False
    source.flags.writeable = False
    return FrozenSubjectAnchor(
        case_id="spring_case",
        logical_entity_id="oscillator_ball",
        dataset_object_id="ball_0",
        entity_class="steel_ball",
        manifest_path=Path("mask_manifest.json"),
        npz_path=Path("ball_0.npz"),
        source_mask=source,
        mask=mask,
        centroid_xy=np.asarray([center_x, center_y], dtype=np.float64),
        area_px2=area,
        equivalent_radius_px=float(math.sqrt(area / math.pi)),
        provenance={"policy": "frozen-like-test-anchor"},
    )


def spring_frame(
    ball_mask: np.ndarray, *, attached: bool, ruler_and_border: bool = True
) -> np.ndarray:
    """Draw a literal current-frame zig-zag spring and unrelated scene edges."""
    frame = np.zeros((*ball_mask.shape, 3), dtype=np.uint8)
    ys, xs = np.where(ball_mask > 0)
    center_x = int(round(float(xs.mean())))
    ball_top = int(ys.min())
    if ruler_and_border:
        cv2.rectangle(frame, (1, 1), (94, 94), (180, 180, 180), 1)
        cv2.line(frame, (10, 4), (10, 88), (255, 255, 255), 2)
        for y in range(8, 72, 8):
            cv2.line(frame, (10, y), (16, y), (255, 255, 255), 1)
    if attached:
        points = [(center_x, 2)]
        direction = -1
        for y in range(6, ball_top, 4):
            points.append((center_x + direction * 4, y))
            direction *= -1
        points.append((center_x, ball_top))
        cv2.polylines(
            frame,
            [np.asarray(points, dtype=np.int32)],
            False,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_8,
        )
    frame[ball_mask > 0] = (160, 160, 160)
    return frame


class VerticalSpringObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = frozen_circle_anchor()

    def test_prompt_expands_around_mask_and_has_one_positive_centroid(self) -> None:
        """Would fail if the prompt box, expansion, or centroid point regresses."""
        prompt = prompt_from_anchor(self.anchor)
        ys, xs = np.where(self.anchor.mask > 0)
        x0, y0, x1, y1 = prompt.box_xyxy.tolist()

        self.assertEqual(0, prompt.frame_index)
        self.assertLess(x0, float(xs.min()))
        self.assertLess(y0, float(ys.min()))
        self.assertGreater(x1, float(xs.max()))
        self.assertGreater(y1, float(ys.max()))
        self.assertGreaterEqual(x0, 0.0)
        self.assertGreaterEqual(y0, 0.0)
        self.assertLessEqual(x1, 95.0)
        self.assertLessEqual(y1, 95.0)
        np.testing.assert_allclose(prompt.points_xy, [[48.0, 72.0]])
        np.testing.assert_array_equal(prompt.point_labels, [1])
        self.assertFalse(prompt.box_xyxy.flags.writeable)
        self.assertFalse(prompt.points_xy.flags.writeable)
        self.assertFalse(prompt.point_labels.flags.writeable)

    def test_identity_accepts_exact_anchor_with_auditable_thresholds(self) -> None:
        """Would fail if exact identity or threshold provenance is omitted."""
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=self.anchor.mask.copy(),
            config=IDENTITY,
        )
        self.assertTrue(decision["accepted"])
        self.assertGreater(decision["anchor_iou"], 0.99)
        self.assertAlmostEqual(0.0, decision["centroid_distance_radii"])
        self.assertAlmostEqual(1.0, decision["area_ratio"])
        self.assertEqual(IDENTITY, dict(decision["thresholds"]))
        for key in ("anchor_iou", "centroid_distance_radii", "area_ratio"):
            self.assertTrue(math.isfinite(decision[key]))
        with self.assertRaises(TypeError):
            decision["accepted"] = False

    def test_identity_rejects_empty_prediction(self) -> None:
        """Would fail if an empty prediction bypasses the identity gate."""
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=np.zeros_like(self.anchor.mask),
            config=IDENTITY,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(0.0, decision["anchor_iou"])
        self.assertEqual(0.0, decision["area_ratio"])
        self.assertTrue(math.isfinite(decision["centroid_distance_radii"]))

    def test_identity_rejects_displaced_prediction(self) -> None:
        """Would fail if centroid displacement and overlap checks are removed."""
        displaced = circle_mask(68, 72)
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=displaced,
            config=IDENTITY,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(0.0, decision["anchor_iou"])
        self.assertGreater(
            decision["centroid_distance_radii"],
            IDENTITY["maximum_centroid_distance_radii"],
        )

    def test_identity_rejects_extreme_area_prediction(self) -> None:
        """Would fail if the frozen-anchor area-ratio gate is removed."""
        bloated = circle_mask(48, 72, radius=20)
        decision = validate_prediction_identity(
            anchor_mask=self.anchor.mask,
            prediction_mask=bloated,
            config=IDENTITY,
        )
        self.assertFalse(decision["accepted"])
        self.assertGreater(
            decision["area_ratio"], IDENTITY["maximum_anchor_area_ratio"]
        )

    def test_identity_mask_tube_normalizes_and_blanks_unavailable_or_bad_masks(
        self,
    ) -> None:
        """Would fail if nonbinary, unavailable, wrong-canvas, or NaN masks leak."""
        nonbinary = self.anchor.mask.astype(np.float32) * 7.5
        wrong_canvas = np.ones((48, 48), dtype=np.uint8)
        nonfinite = self.anchor.mask.astype(np.float32)
        nonfinite[0, 0] = np.nan
        masks = validate_mask_tube(
            [
                nonbinary,
                self.anchor.mask.astype(bool),
                self.anchor.mask,
                wrong_canvas,
                nonfinite,
            ],
            availability=[True, True, False, True, True],
            anchor=self.anchor,
            config=MASK_QUALITY,
        )

        self.assertIsInstance(masks, tuple)
        self.assertEqual(5, len(masks))
        self.assertEqual({0, 255}, set(np.unique(masks[0]).tolist()))
        np.testing.assert_array_equal(masks[0] > 0, self.anchor.mask > 0)
        np.testing.assert_array_equal(masks[1] > 0, self.anchor.mask > 0)
        for mask in masks[2:]:
            self.assertEqual(0, np.count_nonzero(mask))
        for mask in masks:
            self.assertEqual(np.uint8, mask.dtype)
            self.assertEqual((96, 96), mask.shape)
            self.assertFalse(mask.flags.writeable)

    def test_identity_mask_tube_blanks_area_failures_and_bad_config(
        self,
    ) -> None:
        """Would fail if pixel/frame/anchor area gates or config checks fail open."""
        tiny = np.zeros((96, 96), dtype=np.uint8)
        tiny[10:12, 10:12] = 1
        bloated = circle_mask(48, 48, radius=30)
        masks = validate_mask_tube(
            [tiny, bloated],
            availability=[True, True],
            anchor=self.anchor,
            config=MASK_QUALITY,
        )
        self.assertEqual([0, 0], [int(np.count_nonzero(mask)) for mask in masks])

        invalid = dict(MASK_QUALITY, maximum_mask_area_ratio=float("nan"))
        with self.assertRaises(ValueError):
            validate_mask_tube(
                [self.anchor.mask],
                availability=[True],
                anchor=self.anchor,
                config=invalid,
            )

    def test_topology_uses_each_current_ball_corridor(self) -> None:
        """Would fail if topology localizes from a frozen or future ball position."""
        first = circle_mask(48, 72)
        second = circle_mask(68, 66)
        result = observe_spring_topology(
            [spring_frame(first, attached=True), spring_frame(second, attached=True)],
            [first, second],
            availability=[True, True],
            config=TOPOLOGY,
        )
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.all(result.row_coverage > 0.80))
        np.testing.assert_array_equal(result.endpoint_support, [1.0, 1.0])
        self.assertGreater(result.score, 0.90)

    def test_topology_ignores_background_ruler_and_border_when_spring_removed(
        self,
    ) -> None:
        """Would fail if unrelated full-frame edges can earn topology credit."""
        mask = circle_mask(48, 72)
        attached = observe_spring_topology(
            [spring_frame(mask, attached=True)],
            [mask],
            availability=[True],
            config=TOPOLOGY,
        )
        removed = observe_spring_topology(
            [spring_frame(mask, attached=False)],
            [mask],
            availability=[True],
            config=TOPOLOGY,
        )

        self.assertGreater(attached.score, 0.90)
        self.assertLess(removed.row_coverage[0], 0.05)
        self.assertEqual(0.0, removed.endpoint_support[0])
        self.assertLess(removed.score, 0.05)

    def test_topology_fails_closed_per_frame_and_returns_immutable_finite_arrays(
        self,
    ) -> None:
        """Would fail if bad canvases, unavailable frames, or mutable outputs leak."""
        mask = circle_mask(48, 72)
        wrong_frame = np.zeros((48, 48, 3), dtype=np.uint8)
        result = observe_spring_topology(
            [wrong_frame, spring_frame(mask, attached=True)],
            [mask, mask],
            availability=[True, False],
            config=TOPOLOGY,
        )
        np.testing.assert_array_equal(result.valid, [False, False])
        np.testing.assert_array_equal(result.row_coverage, [0.0, 0.0])
        np.testing.assert_array_equal(result.endpoint_support, [0.0, 0.0])
        self.assertEqual(0.0, result.score)
        self.assertTrue(np.all(np.isfinite(result.row_coverage)))
        self.assertTrue(np.all(np.isfinite(result.endpoint_support)))
        self.assertTrue(math.isfinite(result.score))
        self.assertFalse(result.row_coverage.flags.writeable)
        self.assertFalse(result.endpoint_support.flags.writeable)
        self.assertFalse(result.valid.flags.writeable)

        invalid = dict(TOPOLOGY, endpoint_height_radius_ratio=float("inf"))
        with self.assertRaises(ValueError):
            observe_spring_topology(
                [spring_frame(mask, attached=True)],
                [mask],
                availability=[True],
                config=invalid,
            )


if __name__ == "__main__":
    unittest.main()
