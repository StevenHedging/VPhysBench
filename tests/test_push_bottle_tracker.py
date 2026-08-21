from __future__ import annotations

import unittest

import cv2
import numpy as np


class PushBottleTrackerTests(unittest.TestCase):
    def test_clean_binary_mask_removes_detached_warp_artifacts(self) -> None:
        from physbench.reference_observations.curation.push_bottle import (
            clean_binary_mask,
        )

        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(mask, (20, 15), (55, 68), 1, -1)
        mask[3, 4] = 1
        mask[4, 4] = 1

        cleaned = clean_binary_mask(mask, opening_kernel=3)

        count, _, _, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        self.assertEqual(2, count)
        self.assertEqual(0, int(cleaned[3, 4]))
        self.assertGreater(int(cleaned.sum()), 1_800)

    def test_pose_keyframes_keep_directed_bottle_orientation(self) -> None:
        from physbench.reference_observations.curation.push_bottle import (
            apply_pose_keyframes,
        )

        template = np.zeros((180, 220), dtype=np.uint8)
        cv2.rectangle(template, (91, 45), (128, 145), 1, -1)
        # An asymmetric cap makes a 180-degree direction error observable.
        cv2.rectangle(template, (98, 30), (121, 48), 1, -1)
        original = np.stack([template.copy() for _ in range(5)])
        keyframes = {
            1: {"axis_degrees": 160.0, "center_xy": [105.0, 105.0], "scale": 1.0},
            3: {"axis_degrees": 178.0, "center_xy": [125.0, 115.0], "scale": 1.0},
        }

        output = apply_pose_keyframes(
            original,
            template_mask=template,
            keyframes=keyframes,
        )

        # Frames outside the configured interval are untouched.
        self.assertTrue(np.array_equal(template, output[0]))
        self.assertTrue(np.array_equal(template, output[4]))
        # Pose interpolation moves the silhouette smoothly between keyframes.
        y1, x1 = np.nonzero(output[1])
        y2, x2 = np.nonzero(output[2])
        y3, x3 = np.nonzero(output[3])
        self.assertAlmostEqual(105.0, float(x1.mean()), delta=0.5)
        self.assertAlmostEqual(115.0, float(x2.mean()), delta=0.5)
        self.assertAlmostEqual(125.0, float(x3.mean()), delta=0.5)
        # The top cap must follow the direct, unwrapped rotation.  Wrapping the
        # undirected PCA angle at 180 degrees would flip it to the other end.
        template_y, template_x = np.nonzero(template)
        points = np.stack((template_x, template_y), axis=1).astype(np.float64)
        centre = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(points.T))
        direction = vectors[:, int(np.argmax(values))]
        template_axis = float(
            np.degrees(np.arctan2(direction[1], direction[0])) % 180.0
        )
        transform = cv2.getRotationMatrix2D(
            tuple(centre), template_axis - 178.0, 1.0
        )
        transform[:, 2] += np.asarray((125.0, 115.0)) - (
            transform @ np.asarray((*centre, 1.0))
        )
        expected = cv2.warpAffine(
            template,
            transform,
            (template.shape[1], template.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        self.assertTrue(np.array_equal(expected, output[3]))

    def test_dense_rigid_tracking_survives_rotation_and_partial_occlusion(self) -> None:
        from physbench.reference_observations.curation.push_bottle import (
            track_rigid_mask_sequence,
        )

        height = width = 160
        initial = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(initial, (62, 38), (96, 122), 1, -1)
        cv2.circle(initial, (79, 38), 12, 1, -1)

        texture = np.zeros((height, width, 3), dtype=np.uint8)
        texture[:] = (225, 225, 225)
        for y in range(44, 120, 9):
            cv2.line(texture, (64, y), (94, y + 3), (30, 120, 220), 2)
        cv2.circle(texture, (79, 39), 8, (245, 245, 245), -1)

        frames: list[np.ndarray] = []
        truth: list[np.ndarray] = []
        # At 240 fps the real bottle rotates by well under one degree per
        # source frame.  The tracker deliberately consumes every source frame
        # instead of jumping directly between the 24 fps observation samples.
        for index in range(111):
            angle = 0.6 * index
            transform = cv2.getRotationMatrix2D((79, 80), angle, 1.0)
            transform[:, 2] += (0.25 * index, 0.08 * index)
            mask = cv2.warpAffine(
                initial,
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
            )
            object_pixels = cv2.warpAffine(texture, transform, (width, height))
            frame = np.full((height, width, 3), 225, dtype=np.uint8)
            frame[mask.astype(bool)] = object_pixels[mask.astype(bool)]
            # A skin-coloured distractor covers the left edge during the turn.
            if 44 <= index <= 78:
                cv2.rectangle(
                    frame,
                    (45 + index // 4, 70),
                    (68 + index // 4, 105),
                    (90, 135, 190),
                    -1,
                )
            frames.append(frame)
            truth.append(mask)

        tracked = track_rigid_mask_sequence(frames, initial)
        self.assertEqual((111, height, width), tracked.shape)
        ious = []
        for predicted, expected in zip(tracked, truth):
            intersection = np.count_nonzero(predicted & expected)
            union = np.count_nonzero(predicted | expected)
            ious.append(intersection / union)
        self.assertGreater(min(ious), 0.80)
        self.assertGreater(float(np.median(ious)), 0.86)

    def test_edge_and_interior_evidence_corrects_late_pose_drift(self) -> None:
        from physbench.reference_observations.curation.push_bottle import (
            fit_rigid_mask_to_evidence,
        )

        height, width = 220, 260
        template = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(template, (98, 52), (136, 172), 1, -1)
        cv2.circle(template, (117, 52), 15, 1, -1)
        initial_y, initial_x = np.nonzero(template)
        centre = (float(initial_x.mean()), float(initial_y.mean()))

        actual_transform = cv2.getRotationMatrix2D(centre, -82.0, 1.0)
        mapped = actual_transform @ np.asarray((*centre, 1.0))
        actual_transform[:, 2] += np.asarray((145.0, 142.0)) - mapped
        actual = cv2.warpAffine(
            template, actual_transform, (width, height), flags=cv2.INTER_NEAREST
        )

        predicted_transform = cv2.getRotationMatrix2D(centre, -66.0, 1.0)
        mapped = predicted_transform @ np.asarray((*centre, 1.0))
        predicted_transform[:, 2] += np.asarray((137.0, 112.0)) - mapped
        predicted = cv2.warpAffine(
            template, predicted_transform, (width, height), flags=cv2.INTER_NEAREST
        )

        frame = np.full((height, width, 3), 225, dtype=np.uint8)
        frame[actual.astype(bool)] = (175, 175, 175)
        evidence = actual.copy()
        evidence[:142] = 0
        frame[evidence.astype(bool)] = (30, 150, 210)

        fitted = fit_rigid_mask_to_evidence(
            frame=frame,
            template_mask=template,
            predicted_mask=predicted,
            interior_evidence=evidence,
        )
        intersection = np.count_nonzero(fitted & actual)
        union = np.count_nonzero(fitted | actual)
        self.assertGreater(intersection / union, 0.94)


if __name__ == "__main__":
    unittest.main()
