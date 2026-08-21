from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np


def _departing_square_frames() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for x_start in (18, 24, 30, 36, 42):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[24:34, x_start : x_start + 10] = 255
        frames.append(frame)
    return frames


class _SeedEncodedSegmenter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def segment_instances(self, frames, *, prompts, exclusive_masks):
        seed = prompts[0].frame_index
        self.calls.append((seed, exclusive_masks))
        output = []
        for _prompt in prompts:
            tube = []
            for _frame in frames:
                mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
                mask[10:14, seed : seed + 4] = 255
                tube.append(mask)
            output.append(tube)
        return output, {"seed_frame": seed, "backend": "fixture"}


class ReferenceObservationCurationTrackingTests(unittest.TestCase):
    def test_audit_config_pins_large_repair_model_and_joint_competition(self) -> None:
        config = json.loads(
            Path("configs/reference_observations/audit_v1.json").read_text()
        )
        self.assertEqual("facebook/sam2.1-hiera-large", config["model_id"])
        self.assertTrue(config["exclusive_masks"])
        self.assertEqual(8, config["gpu_shards"])

    @staticmethod
    def _anchors_api():
        from physbench.reference_observations.curation import anchors

        return anchors

    @staticmethod
    def _overrides_api():
        from physbench.reference_observations.curation import overrides

        return overrides

    @staticmethod
    def _tracking_api():
        from physbench.reference_observations.curation import tracking

        return tracking

    def test_anchor_candidates_are_independent_of_existing_mask_pixels(self) -> None:
        anchors = self._anchors_api()
        frames = _departing_square_frames()
        wrong_a = np.zeros((64, 64), dtype=np.uint8)
        wrong_a[24:34, 18:28] = 1
        wrong_b = np.zeros((64, 64), dtype=np.uint8)
        wrong_b[50:, 50:] = 1

        candidate_a = anchors.build_independent_anchor_candidates(
            frames,
            scene_id="parabolic_motion",
            expected_count=1,
            comparison_masks={"object_1": wrong_a},
        )[0]
        candidate_b = anchors.build_independent_anchor_candidates(
            frames,
            scene_id="parabolic_motion",
            expected_count=1,
            comparison_masks={"object_1": wrong_b},
        )[0]

        np.testing.assert_array_equal(candidate_a.mask, candidate_b.mask)
        self.assertGreater(candidate_a.centroid_xy[0], 12.0)
        self.assertLess(candidate_a.centroid_xy[0], 32.0)
        self.assertNotEqual(
            candidate_a.comparison_iou,
            candidate_b.comparison_iou,
        )

    def test_anchor_candidates_fail_when_motion_cannot_explain_subject_count(self) -> None:
        anchors = self._anchors_api()
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]

        with self.assertRaisesRegex(ValueError, "independent anchor candidates"):
            anchors.build_independent_anchor_candidates(
                frames,
                scene_id="pendulum",
                expected_count=1,
                comparison_masks={"object_1": np.ones((32, 32), np.uint8)},
            )

    def test_circular_anchor_candidates_use_disk_internal_block_geometry(self) -> None:
        anchors = self._anchors_api()
        frame = np.full((160, 200, 3), (30, 60, 100), np.uint8)
        cv2 = __import__("cv2")
        cv2.circle(frame, (100, 80), 72, (210, 155, 65), -1)
        cv2.rectangle(frame, (45, 58), (72, 91), (225, 225, 225), -1)
        cv2.rectangle(frame, (125, 35), (157, 100), (80, 135, 190), -1)
        candidates = anchors.build_independent_anchor_candidates(
            [frame.copy() for _ in range(3)],
            scene_id="uniform_circular_motion",
            expected_count=2,
            comparison_masks=None,
        )
        self.assertLess(candidates[0].centroid_xy[0], candidates[1].centroid_xy[0])
        self.assertGreater(candidates[0].area_pixels, 500)
        self.assertGreater(candidates[1].area_pixels, 500)

    def test_collision_anchor_candidates_fall_back_to_static_circle_geometry(self) -> None:
        anchors = self._anchors_api()
        cv2 = __import__("cv2")
        frame = np.full((120, 240, 3), 210, np.uint8)
        cv2.line(frame, (0, 88), (239, 88), (80, 80, 80), 3)
        cv2.circle(frame, (55, 72), 13, (30, 30, 30), -1)
        cv2.circle(frame, (170, 70), 18, (45, 45, 45), -1)
        candidates = anchors.build_independent_anchor_candidates(
            [frame.copy() for _ in range(3)],
            scene_id="collision_1d",
            expected_count=2,
            comparison_masks=None,
        )
        self.assertLess(candidates[0].centroid_xy[0], candidates[1].centroid_xy[0])
        self.assertAlmostEqual(55.0, candidates[0].centroid_xy[0], delta=8.0)
        self.assertAlmostEqual(170.0, candidates[1].centroid_xy[0], delta=8.0)

    def test_collision_anchor_candidates_ignore_oversized_static_ring(self) -> None:
        anchors = self._anchors_api()
        cv2 = __import__("cv2")
        frames = []
        for offset in (0, 8, 16):
            frame = np.full((160, 320, 3), 210, np.uint8)
            cv2.line(frame, (0, 118), (319, 118), (80, 80, 80), 3)
            cv2.circle(frame, (70, 102), 10, (25, 25, 25), -1)
            cv2.circle(frame, (220, 101), 14, (40, 40, 40), -1)
            cv2.circle(frame, (150 + offset, 82), 38, (30, 30, 30), 4)
            frames.append(frame)

        candidates = anchors.build_independent_anchor_candidates(
            frames,
            scene_id="collision_1d",
            expected_count=2,
            comparison_masks=None,
        )

        self.assertAlmostEqual(70.0, candidates[0].centroid_xy[0], delta=8.0)
        self.assertAlmostEqual(220.0, candidates[1].centroid_xy[0], delta=8.0)
        self.assertTrue(all(candidate.area_pixels < 1500 for candidate in candidates))

    def test_override_rejects_duplicate_object_prompt_on_same_frame(self) -> None:
        overrides = self._overrides_api()
        value = {
            "schema_version": "1.0",
            "case_id": "case_a",
            "anchor_prompts": [],
            "corrections": [
                {
                    "object_id": "object_1",
                    "frame_index": 3,
                    "box_xyxy": [1, 2, 8, 9],
                    "points_xy": [[5, 5]],
                    "point_labels": [1],
                },
                {
                    "object_id": "object_1",
                    "frame_index": 3,
                    "box_xyxy": [2, 3, 9, 10],
                    "points_xy": [[6, 6]],
                    "point_labels": [1],
                },
            ],
            "lifecycle": [],
            "scene_refinement": "none",
        }

        with self.assertRaisesRegex(ValueError, "duplicate correction prompt"):
            overrides.validate_override(value)

    def test_tracker_uses_nearest_reviewed_seed_and_joint_instance_competition(self) -> None:
        anchors = self._anchors_api()
        overrides = self._overrides_api()
        tracking = self._tracking_api()
        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(5)]
        anchor_mask = np.zeros((24, 24), dtype=np.uint8)
        anchor_mask[10:14, 0:4] = 1
        anchor = anchors.AnchorCandidate.from_mask("object_1", anchor_mask)
        correction = overrides.CorrectionPrompt(
            object_id="object_1",
            frame_index=4,
            box_xyxy=(4.0, 10.0, 7.0, 13.0),
            points_xy=((5.5, 11.5),),
            point_labels=(1,),
        )
        segmenter = _SeedEncodedSegmenter()
        tracker = tracking.CuratedSam2Tracker(segmenter)

        result = tracker.track(
            frames,
            anchors=(anchor,),
            corrections=(correction,),
            lifecycle=(),
        )

        tube = result.masks_by_object["object_1"]
        self.assertTrue(np.all(tube[:3, 10:14, 0:4] == 1))
        self.assertTrue(np.all(tube[3:, 10:14, 4:8] == 1))
        self.assertEqual([(0, True), (4, True)], segmenter.calls)
        self.assertEqual([0, 0, 0, 4, 4], result.seed_frame_by_observation.tolist())

    def test_lifecycle_override_zeroes_masks_after_reviewed_exit(self) -> None:
        anchors = self._anchors_api()
        overrides = self._overrides_api()
        tracking = self._tracking_api()
        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(5)]
        anchor_mask = np.zeros((24, 24), dtype=np.uint8)
        anchor_mask[10:14, 0:4] = 1
        tracker = tracking.CuratedSam2Tracker(_SeedEncodedSegmenter())

        result = tracker.track(
            frames,
            anchors=(anchors.AnchorCandidate.from_mask("object_1", anchor_mask),),
            corrections=(),
            lifecycle=(
                overrides.LifecycleOverride(
                    object_id="object_1",
                    start_index=3,
                    end_index=4,
                    state=2,
                ),
            ),
        )

        self.assertEqual([0, 0, 0, 2, 2], result.states_by_object["object_1"].tolist())
        self.assertEqual(0, int(result.masks_by_object["object_1"][3:].sum()))

    def test_trailing_empty_masks_after_boundary_contact_are_out_of_frame(self) -> None:
        tracking = self._tracking_api()
        masks = np.zeros((5, 12, 16), np.uint8)
        masks[0, 4:8, 8:12] = 1
        masks[1, 4:8, 11:15] = 1
        masks[2, 4:8, 14:16] = 1
        states = np.where(masks.reshape(5, -1).any(axis=1), 0, 3).astype(np.uint8)

        resolved = tracking.resolve_trailing_boundary_exit(masks, states)

        self.assertEqual([0, 0, 0, 2, 2], resolved.tolist())

    def test_interior_tracking_loss_remains_unresolved(self) -> None:
        tracking = self._tracking_api()
        masks = np.zeros((5, 12, 16), np.uint8)
        masks[:3, 4:8, 4:8] = 1
        states = np.where(masks.reshape(5, -1).any(axis=1), 0, 3).astype(np.uint8)

        resolved = tracking.resolve_trailing_boundary_exit(masks, states)

        self.assertEqual([0, 0, 0, 3, 3], resolved.tolist())

    def test_accelerating_object_predicted_beyond_boundary_is_out_of_frame(self) -> None:
        tracking = self._tracking_api()
        masks = np.zeros((6, 96, 128), np.uint8)
        masks[0, 8:16, 72:80] = 1
        masks[1, 28:36, 62:70] = 1
        masks[2, 62:70, 52:60] = 1
        states = np.where(masks.reshape(6, -1).any(axis=1), 0, 3).astype(np.uint8)

        resolved = tracking.resolve_trailing_predicted_exit(masks, states)

        self.assertEqual([0, 0, 0, 2, 2, 2], resolved.tolist())

    def test_motion_extrapolation_does_not_hide_an_interior_tracking_loss(self) -> None:
        tracking = self._tracking_api()
        masks = np.zeros((6, 96, 128), np.uint8)
        masks[0, 36:44, 20:28] = 1
        masks[1, 36:44, 30:38] = 1
        masks[2, 36:44, 40:48] = 1
        states = np.where(masks.reshape(6, -1).any(axis=1), 0, 3).astype(np.uint8)

        resolved = tracking.resolve_trailing_predicted_exit(masks, states)

        self.assertEqual([0, 0, 0, 3, 3, 3], resolved.tolist())

    def test_motion_extrapolation_requires_three_contiguous_visible_samples(self) -> None:
        tracking = self._tracking_api()
        masks = np.zeros((6, 96, 128), np.uint8)
        masks[0, 8:16, 72:80] = 1
        masks[2, 62:70, 52:60] = 1
        states = np.where(masks.reshape(6, -1).any(axis=1), 0, 3).astype(np.uint8)

        resolved = tracking.resolve_trailing_predicted_exit(masks, states)

        self.assertEqual([0, 3, 0, 3, 3, 3], resolved.tolist())

    def test_round_subject_refinement_removes_thin_attached_tether(self) -> None:
        tracking = self._tracking_api()
        cv2 = __import__("cv2")
        masks = np.zeros((3, 80, 80), np.uint8)
        for index, center_x in enumerate((30, 36, 42)):
            cv2.circle(masks[index], (center_x, 52), 9, 1, -1)
            cv2.line(masks[index], (center_x, 8), (center_x, 43), 1, 3)
        states = np.zeros(3, np.uint8)

        refined, refined_states = tracking.isolate_compact_round_subject(
            masks, states
        )

        self.assertTrue(all(refined[i, 52, x] == 1 for i, x in enumerate((30, 36, 42))))
        self.assertEqual(0, int(refined[:, :35].sum()))
        self.assertTrue(np.all(refined.reshape(3, -1).sum(axis=1) < 350))
        self.assertEqual([0, 0, 0], refined_states.tolist())

    def test_round_subject_refinement_fills_only_short_internal_gap(self) -> None:
        tracking = self._tracking_api()
        cv2 = __import__("cv2")
        masks = np.zeros((5, 64, 64), np.uint8)
        cv2.circle(masks[0], (20, 32), 7, 1, -1)
        cv2.circle(masks[1], (24, 32), 7, 1, -1)
        cv2.circle(masks[3], (32, 32), 7, 1, -1)
        cv2.circle(masks[4], (36, 32), 7, 1, -1)
        states = np.asarray([0, 0, 3, 0, 0], np.uint8)

        refined, refined_states = tracking.isolate_compact_round_subject(
            masks, states, maximum_internal_gap=1
        )

        self.assertGreater(int(refined[2].sum()), 100)
        self.assertAlmostEqual(28.0, float(np.nonzero(refined[2])[1].mean()), delta=1.0)
        self.assertEqual([0, 0, 0, 0, 0], refined_states.tolist())


if __name__ == "__main__":
    unittest.main()
