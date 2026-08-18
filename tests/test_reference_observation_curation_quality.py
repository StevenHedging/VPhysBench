from __future__ import annotations

import unittest

import numpy as np


class ReferenceObservationCurationQualityTests(unittest.TestCase):
    @staticmethod
    def _quality_api():
        from physbench.reference_observations.curation import quality

        return quality

    def test_fast_diagnostics_flags_long_unresolved_run(self) -> None:
        quality = self._quality_api()
        result = quality.fast_entity_diagnostics(
            area=np.asarray([100, 100, 0, 0, 0], dtype=np.int64),
            centroid_xy=np.asarray(
                [[0, 0], [1, 0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dtype=np.float32,
            ),
            state=np.asarray([0, 0, 3, 3, 3], dtype=np.uint8),
            physical_time=np.arange(5, dtype=np.float64) / 24.0,
            scene_id="collision_1d",
        )

        unresolved = [event for event in result.events if event.code == "unresolved_run"]
        self.assertEqual([(2, 4)], [event.index_range for event in unresolved])
        self.assertIn("unresolved_run", result.review_reasons)

    def test_projectile_translation_alone_is_not_area_instability(self) -> None:
        quality = self._quality_api()
        result = quality.fast_entity_diagnostics(
            area=np.full(5, 100, dtype=np.int64),
            centroid_xy=np.asarray(
                [[0, 0], [20, 2], [40, 6], [60, 12], [80, 20]],
                dtype=np.float32,
            ),
            state=np.zeros(5, dtype=np.uint8),
            physical_time=np.arange(5, dtype=np.float64) / 24.0,
            scene_id="parabolic_motion",
        )

        self.assertNotIn("area_instability", result.review_reasons)
        self.assertEqual((), tuple(event.code for event in result.events))

    def test_sudden_visible_area_spike_is_localized_for_review(self) -> None:
        quality = self._quality_api()
        result = quality.fast_entity_diagnostics(
            area=np.asarray([100, 102, 101, 520, 103, 101], dtype=np.int64),
            centroid_xy=np.asarray(
                [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
                dtype=np.float32,
            ),
            state=np.zeros(6, dtype=np.uint8),
            physical_time=np.arange(6, dtype=np.float64) / 24.0,
            scene_id="pendulum",
        )

        self.assertIn("area_instability", result.review_reasons)
        area_events = [event for event in result.events if event.code == "area_jump"]
        self.assertEqual([(2, 3), (3, 4)], [event.index_range for event in area_events])

    def test_pixel_diagnostics_flags_fragmented_visible_mask(self) -> None:
        quality = self._quality_api()
        masks = np.zeros((2, 12, 12), dtype=np.uint8)
        masks[0, 3:7, 3:7] = 1
        masks[1, 2:5, 2:5] = 1
        masks[1, 8:10, 8:10] = 1

        result = quality.pixel_entity_diagnostics(
            masks=masks,
            state=np.zeros(2, dtype=np.uint8),
            scene_id="pendulum",
        )

        fragmented = [event for event in result.events if event.code == "fragmented_mask"]
        self.assertEqual([(1, 1)], [event.index_range for event in fragmented])
        self.assertIn("fragmented_mask", result.review_reasons)

    def test_anchor_must_be_bit_exact_with_tube_observation_zero(self) -> None:
        quality = self._quality_api()
        anchor = np.zeros((8, 8), dtype=np.uint8)
        anchor[2:5, 2:5] = 1
        tube_zero = anchor.copy()
        tube_zero[4, 4] = 0

        result = quality.audit_anchor_tube_zero(anchor, tube_zero)

        self.assertEqual("anchor_tube_zero_mismatch", result.code)
        self.assertEqual(1, result.statistics["xor_pixels"])

    def test_mask_reductions_must_equal_stored_trajectory(self) -> None:
        quality = self._quality_api()
        masks = np.zeros((1, 8, 8), dtype=np.uint8)
        masks[0, 2:5, 3:7] = 1

        issues = quality.audit_mask_trajectory_reductions(
            masks=masks,
            area_pixels=np.asarray([11], dtype=np.int64),
            centroid_xy=np.asarray([[4.5, 3.0]], dtype=np.float32),
            bbox_xyxy=np.asarray([[3.0, 2.0, 6.0, 4.0]], dtype=np.float32),
        )

        self.assertEqual(["trajectory_area_mismatch"], [issue.code for issue in issues])


if __name__ == "__main__":
    unittest.main()
