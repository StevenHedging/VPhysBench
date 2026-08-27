from __future__ import annotations

import unittest

import numpy as np


def _disk(
    center_xy: tuple[int, int],
    radius: int,
    *,
    shape: tuple[int, int] = (120, 180),
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cx, cy = center_xy
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2)


class Sam31GtIdentityTests(unittest.TestCase):
    @staticmethod
    def _api():
        from physbench.reference_observations.curation import sam31_gt

        return sam31_gt

    def _candidate(
        self,
        candidate_id: str,
        center_xy: tuple[int, int],
        radius: int,
        confidence: float = 0.9,
    ):
        api = self._api()
        return api.GtCandidate.from_mask(
            candidate_id=candidate_id,
            prompt="small round object",
            backend_object_id=int(candidate_id.rsplit("-", 1)[-1]),
            mask=_disk(center_xy, radius),
            confidence=confidence,
        )

    def test_row_major_candidates_groups_rows_before_sorting_x(self) -> None:
        api = self._api()
        bottom_left = self._candidate("candidate-3", (30, 85), 8)
        top_right = self._candidate("candidate-2", (130, 30), 8)
        top_left = self._candidate("candidate-1", (45, 34), 8)

        ordered = api.row_major_candidates((bottom_left, top_right, top_left))

        self.assertEqual(
            ["candidate-1", "candidate-2", "candidate-3"],
            [item.candidate_id for item in ordered],
        )

    def test_binding_rejects_missing_caption_symbol(self) -> None:
        api = self._api()
        physics = {
            "objects": {
                "object_1": {
                    "mass": {"symbol": "m_1", "value": 0.01},
                    "radius": {"symbol": "r_1", "value": 0.01},
                    "initial_velocity": {"symbol": "v_1", "value": 0.1},
                },
                "object_2": {
                    "mass": {"symbol": "m_2", "value": 0.02},
                    "radius": {"symbol": "r_2", "value": 0.02},
                    "initial_velocity": {"symbol": "v_2", "value": 0.0},
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "caption misses v_2"):
            api.validate_physics_caption_binding(
                physics,
                "The balls have m_1, r_1, v_1, m_2, and r_2.",
            )

    def test_binding_returns_contiguous_object_ids_and_radii(self) -> None:
        api = self._api()
        physics = {
            "physics": {
                "objects": {
                    "object_1": {
                        "mass": {"symbol": "m_1", "value": 0.01},
                        "radius": {"symbol": "r_1", "value": 0.0075},
                        "initial_velocity": {"symbol": "v_1", "value": 0.1},
                    },
                    "object_2": {
                        "mass": {"symbol": "m_2", "value": 0.02},
                        "radius": {"symbol": "r_2", "value": 0.0125},
                        "initial_velocity": {"symbol": "v_2", "value": 0.0},
                    },
                }
            }
        }

        binding = api.validate_physics_caption_binding(
            physics,
            "m_1 r_1 v_1 m_2 r_2 v_2",
        )

        self.assertEqual(("object_1", "object_2"), binding.object_ids)
        self.assertEqual((0.0075, 0.0125), binding.radii_m)

    def test_deduplication_keeps_higher_confidence_same_subject(self) -> None:
        api = self._api()
        lower = self._candidate("candidate-1", (50, 70), 10, 0.72)
        higher = api.GtCandidate.from_mask(
            candidate_id="candidate-2",
            prompt="steel ball",
            backend_object_id=2,
            mask=_disk((51, 70), 10),
            confidence=0.94,
        )

        unique = api.deduplicate_candidates((lower, higher))

        self.assertEqual(("candidate-2",), tuple(item.candidate_id for item in unique))

    def test_selector_ignores_high_confidence_fastener_outside_collision_row(self) -> None:
        api = self._api()
        candidates = (
            self._candidate("candidate-1", (45, 84), 8, 0.89),
            self._candidate("candidate-2", (105, 82), 12, 0.91),
            self._candidate("candidate-3", (142, 28), 7, 0.99),
        )

        selected = api.select_collision_candidates(
            candidates,
            expected_radii=(0.008, 0.012),
            ambiguity_margin=0.01,
        )

        self.assertEqual(
            ("candidate-1", "candidate-2"),
            tuple(item.candidate_id for item in selected),
        )

    def test_selector_preserves_separate_overlapping_subject_masks(self) -> None:
        api = self._api()
        left = self._candidate("candidate-1", (78, 75), 13, 0.91)
        right = self._candidate("candidate-2", (96, 75), 13, 0.92)
        self.assertTrue(np.logical_and(left.mask, right.mask).any())

        selected = api.select_collision_candidates(
            (right, left),
            expected_radii=(0.01, 0.01),
            ambiguity_margin=0.01,
        )

        self.assertEqual(2, len(selected))
        self.assertTrue(np.logical_and(selected[0].mask, selected[1].mask).any())

    def test_selector_rejects_candidate_shortage(self) -> None:
        api = self._api()

        with self.assertRaisesRegex(ValueError, "found 1, expected 2"):
            api.select_collision_candidates(
                (self._candidate("candidate-1", (50, 80), 10),),
                expected_radii=(0.01, 0.01),
            )

    def test_selector_rejects_ambiguous_optimum(self) -> None:
        api = self._api()
        candidates = (
            self._candidate("candidate-1", (30, 80), 9, 0.9),
            self._candidate("candidate-2", (80, 80), 9, 0.9),
            self._candidate("candidate-3", (130, 80), 9, 0.9),
        )

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            api.select_collision_candidates(
                candidates,
                expected_radii=(0.01, 0.01),
                ambiguity_margin=0.2,
            )


if __name__ == "__main__":
    unittest.main()
