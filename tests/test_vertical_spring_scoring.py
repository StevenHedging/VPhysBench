from __future__ import annotations

import math
import unittest

import numpy as np

from physbench.evaluation.scenes.vertical_spring_oscillator.scoring import (
    SpringTrace,
    SpringTraceError,
    extract_spring_trace,
    score_spring_traces,
    theoretical_period_s,
)


QUALITY = {
    "minimum_mask_pixels": 40,
    "minimum_mask_area_ratio": 0.002,
    "maximum_mask_area_ratio": 0.10,
    "minimum_valid_frame_ratio": 0.80,
    "minimum_amplitude_px": 5.0,
    "minimum_s": 0.35,
    "maximum_s": 1.20,
}

SCORING = {
    "weights": {
        "vertical_trajectory": 0.30,
        "period": 0.15,
        "amplitude_envelope": 0.15,
        "equilibrium_release_phase": 0.15,
        "vertical_axis_confinement": 0.10,
        "oscillation_evidence": 0.15,
    },
    "trajectory_scale": 1.0,
    "amplitude_scale": 0.30,
    "equilibrium_scale": 0.30,
    "axis_drift_scale": 0.30,
    "oscillation_amplitude_scale": 0.30,
}


def circle_mask(center_x: float, center_y: float, *, radius: int = 5) -> np.ndarray:
    """A literal, rasterized circle fixture; it has no scoring dependencies."""
    y, x = np.ogrid[:96, :96]
    return ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2).astype(np.uint8)


def sinusoidal_masks(
    times_s: np.ndarray,
    *,
    equilibrium: float = 48.0,
    amplitude: float = 16.0,
    period: float = 0.8,
    phase: float = 0.0,
    horizontal_amplitude: float = 0.0,
) -> list[np.ndarray]:
    return [
        circle_mask(
            48.0 + horizontal_amplitude * math.cos(2.0 * math.pi * time / period),
            equilibrium + amplitude * math.cos(2.0 * math.pi * time / period + phase),
        )
        for time in times_s
    ]


class VerticalSpringScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.arange(96, dtype=float) / 24.0
        self.reference_masks = sinusoidal_masks(self.times)
        self.reference = extract_spring_trace(
            self.reference_masks, self.times, quality_config=QUALITY
        )

    def score(self, prediction: SpringTrace) -> dict[str, object]:
        return score_spring_traces(
            self.reference,
            prediction,
            mass_kg=0.5156,
            stiffness_n_m=32.6213467096774,
            scoring_config=SCORING,
        )

    def test_extracts_period_amplitude_and_positive_release_from_analytic_masks(
        self,
    ) -> None:
        """Would fail if centroids, robust amplitude, or autocorrelation regress."""
        self.assertAlmostEqual(0.8, self.reference.period_s, delta=1 / 24)
        self.assertAlmostEqual(16.0, self.reference.amplitude_px, delta=2.0)
        self.assertGreater(self.reference.release_sign, 0)
        self.assertEqual(1.0, self.reference.valid_ratio)

    def test_rejects_missing_masks_before_interpolation_can_reward_them(self) -> None:
        """Would fail if invalid frames are interpolated before the coverage gate."""
        masks = list(self.reference_masks)
        for index in range(0, len(masks), 2):
            masks[index] = np.zeros_like(masks[index])
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(masks, self.times, quality_config=QUALITY)
        self.assertEqual("insufficient_valid_masks", caught.exception.code)

    def test_rejects_non_monotonic_times_and_subthreshold_motion(self) -> None:
        """Would fail if temporal or minimum-motion quality gates are removed."""
        bad_times = self.times.copy()
        bad_times[4] = bad_times[3]
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(self.reference_masks, bad_times, quality_config=QUALITY)
        self.assertEqual("non_monotonic_times", caught.exception.code)
        nearly_static = sinusoidal_masks(self.times, amplitude=2.0)
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(nearly_static, self.times, quality_config=QUALITY)
        self.assertEqual("insufficient_amplitude", caught.exception.code)

    def test_rejects_an_observable_period_outside_configured_window(self) -> None:
        """Would fail if bounded period search silently aliases a slow oscillator."""
        slow_times = np.arange(240, dtype=float) / 60.0
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(
                sinusoidal_masks(slow_times, period=1.5),
                slow_times,
                quality_config=QUALITY,
            )
        self.assertEqual("period_out_of_bounds", caught.exception.code)

    def test_identity_score_is_one_and_all_components_are_bounded(self) -> None:
        """Would fail if exact agreement or component bounds change."""
        result = self.score(self.reference)
        self.assertEqual(1.0, result["score"])
        self.assertEqual(set(SCORING["weights"]), set(result["components"]))
        self.assertEqual(1.0, sum(result["weights"].values()))
        for value in result["components"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_static_trace_scores_below_quarter(self) -> None:
        """Would fail if a no-motion tube can earn trajectory-only credit."""
        xy = np.column_stack((np.full(len(self.times), 48.0), np.full(len(self.times), 48.0)))
        static = SpringTrace(
            times_s=self.times,
            xy=xy,
            area_px2=np.full(len(self.times), 81.0),
            valid=np.ones(len(self.times), dtype=bool),
            valid_ratio=1.0,
            equilibrium_y_px=48.0,
            amplitude_px=0.0,
            envelope_px=np.zeros(len(self.times)),
            period_s=None,
            horizontal_drift_ratio=0.0,
            release_sign=0,
        )
        self.assertLess(self.score(static)["score"], 0.25)

    def test_wrong_period_and_opposite_release_phase_score_below_match(self) -> None:
        """Would fail if scoring aligns future phases or ignores empirical period."""
        wrong_period = extract_spring_trace(
            sinusoidal_masks(self.times, period=0.6), self.times, quality_config=QUALITY
        )
        opposite_phase = extract_spring_trace(
            sinusoidal_masks(self.times, phase=math.pi),
            self.times,
            quality_config=QUALITY,
        )
        matching = self.score(self.reference)["score"]
        self.assertLess(self.score(wrong_period)["score"], matching)
        self.assertLess(self.score(opposite_phase)["score"], matching)

    def test_horizontal_oscillation_loses_vertical_axis_confinement(self) -> None:
        """Would fail if horizontal drift is not reflected in the physics score."""
        horizontal = extract_spring_trace(
            sinusoidal_masks(self.times, horizontal_amplitude=16.0),
            self.times,
            quality_config=QUALITY,
        )
        result = self.score(horizontal)
        self.assertLess(
            result["components"]["vertical_axis_confinement"],
            self.score(self.reference)["components"]["vertical_axis_confinement"],
        )
        self.assertLess(result["score"], self.score(self.reference)["score"])

    def test_theoretical_period_is_the_mass_spring_formula(self) -> None:
        """Would fail if the physical period calculation uses the wrong units/formula."""
        expected = 2.0 * math.pi * math.sqrt(0.5156 / 32.6213467096774)
        self.assertAlmostEqual(
            expected,
            theoretical_period_s(
                mass_kg=0.5156, stiffness_n_m=32.6213467096774
            ),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
