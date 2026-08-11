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


def masks_for_vertical_positions(positions_y: np.ndarray) -> list[np.ndarray]:
    """Rasterize a hand-provided vertical signal without using scoring helpers."""
    return [circle_mask(48.0, float(position_y)) for position_y in positions_y]


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

    def score_with_physics(
        self, prediction: SpringTrace, *, mass_kg: float, stiffness_n_m: float
    ) -> dict[str, object]:
        return score_spring_traces(
            self.reference,
            prediction,
            mass_kg=mass_kg,
            stiffness_n_m=stiffness_n_m,
            scoring_config=SCORING,
        )

    @staticmethod
    def unchecked_trace(**changes: object) -> SpringTrace:
        """Bypass construction only to exercise score_spring_traces' public boundary."""
        values: dict[str, object] = {
            "times_s": np.arange(96, dtype=float) / 24.0,
            "xy": np.column_stack((np.full(96, 48.0), np.full(96, 48.0))),
            "area_px2": np.full(96, 81.0),
            "valid": np.ones(96, dtype=bool),
            "valid_ratio": 1.0,
            "equilibrium_y_px": 48.0,
            "amplitude_px": 0.0,
            "envelope_px": np.zeros(96),
            "period_s": None,
            "horizontal_drift_ratio": 0.0,
            "release_sign": 0,
        }
        values.update(changes)
        trace = object.__new__(SpringTrace)
        for name, value in values.items():
            object.__setattr__(trace, name, value)
        return trace

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

    def test_score_rejects_a_nonfinite_trace_at_its_public_boundary(self) -> None:
        """Would fail if NaN prediction samples are converted into perfect similarity."""
        prediction = self.unchecked_trace(xy=np.full((96, 2), np.nan))
        with self.assertRaises(SpringTraceError) as caught:
            self.score(prediction)
        self.assertEqual("invalid_trace", caught.exception.code)

    def test_score_rejects_an_empty_trace_at_its_public_boundary(self) -> None:
        """Would fail if empty reductions can create a rewarding score."""
        prediction = self.unchecked_trace(
            times_s=np.empty(0),
            xy=np.empty((0, 2)),
            area_px2=np.empty(0),
            valid=np.empty(0, dtype=bool),
            valid_ratio=0.0,
            envelope_px=np.empty(0),
        )
        reference = self.unchecked_trace(
            times_s=np.empty(0),
            xy=np.empty((0, 2)),
            area_px2=np.empty(0),
            valid=np.empty(0, dtype=bool),
            valid_ratio=0.0,
            envelope_px=np.empty(0),
        )
        with self.assertRaises(SpringTraceError) as caught:
            score_spring_traces(
                reference,
                prediction,
                mass_kg=0.5156,
                stiffness_n_m=32.6213467096774,
                scoring_config=SCORING,
            )
        self.assertEqual("invalid_trace", caught.exception.code)

    def test_nonperiodic_ramp_has_no_period_or_oscillation_evidence(self) -> None:
        """Would fail if a monotonic ramp gets a fallback autocorrelation period."""
        ramp = extract_spring_trace(
            masks_for_vertical_positions(np.linspace(25.0, 70.0, len(self.times))),
            self.times,
            quality_config=QUALITY,
        )
        self.assertIsNone(ramp.period_s)
        self.assertEqual(0.0, self.score(ramp)["components"]["oscillation_evidence"])

    def test_rejects_harmonic_alias_when_fundamental_is_outside_window(self) -> None:
        """Would fail if the 0.75 s harmonic masks a 1.5 s fundamental."""
        times = np.arange(240, dtype=float) / 60.0
        positions = (
            48.0
            + 8.0 * np.cos(2.0 * math.pi * times / 1.5)
            + 20.0 * np.cos(4.0 * math.pi * times / 1.5)
        )
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(
                masks_for_vertical_positions(positions), times, quality_config=QUALITY
            )
        self.assertEqual("period_out_of_bounds", caught.exception.code)

    def test_weak_fundamental_sweep_rejects_harmonic_aliases(self) -> None:
        """Would fail if a 0.75 s harmonic hides any observable 1.5 s fundamental."""
        times = np.arange(240, dtype=float) / 60.0
        for fundamental_amplitude in (1.0, 2.0, 4.0, 8.0):
            with self.subTest(fundamental_amplitude=fundamental_amplitude):
                positions = (
                    48.0
                    + fundamental_amplitude * np.cos(2.0 * math.pi * times / 1.5)
                    + 20.0 * np.cos(4.0 * math.pi * times / 1.5)
                )
                with self.assertRaises(SpringTraceError) as caught:
                    extract_spring_trace(
                        masks_for_vertical_positions(positions),
                        times,
                        quality_config=QUALITY,
                    )
                self.assertEqual("period_out_of_bounds", caught.exception.code)

    def test_single_frequency_multiple_repeats_selects_shortest_fundamental(self) -> None:
        """Would fail if strict harmonic safety chooses an integer repeat of a pure tone."""
        times = np.arange(240, dtype=float) / 60.0
        trace = extract_spring_trace(
            masks_for_vertical_positions(48.0 + 16.0 * np.cos(2.0 * math.pi * times / 0.75)),
            times,
            quality_config=QUALITY,
        )
        self.assertAlmostEqual(0.75, trace.period_s, delta=1 / 60)

    def test_rejects_nonfinite_fundamental_tie_tolerance(self) -> None:
        """Would fail if the harmonic-selection threshold accepts non-finite values."""
        with self.assertRaises(ValueError):
            extract_spring_trace(
                self.reference_masks,
                self.times,
                quality_config={**QUALITY, "fundamental_peak_tie_tolerance": math.nan},
            )

    def test_rejects_irregular_cadence_instead_of_using_index_lags(self) -> None:
        """Would fail if strictly increasing but irregular timestamps are accepted."""
        irregular = self.times.copy()
        irregular[20:] += 0.015
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(
                sinusoidal_masks(irregular), irregular, quality_config=QUALITY
            )
        self.assertEqual("irregular_cadence", caught.exception.code)

    def test_extraction_honors_configured_cadence_tolerance(self) -> None:
        """Would fail if quality_config's cadence tolerance is silently ignored."""
        slightly_irregular = self.times.copy()
        slightly_irregular[20:] += 0.0001
        strict_quality = {**QUALITY, "maximum_cadence_relative_deviation": 0.001}
        with self.assertRaises(SpringTraceError) as caught:
            extract_spring_trace(
                sinusoidal_masks(slightly_irregular),
                slightly_irregular,
                quality_config=strict_quality,
            )
        self.assertEqual("irregular_cadence", caught.exception.code)

    def test_equilibrium_similarity_is_invariant_to_a_common_y_origin_shift(self) -> None:
        """Would fail if equilibrium error is divided by an absolute pixel coordinate."""
        high_reference = extract_spring_trace(
            sinusoidal_masks(self.times, equilibrium=48.0), self.times, quality_config=QUALITY
        )
        high_prediction = extract_spring_trace(
            sinusoidal_masks(self.times, equilibrium=52.0), self.times, quality_config=QUALITY
        )
        low_reference = extract_spring_trace(
            sinusoidal_masks(self.times, equilibrium=24.0), self.times, quality_config=QUALITY
        )
        low_prediction = extract_spring_trace(
            sinusoidal_masks(self.times, equilibrium=28.0), self.times, quality_config=QUALITY
        )
        high = score_spring_traces(
            high_reference,
            high_prediction,
            mass_kg=0.5156,
            stiffness_n_m=32.6213467096774,
            scoring_config=SCORING,
        )
        low = score_spring_traces(
            low_reference,
            low_prediction,
            mass_kg=0.5156,
            stiffness_n_m=32.6213467096774,
            scoring_config=SCORING,
        )
        self.assertAlmostEqual(
            high["components"]["equilibrium_release_phase"],
            low["components"]["equilibrium_release_phase"],
            places=12,
        )

    def test_changed_displacement_envelope_loses_amplitude_envelope_credit(self) -> None:
        """Would fail if only a scalar amplitude is compared and the envelope is ignored."""
        decay = 1.0 - 0.45 * self.times / self.times[-1]
        positions = 48.0 + 16.0 * decay * np.cos(2.0 * math.pi * self.times / 0.8)
        prediction = extract_spring_trace(
            masks_for_vertical_positions(positions), self.times, quality_config=QUALITY
        )
        self.assertLess(
            self.score(prediction)["components"]["amplitude_envelope"],
            self.score(self.reference)["components"]["amplitude_envelope"],
        )

    def test_same_amplitude_period_phase_with_changed_envelope_loses_credit(self) -> None:
        """Would fail if amplitude_envelope ignores a physically different envelope."""
        raw = (1.0 - 0.45 * self.times / self.times[-1]) * np.cos(
            2.0 * math.pi * self.times / 0.8
        )
        centered = raw - np.median(raw)
        raw_amplitude = 0.5 * (
            np.quantile(centered, 0.95) - np.quantile(centered, 0.05)
        )
        positions = 48.0 + self.reference.amplitude_px * centered / raw_amplitude
        equilibrium = float(np.median(positions))
        prediction = SpringTrace(
            times_s=self.times,
            xy=np.column_stack((np.full(len(self.times), 48.0), positions)),
            area_px2=np.full(len(self.times), 81.0),
            valid=np.ones(len(self.times), dtype=bool),
            valid_ratio=1.0,
            equilibrium_y_px=equilibrium,
            amplitude_px=self.reference.amplitude_px,
            envelope_px=np.abs(positions - equilibrium),
            period_s=self.reference.period_s,
            horizontal_drift_ratio=0.0,
            release_sign=1,
        )
        self.assertAlmostEqual(
            self.reference.amplitude_px, prediction.amplitude_px, places=12
        )
        self.assertEqual(self.reference.period_s, prediction.period_s)
        self.assertEqual(self.reference.release_sign, prediction.release_sign)
        self.assertLess(
            self.score(prediction)["components"]["amplitude_envelope"], 0.90
        )

    def test_score_rejects_an_envelope_inconsistent_with_trace_positions(self) -> None:
        """Would fail if a hand-built envelope can override the measured displacement."""
        prediction = self.unchecked_trace(
            times_s=self.reference.times_s,
            xy=self.reference.xy,
            area_px2=self.reference.area_px2,
            valid=self.reference.valid,
            valid_ratio=self.reference.valid_ratio,
            equilibrium_y_px=self.reference.equilibrium_y_px,
            amplitude_px=self.reference.amplitude_px,
            envelope_px=np.zeros(len(self.reference.times_s)),
            period_s=self.reference.period_s,
            horizontal_drift_ratio=self.reference.horizontal_drift_ratio,
            release_sign=self.reference.release_sign,
        )
        with self.assertRaises(SpringTraceError) as caught:
            self.score(prediction)
        self.assertEqual("invalid_trace", caught.exception.code)

    def test_period_component_requires_reference_agreement_when_theory_matches(
        self,
    ) -> None:
        """Would fail if a prediction matching only theory earns a perfect period score."""
        prediction = extract_spring_trace(
            sinusoidal_masks(self.times, period=0.6), self.times, quality_config=QUALITY
        )
        stiffness = 0.5156 * (2.0 * math.pi / 0.6) ** 2
        self.assertLess(
            self.score_with_physics(
                prediction, mass_kg=0.5156, stiffness_n_m=stiffness
            )["components"]["period"],
            1.0,
        )

    def test_period_component_requires_theory_agreement_when_reference_matches(
        self,
    ) -> None:
        """Would fail if an empirical match can ignore an incompatible physical period."""
        stiffness = 0.5156 * (2.0 * math.pi / 0.6) ** 2
        self.assertLess(
            self.score_with_physics(
                self.reference, mass_kg=0.5156, stiffness_n_m=stiffness
            )["components"]["period"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
