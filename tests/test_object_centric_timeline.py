from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.entities import (
    CommonTimeGrid,
    ExposureType,
    ObjectTrack,
    VisibilityState,
    build_common_time_grid,
    exposure_by_id,
)


class CommonTimeGridTests(unittest.TestCase):
    def test_regular_grid_uses_trapezoidal_cell_weights(self) -> None:
        grid = build_common_time_grid([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            grid.cell_weights_s,
            [0.125, 0.25, 0.25, 0.25, 0.125],
        )
        self.assertAlmostEqual(1.0, grid.duration_s)
        self.assertAlmostEqual(1.0, grid.integrate(np.ones(5)))

    def test_irregular_grid_integrates_real_elapsed_time(self) -> None:
        grid = build_common_time_grid([0.0, 0.1, 0.4, 1.0])
        np.testing.assert_allclose(
            grid.cell_weights_s,
            [0.05, 0.2, 0.45, 0.3],
        )
        self.assertAlmostEqual(0.25, grid.integrate([1, 1, 0, 0]))

    def test_explicit_interval_bounds_cover_edge_cells(self) -> None:
        grid = build_common_time_grid(
            [0.25, 0.75],
            interval_start_s=0.0,
            interval_end_s=1.0,
        )
        np.testing.assert_allclose(grid.cell_weights_s, [0.5, 0.5])
        self.assertEqual(
            {
                "times_s": [0.25, 0.75],
                "cell_weights_s": [0.5, 0.5],
                "interval_start_s": 0.0,
                "interval_end_s": 1.0,
                "duration_s": 1.0,
            },
            grid.to_dict(),
        )

    def test_empty_and_single_sample_grids_have_deterministic_duration(
        self,
    ) -> None:
        empty = build_common_time_grid([])
        self.assertEqual(0, len(empty.times_s))
        self.assertEqual(0.0, empty.duration_s)
        self.assertEqual(0.0, empty.integrate([]))

        instantaneous = build_common_time_grid([0.5])
        np.testing.assert_array_equal(
            instantaneous.cell_weights_s,
            [0.0],
        )
        self.assertEqual(0.0, instantaneous.duration_s)

        bounded = build_common_time_grid(
            [0.5],
            interval_start_s=0.0,
            interval_end_s=1.0,
        )
        np.testing.assert_array_equal(bounded.cell_weights_s, [1.0])
        self.assertEqual(1.0, bounded.duration_s)

    def test_time_grid_rejects_ambiguous_or_invalid_inputs(self) -> None:
        invalid = (
            ([0.0, 0.0], {}, "strictly increasing"),
            ([0.1, 0.0], {}, "strictly increasing"),
            ([float("nan")], {}, "finite"),
            ([-0.1], {}, "non-negative"),
            (
                [0.5],
                {"interval_start_s": 0.6},
                "must not follow",
            ),
            (
                [0.5],
                {"interval_end_s": 0.4},
                "must not precede",
            ),
            (
                [],
                {"interval_start_s": 0.0, "interval_end_s": 1.0},
                "empty time grid",
            ),
        )
        for times, kwargs, message in invalid:
            with self.subTest(times=times, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    build_common_time_grid(times, **kwargs)

        with self.assertRaisesRegex(ValueError, "sum"):
            CommonTimeGrid(
                times_s=np.asarray([0.0, 1.0]),
                cell_weights_s=np.asarray([0.2, 0.2]),
                interval_start_s=0.0,
                interval_end_s=1.0,
            )

    def test_grid_arrays_are_immutable_snapshots(self) -> None:
        source = np.asarray([0.0, 1.0])
        grid = build_common_time_grid(source)
        source[1] = 2.0
        np.testing.assert_array_equal(grid.times_s, [0.0, 1.0])
        with self.assertRaises(ValueError):
            grid.cell_weights_s[0] = 2.0

    def test_integrate_rejects_wrong_shape_and_nonfinite_values(self) -> None:
        grid = build_common_time_grid([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "one value"):
            grid.integrate([1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            grid.integrate([1.0, float("nan")])


class WeightedObjectTrackExposureTests(unittest.TestCase):
    @staticmethod
    def _track(**overrides: object) -> ObjectTrack:
        values: dict[str, object] = {
            "track_id": "track",
            "matched_entity_id": "ball",
            "xy": np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
            ),
            "observed": np.asarray([True, True, True, True]),
            "visibility": (VisibilityState.VISIBLE,) * 4,
        }
        values.update(overrides)
        return ObjectTrack(**values)

    def test_legacy_exposure_remains_a_visible_frame_count(self) -> None:
        track = self._track(
            observed=np.asarray([True, False, True, True]),
            visibility=(
                VisibilityState.VISIBLE,
                VisibilityState.VISIBLE,
                VisibilityState.OCCLUDED,
                VisibilityState.VISIBLE,
            ),
        )
        self.assertEqual(2.0, track.exposure())
        self.assertEqual(2.0, track.exposure("localization"))
        self.assertEqual(2.0, track.exposure("association"))

    def test_exposure_channels_are_distinct_and_time_weighted(self) -> None:
        grid = build_common_time_grid([0.0, 0.1, 0.4, 1.0])
        track = self._track(
            existence_observed=np.asarray([True, True, True, True]),
            localization_eligible=np.asarray([True, False, True, True]),
            association_eligible=np.asarray([True, False, False, True]),
            time_weights_s=grid.cell_weights_s,
        )
        self.assertAlmostEqual(
            1.0,
            track.exposure(ExposureType.EXISTENCE),
        )
        self.assertAlmostEqual(
            0.8,
            track.exposure(ExposureType.LOCALIZATION),
        )
        self.assertAlmostEqual(
            0.35,
            track.exposure(ExposureType.ASSOCIATION),
        )

    def test_method_weights_override_track_weights(self) -> None:
        track = self._track(time_weights_s=np.ones(4))
        self.assertEqual(4.0, track.exposure())
        self.assertAlmostEqual(
            1.0,
            track.exposure(time_weights_s=[0.1, 0.2, 0.3, 0.4]),
        )

    def test_exposure_by_id_supports_each_channel_and_shared_weights(
        self,
    ) -> None:
        track = self._track(
            association_eligible=np.asarray([True, False, False, True]),
        )
        result = exposure_by_id(
            [track],
            use_entity_ids=True,
            exposure_type=ExposureType.ASSOCIATION,
            time_weights_s=[0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual({"ball": 0.5}, result)

    def test_evidence_channels_must_be_nested_boolean_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "boolean"):
            self._track(existence_observed=np.asarray([1, 1, 1, 1]))
        with self.assertRaisesRegex(ValueError, "subset of observed"):
            self._track(
                observed=np.asarray([True, False, True, True]),
                existence_observed=np.asarray([True, True, True, True]),
            )
        with self.assertRaisesRegex(
            ValueError,
            "localization_eligible must be a subset",
        ):
            self._track(
                existence_observed=np.asarray([True, False, True, True]),
                localization_eligible=np.asarray([True, True, True, True]),
            )
        with self.assertRaisesRegex(
            ValueError,
            "association_eligible must be a subset",
        ):
            self._track(
                localization_eligible=np.asarray(
                    [True, False, True, True]
                ),
                association_eligible=np.asarray([True, True, True, True]),
            )

    def test_time_weights_are_validated_even_without_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be infinite"):
            self._track(time_weights_s=np.asarray([1.0, 1.0, np.inf, 1.0]))
        track = self._track()
        with self.assertRaisesRegex(ValueError, "one value"):
            track.exposure(time_weights_s=[1.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            track.exposure(time_weights_s=[1.0, -1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
