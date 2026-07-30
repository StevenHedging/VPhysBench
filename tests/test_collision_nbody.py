from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.entities.timeline import (
    build_common_time_grid,
)
from physbench.evaluation.scenes.collision.nbody import (
    NBodyExtractionConfig,
    extract_nbody_collision_state,
    infer_initial_active_entities,
    score_contact_events,
    score_momentum,
    score_nbody_collision,
    score_nonpenetration,
    score_track_coordinates,
    score_velocity,
)


def _state(
    x: np.ndarray,
    *,
    entity_ids: tuple[str, ...] | None = None,
    y: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    masses: np.ndarray | None = None,
):
    positions = np.asarray(x, dtype=float)
    frame_count, body_count = positions.shape
    if y is None:
        y = np.zeros_like(positions)
    centers = np.stack([positions, np.asarray(y, dtype=float)], axis=-1)
    return extract_nbody_collision_state(
        centers,
        np.ones((frame_count, body_count), dtype=bool),
        np.arange(frame_count, dtype=float).tolist(),
        entity_ids=entity_ids
        or tuple(f"body_{index}" for index in range(body_count)),
        radii_px=(
            np.asarray(radii, dtype=float)
            if radii is not None
            else np.ones(body_count, dtype=float)
        ),
        masses_kg=(
            np.asarray(masses, dtype=float)
            if masses is not None
            else np.ones(body_count, dtype=float)
        ),
        axis_origin_xy=np.asarray([0.0, 0.0]),
        axis_direction_xy=np.asarray([1.0, 0.0]),
        config=NBodyExtractionConfig(
            contact_tolerance_px=0.25,
            contact_hysteresis_px=0.25,
            velocity_window_frames=1,
        ),
    )


class NBodyCollisionTests(unittest.TestCase):
    def test_two_body_collision_extracts_pair_event_and_scores_identity(
        self,
    ) -> None:
        trace = _state(
            np.asarray(
                [
                    [0.0, 6.0],
                    [2.0, 6.0],
                    [4.0, 6.0],
                    [2.0, 8.0],
                    [0.0, 10.0],
                ]
            ),
            entity_ids=("left", "right"),
            masses=np.asarray([2.0, 3.0]),
        )
        self.assertEqual(2, len(trace.entity_ids))
        self.assertEqual(1, len(trace.contact_events))
        self.assertEqual(
            ("left", "right"), trace.contact_events[0].entity_pair
        )
        result = score_nbody_collision(
            trace, trace, frame_diagonal_px=100.0
        )
        self.assertAlmostEqual(1.0, result["score"])
        self.assertTrue(
            all(value == 1.0 for value in result["components"].values())
        )

    def test_four_bodies_and_multiple_initially_active_bodies(self) -> None:
        trace = _state(
            np.asarray(
                [
                    [0.0, 5.0, 10.0, 15.0],
                    [1.0, 5.0, 10.0, 14.0],
                    [2.0, 5.0, 10.0, 13.0],
                    [3.0, 5.0, 10.0, 12.0],
                ]
            ),
            entity_ids=("a", "b", "c", "d"),
        )
        self.assertEqual(
            ("a", "d"),
            infer_initial_active_entities(
                trace,
                minimum_speed_px_s=0.5,
                initial_window_frames=3,
            ),
        )
        result = score_nbody_collision(
            trace, trace, frame_diagonal_px=100.0
        )
        self.assertEqual(4, result["reference_body_count"])
        self.assertAlmostEqual(1.0, result["score"])

    def test_extra_contact_event_is_penalized(self) -> None:
        reference = _state(
            np.asarray(
                [
                    [0.0, 6.0],
                    [2.0, 6.0],
                    [4.0, 6.0],
                    [2.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                ]
            ),
            entity_ids=("a", "b"),
        )
        prediction = _state(
            np.asarray(
                [
                    [0.0, 6.0],
                    [2.0, 6.0],
                    [4.0, 6.0],
                    [2.0, 6.0],
                    [4.0, 6.0],
                    [2.0, 6.0],
                    [0.0, 6.0],
                ]
            ),
            entity_ids=("a", "b"),
        )
        self.assertEqual(1, len(reference.contact_events))
        self.assertEqual(2, len(prediction.contact_events))
        result = score_contact_events(
            reference.contact_events,
            prediction.contact_events,
            time_scale_s=1.0,
            gap_scale_px=0.5,
        )
        self.assertEqual(1, result["extra_event_count"])
        self.assertLess(result["score"], 1.0)
        self.assertGreater(result["score"], 0.0)

    def test_contact_by_a_residual_body_is_an_extra_pair_event(self) -> None:
        reference = _state(
            np.asarray(
                [
                    [0.0, 20.0],
                    [0.0, 20.0],
                    [0.0, 20.0],
                    [0.0, 20.0],
                    [0.0, 20.0],
                ]
            ),
            entity_ids=("a", "b"),
        )
        prediction = _state(
            np.asarray(
                [
                    [0.0, 20.0, 6.0],
                    [0.0, 20.0, 4.0],
                    [0.0, 20.0, 2.0],
                    [0.0, 20.0, 4.0],
                    [0.0, 20.0, 6.0],
                ]
            ),
            entity_ids=("a", "b", "residual"),
        )
        result = score_contact_events(
            reference.contact_events,
            prediction.contact_events,
            time_scale_s=1.0,
            gap_scale_px=0.5,
        )
        self.assertEqual(0, result["reference_event_count"])
        self.assertEqual(1, result["prediction_event_count"])
        self.assertEqual(1, result["extra_event_count"])
        self.assertEqual(0.0, result["score"])

    def test_no_event_has_a_well_defined_perfect_event_score(self) -> None:
        no_event = _state(
            np.asarray(
                [
                    [0.0, 8.0],
                    [0.0, 8.0],
                    [0.0, 8.0],
                    [0.0, 8.0],
                ]
            ),
            entity_ids=("a", "b"),
        )
        self.assertEqual((), no_event.contact_events)
        result = score_contact_events(
            no_event.contact_events,
            no_event.contact_events,
            time_scale_s=1.0,
            gap_scale_px=1.0,
        )
        self.assertEqual(1.0, result["score"])
        self.assertEqual(0, result["matched_event_count"])

    def test_missing_contact_frame_cannot_erase_reference_event(self) -> None:
        reference = _state(
            np.asarray(
                [
                    [0.0, 6.0],
                    [2.0, 6.0],
                    [4.0, 6.0],
                    [2.0, 8.0],
                    [0.0, 10.0],
                ]
            ),
            entity_ids=("left", "right"),
        )
        wrong_positions = np.asarray(
            [
                [0.0, 6.0],
                [2.0, 6.0],
                [0.0, 6.0],
                [2.0, 8.0],
                [0.0, 10.0],
            ]
        )
        complete_wrong = _state(
            wrong_positions,
            entity_ids=("left", "right"),
        )
        missing_valid = np.ones((5, 2), dtype=bool)
        missing_valid[2, 0] = False
        missing_centers = np.stack(
            [wrong_positions, np.zeros_like(wrong_positions)],
            axis=-1,
        )
        missing_centers[~missing_valid] = np.nan
        selectively_missing = extract_nbody_collision_state(
            missing_centers,
            missing_valid,
            reference.times_s.tolist(),
            entity_ids=reference.entity_ids,
            radii_px=np.ones(2),
            masses_kg=np.ones(2),
            axis_origin_xy=reference.axis_origin_xy,
            axis_direction_xy=reference.axis_direction_xy,
            config=NBodyExtractionConfig(
                contact_tolerance_px=0.25,
                contact_hysteresis_px=0.25,
                velocity_window_frames=1,
            ),
        )
        self.assertEqual(1, len(reference.contact_events))
        self.assertEqual(0, len(complete_wrong.contact_events))
        self.assertEqual(0, len(selectively_missing.contact_events))
        complete_score = score_nbody_collision(
            reference,
            complete_wrong,
            frame_diagonal_px=100.0,
        )
        missing_score = score_nbody_collision(
            reference,
            selectively_missing,
            frame_diagonal_px=100.0,
        )
        self.assertEqual(
            0.0, complete_score["components"]["contact_graph"]
        )
        self.assertEqual(
            0.0, missing_score["components"]["contact_graph"]
        )
        self.assertEqual(
            0,
            missing_score["details"]["contact_graph"][
                "omitted_reference_event_count"
            ],
        )
        self.assertLessEqual(missing_score["score"], complete_score["score"])

    def test_missing_false_contact_frame_cannot_turn_zero_into_positive(
        self,
    ) -> None:
        reference = _state(
            np.asarray(
                [
                    [0.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                    [0.0, 6.0],
                ]
            ),
            entity_ids=("left", "right"),
        )
        false_contact_positions = np.asarray(
            [
                [0.0, 6.0],
                [0.0, 6.0],
                [2.0, 6.0],
                [4.0, 6.0],
                [2.0, 6.0],
                [0.0, 6.0],
                [0.0, 6.0],
            ]
        )
        complete_false_contact = _state(
            false_contact_positions,
            entity_ids=("left", "right"),
        )
        missing_valid = np.ones((7, 2), dtype=bool)
        missing_valid[3] = False
        missing_centers = np.stack(
            [
                false_contact_positions,
                np.zeros_like(false_contact_positions),
            ],
            axis=-1,
        )
        missing_centers[~missing_valid] = np.nan
        selectively_missing = extract_nbody_collision_state(
            missing_centers,
            missing_valid,
            reference.times_s.tolist(),
            entity_ids=reference.entity_ids,
            radii_px=np.ones(2),
            masses_kg=np.ones(2),
            axis_origin_xy=reference.axis_origin_xy,
            axis_direction_xy=reference.axis_direction_xy,
            config=NBodyExtractionConfig(
                contact_tolerance_px=0.25,
                contact_hysteresis_px=0.25,
                velocity_window_frames=1,
            ),
        )
        self.assertEqual((), reference.contact_events)
        self.assertEqual(1, len(complete_false_contact.contact_events))
        self.assertEqual((), selectively_missing.contact_events)

        complete_score = score_nbody_collision(
            reference,
            complete_false_contact,
            frame_diagonal_px=100.0,
        )
        missing_score = score_nbody_collision(
            reference,
            selectively_missing,
            frame_diagonal_px=100.0,
        )

        self.assertEqual(
            0.0, complete_score["components"]["contact_graph"]
        )
        self.assertEqual(
            0.0, missing_score["components"]["contact_graph"]
        )
        self.assertEqual(0.0, complete_score["score"])
        self.assertEqual(0.0, missing_score["score"])
        contact = missing_score["details"]["contact_graph"]
        self.assertEqual(7, contact["reference_pair_exposure_frames"])
        self.assertEqual(6, contact["comparable_pair_exposure_frames"])
        self.assertFalse(contact["contact_absence_certified"])
        self.assertEqual(
            "conservative_zero",
            contact["incomplete_exposure_policy"],
        )

    def test_selective_missing_cannot_improve_continuous_physics(
        self,
    ) -> None:
        times = np.asarray([0.0, 0.1, 0.2, 1.2, 1.3, 1.4, 1.5])
        reference_x = np.column_stack(
            [4.0 * times, np.full(len(times), 30.0)]
        )
        wrong_x = reference_x.copy()
        wrong_x[3, 0] = 1000.0

        def make_state(x: np.ndarray, valid: np.ndarray):
            centers = np.stack([x, np.zeros_like(x)], axis=-1)
            centers[~valid] = np.nan
            return extract_nbody_collision_state(
                centers,
                valid,
                times.tolist(),
                entity_ids=("a", "b"),
                radii_px=np.ones(2),
                masses_kg=np.ones(2),
                axis_origin_xy=np.asarray([0.0, 0.0]),
                axis_direction_xy=np.asarray([1.0, 0.0]),
                config=NBodyExtractionConfig(
                    contact_tolerance_px=0.25,
                    contact_hysteresis_px=0.25,
                    velocity_window_frames=1,
                ),
            )

        reference = make_state(
            reference_x,
            np.ones((len(times), 2), dtype=bool),
        )
        complete_wrong = make_state(
            wrong_x,
            np.ones((len(times), 2), dtype=bool),
        )
        missing_valid = np.ones((len(times), 2), dtype=bool)
        missing_valid[3, 0] = False
        selectively_missing = make_state(wrong_x, missing_valid)
        complete_score = score_nbody_collision(
            reference,
            complete_wrong,
            frame_diagonal_px=1100.0,
        )
        missing_score = score_nbody_collision(
            reference,
            selectively_missing,
            frame_diagonal_px=1100.0,
        )
        for component in (
            "track_position",
            "velocity",
            "momentum",
            "nonpenetration",
        ):
            with self.subTest(component=component):
                self.assertIsNotNone(
                    complete_score["components"][component]
                )
                self.assertIsNotNone(
                    missing_score["components"][component]
                )
                self.assertLessEqual(
                    missing_score["components"][component],
                    complete_score["components"][component] + 1e-12,
                )
                detail = missing_score["details"][component]
                self.assertGreater(detail["reference_exposure_s"], 0.0)
                self.assertGreaterEqual(detail["matched_exposure_s"], 0.0)
                self.assertLess(
                    detail["matched_exposure_s"],
                    detail["reference_exposure_s"],
                )
                self.assertAlmostEqual(
                    detail["coverage"],
                    detail["matched_exposure_s"]
                    / detail["reference_exposure_s"],
                )
        self.assertLessEqual(
            missing_score["score"], complete_score["score"] + 1e-12
        )

        grid = build_common_time_grid(times)
        removed_weight = float(grid.cell_weights_s[3])
        position = missing_score["details"]["track_position"]
        self.assertAlmostEqual(
            2.0 * grid.duration_s,
            position["reference_exposure_s"],
        )
        self.assertAlmostEqual(
            2.0 * grid.duration_s - removed_weight,
            position["matched_exposure_s"],
        )
        self.assertNotAlmostEqual(
            13.0 / 14.0,
            position["coverage"],
            places=4,
        )

    def test_close_disjoint_position_scores_above_far_position(self) -> None:
        reference = _state(
            np.asarray(
                [[0.0, 10.0], [1.0, 10.0], [2.0, 10.0]]
            )
        )
        close = _state(
            np.asarray(
                [[2.5, 12.5], [3.5, 12.5], [4.5, 12.5]]
            )
        )
        far = _state(
            np.asarray(
                [[25.0, 35.0], [26.0, 35.0], [27.0, 35.0]]
            )
        )
        close_score = score_track_coordinates(
            reference, close, frame_diagonal_px=100.0
        )["score"]
        far_score = score_track_coordinates(
            reference, far, frame_diagonal_px=100.0
        )["score"]
        self.assertGreater(close_score, far_score)
        self.assertGreater(far_score, 0.0)

    def test_excess_penetration_is_penalized_relative_to_reference(
        self,
    ) -> None:
        reference = _state(
            np.asarray(
                [[0.0, 6.0], [1.0, 6.0], [2.0, 6.0], [1.0, 6.0]]
            )
        )
        penetrating = _state(
            np.asarray(
                [[0.0, 6.0], [1.0, 6.0], [5.0, 6.0], [1.0, 6.0]]
            )
        )
        identity = score_nonpenetration(reference, reference)
        degraded = score_nonpenetration(reference, penetrating)
        self.assertEqual(1.0, identity["score"])
        self.assertLess(degraded["score"], 1.0)
        self.assertGreater(degraded["mean_excess_penetration"], 0.0)

    def test_velocity_and_system_momentum_drift_are_penalized(self) -> None:
        reference = _state(
            np.asarray(
                [
                    [0.0, 20.0],
                    [1.0, 20.0],
                    [2.0, 20.0],
                    [3.0, 20.0],
                    [4.0, 20.0],
                ]
            ),
            entity_ids=("moving", "stationary"),
            masses=np.asarray([2.0, 1.0]),
        )
        accelerating = _state(
            np.asarray(
                [
                    [0.0, 20.0],
                    [1.0, 20.0],
                    [3.0, 20.0],
                    [6.0, 20.0],
                    [10.0, 20.0],
                ]
            ),
            entity_ids=("moving", "stationary"),
            masses=np.asarray([2.0, 1.0]),
        )
        self.assertEqual(1.0, score_velocity(reference, reference)["score"])
        self.assertEqual(1.0, score_momentum(reference, reference)["score"])
        self.assertLess(
            score_velocity(reference, accelerating)["score"], 1.0
        )
        self.assertLess(
            score_momentum(reference, accelerating)["score"], 1.0
        )

    def test_entity_order_does_not_change_id_based_scoring(self) -> None:
        reference_x = np.asarray(
            [
                [0.0, 6.0, 12.0, 18.0],
                [1.0, 6.0, 12.0, 17.0],
                [2.0, 6.0, 12.0, 16.0],
                [3.0, 6.0, 12.0, 15.0],
            ]
        )
        ids = ("a", "b", "c", "d")
        masses = np.asarray([1.0, 2.0, 3.0, 4.0])
        reference = _state(
            reference_x,
            entity_ids=ids,
            masses=masses,
        )
        order = np.asarray([3, 1, 0, 2])
        prediction = _state(
            reference_x[:, order],
            entity_ids=tuple(ids[index] for index in order),
            masses=masses[order],
        )
        result = score_nbody_collision(
            reference, prediction, frame_diagonal_px=100.0
        )
        self.assertAlmostEqual(1.0, result["score"])

    def test_missing_observations_produce_finite_low_score(self) -> None:
        reference = _state(
            np.asarray(
                [[0.0, 6.0], [1.0, 6.0], [2.0, 6.0], [3.0, 6.0]]
            )
        )
        centers = reference.centers_xy.copy()
        centers[:] = np.nan
        prediction = extract_nbody_collision_state(
            centers,
            np.zeros((4, 2), dtype=bool),
            reference.times_s.tolist(),
            entity_ids=reference.entity_ids,
            radii_px=np.ones(2),
            masses_kg=np.ones(2),
            axis_origin_xy=reference.axis_origin_xy,
            axis_direction_xy=reference.axis_direction_xy,
        )
        result = score_nbody_collision(
            reference, prediction, frame_diagonal_px=100.0
        )
        self.assertTrue(np.isfinite(result["score"]))
        self.assertEqual(0.0, result["score"])
        self.assertEqual(
            0.0, result["details"]["track_position"]["coverage"]
        )
        self.assertEqual(
            "conservative_zero",
            result["details"]["contact_graph"][
                "incomplete_exposure_policy"
            ],
        )

    def test_missing_body_has_conservative_zero_pair_physics(
        self,
    ) -> None:
        reference = _state(
            np.asarray(
                [[0.0, 8.0], [1.0, 8.0], [2.0, 8.0], [3.0, 8.0]]
            ),
            entity_ids=("visible", "missing"),
        )
        centers = reference.centers_xy.copy()
        centers[:, 1] = np.nan
        valid = np.ones((4, 2), dtype=bool)
        valid[:, 1] = False
        prediction = extract_nbody_collision_state(
            centers,
            valid,
            reference.times_s.tolist(),
            entity_ids=reference.entity_ids,
            radii_px=np.ones(2),
            masses_kg=np.ones(2),
            axis_origin_xy=reference.axis_origin_xy,
            axis_direction_xy=reference.axis_direction_xy,
        )
        result = score_nbody_collision(
            reference, prediction, frame_diagonal_px=100.0
        )
        self.assertEqual(0.0, result["score"])
        self.assertEqual([], result["omitted_components"])
        for component in (
            "contact_graph",
            "momentum",
            "nonpenetration",
        ):
            self.assertEqual(0.0, result["components"][component])
        self.assertEqual(
            "conservative_zero",
            result["details"]["contact_graph"][
                "incomplete_exposure_policy"
            ],
        )

    def test_malformed_or_incompatible_inputs_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two physical bodies"):
            _state(np.asarray([[0.0], [1.0]]))
        reference = _state(np.asarray([[0.0, 6.0], [1.0, 6.0]]))
        shifted_time_centers = np.asarray(
            [[[0.0, 0.0], [6.0, 0.0]], [[1.0, 0.0], [6.0, 0.0]]]
        )
        shifted = extract_nbody_collision_state(
            shifted_time_centers,
            np.ones((2, 2), dtype=bool),
            [0.0, 1.5],
            entity_ids=("body_0", "body_1"),
            radii_px=np.ones(2),
            masses_kg=np.ones(2),
            axis_origin_xy=np.asarray([0.0, 0.0]),
            axis_direction_xy=np.asarray([1.0, 0.0]),
        )
        with self.assertRaisesRegex(ValueError, "common timeline"):
            score_track_coordinates(
                reference, shifted, frame_diagonal_px=100.0
            )


if __name__ == "__main__":
    unittest.main()
