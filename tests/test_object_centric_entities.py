from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from physbench.evaluation.common.entities import (
    EntityMatch,
    EntitySpec,
    ObjectTrack,
    VisibilityState,
    compare_positions,
    compose_gated_case_score,
    compose_weighted_geometric,
    distance_similarity,
    freeze_initial_assignment,
    score_entity_integrity,
)


def _matches(
    entity_id: str,
    track_id: str,
    qualities: list[float],
) -> list[EntityMatch]:
    return [
        EntityMatch(
            frame_index=index,
            entity_id=entity_id,
            track_id=track_id,
            localization_quality=quality,
            normalized_distance=math.sqrt(max(1.0 / quality - 1.0, 0.0))
            if quality > 0.0
            else 1_000.0,
        )
        for index, quality in enumerate(qualities)
    ]


class ObjectCentricEntityTests(unittest.TestCase):
    def test_object_track_counts_only_visible_observations_as_exposure(
        self,
    ) -> None:
        track = ObjectTrack(
            track_id="track",
            matched_entity_id="ball",
            xy=np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float),
            observed=np.asarray([True, False, True, True]),
            visibility=(
                VisibilityState.VISIBLE,
                VisibilityState.VISIBLE,
                VisibilityState.OCCLUDED,
                VisibilityState.VISIBLE,
            ),
        )
        self.assertEqual(2.0, track.exposure())

    def test_object_track_rejects_infinite_optional_observations(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "must not be infinite"):
            ObjectTrack(
                track_id="track",
                xy=np.asarray([[0.0, 0.0]]),
                observed=np.asarray([False]),
                visibility=(VisibilityState.UNKNOWN,),
                confidence=np.asarray([float("inf")]),
            )

    def test_initial_assignment_is_rectangular_and_preserves_residuals(
        self,
    ) -> None:
        entities = [
            EntitySpec("striker", "striker", "ball"),
            EntitySpec("target", "target", "ball"),
        ]
        assignment = freeze_initial_assignment(
            entities,
            ["left", "middle", "extra"],
            np.asarray([[0.05, 0.9, 0.7], [0.8, 0.04, 0.6]]),
            maximum_match_cost=0.5,
        )
        self.assertEqual(
            {"striker": "left", "target": "middle"},
            assignment.entity_to_track,
        )
        self.assertEqual(("extra",), assignment.residual_track_ids)
        self.assertEqual((), assignment.unmatched_entity_ids)

    def test_initial_assignment_can_leave_an_entity_missing(self) -> None:
        entities = [
            EntitySpec("first", "first", "ball"),
            EntitySpec("second", "second", "ball"),
        ]
        assignment = freeze_initial_assignment(
            entities,
            ["candidate"],
            np.asarray([[0.1], [float("inf")]]),
            maximum_match_cost=0.5,
        )
        self.assertEqual("candidate", assignment.entity_to_track["first"])
        self.assertIsNone(assignment.entity_to_track["second"])
        self.assertEqual(("second",), assignment.unmatched_entity_ids)

    def test_initial_assignment_retains_overflow_as_residuals(self) -> None:
        entities = [EntitySpec("body", "body", "ball")]
        track_ids = [f"noise_{index:02d}" for index in range(19)] + [
            "body_track"
        ]
        costs = np.full((1, len(track_ids)), 0.4, dtype=float)
        costs[0, -1] = 0.01
        assignment = freeze_initial_assignment(
            entities,
            track_ids,
            costs,
            maximum_match_cost=0.5,
            maximum_entities=16,
        )
        self.assertEqual("body_track", assignment.entity_to_track["body"])
        self.assertEqual(19, len(assignment.residual_track_ids))
        self.assertEqual(set(track_ids[:-1]), set(assignment.residual_track_ids))

        no_expected = freeze_initial_assignment(
            [],
            track_ids,
            np.empty((0, len(track_ids)), dtype=float),
            maximum_match_cost=0.5,
            maximum_entities=16,
        )
        self.assertEqual({}, no_expected.entity_to_track)
        self.assertEqual(tuple(track_ids), no_expected.residual_track_ids)
        self.assertAlmostEqual(20.0, no_expected.total_cost)

    def test_many_candidates_do_not_change_exact_assignment(self) -> None:
        entities = [
            EntitySpec("a", "a", "ball"),
            EntitySpec("b", "b", "ball"),
        ]
        assignment = freeze_initial_assignment(
            entities,
            ["a_best", "a_duplicate", "b_only"],
            np.asarray(
                [
                    [0.0, 0.01, 0.02],
                    [float("inf"), float("inf"), 0.02],
                ]
            ),
            maximum_match_cost=0.5,
        )
        self.assertEqual(
            {"a": "a_best", "b": "b_only"},
            assignment.entity_to_track,
        )
        self.assertEqual(
            ("a_duplicate",),
            assignment.residual_track_ids,
        )
        self.assertAlmostEqual(1.02, assignment.total_cost)

    def test_assignment_matches_bruteforce_on_random_rectangles(self) -> None:
        rng = np.random.default_rng(20260730)
        for _ in range(200):
            entity_count = int(rng.integers(0, 5))
            track_count = int(rng.integers(0, 7))
            entities = [
                EntitySpec(f"e{index}", f"r{index}", "body")
                for index in range(entity_count)
            ]
            track_ids = [f"t{index}" for index in range(track_count)]
            costs = rng.uniform(0.0, 1.5, (entity_count, track_count))
            if costs.size:
                costs[rng.random(costs.shape) < 0.2] = float("inf")
            threshold = 0.8
            missing_cost = 0.9
            residual_cost = 0.7
            result = freeze_initial_assignment(
                entities,
                track_ids,
                costs,
                maximum_match_cost=threshold,
                unmatched_entity_cost=missing_cost,
                unmatched_track_cost=residual_cost,
                maximum_entities=4,
            )

            brute_cost = float("inf")
            choices = [None, *range(track_count)]
            for assignment in itertools.product(
                choices,
                repeat=entity_count,
            ):
                used = [index for index in assignment if index is not None]
                if len(set(used)) != len(used):
                    continue
                if any(
                    not math.isfinite(float(costs[entity_index, track_index]))
                    or float(costs[entity_index, track_index]) > threshold
                    for entity_index, track_index in enumerate(assignment)
                    if track_index is not None
                ):
                    continue
                candidate_cost = sum(
                    missing_cost
                    if track_index is None
                    else float(costs[entity_index, track_index])
                    for entity_index, track_index in enumerate(assignment)
                )
                candidate_cost += (
                    track_count - len(used)
                ) * residual_cost
                brute_cost = min(brute_cost, candidate_cost)
            self.assertAlmostEqual(brute_cost, result.total_cost)

    def test_initial_assignment_rejects_negative_costs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            freeze_initial_assignment(
                [EntitySpec("body", "body", "ball")],
                ["track"],
                np.asarray([[-0.1]]),
                maximum_match_cost=0.5,
            )

    def test_reference_scaled_position_score_is_strictly_monotonic(
        self,
    ) -> None:
        common = {
            "reference_area_px2": math.pi * 10.0**2,
            "frame_diagonal_px": 1_000.0,
            "jitter_tolerance_radius_fraction": 0.0,
        }
        identity = compare_positions([0, 0], [0, 0], **common)
        close_disjoint = compare_positions([0, 0], [25, 0], **common)
        far = compare_positions([0, 0], [250, 0], **common)
        self.assertEqual(1.0, identity.score)
        self.assertGreater(close_disjoint.score, far.score)
        self.assertGreater(far.score, 0.0)

    def test_identity_entity_set_scores_one(self) -> None:
        result = score_entity_integrity(
            _matches("ball", "track", [1.0, 1.0, 1.0]),
            reference_exposure={"ball": 3.0},
            prediction_exposure={"track": 3.0},
        )
        self.assertAlmostEqual(1.0, result.score)
        self.assertAlmostEqual(1.0, result.soft_detection_accuracy)
        self.assertAlmostEqual(1.0, result.association_accuracy)
        self.assertAlmostEqual(0.0, result.gospa.distance)

    def test_distance_lowers_entity_score_without_an_iou_cliff(self) -> None:
        close = score_entity_integrity(
            _matches("ball", "track", [0.8, 0.8]),
            reference_exposure={"ball": 2.0},
            prediction_exposure={"track": 2.0},
        )
        far = score_entity_integrity(
            _matches("ball", "track", [0.1, 0.1]),
            reference_exposure={"ball": 2.0},
            prediction_exposure={"track": 2.0},
        )
        self.assertGreater(close.score, far.score)
        self.assertGreater(far.score, 0.0)
        self.assertAlmostEqual(
            close.integrity_gate,
            far.integrity_gate,
        )

    def test_invalid_reference_position_is_not_a_prediction_zero(self) -> None:
        common = {
            "reference_area_px2": 100.0,
            "frame_diagonal_px": 100.0,
        }
        with self.assertRaisesRegex(ValueError, "reference positions"):
            compare_positions([float("nan"), 0.0], [0.0, 0.0], **common)
        prediction_invalid = compare_positions(
            [0.0, 0.0],
            [float("nan"), 0.0],
            **common,
        )
        self.assertEqual(0.0, prediction_invalid.score)
        with self.assertRaisesRegex(ValueError, "reference_area_px2"):
            compare_positions(
                [0.0, 0.0],
                [1.0, 0.0],
                reference_area_px2=-1.0,
                frame_diagonal_px=100.0,
            )
        with self.assertRaisesRegex(ValueError, "frame_diagonal_px"):
            compare_positions(
                [0.0, 0.0],
                [1.0, 0.0],
                reference_area_px2=100.0,
                frame_diagonal_px=0.0,
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            compare_positions(
                [],
                [],
                reference_area_px2=100.0,
                frame_diagonal_px=100.0,
            )
        with self.assertRaisesRegex(ValueError, "unsupported distance kernel"):
            compare_positions(
                [0.0, 0.0],
                [0.0, 0.0],
                reference_area_px2=100.0,
                frame_diagonal_px=100.0,
                kernel="unknown",
            )

    def test_distance_similarity_rejects_negative_infinity_as_perfect(
        self,
    ) -> None:
        self.assertEqual(0.0, distance_similarity(float("-inf")))

    def test_persistent_duplicate_is_penalized_more_than_transient_extra(
        self,
    ) -> None:
        matches = _matches("ball", "primary", [1.0, 1.0])
        transient = score_entity_integrity(
            matches,
            reference_exposure={"ball": 2.0},
            prediction_exposure={"primary": 2.0, "duplicate": 1.0},
        )
        persistent = score_entity_integrity(
            matches,
            reference_exposure={"ball": 2.0},
            prediction_exposure={"primary": 2.0, "duplicate": 2.0},
        )
        self.assertGreater(transient.score, persistent.score)
        self.assertGreater(
            persistent.gospa.false_exposure,
            transient.gospa.false_exposure,
        )

    def test_integrity_is_a_non_dilutable_case_gate(self) -> None:
        matches = [
            EntityMatch(
                frame_index=frame,
                entity_id=entity,
                track_id=f"track_{entity}",
                localization_quality=1.0,
                normalized_distance=0.0,
            )
            for frame in range(2)
            for entity in ("a", "b", "c")
        ]
        integrity = score_entity_integrity(
            matches,
            reference_exposure={"a": 2.0, "b": 2.0, "c": 2.0},
            prediction_exposure={
                "track_a": 2.0,
                "track_b": 2.0,
                "track_c": 2.0,
                "persistent_extra": 2.0,
            },
        )
        result = compose_gated_case_score(
            integrity.integrity_gate,
            {"physics": 1.0, "visual": 1.0},
            {"physics": 0.5, "visual": 0.5},
        )
        self.assertAlmostEqual(0.75, integrity.integrity_gate)
        self.assertAlmostEqual(0.75, result.score)

    def test_cardinality_and_switch_counterfactual_bounds(self) -> None:
        stable_matches = [
            EntityMatch(frame, entity, f"track_{entity}", 1.0, 0.0)
            for frame in range(2)
            for entity in ("a", "b", "c")
        ]
        reference = {"a": 2.0, "b": 2.0, "c": 2.0}
        missing = score_entity_integrity(
            [
                match
                for match in stable_matches
                if match.entity_id != "c"
            ],
            reference_exposure=reference,
            prediction_exposure={"track_a": 2.0, "track_b": 2.0},
        )
        half_extra = score_entity_integrity(
            stable_matches,
            reference_exposure=reference,
            prediction_exposure={
                "track_a": 2.0,
                "track_b": 2.0,
                "track_c": 2.0,
                "extra": 1.0,
            },
        )
        half_switch = score_entity_integrity(
            [
                EntityMatch(
                    frame,
                    entity,
                    (
                        f"track_{entity}"
                        if frame == 0 or entity == "c"
                        else ("track_b" if entity == "a" else "track_a")
                    ),
                    1.0,
                    0.0,
                )
                for frame in range(2)
                for entity in ("a", "b", "c")
            ],
            reference_exposure=reference,
            prediction_exposure={
                "track_a": 2.0,
                "track_b": 2.0,
                "track_c": 2.0,
            },
        )
        self.assertAlmostEqual(2.0 / 3.0, missing.integrity_gate)
        self.assertAlmostEqual(6.0 / 7.0, half_extra.integrity_gate)
        self.assertAlmostEqual(5.0 / 9.0, half_switch.integrity_gate)

    def test_disappearance_is_a_finite_low_score_not_an_error(self) -> None:
        result = score_entity_integrity(
            _matches("ball", "track", [1.0]),
            reference_exposure={"ball": 4.0},
            prediction_exposure={"track": 1.0},
        )
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLess(result.score, 1.0)
        self.assertAlmostEqual(3.0, result.gospa.missed_exposure)

    def test_identity_switch_lowers_association(self) -> None:
        stable = [
            EntityMatch(index, entity, track, 1.0, 0.0)
            for index in range(4)
            for entity, track in (("a", "ta"), ("b", "tb"))
        ]
        switched = [
            EntityMatch(
                index,
                entity,
                track,
                1.0,
                0.0,
            )
            for index in range(4)
            for entity, track in (
                (("a", "ta"), ("b", "tb"))
                if index < 2
                else (("a", "tb"), ("b", "ta"))
            )
        ]
        exposures = {"a": 4.0, "b": 4.0}
        tracks = {"ta": 4.0, "tb": 4.0}
        stable_score = score_entity_integrity(
            stable,
            reference_exposure=exposures,
            prediction_exposure=tracks,
        )
        switched_score = score_entity_integrity(
            switched,
            reference_exposure=exposures,
            prediction_exposure=tracks,
        )
        self.assertAlmostEqual(1.0, stable_score.association_accuracy)
        self.assertLess(
            switched_score.association_accuracy,
            stable_score.association_accuracy,
        )
        self.assertLess(switched_score.score, stable_score.score)

    def test_unobservable_identity_slots_do_not_create_fake_switches(
        self,
    ) -> None:
        matches = [
            EntityMatch(0, "ball", "track_before", 1.0, 0.0),
            EntityMatch(
                1,
                "ball",
                "track_after",
                1.0,
                0.0,
                association_eligible=False,
            ),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "association exposure mappings",
        ):
            score_entity_integrity(
                matches,
                reference_exposure={"ball": 2.0},
                prediction_exposure={
                    "track_before": 1.0,
                    "track_after": 1.0,
                },
            )
        result = score_entity_integrity(
            matches,
            reference_exposure={"ball": 2.0},
            prediction_exposure={
                "track_before": 1.0,
                "track_after": 1.0,
            },
            reference_association_exposure={"ball": 1.0},
            prediction_association_exposure={
                "track_before": 1.0,
                "track_after": 0.0,
            },
        )
        self.assertAlmostEqual(1.0, result.integrity_gate)
        self.assertAlmostEqual(1.0, result.association_accuracy)
        self.assertAlmostEqual(1.0, result.association_matched_exposure)

    def test_match_exposure_is_conserved_per_entity_and_track(self) -> None:
        impossible_entity_reuse = [
            EntityMatch(0, "a", "t1", 1.0, 0.0),
            EntityMatch(1, "a", "t2", 1.0, 0.0),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "lifecycle for entity 'a'",
        ):
            score_entity_integrity(
                impossible_entity_reuse,
                reference_exposure={"a": 1.0, "b": 1.0},
                prediction_exposure={"t1": 1.0, "t2": 1.0},
            )

        impossible_track_reuse = [
            EntityMatch(0, "a", "t1", 1.0, 0.0),
            EntityMatch(1, "b", "t1", 1.0, 0.0),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "lifecycle for track 't1'",
        ):
            score_entity_integrity(
                impossible_track_reuse,
                reference_exposure={"a": 1.0, "b": 1.0},
                prediction_exposure={"t1": 1.0, "t2": 1.0},
            )

        fractional_reuse = [
            EntityMatch(0, "a", "t1", 1.0, 0.0, weight=0.25),
            EntityMatch(
                1,
                "a",
                "t2",
                1.0,
                0.0,
                weight=0.250000002,
            ),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "lifecycle for entity 'a'",
        ):
            score_entity_integrity(
                fractional_reuse,
                reference_exposure={"a": 0.5, "b": 0.5},
                prediction_exposure={"t1": 0.5, "t2": 0.5},
            )

    def test_gospa_parameters_must_be_finite(self) -> None:
        for kwargs in (
            {"gospa_order": float("nan")},
            {"gospa_order": float("inf")},
            {"gospa_cutoff": float("nan")},
            {"gospa_cutoff": float("inf")},
            {"gospa_alpha": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "GOSPA"):
                    score_entity_integrity(
                        [],
                        reference_exposure={},
                        prediction_exposure={},
                        **kwargs,
                    )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            score_entity_integrity(
                [],
                reference_exposure={1: 1.0},  # type: ignore[dict-item]
                prediction_exposure={},
            )

    def test_geometric_composition_gates_zero_and_omits_unavailable(
        self,
    ) -> None:
        gated = compose_weighted_geometric(
            {"shape": 0.0, "physics": 1.0, "appearance": None},
            {"shape": 0.5, "physics": 0.4, "appearance": 0.1},
        )
        self.assertEqual(0.0, gated.score)
        self.assertEqual(("appearance",), gated.omitted_components)
        self.assertAlmostEqual(
            1.0, sum(gated.weights_used.values())
        )


if __name__ == "__main__":
    unittest.main()
