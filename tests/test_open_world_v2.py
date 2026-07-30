from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.entities.contracts import (
    EntitySpec,
    LifecyclePolicy,
    ObjectTrack,
    ReferenceCapability,
    VisibilityState,
)
from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
)
from physbench.evaluation.common.entities.matching import FrozenAssignment
from physbench.evaluation.common.entities.timeline import (
    build_common_time_grid,
)
from physbench.evaluation.common.entities.v2 import (
    ExpectedPositionSample,
    compare_open_world_v2,
    compose_object_centric_v2,
    fail_closed_open_world_v2,
    freeze_condition_identity,
    resolve_expected_entity_timeline,
    safe_compare_open_world_v2,
)


def _grid():
    return build_common_time_grid(
        [0.5, 1.5, 2.5, 3.5],
        interval_start_s=0.0,
        interval_end_s=4.0,
    )


def _entity(
    *,
    lifecycle: LifecyclePolicy = LifecyclePolicy.PERSISTENT,
) -> EntitySpec:
    return EntitySpec(
        entity_id="ball_0",
        role_id="subject",
        entity_class="ball",
        lifecycle=lifecycle,
        anchor={"color": "red"},
    )


def _reference_track(
    xy: np.ndarray | None = None,
    *,
    visibility: tuple[VisibilityState, ...] | None = None,
) -> ObjectTrack:
    points = (
        np.asarray(
            [[8.0, 8.0], [10.0, 8.0], [12.0, 8.0], [14.0, 8.0]],
            dtype=np.float64,
        )
        if xy is None
        else np.asarray(xy, dtype=np.float64)
    )
    observed = np.ones(4, dtype=bool)
    states = visibility or (VisibilityState.VISIBLE,) * 4
    return ObjectTrack(
        track_id="reference",
        matched_entity_id="ball_0",
        xy=points,
        observed=observed,
        visibility=states,
        areas_px2=np.full(4, 16.0),
        existence_observed=observed,
        localization_eligible=observed,
        association_eligible=observed,
        time_weights_s=np.ones(4),
        metadata={"entity_class": "ball"},
    )


def _mask(x: int, y: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = 255
    return mask


def _track(
    track_id: str,
    samples: dict[int, tuple[float, float]],
) -> OpenWorldTrack:
    detections = tuple(
        ObjectDetection(
            frame_index=frame,
            detection_id=f"{track_id}_{frame}",
            xy=np.asarray(xy, dtype=np.float64),
            area_px2=9.0,
            entity_class="ball",
            mask=_mask(round(xy[0]), round(xy[1])),
            confidence=1.0,
            evidence_tier=EvidenceTier.PARTICIPANT,
            sources=("unit_test",),
        )
        for frame, xy in sorted(samples.items())
    )
    return OpenWorldTrack(
        track_id=track_id,
        detections=detections,
        confirmed=True,
        evidence_tier=EvidenceTier.PARTICIPANT,
    )


def _observation(
    *tracks: OpenWorldTrack,
    overflow: np.ndarray | None = None,
) -> OpenWorldObservation:
    return OpenWorldObservation(
        tracks=tuple(tracks),
        overflow_counts=(
            np.zeros(4, dtype=np.float64)
            if overflow is None
            else overflow
        ),
    )


def _same_case_timeline(
    *,
    entity: EntitySpec | None = None,
    expected_mask: np.ndarray | None = None,
):
    return resolve_expected_entity_timeline(
        entity or _entity(),
        time_grid=_grid(),
        capability=ReferenceCapability.SAME_CASE_GT,
        reference_track=_reference_track(),
        declared_expected_mask=expected_mask,
        reference_masks=[
            _mask(8, 8),
            _mask(10, 8),
            _mask(12, 8),
            _mask(14, 8),
        ],
    )


class OpenWorldV2Test(unittest.TestCase):
    def test_physics_parent_supervises_persistent_cardinality_not_future_pixels(
        self,
    ) -> None:
        timeline = resolve_expected_entity_timeline(
            _entity(),
            time_grid=_grid(),
            capability=ReferenceCapability.PHYSICS_PARENT,
            condition_position=ExpectedPositionSample(
                xy=np.asarray([8.0, 8.0]),
                area_px2=16.0,
                mask=_mask(8, 8),
            ),
        )
        np.testing.assert_array_equal(
            timeline.expected_exists,
            [True, True, True, True],
        )
        np.testing.assert_array_equal(
            timeline.existence_supervised,
            [True, True, True, True],
        )
        np.testing.assert_array_equal(
            timeline.localization_supervised,
            [True, False, False, False],
        )
        prediction = _track(
            "generated",
            {
                0: (8.0, 8.0),
                1: (25.0, 25.0),
                2: (4.0, 27.0),
                3: (28.0, 3.0),
            },
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(prediction),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        self.assertAlmostEqual(1.0, result.integrity.integrity_gate)
        self.assertEqual(
            [None, None, None],
            [row["position_score"] for row in result.per_frame[1:]],
        )

    def test_parent_visibility_cannot_choose_a_nonpersistent_denominator(
        self,
    ) -> None:
        entity = _entity(lifecycle=LifecyclePolicy.MAY_EXIT)
        parent = _reference_track(
            visibility=(
                VisibilityState.VISIBLE,
                VisibilityState.VISIBLE,
                VisibilityState.OUT_OF_FRAME,
                VisibilityState.OUT_OF_FRAME,
            )
        )
        timeline = resolve_expected_entity_timeline(
            entity,
            time_grid=_grid(),
            capability=ReferenceCapability.PHYSICS_PARENT,
            reference_track=parent,
            condition_position=[8.0, 8.0],
        )
        np.testing.assert_array_equal(
            timeline.existence_supervised,
            [True, False, False, False],
        )
        np.testing.assert_array_equal(
            timeline.localization_supervised,
            [True, False, False, False],
        )

    def test_parent_future_identity_requires_condition_frozen_assignment(
        self,
    ) -> None:
        timeline = resolve_expected_entity_timeline(
            _entity(),
            time_grid=_grid(),
            capability=ReferenceCapability.PHYSICS_PARENT,
            condition_position=[8.0, 8.0],
            condition_identity_supervised=True,
        )
        prediction = _observation(
            _track(
                "generated",
                {
                    0: (8.0, 8.0),
                    1: (9.0, 8.0),
                    2: (10.0, 8.0),
                    3: (11.0, 8.0),
                },
            )
        )
        with self.assertRaisesRegex(ValueError, "condition-frozen"):
            compare_open_world_v2(
                expected_timelines=[timeline],
                prediction_observation=prediction,
                time_grid=_grid(),
                frame_diagonal_px=45.0,
            )
        frozen = FrozenAssignment(
            entity_to_track={"ball_0": "generated"},
            residual_track_ids=(),
            unmatched_entity_ids=(),
            total_cost=0.0,
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=prediction,
            time_grid=_grid(),
            frame_diagonal_px=45.0,
            frozen_identity=frozen,
        )
        self.assertAlmostEqual(1.0, result.integrity.integrity_gate)

    def test_missing_and_extra_exposure_are_both_penalized(self) -> None:
        timeline = _same_case_timeline()
        missing = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                _track("short", {0: (8.0, 8.0), 1: (10.0, 8.0)})
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        extra = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                _track(
                    "expected",
                    {
                        0: (8.0, 8.0),
                        1: (10.0, 8.0),
                        2: (12.0, 8.0),
                        3: (14.0, 8.0),
                    },
                ),
                _track(
                    "duplicate",
                    {
                        0: (8.0, 16.0),
                        1: (10.0, 16.0),
                        2: (12.0, 16.0),
                        3: (14.0, 16.0),
                    },
                ),
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        self.assertLess(missing.integrity.integrity_gate, 1.0)
        self.assertLess(extra.integrity.integrity_gate, 1.0)
        self.assertEqual(["ball_0"], missing.per_frame[3]["missing_entity_ids"])
        self.assertTrue(extra.per_frame[3]["extra_track_ids"])

    def test_null_assignment_reports_far_replacement_as_missing_and_extra(
        self,
    ) -> None:
        timeline = _same_case_timeline()
        far = _track(
            "far",
            {
                0: (300.0, 300.0),
                1: (300.0, 300.0),
                2: (300.0, 300.0),
                3: (300.0, 300.0),
            },
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(far),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
            minimum_match_position_similarity=0.1,
        )
        self.assertEqual((), result.matches)
        self.assertEqual(["ball_0"], result.per_frame[0]["missing_entity_ids"])
        self.assertEqual(["far"], result.per_frame[0]["extra_track_ids"])
        self.assertTrue(
            result.per_frame[0]["rejected_candidate_matches"]
        )
        self.assertEqual(0.0, result.integrity.integrity_gate)

    def test_id_switch_is_audited_and_lowers_association(self) -> None:
        timeline = _same_case_timeline()
        split = _observation(
            _track("first", {0: (8.0, 8.0), 1: (10.0, 8.0)}),
            _track("second", {2: (12.0, 8.0), 3: (14.0, 8.0)}),
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=split,
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        self.assertTrue(result.per_frame[2]["id_switches"])
        self.assertLess(result.integrity.association_accuracy, 1.0)
        self.assertLess(result.integrity.integrity_gate, 1.0)

    def test_frozen_id_does_not_let_a_replacement_fill_a_missing_object(
        self,
    ) -> None:
        timeline = _same_case_timeline()
        observation = _observation(
            _track("original", {0: (8.0, 8.0), 1: (10.0, 8.0)}),
            _track(
                "replacement",
                {2: (12.0, 8.0), 3: (14.0, 8.0)},
            ),
        )
        frozen = FrozenAssignment(
            entity_to_track={"ball_0": "original"},
            residual_track_ids=("replacement",),
            unmatched_entity_ids=(),
            total_cost=0.0,
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=observation,
            time_grid=_grid(),
            frame_diagonal_px=45.0,
            frozen_identity=frozen,
        )
        self.assertEqual(
            ["ball_0"], result.per_frame[2]["missing_entity_ids"]
        )
        self.assertEqual(
            ["replacement"], result.per_frame[2]["extra_track_ids"]
        )

    def test_deleting_wrong_replacement_frames_cannot_raise_v2_gate(
        self,
    ) -> None:
        timeline = _same_case_timeline()
        frozen = FrozenAssignment(
            entity_to_track={"ball_0": "original"},
            residual_track_ids=("replacement",),
            unmatched_entity_ids=(),
            total_cost=0.0,
        )
        deleted = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                _track(
                    "original",
                    {0: (8.0, 8.0), 1: (10.0, 8.0)},
                )
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
            frozen_identity=frozen,
        )
        replaced = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                _track(
                    "original",
                    {0: (8.0, 8.0), 1: (10.0, 8.0)},
                ),
                _track(
                    "replacement",
                    {2: (12.0, 8.0), 3: (14.0, 8.0)},
                ),
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
            frozen_identity=frozen,
        )
        self.assertLessEqual(
            deleted.integrity.integrity_gate,
            replaced.integrity.integrity_gate + 1e-12,
        )
        self.assertAlmostEqual(
            0.125,
            deleted.deletion_resistant_gate_ceiling,
        )
        self.assertGreater(
            deleted.raw_integrity_gate,
            deleted.integrity.integrity_gate,
        )

    def test_declared_may_exit_is_not_counted_as_disappearance(self) -> None:
        entity = _entity(lifecycle=LifecyclePolicy.MAY_EXIT)
        timeline = _same_case_timeline(
            entity=entity,
            expected_mask=np.asarray([True, True, False, False]),
        )
        exited = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                _track("ball", {0: (8.0, 8.0), 1: (10.0, 8.0)})
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        persisted = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                _track(
                    "ball",
                    {
                        0: (8.0, 8.0),
                        1: (10.0, 8.0),
                        2: (12.0, 8.0),
                        3: (14.0, 8.0),
                    },
                )
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        self.assertAlmostEqual(1.0, exited.integrity.integrity_gate)
        self.assertLess(persisted.integrity.integrity_gate, 1.0)
        self.assertEqual(
            ["ball_0"],
            exited.per_frame[2]["legally_absent_entity_ids"],
        )

    def test_overflow_is_formal_false_exposure(self) -> None:
        timeline = _same_case_timeline()
        expected = _track(
            "expected",
            {
                0: (8.0, 8.0),
                1: (10.0, 8.0),
                2: (12.0, 8.0),
                3: (14.0, 8.0),
            },
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(
                expected,
                overflow=np.ones(4, dtype=np.float64),
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        self.assertEqual(4.0, result.prediction_exposure["__overflow__"])
        self.assertLess(result.integrity.exposure_precision, 1.0)
        self.assertLess(result.integrity.integrity_gate, 1.0)

    def test_condition_identity_hook_is_frozen(self) -> None:
        first = EntitySpec("red", "left", "ball", anchor={"color": "red"})
        second = EntitySpec("blue", "right", "ball", anchor={"color": "blue"})
        color = {"track_a": "blue", "track_b": "red"}
        assignment = freeze_condition_identity(
            [first, second],
            ["track_a", "track_b"],
            identity_cost_hook=lambda entity, track_id: (
                0.0
                if entity.anchor["color"] == color[track_id]
                else 2.0
            ),
            maximum_match_cost=1.0,
        )
        self.assertEqual("track_b", assignment.entity_to_track["red"])
        self.assertEqual("track_a", assignment.entity_to_track["blue"])

    def test_failure_helpers_return_conservative_finite_zero(self) -> None:
        timeline = _same_case_timeline()
        direct = fail_closed_open_world_v2(
            expected_timelines=[timeline],
            time_grid=_grid(),
            reason="segmentation unavailable",
        )
        safe = safe_compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("observer crashed")
            ),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        for result in (direct, safe):
            self.assertTrue(result.failed)
            self.assertEqual(0.0, result.integrity.integrity_gate)
            self.assertTrue(np.isfinite(result.integrity.score))
            self.assertEqual([], result.to_dict()["matches"])
        composition = compose_object_centric_v2(
            safe,
            content_components={"physics": None},
            content_weights={"physics": 1.0},
        )
        self.assertEqual(0.0, composition.score)

    def test_matched_mask_and_iou_artifacts_are_kept(self) -> None:
        timeline = _same_case_timeline()
        prediction = _track(
            "expected",
            {
                0: (8.0, 8.0),
                1: (10.0, 8.0),
                2: (12.0, 8.0),
                3: (14.0, 8.0),
            },
        )
        result = compare_open_world_v2(
            expected_timelines=[timeline],
            prediction_observation=_observation(prediction),
            time_grid=_grid(),
            frame_diagonal_px=45.0,
        )
        self.assertTrue(all(mask is not None for mask in result.matched_masks))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), result.matched_mask_iou)
        self.assertEqual(
            [9, 9, 9, 9],
            result.to_dict()["matched_mask_area_px2"],
        )


if __name__ == "__main__":
    unittest.main()
