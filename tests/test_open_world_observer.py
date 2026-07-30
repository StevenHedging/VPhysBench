from __future__ import annotations

import unittest

import cv2
import numpy as np

from physbench.evaluation.common.entities.contracts import (
    ObjectTrack,
    VisibilityState,
)
from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
    compare_open_world_tracks,
    deduplicate_frame_detections,
    track_open_world_detections,
)
from physbench.evaluation.common.entities.timeline import (
    build_common_time_grid,
)


def _mask(x: int, y: int = 20, radius: int = 3) -> np.ndarray:
    value = np.zeros((48, 96), dtype=np.uint8)
    cv2.circle(value, (x, y), radius, 255, -1)
    return value


def _detection(
    frame: int,
    name: str,
    x: float,
    *,
    y: float = 20.0,
    tier: EvidenceTier = EvidenceTier.PARTICIPANT,
    sources: tuple[str, ...] = ("circle", "motion"),
    metadata: dict[str, object] | None = None,
) -> ObjectDetection:
    mask = _mask(int(round(x)), int(round(y)))
    return ObjectDetection(
        frame_index=frame,
        detection_id=name,
        xy=np.asarray([x, y]),
        area_px2=float(np.count_nonzero(mask)),
        entity_class="ball",
        mask=mask,
        confidence=0.95,
        evidence_tier=tier,
        sources=sources,
        metadata=metadata or {},
    )


def _reference_track(
    entity_id: str,
    xs: list[float],
    *,
    grid,
) -> ObjectTrack:
    observed = np.ones(len(xs), dtype=bool)
    areas = np.full(len(xs), np.count_nonzero(_mask(10)), dtype=float)
    return ObjectTrack(
        track_id=f"gt_{entity_id}",
        matched_entity_id=entity_id,
        xy=np.column_stack([xs, np.full(len(xs), 20.0)]),
        observed=observed,
        visibility=tuple(VisibilityState.VISIBLE for _ in xs),
        areas_px2=areas,
        existence_observed=observed,
        localization_eligible=observed,
        association_eligible=observed,
        time_weights_s=grid.cell_weights_s,
        metadata={"entity_class": "ball"},
    )


class OpenWorldObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_common_time_grid([0.0, 0.125, 0.25])

    def test_pixel_duplicate_is_removed_but_nearby_object_is_not(self) -> None:
        first = _detection(0, "a", 10)
        duplicate = _detection(0, "b", 10)
        nearby = _detection(0, "c", 18)
        retained = deduplicate_frame_detections(
            [first, duplicate, nearby]
        )
        self.assertEqual(2, len(retained))

    def test_synchronous_tracks_remain_distinct(self) -> None:
        frames = [
            [
                _detection(index, f"a{index}", 10 + index),
                _detection(index, f"b{index}", 30 + index),
            ]
            for index in range(3)
        ]
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        self.assertEqual(2, len(observed.tracks))
        self.assertTrue(all(track.confirmed for track in observed.tracks))

    def test_tracking_partition_is_opt_in_and_blocks_identity_takeover(
        self,
    ) -> None:
        unpartitioned = track_open_world_detections(
            [
                [_detection(0, "direct_0", 10)],
                [_detection(1, "residual_1", 11)],
                [_detection(2, "residual_2", 12)],
            ],
            time_grid=self.grid,
        )
        self.assertEqual(1, len(unpartitioned.tracks))

        partitioned = track_open_world_detections(
            [
                [
                    _detection(
                        0,
                        "direct_0",
                        10,
                        metadata={
                            "exclusive_tracking_partition": "directed"
                        },
                    )
                ],
                [
                    _detection(
                        1,
                        "residual_1",
                        11,
                        metadata={
                            "exclusive_tracking_partition": "residual"
                        },
                    )
                ],
                [
                    _detection(
                        2,
                        "residual_2",
                        12,
                        metadata={
                            "exclusive_tracking_partition": "residual"
                        },
                    )
                ],
            ],
            time_grid=self.grid,
        )
        self.assertEqual(2, len(partitioned.tracks))
        self.assertEqual(
            {1, 2},
            {len(track.detections) for track in partitioned.tracks},
        )

    def test_full_duration_extra_caps_presence_at_two_thirds(self) -> None:
        references = [
            _reference_track("ball_1", [10, 11, 12], grid=self.grid),
            _reference_track("ball_2", [30, 31, 32], grid=self.grid),
        ]
        frames = [
            [
                _detection(index, f"a{index}", 10 + index),
                _detection(index, f"b{index}", 30 + index),
                _detection(index, f"x{index}", 60 + index),
            ]
            for index in range(3)
        ]
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        result = compare_open_world_tracks(
            reference_tracks=references,
            prediction_observation=observed,
            time_grid=self.grid,
            frame_diagonal_px=float(np.hypot(96, 48)),
        )
        self.assertAlmostEqual(
            2.0 / 3.0,
            result.integrity.presence_detection_accuracy,
            places=7,
        )

    def test_missing_and_distance_are_separate(self) -> None:
        references = [
            _reference_track("ball_1", [10, 10, 10], grid=self.grid),
            _reference_track("ball_2", [30, 30, 30], grid=self.grid),
        ]
        frames = [
            [_detection(index, f"a{index}", 70)]
            for index in range(3)
        ]
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        result = compare_open_world_tracks(
            reference_tracks=references,
            prediction_observation=observed,
            time_grid=self.grid,
            frame_diagonal_px=float(np.hypot(96, 48)),
        )
        self.assertAlmostEqual(
            0.5,
            result.integrity.presence_detection_accuracy,
            places=7,
        )
        self.assertLess(result.integrity.soft_detection_accuracy, 0.1)

    def test_null_assignment_separates_far_replacement_from_near_disjoint(
        self,
    ) -> None:
        reference = _reference_track(
            "ball_1",
            [10.0, 10.0, 10.0],
            grid=self.grid,
        )

        def compare_at(x: float):
            observed = track_open_world_detections(
                [
                    [_detection(index, f"candidate_{index}", x)]
                    for index in range(3)
                ],
                time_grid=self.grid,
            )
            return compare_open_world_tracks(
                reference_tracks=[reference],
                prediction_observation=observed,
                time_grid=self.grid,
                frame_diagonal_px=float(np.hypot(96, 48)),
                minimum_match_position_similarity=0.1,
            )

        near = compare_at(18.0)
        far = compare_at(90.0)
        for frame in near.per_frame:
            self.assertEqual(1, len(frame["matches"]))
            self.assertEqual([], frame["missing_entity_ids"])
            self.assertEqual([], frame["residual_track_ids"])
            self.assertEqual([], frame["rejected_candidate_matches"])
        for frame in far.per_frame:
            self.assertEqual([], frame["matches"])
            self.assertEqual(["ball_1"], frame["missing_entity_ids"])
            self.assertEqual(1, len(frame["residual_track_ids"]))
            self.assertEqual(1, len(frame["rejected_candidate_matches"]))
            rejected = frame["rejected_candidate_matches"][0]
            self.assertEqual(
                "position_similarity_below_minimum",
                rejected["rejection_reason"],
            )
            self.assertLess(rejected["position_score"], 0.1)
            self.assertGreater(frame["position_diagnostic_score"], 0.0)
            self.assertGreater(
                near.per_frame[frame["frame"]][
                    "position_diagnostic_score"
                ],
                frame["position_diagnostic_score"],
            )
        self.assertEqual(0.0, far.integrity.presence_detection_accuracy)
        self.assertEqual(0.0, far.integrity.exposure_recall)
        self.assertEqual(0.0, far.integrity.exposure_precision)

    def test_rejected_participant_does_not_block_close_tentative_tier(
        self,
    ) -> None:
        reference = _reference_track(
            "ball_1",
            [10.0, 10.0, 10.0],
            grid=self.grid,
        )
        participant = OpenWorldTrack(
            track_id="far_participant",
            detections=tuple(
                _detection(
                    index,
                    f"far_{index}",
                    90.0,
                    tier=EvidenceTier.PARTICIPANT,
                )
                for index in range(3)
            ),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        )
        tentative = OpenWorldTrack(
            track_id="near_tentative",
            detections=tuple(
                _detection(
                    index,
                    f"near_{index}",
                    11.0,
                    tier=EvidenceTier.TENTATIVE,
                    sources=("motion_circle",),
                )
                for index in range(3)
            ),
            confirmed=False,
            evidence_tier=EvidenceTier.TENTATIVE,
        )
        result = compare_open_world_tracks(
            reference_tracks=[reference],
            prediction_observation=OpenWorldObservation(
                tracks=(participant, tentative),
                overflow_counts=np.zeros(3),
            ),
            time_grid=self.grid,
            frame_diagonal_px=float(np.hypot(96, 48)),
            minimum_match_position_similarity=0.1,
        )
        for frame in result.per_frame:
            self.assertEqual(
                "near_tentative",
                frame["matches"][0]["prediction_track_id"],
            )
            self.assertEqual(
                ["far_participant"],
                frame["residual_track_ids"],
            )
            self.assertEqual([], frame["missing_entity_ids"])

    def test_null_assignment_similarity_threshold_is_validated(self) -> None:
        reference = _reference_track(
            "ball_1",
            [10.0, 10.0, 10.0],
            grid=self.grid,
        )
        observation = track_open_world_detections(
            [
                [_detection(index, f"candidate_{index}", 10.0)]
                for index in range(3)
            ],
            time_grid=self.grid,
        )
        for invalid in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare_open_world_tracks(
                    reference_tracks=[reference],
                    prediction_observation=observation,
                    time_grid=self.grid,
                    frame_diagonal_px=float(np.hypot(96, 48)),
                    minimum_match_position_similarity=invalid,
                )

    def test_single_frame_tentative_extra_is_not_free(self) -> None:
        references = [
            _reference_track("ball_1", [10, 10, 10], grid=self.grid)
        ]
        frames = [
            [_detection(index, f"a{index}", 10)]
            for index in range(3)
        ]
        frames[1].append(
            _detection(
                1,
                "tentative",
                60,
                tier=EvidenceTier.TENTATIVE,
                sources=("motion",),
            )
        )
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        result = compare_open_world_tracks(
            reference_tracks=references,
            prediction_observation=observed,
            time_grid=self.grid,
            frame_diagonal_px=float(np.hypot(96, 48)),
        )
        self.assertLess(
            result.integrity.presence_detection_accuracy,
            1.0,
        )
        self.assertGreater(
            result.integrity.presence_detection_accuracy,
            0.8,
        )

    def test_singleton_ambiguous_proposal_has_no_formal_exposure(self) -> None:
        frames = [
            [],
            [
                _detection(
                    1,
                    "ambiguous",
                    60,
                    tier=EvidenceTier.AMBIGUOUS,
                    sources=("hough_circle",),
                )
            ],
            [],
        ]
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        self.assertEqual(1, len(observed.tracks))
        self.assertFalse(observed.tracks[0].confirmed)
        self.assertEqual(
            EvidenceTier.AMBIGUOUS,
            observed.tracks[0].evidence_tier,
        )
        self.assertEqual(0.0, observed.tracks[0].formal_exposure_weight)

    def test_persistent_ambiguous_track_is_weak_not_participant(self) -> None:
        frames = [
            [
                _detection(
                    index,
                    f"hough_{index}",
                    60,
                    tier=EvidenceTier.AMBIGUOUS,
                    sources=("hough_circle",),
                )
            ]
            for index in range(3)
        ]
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        self.assertEqual(1, len(observed.tracks))
        self.assertTrue(observed.tracks[0].confirmed)
        self.assertEqual(
            EvidenceTier.TENTATIVE,
            observed.tracks[0].evidence_tier,
        )
        self.assertEqual(0.25, observed.tracks[0].formal_exposure_weight)

    def test_persistence_suppression_requires_an_explicit_marker(self) -> None:
        frames = [
            [
                _detection(
                    index,
                    f"apparatus_{index}",
                    60,
                    tier=EvidenceTier.PARTICIPANT,
                    metadata={
                        "suppress_persistence_only_promotion": True
                    },
                )
            ]
            for index in range(3)
        ]
        observed = track_open_world_detections(
            frames,
            time_grid=self.grid,
        )
        self.assertTrue(observed.tracks[0].confirmed)
        self.assertEqual(
            EvidenceTier.AMBIGUOUS,
            observed.tracks[0].evidence_tier,
        )
        self.assertEqual(0.0, observed.tracks[0].formal_exposure_weight)

    def test_motion_participant_remains_full_exposure(self) -> None:
        frames = [
            [
                _detection(
                    index,
                    f"motion_{index}",
                    60 + index,
                    tier=EvidenceTier.PARTICIPANT,
                    sources=("motion_circle",),
                )
            ]
            for index in range(3)
        ]
        observed = track_open_world_detections(
            frames, time_grid=self.grid
        )
        self.assertEqual(
            EvidenceTier.PARTICIPANT,
            observed.tracks[0].evidence_tier,
        )
        self.assertEqual(1.0, observed.tracks[0].formal_exposure_weight)

    def test_ambiguous_candidate_cannot_steal_hungarian_match(self) -> None:
        reference = _reference_track(
            "ball_1",
            [10.0, 10.0, 10.0],
            grid=self.grid,
        )
        participant_detections = tuple(
            _detection(
                index,
                f"participant_{index}",
                14.0,
                tier=EvidenceTier.PARTICIPANT,
                sources=("directed",),
            )
            for index in range(3)
        )
        ambiguous_detection = _detection(
            1,
            "ambiguous_exact",
            10.0,
            tier=EvidenceTier.AMBIGUOUS,
            sources=("hough_circle",),
        )
        observation = OpenWorldObservation(
            tracks=(
                OpenWorldTrack(
                    track_id="participant",
                    detections=participant_detections,
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
                OpenWorldTrack(
                    track_id="ambiguous",
                    detections=(ambiguous_detection,),
                    confirmed=False,
                    evidence_tier=EvidenceTier.AMBIGUOUS,
                ),
            ),
            overflow_counts=np.zeros(3),
        )
        result = compare_open_world_tracks(
            reference_tracks=[reference],
            prediction_observation=observation,
            time_grid=self.grid,
            frame_diagonal_px=float(np.hypot(96, 48)),
        )
        frame = result.per_frame[1]
        self.assertEqual(
            "participant",
            frame["matches"][0]["prediction_track_id"],
        )
        self.assertEqual([], frame["residual_track_ids"])
        self.assertEqual(
            ["ambiguous"],
            frame["ambiguous_candidate_track_ids"],
        )
        self.assertAlmostEqual(
            1.0,
            result.integrity.presence_detection_accuracy,
        )

    def test_overflow_is_exposure_not_failure(self) -> None:
        frames = [
            [
                _detection(0, f"d{index}", 5 + 8 * index)
                for index in range(8)
            ],
            [],
            [],
        ]
        observed = track_open_world_detections(
            frames,
            time_grid=self.grid,
            maximum_tracks=2,
        )
        self.assertEqual(2, len(observed.tracks))
        self.assertGreater(observed.overflow_counts[0], 0.0)


if __name__ == "__main__":
    unittest.main()
