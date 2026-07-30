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
