from __future__ import annotations

import unittest

import cv2
import numpy as np

from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldTrack,
    compare_open_world_tracks,
)
from physbench.evaluation.common.entities.timeline import (
    build_common_time_grid,
)
from physbench.evaluation.scenes.collision.open_world import (
    _calibrate_residual_track_evidence,
    _matches_direct_mask,
    _select_entity_set,
    discover_prediction_objects,
    reference_tracks_from_instance_masks,
)


class CollisionOpenWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_common_time_grid(
            [0.0, 0.0625, 0.125, 0.1875]
        )
        self.observation = {
            "track_top_ratio": 0.5,
            "track_bottom_ratio": 0.9,
            "maximum_entity_y_spread_px": 20.0,
            "maximum_radius_ratio": 4.0,
            "minimum_circle_radius_px": 3,
            "maximum_circle_radius_px": 10,
            "minimum_circle_center_distance_px": 6,
            "circle_nms_overlap_distance_factor": 0.55,
            "maximum_candidates": 20,
            "hough_accumulator_thresholds": [30, 26],
            "hough_dp": 1.2,
            "edge_threshold": 100.0,
            "minimum_target_separation_px": 5.0,
            "maximum_target_separation_px": 40.0,
            "minimum_target_gap_radius_factor": 0.4,
            "maximum_target_gap_radius_factor": 4.0,
            "preferred_target_gap_radius_factor": 1.1,
            "minimum_striker_separation_px": 8.0,
            "minimum_striker_gap_radius_factor": 0.8,
            "motion_difference_threshold": 8,
            "motion_minimum_area": 5,
            "residual_minimum_radius_ratio": 0.4,
            "residual_maximum_radius_ratio": 2.5,
            "residual_direct_duplicate_center_fraction": 0.8,
            "residual_hough_confirmation_frames": 3,
            "residual_motion_confirmation_frames": 3,
            "residual_motion_minimum_displacement_radius_ratio": 2.0,
            "maximum_residual_tracks": 8,
        }
        self.quality = {
            "minimum_mask_pixels": 8,
            "maximum_mask_area_ratio": 0.08,
        }

    @staticmethod
    def _instances(
        xs: list[list[int]],
        *,
        width: int = 120,
        height: int = 60,
    ) -> list[list[np.ndarray]]:
        output = []
        for instance in xs:
            masks = []
            for x in instance:
                value = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(value, (x, 40), 5, 255, -1)
                masks.append(value)
            output.append(masks)
        return output

    def test_generic_selector_accepts_two_and_four_bodies(self) -> None:
        circles = [
            (10.0, 40.0, 5.0),
            (30.0, 40.0, 5.0),
            (50.0, 40.0, 5.0),
            (70.0, 40.0, 5.0),
        ]
        for count in (2, 4):
            selected = _select_entity_set(
                circles,
                expected_count=count,
                frame_width=120,
                config=self.observation,
            )
            self.assertIsNotNone(selected)
            self.assertEqual(count, len(selected[0]))

    def test_motion_supported_fourth_ball_is_residual(self) -> None:
        reference_masks = self._instances(
            [
                [20, 20, 20, 20],
                [50, 50, 50, 50],
                [80, 80, 80, 80],
            ]
        )
        prediction_masks = self._instances(
            [
                [20, 20, 20, 20],
                [50, 50, 50, 50],
                [80, 80, 80, 80],
            ]
        )
        frames = []
        for frame_index in range(4):
            frame = np.full((60, 120, 3), 120, dtype=np.uint8)
            for x in (20, 50, 80):
                cv2.circle(frame, (x, 40), 5, (230, 230, 230), -1)
            cv2.circle(
                frame,
                (100 + min(frame_index, 1), 40),
                5,
                (250, 250, 250),
                -1,
            )
            frames.append(frame)
        reference = reference_tracks_from_instance_masks(
            reference_masks,
            entity_ids=("ball_1", "ball_2", "ball_3"),
            entity_class="ball",
            time_grid=self.grid,
            minimum_area=8,
            maximum_area_ratio=0.08,
        )
        prediction = discover_prediction_objects(
            frames,
            directed_instance_masks=prediction_masks,
            time_grid=self.grid,
            observation_config=self.observation,
            quality_config=self.quality,
        )
        comparison = compare_open_world_tracks(
            reference_tracks=reference,
            prediction_observation=prediction,
            time_grid=self.grid,
            frame_diagonal_px=float(np.hypot(120, 60)),
        )
        self.assertLess(
            comparison.integrity.presence_detection_accuracy,
            1.0,
        )
        self.assertGreaterEqual(
            len(prediction.tracks),
            4,
        )

    def test_direct_duplicate_gate_does_not_merge_tangent_ball(self) -> None:
        direct = self._instances([[20, 20, 20, 20]])[0]

        def detection(x: float, radius: int) -> ObjectDetection:
            mask = np.zeros((60, 120), dtype=np.uint8)
            cv2.circle(mask, (int(round(x)), 40), radius, 255, -1)
            return ObjectDetection(
                frame_index=0,
                detection_id=f"candidate_{x:g}",
                xy=np.asarray([x, 40.0]),
                area_px2=float(np.count_nonzero(mask)),
                entity_class="ball",
                mask=mask,
                evidence_tier=EvidenceTier.TENTATIVE,
                sources=("hough_circle",),
            )

        shifted_same_body = detection(25.0, 8)
        tangent_distinct_body = detection(33.0, 8)
        self.assertTrue(
            _matches_direct_mask(
                shifted_same_body,
                [direct],
                center_distance_fraction=0.8,
            )
        )
        self.assertFalse(
            _matches_direct_mask(
                tangent_distinct_body,
                [direct],
                center_distance_fraction=0.8,
            )
        )

    def test_hough_only_evidence_never_becomes_full_participant(self) -> None:
        def track(
            track_id: str,
            *,
            source: str,
            frames: int,
            step_px: float = 0.0,
        ) -> OpenWorldTrack:
            detections = []
            for frame_index in range(frames):
                mask = np.zeros((60, 120), dtype=np.uint8)
                cv2.circle(mask, (100, 40), 5, 255, -1)
                detections.append(
                    ObjectDetection(
                        frame_index=frame_index,
                        detection_id=f"{track_id}_{frame_index}",
                        xy=np.asarray(
                            [100.0 + step_px * frame_index, 40.0]
                        ),
                        area_px2=float(np.count_nonzero(mask)),
                        entity_class="ball",
                        mask=mask,
                        confidence=0.95,
                        evidence_tier=(
                            EvidenceTier.PARTICIPANT
                            if source == "motion_circle"
                            else EvidenceTier.TENTATIVE
                        ),
                        sources=(source,),
                    )
                )
            return OpenWorldTrack(
                track_id=track_id,
                detections=tuple(detections),
                confirmed=True,
                evidence_tier=EvidenceTier.PARTICIPANT,
            )

        calibrated, diagnostics = _calibrate_residual_track_evidence(
            (
                track("short_hough", source="hough_circle", frames=1),
                track("persistent_hough", source="hough_circle", frames=3),
                track("motion", source="motion_circle", frames=1),
            ),
            hough_confirmation_frames=3,
            motion_confirmation_frames=3,
            motion_minimum_displacement_radius_ratio=2.0,
        )
        self.assertEqual(
            [
                EvidenceTier.AMBIGUOUS,
                EvidenceTier.TENTATIVE,
                EvidenceTier.TENTATIVE,
            ],
            [value.evidence_tier for value in calibrated],
        )
        self.assertEqual(
            {
                "ambiguous_tracks": 1,
                "tentative_tracks": 2,
                "participant_tracks": 0,
                "static_motion_rejections": 0,
                "insufficient_motion_support_tracks": 1,
            },
            diagnostics,
        )

        persistent_motion, moving_diagnostics = (
            _calibrate_residual_track_evidence(
            (
                track(
                    "persistent_motion",
                    source="motion_circle",
                    frames=3,
                    step_px=6.0,
                ),
            ),
            hough_confirmation_frames=3,
            motion_confirmation_frames=3,
            motion_minimum_displacement_radius_ratio=2.0,
            )
        )
        self.assertEqual(
            EvidenceTier.PARTICIPANT,
            persistent_motion[0].evidence_tier,
        )
        self.assertEqual(0, moving_diagnostics["static_motion_rejections"])

        static_motion, static_diagnostics = (
            _calibrate_residual_track_evidence(
                (
                    track(
                        "static_motion_ghost",
                        source="motion_circle",
                        frames=6,
                    ),
                ),
                hough_confirmation_frames=3,
                motion_confirmation_frames=3,
                motion_minimum_displacement_radius_ratio=2.0,
            )
        )
        self.assertEqual(
            EvidenceTier.TENTATIVE,
            static_motion[0].evidence_tier,
        )
        self.assertEqual(
            1,
            static_diagnostics["static_motion_rejections"],
        )


if __name__ == "__main__":
    unittest.main()
