from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from _paths import ROOT
from physbench.evaluation.common.entities import build_common_time_grid
from physbench.evaluation.scenes.pendulum.open_world import (
    PendulumStructureSpec,
    _weak_directed_string_circle,
    audit_pendulum_topology,
    detect_condition_structure,
    discover_pendulum_objects,
    extract_bob_masks,
    letterbox_condition_image,
    prompt_from_structure,
)
from physbench.evaluation.scenes.pendulum.v6_evaluator import (
    _time_coverage,
)


_OBSERVATION_CONFIG = {
    "blur_kernel": 7,
    "hough_dp": 1.2,
    "hough_accumulator_thresholds": [32, 27, 23, 19, 15, 12],
    "edge_threshold": 100.0,
    "minimum_center_distance_px": 12.0,
    "minimum_radius_ratio": 0.55,
    "maximum_radius_ratio": 1.8,
    "maximum_circle_candidates": 48,
    "maximum_condition_circle_candidates": 200,
    "line_canny_low": 35.0,
    "line_canny_high": 120.0,
    "line_hough_threshold": 20,
    "minimum_string_segment_length_px": 14,
    "maximum_string_line_gap_px": 14,
    "minimum_length_radius_ratio": 2.5,
    "maximum_length_canvas_ratio": 0.8,
    "minimum_condition_bob_angle_deg": 7.0,
    "minimum_condition_structure_score": 0.18,
    "support_exclusion_radius_ratio": 1.2,
    "minimum_bob_pixels": 8,
    "direct_minimum_area_ratio": 0.2,
    "direct_maximum_area_ratio": 4.0,
    "minimum_anchor_color_similarity": 0.1,
    "condition_change_threshold": 24.0,
    "minimum_residual_length_ratio": 0.35,
    "maximum_residual_length_ratio": 2.0,
    "direct_duplicate_overlap": 0.45,
    "direct_duplicate_center_radius_ratio": 2.25,
    "participant_string_score": 0.55,
    "minimum_change_overlap": 0.25,
    "minimum_string_candidate_change_overlap": 0.3,
    "minimum_residual_color_similarity": 0.05,
    "maximum_residual_pivot_error_ratio": 0.25,
    "maximum_tracking_gap_s": 0.25,
    "maximum_tracking_assignment_cost": 4.0,
    "maximum_tracks": 24,
}


def _assembly(
    *,
    pivot: tuple[int, int] = (60, 10),
    bob: tuple[int, int] = (82, 92),
    radius: int = 8,
    string: bool = True,
    branch: bool = False,
    apparatus: bool = False,
) -> np.ndarray:
    mask = np.zeros((120, 140), dtype=np.uint8)
    if string:
        cv2.line(mask, pivot, bob, 255, 3, cv2.LINE_AA)
    if branch:
        cv2.line(mask, (70, 48), (112, 58), 255, 3, cv2.LINE_AA)
    if apparatus:
        cv2.line(mask, pivot, (60, 78), 255, 3, cv2.LINE_AA)
        cv2.line(mask, (24, 10), (98, 10), 255, 3, cv2.LINE_AA)
    cv2.circle(mask, bob, radius, 255, -1, cv2.LINE_AA)
    return mask


class PendulumOpenWorldV6Tests(unittest.TestCase):
    def test_all_five_physics_parent_conditions_freeze_the_current_bob(
        self,
    ) -> None:
        root = ROOT / "datasets" / "assets" / "pendulum"
        expected = {
            "pendulum_ltot0110mm_lrope0100mm_r010mm_a010deg_ood01": (
                (329.0, 291.0),
                0.01 / 0.11,
            ),
            "pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood02": (
                (137.0, 334.0),
                0.01 / 0.13,
            ),
            "pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood03": (
                (140.0, 365.0),
                0.01 / 0.13,
            ),
            "pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood04": (
                (350.0, 415.0),
                0.01 / 0.155,
            ),
            "pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood05": (
                (345.0, 415.0),
                0.01 / 0.155,
            ),
        }
        if any(
            not list((root / case_id / "canonical").glob("first_frame.*"))
            for case_id in expected
        ):
            self.skipTest("full pendulum media assets are not published")
        for case_id, (target, ratio) in expected.items():
            with self.subTest(case_id=case_id):
                candidates = list(
                    (root / case_id / "canonical").glob("first_frame.*")
                )
                self.assertEqual(1, len(candidates))
                frame, _ = letterbox_condition_image(
                    candidates[0], width=480, height=832
                )
                structure = detect_condition_structure(
                    frame,
                    config=_OBSERVATION_CONFIG,
                    expected_radius_length_ratio=ratio,
                )
                self.assertLess(
                    float(
                        np.linalg.norm(
                            structure.bob_xy
                            - np.asarray(target, dtype=np.float64)
                        )
                    ),
                    5.0,
                )
                self.assertGreater(structure.confidence, 0.5)
                self.assertGreater(
                    np.count_nonzero(structure.bob_mask), 20
                )

    def test_bob_extraction_and_topology_distinguish_broken_string(
        self,
    ) -> None:
        intact = _assembly()
        broken = _assembly(string=False)
        bobs, xy, radii = extract_bob_masks(
            [intact, broken], expected_radius_px=8.0
        )
        self.assertTrue(np.isfinite(xy).all())
        self.assertTrue(np.isfinite(radii).all())
        self.assertTrue(all(np.count_nonzero(mask) > 20 for mask in bobs))
        intact_audit = audit_pendulum_topology(
            [intact],
            pivot_xy=[60.0, 10.0],
            bob_xy=xy[:1],
            bob_observed=[True],
            minimum_string_occupancy=0.25,
        )
        broken_audit = audit_pendulum_topology(
            [broken],
            pivot_xy=[60.0, 10.0],
            bob_xy=xy[1:],
            bob_observed=[True],
            minimum_string_occupancy=0.25,
        )
        self.assertGreater(
            float(intact_audit["score"]),
            float(broken_audit["score"]),
        )
        self.assertEqual(0, intact_audit["broken_frame_count"])
        self.assertEqual(1, broken_audit["broken_frame_count"])

    def test_branch_string_without_second_bob_has_duration_penalty(
        self,
    ) -> None:
        frame_count = 8
        normal = _assembly()
        forked = _assembly(branch=True)
        bob_mask = np.zeros_like(normal)
        cv2.circle(bob_mask, (82, 92), 8, 255, -1)
        xy = np.repeat(
            np.asarray([[82.0, 92.0]], dtype=np.float64),
            frame_count,
            axis=0,
        )
        common = {
            "pivot_xy": [60.0, 10.0],
            "bob_xy": xy,
            "bob_observed": [True] * frame_count,
            "bob_masks": [bob_mask] * frame_count,
            "minimum_string_occupancy": 0.25,
        }
        normal_audit = audit_pendulum_topology(
            [normal] * frame_count,
            **common,
        )
        apparatus_audit = audit_pendulum_topology(
            [_assembly(apparatus=True)] * frame_count,
            **common,
        )
        short_branch_audit = audit_pendulum_topology(
            [forked, *([normal] * (frame_count - 1))],
            **common,
        )
        long_branch_audit = audit_pendulum_topology(
            [forked] * (frame_count // 2)
            + [normal] * (frame_count // 2),
            **common,
        )
        self.assertGreater(float(normal_audit["score"]), 0.95)
        self.assertGreater(float(apparatus_audit["score"]), 0.95)
        self.assertGreater(
            float(normal_audit["score"]),
            float(short_branch_audit["score"]),
        )
        self.assertGreater(
            float(short_branch_audit["score"]),
            float(long_branch_audit["score"]),
        )
        self.assertEqual(0, normal_audit["branch_frame_count"])
        self.assertEqual(0, apparatus_audit["branch_frame_count"])
        self.assertEqual(1, short_branch_audit["branch_frame_count"])
        self.assertEqual(
            frame_count // 2,
            long_branch_audit["branch_frame_count"],
        )
        fork_row = long_branch_audit["per_frame"][0]
        self.assertTrue(fork_row["branch_string_detected"])
        self.assertGreater(fork_row["branch_string_evidence"], 0.5)
        self.assertLess(
            fork_row["topology_frame_score"],
            fork_row["string_intact_score"],
        )

    def test_condition_prompt_is_not_derived_from_prediction_pixels(
        self,
    ) -> None:
        bob = np.zeros((120, 140), dtype=np.uint8)
        subject = _assembly()
        cv2.circle(bob, (82, 92), 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray([60.0, 10.0]),
            bob_xy=np.asarray([82.0, 92.0]),
            bob_radius_px=8.0,
            bob_mask=bob,
            subject_mask=subject,
            confidence=1.0,
            source="unit_condition",
        )
        prompt = prompt_from_structure(structure)
        self.assertEqual(0, prompt.frame_index)
        np.testing.assert_allclose(prompt.points_xy[-1], structure.bob_xy)
        self.assertEqual(
            "unit_condition",
            structure.to_dict()["source"],
        )

    def test_missing_duration_monotonically_reduces_physics_coverage(
        self,
    ) -> None:
        weights = build_common_time_grid(
            np.arange(21, dtype=np.float64) / 20.0
        ).cell_weights_s
        expected = np.ones(21, dtype=bool)
        values = []
        for missing in (0, 2, 5, 10):
            observed = np.ones(21, dtype=bool)
            if missing:
                observed[-missing:] = False
            values.append(
                _time_coverage(observed, expected, weights)
            )
        self.assertEqual(sorted(values, reverse=True), values)
        self.assertEqual(len(values), len(set(values)))

    def test_extra_bob_is_retained_as_a_second_open_world_track(self) -> None:
        times = [0.0, 0.1, 0.2, 0.3]
        grid = build_common_time_grid(times)
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        cv2.line(condition, (60, 10), (82, 92), (235, 235, 235), 2)
        cv2.circle(condition, (82, 92), 8, (240, 240, 240), -1)
        structure_mask = _assembly()
        bob_mask = np.zeros((120, 140), dtype=np.uint8)
        cv2.circle(bob_mask, (82, 92), 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray([60.0, 10.0]),
            bob_xy=np.asarray([82.0, 92.0]),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=structure_mask,
            confidence=1.0,
            source="unit_condition",
        )
        frames = []
        directed = []
        for index in range(len(times)):
            frame = np.zeros_like(condition)
            first = (82 - index * 2, 92)
            second = (112 - index * 2, 85)
            cv2.line(frame, (60, 10), first, (235, 235, 235), 2)
            cv2.circle(frame, first, 8, (240, 240, 240), -1)
            cv2.line(frame, (60, 10), second, (225, 225, 225), 2)
            cv2.circle(frame, second, 8, (250, 250, 250), -1)
            frames.append(frame)
            mask = np.zeros((120, 140), dtype=np.uint8)
            cv2.circle(mask, first, 8, 255, -1)
            directed.append(mask)
        observation = discover_pendulum_objects(
            frames,
            directed_bob_masks=directed,
            condition_frame=condition,
            structure=structure,
            time_grid=grid,
            config={
                **_OBSERVATION_CONFIG,
                "hough_accumulator_thresholds": [14, 10, 8],
                "maximum_circle_candidates": 32,
            },
        )
        participant_tracks = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        self.assertGreaterEqual(len(participant_tracks), 2)

    def test_directed_drift_to_a_replacement_splits_the_frozen_identity(
        self,
    ) -> None:
        times = [0.0, 0.1, 0.2, 0.3]
        grid = build_common_time_grid(times)
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        cv2.line(condition, (60, 10), (82, 92), (235, 235, 235), 2)
        cv2.circle(condition, (82, 92), 8, (245, 245, 245), -1)
        bob_mask = np.zeros((120, 140), dtype=np.uint8)
        cv2.circle(bob_mask, (82, 92), 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray([60.0, 10.0]),
            bob_xy=np.asarray([82.0, 92.0]),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=_assembly(),
            confidence=1.0,
            source="unit_condition",
        )
        frames: list[np.ndarray] = []
        directed: list[np.ndarray] = []
        for index in range(len(times)):
            center = (82 - 2 * index, 92)
            frame = np.zeros_like(condition)
            color = (
                (245, 245, 245)
                if index < 2
                else (0, 0, 245)
            )
            cv2.line(frame, (60, 10), center, color, 2)
            cv2.circle(frame, center, 8, color, -1)
            mask = np.zeros(condition.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, 8, 255, -1)
            frames.append(frame)
            directed.append(mask)
        observation = discover_pendulum_objects(
            frames,
            directed_bob_masks=directed,
            condition_frame=condition,
            structure=structure,
            time_grid=grid,
            config=_OBSERVATION_CONFIG,
        )
        classes = {
            track.entity_class: {
                detection.frame_index
                for detection in track.detections
            }
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        }
        self.assertEqual({0, 1}, classes["pendulum_bob"])
        self.assertEqual(
            {2, 3},
            classes["pendulum_bob__replacement"],
        )
        self.assertEqual(
            2,
            observation.diagnostics["identity_rejected_frames"],
        )

    def test_offset_hough_fit_at_bob_string_junction_is_not_a_ghost_bob(
        self,
    ) -> None:
        times = (np.arange(20, dtype=np.float64) / 20.0).tolist()
        grid = build_common_time_grid(times)
        pivot = (100, 20)
        center = (100, 160)
        condition = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.line(condition, pivot, center, (245, 245, 245), 3)
        cv2.circle(condition, center, 9, (245, 245, 245), -1)
        bob_mask = np.zeros(condition.shape[:2], dtype=np.uint8)
        cv2.circle(bob_mask, center, 9, 255, -1)
        subject = np.zeros_like(bob_mask)
        cv2.line(subject, pivot, center, 255, 3)
        cv2.circle(subject, center, 9, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray(pivot, dtype=np.float64),
            bob_xy=np.asarray(center, dtype=np.float64),
            bob_radius_px=9.0,
            bob_mask=bob_mask,
            subject_mask=subject,
            confidence=1.0,
            source="unit_condition",
        )
        observation = discover_pendulum_objects(
            [condition.copy() for _ in times],
            directed_bob_masks=[bob_mask.copy() for _ in times],
            condition_frame=condition,
            structure=structure,
            time_grid=grid,
            config={
                **_OBSERVATION_CONFIG,
                "hough_accumulator_thresholds": [14, 10, 8],
            },
        )
        participants = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        self.assertEqual(1, len(participants))
        self.assertEqual("pendulum_bob", participants[0].entity_class)

    def test_mid_string_hough_fit_requires_independent_bob_evidence(
        self,
    ) -> None:
        # Values come from the real GT-self frame-70 false positive.  Its
        # Hough center is halfway along the directed string and only 1.68 px
        # from the centerline, but the circle has weak body-change support.
        common = {
            "radius_px": 15.960000991821289,
            "pivot_xy": [237.0, 88.0],
            "directed_bob_xy": [
                199.88771929824563,
                423.95438596491226,
            ],
            "directed_bob_radius_px": math.sqrt(570.0 / math.pi),
            "minimum_body_change_overlap": 0.3,
            "bob_clearance_radius_ratio": 2.25,
        }
        rejected, diagnostics = _weak_directed_string_circle(
            [216.60000610351562, 257.4000244140625],
            change_overlap=0.2572145545796738,
            **common,
        )
        self.assertTrue(rejected)
        self.assertTrue(diagnostics["directed_string_interior"])
        self.assertLess(
            float(
                diagnostics[
                    "directed_string_centerline_distance_px"
                ]
            ),
            2.0,
        )

        # A filled inline second bob has independent body-change evidence;
        # a bob at the end of a branch is off the frozen main centerline.
        inline_body, _ = _weak_directed_string_circle(
            [216.60000610351562, 257.4000244140625],
            change_overlap=0.8,
            **common,
        )
        branch_body, branch_diagnostics = (
            _weak_directed_string_circle(
                [300.0, 260.0],
                change_overlap=0.26,
                **common,
            )
        )
        self.assertFalse(inline_body)
        self.assertFalse(branch_body)
        self.assertFalse(
            branch_diagnostics["directed_string_interior"]
        )


if __name__ == "__main__":
    unittest.main()
