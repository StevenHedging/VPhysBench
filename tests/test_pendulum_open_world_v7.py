from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from _paths import ROOT
from physbench.evaluation.common.entities import build_common_time_grid
from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldTrack,
)
from physbench.evaluation.scenes.pendulum.open_world import (
    PendulumStructureSpec,
    letterbox_condition_image,
)
from physbench.evaluation.scenes.pendulum.scoring import score_traces
from physbench.evaluation.scenes.pendulum.v7_open_world import (
    _calibrate_residual_tracks,
    _condition_preexistence_score,
    _directed_continuity,
    _terminal_residual_relation,
    compare_pendulum_topology_v7,
    detect_condition_structure_v7,
    discover_pendulum_objects_v7,
    extract_bob_trace_v7,
    fuse_pendulum_proposals_v7,
    observe_pendulum_topology_v7,
    track_masks,
)


_OBSERVATION_CONFIG = {
    "blur_kernel": 7,
    "hough_dp": 1.2,
    "hough_accumulator_thresholds": [32, 27, 23, 19, 15, 12],
    "edge_threshold": 100.0,
    "minimum_center_distance_px": 12.0,
    "minimum_radius_ratio": 0.55,
    "maximum_radius_ratio": 1.8,
    "maximum_condition_circle_candidates": 200,
    "line_canny_low": 35.0,
    "line_canny_high": 120.0,
    "line_hough_threshold": 20,
    "minimum_string_segment_length_px": 14,
    "maximum_string_line_gap_px": 14,
    "minimum_length_radius_ratio": 2.5,
    "maximum_length_canvas_ratio": 0.8,
    "v7_condition_minimum_visible_angle_deg": 7.0,
    "v7_condition_support_continuation_minimum_alignment": 0.90,
    "v7_condition_support_continuation_maximum_axis_distance_radius_ratio": (
        1.20
    ),
    "v7_condition_support_continuation_minimum_extension_radius_ratio": 2.50,
}

_TOPOLOGY_CONFIG = {
    "string_half_width_px": 2,
    "minimum_string_occupancy": 0.3,
    "branch_string_penalty_weight": 0.65,
    "branch_minimum_length_ratio": 0.12,
    "branch_minimum_elongation": 3.0,
    "branch_maximum_thickness_radius_ratio": 0.75,
    "branch_bob_exclusion_margin_ratio": 0.4,
    "branch_support_half_width_radius_ratio": 1.25,
    "branch_support_verticality_threshold": 0.9,
    "branch_pivot_support_verticality_threshold": 0.65,
    "v7_topology_persistence_frames": 3,
}


def _mask(
    center: tuple[int, int],
    *,
    shape: tuple[int, int] = (120, 140),
    radius: int = 8,
) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    cv2.circle(output, center, radius, 255, -1)
    return output


def _detection(
    frame_index: int,
    center: tuple[int, int],
    *,
    detection_id: str,
    sources: tuple[str, ...],
    metadata: dict[str, object],
    tier: EvidenceTier = EvidenceTier.TENTATIVE,
) -> ObjectDetection:
    mask = _mask(center)
    return ObjectDetection(
        frame_index=frame_index,
        detection_id=detection_id,
        xy=np.asarray(center, dtype=np.float64),
        area_px2=float(np.count_nonzero(mask)),
        entity_class=(
            "pendulum_bob__replacement"
            if metadata.get("identity_anchor_valid") is False
            else "pendulum_bob"
        ),
        mask=mask,
        confidence=0.8,
        evidence_tier=tier,
        sources=sources,
        metadata=metadata,
    )


class PendulumOpenWorldV7Tests(unittest.TestCase):
    def test_short_gap_velocity_recovery_does_not_admit_replacement(
        self,
    ) -> None:
        grid = build_common_time_grid([0.0, 0.1, 0.2, 0.3])
        pivot = np.asarray([60.0, 10.0])
        anchor = np.asarray([100.0, 80.0])
        structure = PendulumStructureSpec(
            pivot_xy=pivot,
            bob_xy=anchor,
            bob_radius_px=8.0,
            bob_mask=_mask((100, 80)),
            subject_mask=_mask((100, 80)),
            confidence=1.0,
            source="unit_condition",
        )

        def directed(index: int, center: tuple[int, int]) -> ObjectDetection:
            return _detection(
                index,
                center,
                detection_id=f"directed_{index}",
                sources=("condition_directed_sam2",),
                metadata={"identity_anchor_valid": True},
                tier=EvidenceTier.PARTICIPANT,
            )

        accepted = [
            directed(0, (100, 80)),
            directed(1, (82, 92)),
        ]
        recovered, recovered_audit = _directed_continuity(
            directed(3, (46, 116)),
            accepted=accepted,
            structure=structure,
            time_grid=grid,
            config={
                "v7_directed_maximum_jump_radius_ratio": 5.0,
                "v7_directed_maximum_velocity_error_radius_ratio": 4.0,
                "v7_directed_maximum_recovery_gap_steps": 3.0,
            },
        )
        replacement, replacement_audit = _directed_continuity(
            directed(2, (46, 116)),
            accepted=accepted,
            structure=structure,
            time_grid=grid,
            config={
                "v7_directed_maximum_jump_radius_ratio": 5.0,
                "v7_directed_maximum_velocity_error_radius_ratio": 4.0,
                "v7_directed_maximum_recovery_gap_steps": 3.0,
            },
        )
        self.assertTrue(recovered)
        self.assertGreater(
            float(recovered_audit["directed_displacement_radius_ratio"]),
            5.0,
        )
        self.assertAlmostEqual(
            2.0,
            float(recovered_audit["directed_elapsed_gap_steps"]),
            places=12,
        )
        self.assertFalse(replacement)
        self.assertFalse(replacement_audit["directed_jump_valid"])

    def test_condition_ncc_rejects_jittered_apparatus_not_new_extra(
        self,
    ) -> None:
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        cv2.rectangle(condition, (65, 35), (91, 63), (90, 180, 230), -1)
        cv2.line(condition, (64, 34), (92, 64), (255, 255, 255), 2)
        shifted = cv2.warpAffine(
            condition,
            np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]),
            (140, 120),
        )
        shifted = cv2.convertScaleAbs(shifted, alpha=1.0, beta=8)
        apparatus = _condition_preexistence_score(
            shifted,
            condition,
            center_xy=(80.0, 48.0),
            radius_px=10.0,
            search_radius_px=3,
            patch_radius_ratio=1.5,
        )
        with_extra = np.array(condition, copy=True)
        cv2.circle(with_extra, (112, 84), 9, (240, 240, 240), -1)
        extra = _condition_preexistence_score(
            with_extra,
            condition,
            center_xy=(112.0, 84.0),
            radius_px=9.0,
            search_radius_px=3,
            patch_radius_ratio=1.5,
        )
        self.assertGreater(apparatus, 0.84)
        self.assertLess(extra, 0.84)

    def test_terminal_residual_repairs_mask_without_merging_second_branch(
        self,
    ) -> None:
        pivot = np.asarray([60.0, 10.0])
        bob = np.asarray([82.0, 92.0])
        structure = PendulumStructureSpec(
            pivot_xy=pivot,
            bob_xy=bob,
            bob_radius_px=8.0,
            bob_mask=_mask((82, 92)),
            subject_mask=_mask((82, 92)),
            confidence=1.0,
            source="unit_condition",
        )
        directed_mask = _mask((80, 84), radius=5)
        directed = ObjectDetection(
            frame_index=0,
            detection_id="directed_fragment",
            xy=np.asarray([80.0, 84.0]),
            area_px2=float(np.count_nonzero(directed_mask)),
            entity_class="pendulum_bob",
            mask=directed_mask,
            confidence=0.95,
            evidence_tier=EvidenceTier.PARTICIPANT,
            sources=("condition_directed_sam2",),
            metadata={"identity_anchor_valid": True},
        )
        residual_mask = _mask((82, 92))
        relation = _terminal_residual_relation(
            center_xy=np.asarray([82.0, 92.0]),
            radius_px=8.0,
            area_px2=float(np.count_nonzero(residual_mask)),
            directed_center_xy=directed.xy,
            directed_area_px2=directed.area_px2,
            structure=structure,
            config={},
        )
        self.assertTrue(relation["terminal_residual_repair"])
        residual = ObjectDetection(
            frame_index=0,
            detection_id="terminal_residual",
            xy=np.asarray([82.0, 92.0]),
            area_px2=float(np.count_nonzero(residual_mask)),
            entity_class="pendulum_bob",
            mask=residual_mask,
            confidence=0.85,
            evidence_tier=EvidenceTier.TENTATIVE,
            sources=("pendulum_v7_residual_candidate",),
            metadata={
                "identity_anchor_valid": True,
                "independent_residual_evidence": True,
                **relation,
            },
        )
        fused = fuse_pendulum_proposals_v7([directed, residual])
        self.assertEqual(1, len(fused))
        np.testing.assert_allclose(fused[0].xy, residual.xy)
        self.assertEqual(
            "condition_directed_sam2",
            fused[0].metadata["identity_evidence_source"],
        )
        self.assertEqual(
            "terminal_residual",
            fused[0].metadata["localization_evidence_source"],
        )

        other_relation = _terminal_residual_relation(
            center_xy=np.asarray([25.0, 80.0]),
            radius_px=8.0,
            area_px2=float(np.count_nonzero(residual_mask)),
            directed_center_xy=directed.xy,
            directed_area_px2=directed.area_px2,
            structure=structure,
            config={},
        )
        self.assertFalse(other_relation["terminal_residual_repair"])

    def test_directed_subject_residual_recovers_one_missing_sam_bob(
        self,
    ) -> None:
        times = [0.0, 0.1, 0.2, 0.3]
        grid = build_common_time_grid(times)
        pivot = (60, 10)
        centers = [(82, 92), (78, 92), (74, 92), (70, 90)]
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        cv2.line(condition, pivot, centers[0], (235, 235, 235), 2)
        cv2.circle(condition, centers[0], 8, (240, 240, 240), -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray(pivot, dtype=np.float64),
            bob_xy=np.asarray(centers[0], dtype=np.float64),
            bob_radius_px=8.0,
            bob_mask=_mask(centers[0]),
            subject_mask=_mask(centers[0]),
            confidence=1.0,
            source="unit_condition",
        )
        frames: list[np.ndarray] = []
        directed: list[np.ndarray] = []
        for index, center in enumerate(centers):
            frame = np.zeros_like(condition)
            cv2.line(frame, pivot, center, (235, 235, 235), 2)
            cv2.circle(frame, center, 8, (240, 240, 240), -1)
            frames.append(frame)
            directed.append(
                np.zeros(condition.shape[:2], dtype=np.uint8)
                if index == 2
                else _mask(center)
            )
        residual = [[] for _ in times]
        residual[2] = [
            _detection(
                2,
                centers[2],
                detection_id="sam_terminal_recovery",
                sources=("pendulum_v7_residual_candidate",),
                metadata={
                    "identity_anchor_valid": True,
                    "independent_residual_evidence": True,
                    "condition_identity_recovery_candidate": True,
                    "directed_subject_overlap": 0.75,
                },
            )
        ]
        with patch(
            "physbench.evaluation.scenes.pendulum.v7_open_world."
            "_residual_detections",
            return_value=(
                residual,
                {
                    "weak_string_circle": 0,
                    "apparatus": 0,
                    "condition_preexisting_apparatus": 0,
                    "single_source": 0,
                },
            ),
        ):
            observation = discover_pendulum_objects_v7(
                frames,
                directed_bob_masks=directed,
                directed_subject_masks=[
                    np.array(value, copy=True) for value in directed
                ],
                condition_frame=condition,
                structure=structure,
                time_grid=grid,
                config={
                    **_OBSERVATION_CONFIG,
                    "minimum_bob_pixels": 8,
                    "direct_minimum_area_ratio": 0.2,
                    "direct_maximum_area_ratio": 4.0,
                    "minimum_anchor_color_similarity": 0.1,
                    "maximum_tracking_gap_s": 0.25,
                    "maximum_tracking_assignment_cost": 4.0,
                    "maximum_tracks": 24,
                },
            )
        condition_track = next(
            value
            for value in observation.tracks
            if value.track_id == "track_condition_bob"
        )
        self.assertEqual(
            [0, 1, 2, 3],
            [value.frame_index for value in condition_track.detections],
        )
        self.assertEqual(
            1, observation.diagnostics["residual_identity_recoveries"]
        )

    def test_recovery_ranks_causal_geometry_before_confidence(self) -> None:
        times = [0.0, 0.1, 0.2, 0.3, 0.4]
        grid = build_common_time_grid(times)
        pivot = (60, 10)
        centers = [(82, 92), (78, 92), (74, 92), (70, 92), (66, 92)]
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        cv2.line(condition, pivot, centers[0], (235, 235, 235), 2)
        cv2.circle(condition, centers[0], 8, (240, 240, 240), -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray(pivot, dtype=np.float64),
            bob_xy=np.asarray(centers[0], dtype=np.float64),
            bob_radius_px=8.0,
            bob_mask=_mask(centers[0]),
            subject_mask=_mask(centers[0]),
            confidence=1.0,
            source="unit_condition",
        )
        frames: list[np.ndarray] = []
        directed: list[np.ndarray] = []
        for index, center in enumerate(centers):
            frame = np.zeros_like(condition)
            cv2.line(frame, pivot, center, (235, 235, 235), 2)
            cv2.circle(frame, center, 8, (240, 240, 240), -1)
            frames.append(frame)
            directed.append(
                np.zeros(condition.shape[:2], dtype=np.uint8)
                if index == 2
                else _mask(center)
            )
        residual = [[] for _ in times]
        causal = _detection(
            2,
            centers[2],
            detection_id="causal_geometry",
            sources=("pendulum_v7_residual_candidate",),
            metadata={
                "identity_anchor_valid": True,
                "independent_residual_evidence": True,
                "condition_geometry_recovery_candidate": True,
                "body_contrast_score": 1.0,
                "circle_edge_support": 1.0,
                "string_score": 0.9,
            },
        )
        high_confidence_wrong = _detection(
            2,
            (74, 80),
            detection_id="high_confidence_wrong",
            sources=("pendulum_v7_residual_candidate",),
            metadata={
                "identity_anchor_valid": True,
                "independent_residual_evidence": True,
                "condition_geometry_recovery_candidate": True,
                "body_contrast_score": 1.0,
                "circle_edge_support": 1.0,
                "string_score": 0.9,
            },
        )
        high_confidence_wrong = replace(
            high_confidence_wrong,
            confidence=0.99,
        )
        residual[2] = [causal, high_confidence_wrong]
        with patch(
            "physbench.evaluation.scenes.pendulum.v7_open_world."
            "_residual_detections",
            return_value=(
                residual,
                {
                    "weak_string_circle": 0,
                    "apparatus": 0,
                    "condition_preexisting_apparatus": 0,
                    "single_source": 0,
                },
            ),
        ):
            observation = discover_pendulum_objects_v7(
                frames,
                directed_bob_masks=directed,
                condition_frame=condition,
                structure=structure,
                time_grid=grid,
                config={
                    **_OBSERVATION_CONFIG,
                    "minimum_bob_pixels": 8,
                    "direct_minimum_area_ratio": 0.2,
                    "direct_maximum_area_ratio": 4.0,
                    "minimum_anchor_color_similarity": 0.1,
                    "maximum_tracking_gap_s": 0.25,
                    "maximum_tracking_assignment_cost": 4.0,
                    "maximum_tracks": 24,
                    "v7_recovery_maximum_velocity_error_radius_ratio": 2.0,
                },
            )
        condition_track = next(
            value
            for value in observation.tracks
            if value.track_id == "track_condition_bob"
        )
        recovered = next(
            value
            for value in condition_track.detections
            if value.frame_index == 2
        )
        np.testing.assert_allclose(recovered.xy, centers[2])
        self.assertEqual(
            "condition_geometry_causal_continuity",
            recovered.metadata["identity_evidence_source"],
        )

    def test_fusion_is_transitive_and_permutation_invariant(self) -> None:
        values = [
            _detection(
                0,
                (20 + 8 * index, 70),
                detection_id=f"proposal_{index}",
                sources=("pendulum_v7_residual_candidate",),
                metadata={
                    "identity_anchor_valid": True,
                    "independent_residual_evidence": True,
                },
            )
            for index in range(3)
        ]
        forward = fuse_pendulum_proposals_v7(values)
        reverse = fuse_pendulum_proposals_v7(list(reversed(values)))
        self.assertEqual(1, len(forward))
        self.assertEqual(1, len(reverse))
        self.assertEqual(
            forward[0].metadata["fused_detection_ids"],
            reverse[0].metadata["fused_detection_ids"],
        )
        np.testing.assert_allclose(forward[0].xy, reverse[0].xy)

    def test_all_physics_parent_conditions_freeze_visible_child_bob(
        self,
    ) -> None:
        root = (
            ROOT
            / "datasets"
            / "physics_video"
            / "assets"
            / "pendulum"
        )
        expected = {
            "pendulum_ltot0110mm_lrope0100mm_r010mm_a010deg_ood01": (
                (329.0, 291.0),
                0.01 / 0.11,
                10.0,
            ),
            "pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood02": (
                (137.0, 334.0),
                0.01 / 0.13,
                30.0,
            ),
            "pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood03": (
                (140.0, 365.0),
                0.01 / 0.13,
                30.0,
            ),
            "pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood04": (
                (350.0, 415.0),
                0.01 / 0.155,
                20.0,
            ),
            "pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood05": (
                (345.0, 415.0),
                0.01 / 0.155,
                20.0,
            ),
        }
        ood01_support_rejections = 0
        for case_id, (target, ratio, angle_deg) in expected.items():
            with self.subTest(case_id=case_id):
                candidates = list(
                    (root / case_id / "canonical").glob("first_frame.*")
                )
                self.assertEqual(1, len(candidates))
                frame, _ = letterbox_condition_image(
                    candidates[0], width=480, height=832
                )
                decision = detect_condition_structure_v7(
                    frame,
                    config=_OBSERVATION_CONFIG,
                    expected_radius_length_ratio=ratio,
                    expected_initial_angle_deg=angle_deg,
                )
                self.assertLess(
                    float(
                        np.linalg.norm(
                            decision.structure.bob_xy
                            - np.asarray(target, dtype=np.float64)
                        )
                    ),
                    5.0,
                )
                self.assertGreater(
                    np.count_nonzero(decision.structure.bob_mask), 20
                )
                if case_id.endswith("ood01"):
                    ood01_support_rejections = decision.rejection_counts[
                        "support_continues_below_candidate"
                    ]
        # OOD01 used to select a circular fit on the through-going stand at
        # [246.6, 378.6].  The terminal-body veto must remain exercised.
        self.assertGreater(ood01_support_rejections, 0)

    def test_condition_only_multisource_anchor_fixes_r1_small_angle_case(
        self,
    ) -> None:
        path = (
            ROOT
            / "datasets"
            / "physics_video"
            / "assets"
            / "pendulum"
            / "pendulum_r1_ltot0210mm_lrope0200mm_r010mm_a005deg"
            / "canonical"
            / "first_frame.png"
        )
        frame, _ = letterbox_condition_image(
            path, width=480, height=832
        )
        decision = detect_condition_structure_v7(
            frame,
            config=_OBSERVATION_CONFIG,
            expected_radius_length_ratio=0.01 / 0.21,
            expected_initial_angle_deg=5.0,
        )
        np.testing.assert_allclose(
            decision.structure.bob_xy,
            np.asarray([292.2, 431.4]),
            atol=4.0,
        )
        self.assertGreaterEqual(decision.source_agreement, 0.8)
        self.assertIn(
            "condition_v7",
            decision.structure.source,
        )

    def test_partial_hough_string_extends_to_long_pendulum_pivot(
        self,
    ) -> None:
        path = (
            ROOT
            / "datasets"
            / "physics_video"
            / "assets"
            / "pendulum"
            / "pendulum_r1_ltot0280mm_lrope0270mm_r010mm_a025deg"
            / "canonical"
            / "first_frame.png"
        )
        frame, _ = letterbox_condition_image(
            path, width=480, height=832
        )
        decision = detect_condition_structure_v7(
            frame,
            config=_OBSERVATION_CONFIG,
            expected_radius_length_ratio=0.01 / 0.28,
            expected_initial_angle_deg=25.0,
        )
        np.testing.assert_allclose(
            decision.structure.bob_xy,
            np.asarray([405.0, 522.0]),
            atol=4.0,
        )
        np.testing.assert_allclose(
            decision.structure.pivot_xy,
            np.asarray([233.0, 95.0]),
            atol=12.0,
        )
        self.assertTrue(
            decision.hypotheses[0]["physical_pivot_extended"]
        )
        self.assertGreaterEqual(
            float(decision.hypotheses[0]["full_string_edge_score"]),
            0.9,
        )

    def test_overlapping_sources_fuse_before_identity_tracking(self) -> None:
        invalid_direct = _detection(
            0,
            (82, 92),
            detection_id="direct",
            sources=("condition_directed_sam2", "identity_anchor_rejected"),
            metadata={"identity_anchor_valid": False},
            tier=EvidenceTier.PARTICIPANT,
        )
        residual = _detection(
            0,
            (83, 92),
            detection_id="residual",
            sources=("pendulum_v7_residual_candidate",),
            metadata={
                "identity_anchor_valid": True,
                "independent_residual_evidence": True,
                "proposal_sources": [
                    "hough_circle",
                    "condition_frame_change",
                    "pivot_string_geometry",
                ],
            },
        )
        fused = fuse_pendulum_proposals_v7(
            [invalid_direct, residual]
        )
        self.assertEqual(1, len(fused))
        self.assertEqual("pendulum_bob", fused[0].entity_class)
        self.assertEqual(2, fused[0].metadata["proposal_fusion_count"])
        self.assertTrue(fused[0].metadata["identity_anchor_valid"])
        self.assertIn(
            "condition_directed_sam2", fused[0].sources
        )
        self.assertIn(
            "pendulum_v7_residual_candidate", fused[0].sources
        )

    def test_residual_requires_persistence_before_formal_promotion(
        self,
    ) -> None:
        grid = build_common_time_grid([0.0, 0.1, 0.2, 0.3])

        def track(frame_indices: list[int], track_id: str) -> OpenWorldTrack:
            detections = tuple(
                _detection(
                    index,
                    (100 + index, 80),
                    detection_id=f"{track_id}_{index}",
                    sources=("pendulum_v7_residual_candidate",),
                    metadata={
                        "independent_residual_evidence": True,
                        "proposal_sources": [
                            "hough_circle",
                            "condition_frame_change",
                        ],
                    },
                )
                for index in frame_indices
            )
            return OpenWorldTrack(
                track_id=track_id,
                detections=detections,
                confirmed=True,
                evidence_tier=EvidenceTier.PARTICIPANT,
            )

        calibrated, promoted = _calibrate_residual_tracks(
            [track([0], "flash"), track([0, 1, 2], "persistent")],
            time_grid=grid,
            config={
                "v7_residual_minimum_frames": 3,
                "v7_residual_minimum_duration_s": 0.12,
            },
        )
        by_id = {value.track_id: value for value in calibrated}
        self.assertEqual(
            EvidenceTier.AMBIGUOUS, by_id["flash"].evidence_tier
        )
        self.assertEqual(
            EvidenceTier.PARTICIPANT,
            by_id["persistent"].evidence_tier,
        )
        self.assertEqual(1, promoted)

    def test_persistent_second_bob_survives_fusion_as_an_extra_track(
        self,
    ) -> None:
        times = [index * 0.1 for index in range(8)]
        grid = build_common_time_grid(times)
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        cv2.line(
            condition, (60, 10), (82, 92), (235, 235, 235), 2
        )
        cv2.circle(condition, (82, 92), 8, (240, 240, 240), -1)
        bob_mask = _mask((82, 92))
        subject_mask = np.zeros((120, 140), dtype=np.uint8)
        cv2.line(subject_mask, (60, 10), (82, 92), 255, 2)
        cv2.circle(subject_mask, (82, 92), 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray([60.0, 10.0]),
            bob_xy=np.asarray([82.0, 92.0]),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=subject_mask,
            confidence=1.0,
            source="unit_condition",
        )
        frames: list[np.ndarray] = []
        directed: list[np.ndarray] = []
        for index in range(len(times)):
            frame = np.zeros_like(condition)
            declared = (82 - index, 92)
            extra = (112 - index, 85)
            cv2.line(
                frame, (60, 10), declared, (235, 235, 235), 2
            )
            cv2.circle(frame, declared, 8, (240, 240, 240), -1)
            cv2.line(frame, (60, 10), extra, (225, 225, 225), 2)
            cv2.circle(frame, extra, 8, (250, 250, 250), -1)
            frames.append(frame)
            directed.append(_mask(declared))
        observation = discover_pendulum_objects_v7(
            frames,
            directed_bob_masks=directed,
            condition_frame=condition,
            structure=structure,
            time_grid=grid,
            config={
                **_OBSERVATION_CONFIG,
                "maximum_circle_candidates": 32,
                "hough_accumulator_thresholds": [14, 10, 8],
                "minimum_bob_pixels": 8,
                "direct_minimum_area_ratio": 0.2,
                "direct_maximum_area_ratio": 4.0,
                "minimum_anchor_color_similarity": 0.1,
                "condition_change_threshold": 24.0,
                "support_exclusion_radius_ratio": 1.2,
                "participant_string_score": 0.55,
                "maximum_residual_pivot_error_ratio": 0.25,
                "maximum_tracking_gap_s": 0.25,
                "maximum_tracking_assignment_cost": 4.0,
                "maximum_tracks": 24,
                "v7_residual_minimum_duration_s": 0.12,
            },
        )
        formal = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        self.assertEqual(2, len(formal))
        self.assertEqual(
            {"track_condition_bob", "track_0000"},
            {track.track_id for track in formal},
        )

    def test_condition_anchor_ghost_is_promoted_after_bob_departure(
        self,
    ) -> None:
        times = [index * 0.1 for index in range(5)]
        grid = build_common_time_grid(times)
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        pivot = (60, 10)
        anchor = (82, 92)
        cv2.line(condition, pivot, anchor, (235, 235, 235), 2)
        cv2.circle(condition, anchor, 8, (240, 240, 240), -1)
        bob_mask = _mask(anchor)
        subject_mask = np.zeros((120, 140), dtype=np.uint8)
        cv2.line(subject_mask, pivot, anchor, 255, 2)
        cv2.circle(subject_mask, anchor, 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray(pivot, dtype=np.float64),
            bob_xy=np.asarray(anchor, dtype=np.float64),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=subject_mask,
            confidence=1.0,
            source="unit_condition",
        )
        directed_centers = [
            anchor,
            (97, 90),
            (105, 88),
            (109, 86),
            (112, 85),
        ]
        frames: list[np.ndarray] = []
        directed: list[np.ndarray] = []
        for center in directed_centers:
            # The original condition bob remains byte-identical after the
            # directed physical bob departs: a static duplication/ghost.
            frame = np.array(condition, copy=True)
            cv2.line(frame, pivot, center, (235, 235, 235), 2)
            cv2.circle(frame, center, 8, (240, 240, 240), -1)
            frames.append(frame)
            directed.append(_mask(center))
        with patch(
            "physbench.evaluation.scenes.pendulum.v7_open_world."
            "_circle_candidates",
            return_value=[(82.0, 92.0, 8.0)],
        ), patch(
            "physbench.evaluation.scenes.pendulum.v7_open_world."
            "_line_segments",
            return_value=[],
        ):
            observation = discover_pendulum_objects_v7(
                frames,
                directed_bob_masks=directed,
                condition_frame=condition,
                structure=structure,
                time_grid=grid,
                config={
                    **_OBSERVATION_CONFIG,
                    "minimum_bob_pixels": 8,
                    "direct_minimum_area_ratio": 0.2,
                    "direct_maximum_area_ratio": 4.0,
                    "minimum_anchor_color_similarity": 0.1,
                    "condition_change_threshold": 24.0,
                    "support_exclusion_radius_ratio": 1.2,
                    "participant_string_score": 0.55,
                    "maximum_residual_pivot_error_ratio": 0.25,
                    "maximum_tracking_gap_s": 0.25,
                    "maximum_tracking_assignment_cost": 4.0,
                    "maximum_tracks": 24,
                    "v7_residual_minimum_frames": 3,
                    "v7_residual_minimum_duration_s": 0.12,
                },
            )
        formal = [
            track
            for track in observation.tracks
            if track.formal_exposure_weight > 0.0
        ]
        self.assertEqual(2, len(formal))
        ghost = next(
            track
            for track in formal
            if track.track_id != "track_condition_bob"
        )
        self.assertEqual([2, 3, 4], [
            value.frame_index for value in ghost.detections
        ])
        self.assertTrue(
            all(
                value.metadata["condition_anchor_copy_evidence"]
                for value in ghost.detections
            )
        )

    def test_directed_far_replacement_cannot_take_condition_identity(
        self,
    ) -> None:
        times = [index * 0.1 for index in range(4)]
        grid = build_common_time_grid(times)
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        pivot = (60, 10)
        anchor = (82, 92)
        cv2.line(condition, pivot, anchor, (235, 235, 235), 2)
        cv2.circle(condition, anchor, 8, (240, 240, 240), -1)
        bob_mask = _mask(anchor)
        subject_mask = np.zeros((120, 140), dtype=np.uint8)
        cv2.line(subject_mask, pivot, anchor, 255, 2)
        cv2.circle(subject_mask, anchor, 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray(pivot, dtype=np.float64),
            bob_xy=np.asarray(anchor, dtype=np.float64),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=subject_mask,
            confidence=1.0,
            source="unit_condition",
        )
        centers = [anchor, (78, 92), (18, 80), (16, 80)]
        frames: list[np.ndarray] = []
        directed: list[np.ndarray] = []
        for center in centers:
            frame = np.zeros_like(condition)
            cv2.line(frame, pivot, center, (235, 235, 235), 2)
            cv2.circle(frame, center, 8, (240, 240, 240), -1)
            frames.append(frame)
            directed.append(_mask(center))
        with patch(
            "physbench.evaluation.scenes.pendulum.v7_open_world."
            "_residual_detections",
            return_value=([[] for _ in times], {
                "weak_string_circle": 0,
                "apparatus": 0,
                "single_source": 0,
            }),
        ):
            observation = discover_pendulum_objects_v7(
                frames,
                directed_bob_masks=directed,
                condition_frame=condition,
                structure=structure,
                time_grid=grid,
                config={
                    **_OBSERVATION_CONFIG,
                    "minimum_bob_pixels": 8,
                    "direct_minimum_area_ratio": 0.2,
                    "direct_maximum_area_ratio": 4.0,
                    "minimum_anchor_color_similarity": 0.1,
                    "maximum_tracking_gap_s": 0.25,
                    "maximum_tracking_assignment_cost": 4.0,
                    "maximum_tracks": 24,
                    "v7_directed_maximum_jump_radius_ratio": 5.0,
                    "v7_directed_maximum_velocity_error_radius_ratio": 4.0,
                },
            )
        condition_track = next(
            track
            for track in observation.tracks
            if track.track_id == "track_condition_bob"
        )
        self.assertEqual(
            [0, 1],
            [value.frame_index for value in condition_track.detections],
        )
        replacement = next(
            track
            for track in observation.tracks
            if track.entity_class == "pendulum_bob__replacement"
        )
        self.assertEqual(
            [2, 3],
            [value.frame_index for value in replacement.detections],
        )
        self.assertEqual(EvidenceTier.PARTICIPANT, replacement.evidence_tier)
        self.assertEqual(
            2,
            observation.diagnostics["directed_continuity_rejections"],
        )

    def test_disappearance_remains_missing_and_is_not_interpolated_by_observer(
        self,
    ) -> None:
        times = [index * 0.1 for index in range(8)]
        grid = build_common_time_grid(times)
        condition = np.zeros((120, 140, 3), dtype=np.uint8)
        pivot = (60, 10)
        bob = (82, 92)
        cv2.line(condition, pivot, bob, (235, 235, 235), 2)
        cv2.circle(condition, bob, 8, (240, 240, 240), -1)
        bob_mask = _mask(bob)
        subject_mask = np.zeros(condition.shape[:2], dtype=np.uint8)
        cv2.line(subject_mask, pivot, bob, 255, 2)
        cv2.circle(subject_mask, bob, 8, 255, -1)
        structure = PendulumStructureSpec(
            pivot_xy=np.asarray(pivot, dtype=np.float64),
            bob_xy=np.asarray(bob, dtype=np.float64),
            bob_radius_px=8.0,
            bob_mask=bob_mask,
            subject_mask=subject_mask,
            confidence=1.0,
            source="unit_condition",
        )
        coverages: list[float] = []
        for missing in (0, 2, 4):
            directed = [
                (
                    np.array(bob_mask, copy=True)
                    if index < len(times) - missing
                    else np.zeros_like(bob_mask)
                )
                for index in range(len(times))
            ]
            with patch(
                "physbench.evaluation.scenes.pendulum.v7_open_world."
                "_residual_detections",
                return_value=(
                    [[] for _ in times],
                    {
                        "weak_string_circle": 0,
                        "apparatus": 0,
                        "single_source": 0,
                    },
                ),
            ):
                observation = discover_pendulum_objects_v7(
                    [np.array(condition, copy=True) for _ in times],
                    directed_bob_masks=directed,
                    condition_frame=condition,
                    structure=structure,
                    time_grid=grid,
                    config={
                        **_OBSERVATION_CONFIG,
                        "minimum_bob_pixels": 8,
                        "direct_minimum_area_ratio": 0.2,
                        "direct_maximum_area_ratio": 4.0,
                        "minimum_anchor_color_similarity": 0.1,
                        "maximum_tracking_gap_s": 0.25,
                        "maximum_tracking_assignment_cost": 4.0,
                        "maximum_tracks": 24,
                    },
                )
            condition_track = next(
                track
                for track in observation.tracks
                if track.track_id == "track_condition_bob"
            )
            _, _, observed = track_masks(
                condition_track,
                frame_count=len(times),
                shape=condition.shape[:2],
            )
            self.assertEqual(
                len(times) - missing,
                int(np.count_nonzero(observed)),
            )
            coverages.append(float(np.mean(observed)))
        self.assertEqual(sorted(coverages, reverse=True), coverages)
        self.assertEqual(3, len(set(coverages)))

    def test_matched_bob_gt_self_trace_is_exactly_symmetric(self) -> None:
        times = np.arange(41, dtype=np.float64) / 20.0
        pivot = np.asarray([70.0, 15.0])
        angle = 0.25 * np.cos(2.0 * math.pi * times / 1.2)
        length = 75.0
        xy = np.column_stack(
            (
                pivot[0] + length * np.sin(angle),
                pivot[1] + length * np.cos(angle),
            )
        )
        observed = np.ones(len(times), dtype=bool)
        reference = extract_bob_trace_v7(
            xy,
            observed,
            times,
            pivot_xy=pivot,
            period_config={"minimum_s": 0.3, "maximum_s": 2.5},
        )
        prediction = extract_bob_trace_v7(
            np.array(xy, copy=True),
            np.array(observed, copy=True),
            times,
            pivot_xy=pivot,
            period_config={"minimum_s": 0.3, "maximum_s": 2.5},
        )
        result = score_traces(
            reference,
            prediction,
            scoring_config={
                "minimum_angle_scale_deg": 5.0,
                "pivot_drift_scale": 0.03,
                "length_cv_scale": 0.05,
                "weights": {
                    "angle_trajectory": 0.6,
                    "period": 0.2,
                    "amplitude": 0.1,
                    "structural_consistency": 0.1,
                },
            },
        )
        self.assertAlmostEqual(1.0, result["score"], places=12)

    def test_topology_is_symmetric_and_single_frame_break_is_not_formal(
        self,
    ) -> None:
        count = 7
        pivot = np.asarray([60.0, 10.0])
        bob = np.repeat(
            np.asarray([[82.0, 92.0]]), count, axis=0
        )
        bob_masks = [_mask((82, 92)) for _ in range(count)]
        frames = []
        assemblies = []
        for index in range(count):
            frame = np.zeros((120, 140, 3), dtype=np.uint8)
            assembly = np.zeros((120, 140), dtype=np.uint8)
            if index != 3:
                cv2.line(
                    frame, (60, 10), (82, 92), (230, 230, 230), 2
                )
                cv2.line(assembly, (60, 10), (82, 92), 255, 2)
            cv2.circle(frame, (82, 92), 8, (240, 240, 240), -1)
            cv2.circle(assembly, (82, 92), 8, 255, -1)
            frames.append(frame)
            assemblies.append(assembly)
        observation = observe_pendulum_topology_v7(
            frames,
            assemblies,
            pivot_xy=pivot,
            bob_xy=bob,
            bob_observed=[True] * count,
            bob_masks=bob_masks,
            frame_weights=np.ones(count),
            config=_TOPOLOGY_CONFIG,
        )
        comparison = compare_pendulum_topology_v7(
            observation,
            observation,
            frame_weights=np.ones(count),
        )
        self.assertEqual(0, observation["broken_frame_count"])
        self.assertAlmostEqual(1.0, comparison["score"], places=12)
        self.assertEqual(
            "symmetric_reference_prediction_multisource",
            comparison["mode"],
        )

    def test_topology_source_disagreement_is_finite_and_auditable(
        self,
    ) -> None:
        count = 5
        pivot = np.asarray([60.0, 10.0])
        bob = np.repeat(
            np.asarray([[82.0, 92.0]]), count, axis=0
        )
        frames = []
        assemblies = []
        for _ in range(count):
            frame = np.zeros((120, 140, 3), dtype=np.uint8)
            cv2.line(
                frame, (60, 10), (82, 92), (240, 240, 240), 2
            )
            cv2.circle(frame, (82, 92), 8, (240, 240, 240), -1)
            frames.append(frame)
            assemblies.append(_mask((82, 92)))
        observation = observe_pendulum_topology_v7(
            frames,
            assemblies,
            pivot_xy=pivot,
            bob_xy=bob,
            bob_observed=[True] * count,
            bob_masks=[_mask((82, 92)) for _ in range(count)],
            frame_weights=np.ones(count),
            config=_TOPOLOGY_CONFIG,
        )
        metric = compare_pendulum_topology_v7(
            None,
            observation,
            frame_weights=np.ones(count),
        )
        self.assertTrue(math.isfinite(float(metric["score"])))
        self.assertGreater(
            float(metric["source_disagreement_ratio"]), 0.45
        )
        self.assertTrue(metric["uncertain"])
        self.assertLess(float(metric["score"]), 1.0)
        self.assertEqual(count, observation["broken_frame_count"])
        # A physics parent supplies normalized dynamics only.  Its resolved
        # video is not child GT and must never become a future-pixel topology
        # target; scoring the parent itself as a prediction is therefore a
        # condition-consistency negative probe, not a GT-self identity check.
        self.assertEqual(
            "condition_absolute_no_parent_future_pixels",
            metric["mode"],
        )

    def test_persistent_broken_string_is_formally_penalized(self) -> None:
        count = 7
        pivot = np.asarray([60.0, 10.0])
        bob = np.repeat(
            np.asarray([[82.0, 92.0]]), count, axis=0
        )
        bob_masks = [_mask((82, 92)) for _ in range(count)]
        intact_frames: list[np.ndarray] = []
        intact_assemblies: list[np.ndarray] = []
        broken_frames: list[np.ndarray] = []
        broken_assemblies: list[np.ndarray] = []
        for index in range(count):
            intact_frame = np.zeros((120, 140, 3), dtype=np.uint8)
            intact_assembly = np.zeros((120, 140), dtype=np.uint8)
            cv2.line(
                intact_frame, (60, 10), (82, 92), (230, 230, 230), 2
            )
            cv2.line(intact_assembly, (60, 10), (82, 92), 255, 2)
            cv2.circle(
                intact_frame, (82, 92), 8, (240, 240, 240), -1
            )
            cv2.circle(intact_assembly, (82, 92), 8, 255, -1)
            intact_frames.append(intact_frame)
            intact_assemblies.append(intact_assembly)
            broken_frame = np.array(intact_frame, copy=True)
            broken_assembly = np.array(intact_assembly, copy=True)
            if 2 <= index <= 5:
                broken_frame = np.zeros_like(intact_frame)
                broken_assembly = np.zeros_like(intact_assembly)
                cv2.circle(
                    broken_frame, (82, 92), 8, (240, 240, 240), -1
                )
                cv2.circle(broken_assembly, (82, 92), 8, 255, -1)
            broken_frames.append(broken_frame)
            broken_assemblies.append(broken_assembly)
        reference = observe_pendulum_topology_v7(
            intact_frames,
            intact_assemblies,
            pivot_xy=pivot,
            bob_xy=bob,
            bob_observed=[True] * count,
            bob_masks=bob_masks,
            frame_weights=np.ones(count),
            config=_TOPOLOGY_CONFIG,
        )
        prediction = observe_pendulum_topology_v7(
            broken_frames,
            broken_assemblies,
            pivot_xy=pivot,
            bob_xy=bob,
            bob_observed=[True] * count,
            bob_masks=bob_masks,
            frame_weights=np.ones(count),
            config=_TOPOLOGY_CONFIG,
        )
        metric = compare_pendulum_topology_v7(
            reference,
            prediction,
            frame_weights=np.ones(count),
        )
        self.assertEqual(4, prediction["broken_frame_count"])
        self.assertLess(float(metric["score"]), 0.6)


if __name__ == "__main__":
    unittest.main()
