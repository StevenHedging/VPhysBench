from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.entities import (
    PositionEvidence,
    ReferenceCapability,
    build_common_time_grid,
    compose_object_centric_v2,
    materialize_entity_manifest,
    resolve_expected_entity_timeline,
    safe_compare_open_world_v2,
)
from physbench.evaluation.common.subject import compare_subjects
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.scenes.circular_motion.open_world import (
    empty_open_world_observation,
    freeze_circular_apparatus,
    freeze_reference_identities,
    freeze_prediction_identity,
    matched_prediction_inputs,
    observe_circular_objects,
    polar_position_similarity,
    reference_instance_tracks,
    score_open_world_orbits,
)
from physbench.evaluation.scenes.circular_motion.v6_evaluator import (
    CircularMotionOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.circular_motion.v7_evaluator import (
    CircularMotionOpenWorldCaseEvaluatorV7,
)


_FRAME_COUNT = 9
_TIMES = [index / 8.0 for index in range(_FRAME_COUNT)]
_GRID = build_common_time_grid(_TIMES)
_CSTI_MAPPING = {
    "enabled": True,
    "algorithm": "exact_full_tube_edt",
    "spatial_tolerance_fraction": 0.005,
    "temporal_tolerance_s": 0.05,
    "condition_frame_policy": "exclude_initial_samples",
    "initial_frames_excluded": 1,
    "score_aggregation": "full_tube",
    "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
    "case_aggregation": "mean_gt_entities",
    "timeline_policy": "physical_overlap",
    "mask_resolution": "scene_analysis_native",
}
_CSTI_CONFIG = CSTIConfig.from_mapping(_CSTI_MAPPING)
_COLOR_CONFIG = {
    "green_hsv_lower": [35, 35, 25],
    "green_hsv_upper": [100, 255, 255],
    "disk_erosion_kernel": 5,
    "minimum_disk_area_ratio": 0.05,
    "minimum_component_area": 6,
    "maximum_component_area_ratio": 0.03,
    # Legacy expected-N selector cap: v6 must not use this value.
    "maximum_candidates": 2,
    "minimum_valid_frame_ratio": 0.9,
    "open_world_maximum_gap_s": 0.4,
    "open_world_maximum_candidates_per_frame": 64,
    "open_world_maximum_tracks": 64,
}
_V7_COLOR_CONFIG = {
    **_COLOR_CONFIG,
    "disk_colour_policy": "adaptive_dominant_hue",
    "apparatus_coordinate_policy": "condition_frozen_shared_frame_v1",
    "apparatus_appearance_policy": (
        "condition_lab_value_temporal_residual_v1"
    ),
    "apparatus_palette_frame_shift_policy": (
        "condition_support_robust_median_v1"
    ),
    "apparatus_palette_maximum_frame_shift_lab": 45.0,
    "coordinate_phase_policy": (
        "condition_frozen_absolute_phase_v1"
    ),
    "physics_parent_phase_policy": (
        "separate_condition_relative_dynamics_v1"
    ),
    "apparatus_palette_lab_quantization": 8,
    "apparatus_palette_maximum_colours": 24,
    "apparatus_palette_minimum_fraction": 0.001,
    "apparatus_palette_maximum_lab_distance": 18.0,
    "apparatus_condition_maximum_lab_distance": 22.0,
    "apparatus_temporal_minimum_lab_distance": 30.0,
    "apparatus_temporal_condition_minimum_lab_distance": 30.0,
    "adaptive_disk_minimum_saturation": 35,
    "adaptive_disk_minimum_value": 25,
    "adaptive_disk_hue_smoothing_radius": 4,
    "adaptive_disk_hue_tolerance": 8,
    "adaptive_disk_maximum_radius_ratio": 0.54,
    "adaptive_disk_erosion_kernel": 11,
    "open_world_component_shape_filter": "compact_participant_v1",
    "open_world_minimum_component_disk_area_ratio": 0.003,
}


def _case(count: int = 2) -> dict[str, object]:
    physics: dict[str, object] = {
        "angular_velocity": {
            "value": 1.2,
            "unit": "rad/s",
            "annotated": True,
        },
        "angular_velocity_rad_s": {
            "value": 1.2,
            "unit": "rad/s",
            "annotated": False,
        },
    }
    for index in range(count):
        physics[f"object_{index + 1}_orbit_radius"] = {
            "value": 0.1 + 0.1 * index,
            "unit": "m",
            "annotated": True,
        }
    return {
        "case_id": f"circular_{count}",
        "scene_id": "uniform_circular_motion",
        "has_real_reference_video": True,
        "appearance": {
            "object_count": count,
            "moving_objects": [
                "silver" if index % 2 == 0 else "wood"
                for index in range(count)
            ],
        },
        "physics": physics,
    }


def _frames(
    *,
    radii: tuple[int, ...] = (22, 38),
    present: dict[int, set[int]] | None = None,
    extra_windows: tuple[tuple[int, int, int, float], ...] = (),
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for frame_index in range(_FRAME_COUNT):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(frame, (64, 64), 52, (0, 180, 0), -1)
        for object_index, radius in enumerate(radii):
            if (
                present is not None
                and frame_index not in present.get(
                    object_index, set(range(_FRAME_COUNT))
                )
            ):
                continue
            angle = 0.15 * frame_index + 1.4 * object_index
            _draw_object(frame, radius=radius, angle=angle, index=object_index)
        for radius, first, last, phase in extra_windows:
            if first <= frame_index <= last:
                _draw_object(
                    frame,
                    radius=radius,
                    angle=0.15 * frame_index + phase,
                    index=7,
                )
        output.append(frame)
    return output


def _draw_object(
    frame: np.ndarray,
    *,
    radius: int,
    angle: float,
    index: int,
) -> None:
    x = int(round(64 + radius * math.cos(angle)))
    y = int(round(64 + radius * math.sin(angle)))
    color = (255, 255, 255) if index % 2 == 0 else (40, 100, 180)
    cv2.rectangle(frame, (x - 4, y - 3), (x + 4, y + 3), color, -1)


def _phase_shifted_frames(offset_rad: float) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for frame_index in range(_FRAME_COUNT):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(frame, (64, 64), 52, (0, 180, 0), -1)
        for object_index, radius in enumerate((22, 38)):
            _draw_object(
                frame,
                radius=radius,
                angle=(
                    0.15 * frame_index
                    + 1.4 * object_index
                    + offset_rad
                ),
                index=object_index,
            )
        output.append(frame)
    return output


def _translated_platform_frames(offset_x: int) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for frame_index in range(_FRAME_COUNT):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        center_x = 64 + offset_x
        cv2.circle(frame, (center_x, 64), 52, (0, 180, 0), -1)
        for object_index, radius in enumerate((22, 38)):
            angle = 0.15 * frame_index + 1.4 * object_index
            x = int(round(center_x + radius * math.cos(angle)))
            y = int(round(64 + radius * math.sin(angle)))
            color = (
                (255, 255, 255)
                if object_index % 2 == 0
                else (40, 100, 180)
            )
            cv2.rectangle(
                frame,
                (x - 4, y - 3),
                (x + 4, y + 3),
                color,
                -1,
            )
        output.append(frame)
    return output


def _scaled_platform_frames(scale: float) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    disk_radius = int(round(52 * scale))
    for frame_index in range(_FRAME_COUNT):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(frame, (64, 64), disk_radius, (0, 180, 0), -1)
        for object_index, radius in enumerate((22, 38)):
            _draw_object(
                frame,
                radius=int(round(radius * scale)),
                angle=0.15 * frame_index + 1.4 * object_index,
                index=object_index,
            )
        output.append(frame)
    return output


def _reference(config: dict[str, object] | None = None):
    observation_config = _COLOR_CONFIG if config is None else config
    frames = _frames()
    apparatus = (
        freeze_circular_apparatus(
            frames[0],
            config=observation_config,
            source="unit_condition",
        )
        if observation_config.get("apparatus_coordinate_policy")
        == "condition_frozen_shared_frame_v1"
        else None
    )
    observation = observe_circular_objects(
        frames,
        time_grid=_GRID,
        config=observation_config,
        apparatus_anchor=apparatus,
    )
    manifest = materialize_entity_manifest(_case())
    frozen = freeze_reference_identities(
        observation,
        entities=manifest.entities,
        time_grid=_GRID,
    )
    return observation, frozen


def _integrity(
    frames: list[np.ndarray],
    *,
    config: dict[str, object] | None = None,
):
    observation_config = _COLOR_CONFIG if config is None else config
    _, frozen = _reference(observation_config)
    manifest = materialize_entity_manifest(_case())
    prediction = observe_circular_objects(
        frames,
        time_grid=_GRID,
        config=observation_config,
        apparatus_anchor=(
            freeze_circular_apparatus(
                _frames()[0],
                config=observation_config,
                source="unit_condition",
            )
            if observation_config.get("apparatus_coordinate_policy")
            == "condition_frozen_shared_frame_v1"
            else None
        ),
    )
    reference_by_id = {
        track.matched_entity_id: track for track in frozen.tracks
    }
    masks_by_id = {
        entity_id: masks
        for entity_id, masks in zip(
            frozen.entity_ids, frozen.instance_masks
        )
    }
    timelines = [
        resolve_expected_entity_timeline(
            entity.to_entity_spec(),
            time_grid=_GRID,
            capability=ReferenceCapability.SAME_CASE_GT,
            reference_track=reference_by_id[entity.entity_id],
            reference_masks=masks_by_id[entity.entity_id],
        )
        for entity in manifest.entities
    ]
    assignment = freeze_prediction_identity(
        prediction.objects,
        entities=manifest.entities,
        condition_anchors=frozen.anchors,
    )

    def position(timeline, detection, frame_index, _diagonal):
        value = polar_position_similarity(
            timeline.reference_xy[frame_index], detection.xy
        )
        return PositionEvidence(
            score=float(value["score"]),
            normalized_distance=float(value["normalized_distance"]),
            diagnostics={"mode": value["mode"]},
        )

    comparison = safe_compare_open_world_v2(
        expected_timelines=timelines,
        prediction_factory=lambda: prediction.objects,
        time_grid=_GRID,
        frame_diagonal_px=2.0 * math.sqrt(2.0) * 100.0,
        frozen_identity=assignment,
        minimum_match_position_similarity=0.1,
        position_similarity_callback=position,
    )
    return comparison, prediction


def _formal_v7_score(
    frames: list[np.ndarray],
) -> tuple[float, dict[str, float]]:
    comparison, prediction = _integrity(
        frames, config=_V7_COLOR_CONFIG
    )
    _, frozen = _reference(_V7_COLOR_CONFIG)
    reference_tracks = reference_instance_tracks(frozen)
    prediction_tracks, matched_union = matched_prediction_inputs(
        reference=frozen,
        prediction=prediction.objects,
        matches=comparison.matches,
        frame_count=_FRAME_COUNT,
    )
    orbit = score_open_world_orbits(
        reference_tracks,
        prediction_tracks,
        matches=comparison.matches,
        reference_object_tracks=frozen.tracks,
        prediction_observation=prediction.objects,
        times_s=_TIMES,
        time_grid=_GRID,
        config={
            "angular_trajectory_scale_rad": 0.35,
            "angular_velocity_error_scale": 0.25,
            "radial_cv_scale": 0.08,
            "angular_fit_rmse_scale_rad": 0.2,
            "radius_configuration_scale": 0.12,
            "weights": {
                "angular_trajectory": 0.5,
                "angular_velocity": 0.25,
                "orbit_geometry": 0.15,
                "uniform_motion": 0.1,
            },
            "center_fallback_radius_ratio": 0.08,
            "polar_radial_scale_ratio": 0.15,
            "polar_angular_scale_rad": 0.35,
            "center_cartesian_scale_ratio": 0.15,
        },
    )
    subject = compare_subjects(
        reference_frames=_frames(),
        prediction_frames=frames,
        reference_masks=reference_tracks.union_masks,
        prediction_masks=matched_union,
        reference_mode="same_case_reference",
        config={
            "minimum_observed_pixels": 4,
            "position_distance_scale": 0.08,
            "canonical_crop_size": 64,
            "boundary_tolerance_px": 2,
            "weights": {
                "position": 0.5,
                "shape": 0.2,
                "appearance": 0.3,
            },
        },
    )
    components = {
        "orbit_physics": float(orbit["score"]),
        "shape": float(subject.components["shape"]),
        "appearance": float(subject.components["appearance"]),
    }
    composition = compose_object_centric_v2(
        comparison,
        content_components=components,
        content_weights={
            "orbit_physics": 0.55,
            "shape": 0.15,
            "appearance": 0.30,
        },
    )
    return float(composition.score or 0.0), components


class CircularOpenWorldObservationTests(unittest.TestCase):
    def test_real_large_platform_is_not_rejected_at_half_frame_radius(
        self,
    ) -> None:
        dataset = load_dataset(LATEST_DATASET, check_assets=False)
        case = next(
            case
            for case in dataset.cases
            if case["case_id"] == "circular_r1_wood04cm_img_0390"
        )
        frame = cv2.imread(
            str(dataset.asset_root / case["assets"]["first_frame"]),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        normalized = cv2.resize(
            frame,
            (270, 480),
            interpolation=cv2.INTER_AREA,
        )

        apparatus = freeze_circular_apparatus(
            normalized,
            config=_V7_COLOR_CONFIG,
            source="real_large_platform_regression",
        )

        np.testing.assert_allclose(
            [136.2, 246.0],
            apparatus.center_xy,
            atol=3.0,
        )
        self.assertAlmostEqual(138.4, apparatus.radius_px, delta=4.0)

    def test_adaptive_disk_colour_policy_is_hue_invariant(self) -> None:
        config = {
            **_COLOR_CONFIG,
            "disk_colour_policy": "adaptive_dominant_hue",
            "adaptive_disk_hue_tolerance": 8,
            "adaptive_disk_erosion_kernel": 11,
            "minimum_component_area": 20,
            "open_world_component_shape_filter": (
                "compact_participant_v1"
            ),
            "open_world_minimum_component_disk_area_ratio": 0.0,
            "open_world_minimum_component_rectangularity": 0.30,
        }
        summaries = []
        for disk_colour in (
            (0, 180, 0),
            (220, 150, 20),
            (90, 90, 225),
        ):
            frames = []
            for frame_index in range(_FRAME_COUNT):
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
                cv2.circle(frame, (64, 64), 52, disk_colour, -1)
                _draw_object(
                    frame,
                    radius=30,
                    angle=0.15 * frame_index,
                    index=0,
                )
                frames.append(frame)
            observation = observe_circular_objects(
                frames,
                time_grid=_GRID,
                config=config,
            )
            self.assertEqual(
                {1}, set(observation.candidate_counts.tolist())
            )
            self.assertEqual(1, len(observation.objects.tracks))
            summaries.append(
                (
                    np.median(observation.disk_centers_xy, axis=0),
                    float(np.median(observation.disk_radii_px)),
                )
            )
        for center, radius in summaries:
            np.testing.assert_allclose(center, [64.0, 64.0], atol=1.0)
            self.assertAlmostEqual(52.0, radius, delta=2.0)

    def test_adaptive_disk_geometry_rejects_larger_saturated_background(
        self,
    ) -> None:
        config = {
            **_COLOR_CONFIG,
            "disk_colour_policy": "adaptive_dominant_hue",
            "adaptive_disk_hue_tolerance": 8,
            "adaptive_disk_erosion_kernel": 9,
            "open_world_component_shape_filter": (
                "compact_participant_v1"
            ),
        }
        frames = []
        for frame_index in range(_FRAME_COUNT):
            # The red background has more saturated support than the blue
            # disk. Geometry, rather than global hue frequency, must select
            # the circular apparatus.
            frame = np.full((128, 128, 3), (0, 0, 180), dtype=np.uint8)
            cv2.circle(frame, (64, 64), 48, (180, 80, 20), -1)
            _draw_object(
                frame,
                radius=28,
                angle=0.15 * frame_index,
                index=0,
            )
            frames.append(frame)
        observation = observe_circular_objects(
            frames,
            time_grid=_GRID,
            config=config,
        )
        self.assertEqual(
            {1}, set(observation.candidate_counts.tolist())
        )
        self.assertEqual(1, len(observation.objects.tracks))
        np.testing.assert_allclose(
            np.median(observation.disk_centers_xy, axis=0),
            [64.0, 64.0],
            atol=1.0,
        )
        self.assertAlmostEqual(
            48.0,
            float(np.median(observation.disk_radii_px)),
            delta=2.0,
        )

    def test_adaptive_disk_prefers_platform_over_letterboxed_tabletop(
        self,
    ) -> None:
        config = {
            **_COLOR_CONFIG,
            "disk_colour_policy": "adaptive_dominant_hue",
            "adaptive_disk_hue_tolerance": 8,
            "adaptive_disk_erosion_kernel": 11,
            "adaptive_disk_maximum_area_ratio": 0.75,
            "adaptive_disk_minimum_radius_ratio": 0.08,
            "adaptive_disk_maximum_radius_ratio": 0.48,
            "open_world_component_shape_filter": (
                "compact_participant_v1"
            ),
            "open_world_minimum_component_disk_area_ratio": 0.003,
        }
        frames = []
        for frame_index in range(_FRAME_COUNT):
            hsv = np.zeros((480, 640, 3), dtype=np.uint8)
            # This centred saturated capture region has more area and a
            # larger enclosing circle than the actual apparatus.  It
            # reproduces the portrait-video letterbox geometry from the real
            # circular corpus.
            cv2.rectangle(hsv, (185, 100), (455, 380), (15, 150, 210), -1)
            cv2.circle(hsv, (320, 240), 128, (100, 180, 220), -1)
            # High-saturation platform shading just beyond the narrow
            # geometry interval remains apparatus, not a physical object.
            cv2.circle(hsv, (300, 315), 10, (108, 115, 220), -1)
            angle = 0.15 * frame_index
            x = int(round(320 + 94 * math.cos(angle)))
            y = int(round(240 + 94 * math.sin(angle)))
            cv2.rectangle(hsv, (x - 8, y - 6), (x + 8, y + 6), (0, 0, 255), -1)
            frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        grid = build_common_time_grid(_TIMES)
        observation = observe_circular_objects(
            frames,
            time_grid=grid,
            config=config,
        )
        np.testing.assert_allclose(
            np.median(observation.disk_centers_xy, axis=0),
            [320.0, 240.0],
            atol=1.0,
        )
        self.assertAlmostEqual(
            128.0,
            float(np.median(observation.disk_radii_px)),
            delta=2.0,
        )
        self.assertEqual({1}, set(observation.candidate_counts.tolist()))
        self.assertEqual(1, len(observation.objects.tracks))

    def test_shape_rejected_participant_remains_in_cardinality_audit(
        self,
    ) -> None:
        config = {
            **_COLOR_CONFIG,
            "open_world_component_shape_filter": (
                "compact_participant_v1"
            ),
            "open_world_maximum_component_aspect_ratio": 3.0,
        }
        frames = []
        for frame_index in range(_FRAME_COUNT):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.circle(frame, (64, 64), 52, (0, 180, 0), -1)
            _draw_object(
                frame,
                radius=22,
                angle=0.15 * frame_index,
                index=0,
            )
            angle = 0.15 * frame_index + 2.4
            x = int(round(64 + 34 * math.cos(angle)))
            y = int(round(64 + 34 * math.sin(angle)))
            cv2.rectangle(
                frame,
                (x - 10, y - 1),
                (x + 10, y + 1),
                (255, 255, 255),
                -1,
            )
            frames.append(frame)
        observation = observe_circular_objects(
            frames,
            time_grid=_GRID,
            config=config,
        )
        self.assertEqual(
            {2}, set(observation.candidate_counts.tolist())
        )
        self.assertEqual(2, len(observation.objects.tracks))
        rejected = [
            detection
            for track in observation.objects.tracks
            for detection in track.detections
            if not bool(
                detection.metadata["shape_filter_accepted"]
            )
        ]
        self.assertEqual(_FRAME_COUNT, len(rejected))
        self.assertTrue(
            any(
                track.formal_exposure_weight == 1.0
                and any(
                    not bool(
                        detection.metadata[
                            "shape_filter_accepted"
                        ]
                    )
                    for detection in track.detections
                )
                for track in observation.objects.tracks
            )
        )

        overflow_config = {
            **config,
            "open_world_maximum_candidates_per_frame": 1,
        }
        overflow = observe_circular_objects(
            frames,
            time_grid=_GRID,
            config=overflow_config,
        )
        self.assertEqual(
            {2}, set(overflow.candidate_counts.tolist())
        )
        self.assertTrue(np.all(overflow.objects.overflow_counts >= 1.0))

    def test_preserves_one_two_and_n_candidates_without_expected_n_crop(
        self,
    ) -> None:
        for radii in ((30,), (22, 38), (13, 23, 33, 43)):
            with self.subTest(count=len(radii)):
                observation = observe_circular_objects(
                    _frames(radii=radii),
                    time_grid=_GRID,
                    config=_COLOR_CONFIG,
                )
                self.assertEqual(
                    {len(radii)}, set(observation.candidate_counts.tolist())
                )
                self.assertEqual(len(radii), len(observation.objects.tracks))
                self.assertTrue(
                    all(
                        track.formal_exposure_weight == 1.0
                        for track in observation.objects.tracks
                    )
                )

    def test_disk_is_apparatus_not_a_physical_object(self) -> None:
        frames = []
        for _ in range(_FRAME_COUNT):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.circle(frame, (64, 64), 52, (0, 180, 0), -1)
            frames.append(frame)
        observation = observe_circular_objects(
            frames, time_grid=_GRID, config=_COLOR_CONFIG
        )
        self.assertEqual({0}, set(observation.candidate_counts.tolist()))
        self.assertEqual(0, len(observation.objects.tracks))

    def test_third_orbiter_is_formal_extra_and_lowers_integrity(self) -> None:
        baseline, _ = _integrity(_frames(), config=_V7_COLOR_CONFIG)
        extra, observation = _integrity(
            _frames(extra_windows=((30, 0, _FRAME_COUNT - 1, 2.6),)),
            config=_V7_COLOR_CONFIG,
        )
        self.assertEqual(3, len(observation.objects.tracks))
        self.assertEqual(1.0, baseline.integrity.integrity_gate)
        self.assertLess(extra.integrity.integrity_gate, 1.0)
        self.assertTrue(
            all(row["extra_track_ids"] for row in extra.per_frame)
        )

    def test_static_extra_is_formal_and_lowers_integrity(self) -> None:
        frames = _frames()
        for frame in frames:
            _draw_object(frame, radius=30, angle=2.6, index=7)
        baseline, _ = _integrity(_frames(), config=_V7_COLOR_CONFIG)
        extra, observation = _integrity(
            frames, config=_V7_COLOR_CONFIG
        )
        self.assertEqual(3, len(observation.objects.tracks))
        self.assertEqual(1.0, baseline.integrity.integrity_gate)
        self.assertLess(extra.integrity.integrity_gate, 1.0)
        self.assertTrue(
            any(row["extra_track_ids"] for row in extra.per_frame)
        )

    def test_condition_frozen_geometry_penalizes_platform_translation(
        self,
    ) -> None:
        shifted = _translated_platform_frames(8)
        comparison, observation = _integrity(
            shifted, config=_V7_COLOR_CONFIG
        )
        np.testing.assert_allclose(
            observation.disk_centers_xy,
            np.tile([64.0, 64.0], (_FRAME_COUNT, 1)),
            atol=1.0,
        )
        self.assertLess(
            comparison.integrity.soft_detection_accuracy, 0.99
        )
        self.assertGreater(
            comparison.integrity.gospa.normalized_distance, 0.0
        )
        baseline_score, _ = _formal_v7_score(_frames())
        shifted_score, shifted_components = _formal_v7_score(shifted)
        self.assertGreater(baseline_score, 0.99)
        self.assertLess(shifted_score, 0.9)
        self.assertLess(shifted_components["orbit_physics"], 0.9)

    def test_condition_frozen_phase_does_not_self_align_prediction(
        self,
    ) -> None:
        comparison, _ = _integrity(
            _phase_shifted_frames(0.8),
            config=_V7_COLOR_CONFIG,
        )
        position_scores = [
            float(match["position_score"])
            for row in comparison.per_frame
            for match in row["matches"]
        ]
        self.assertTrue(position_scores)
        self.assertLess(float(np.mean(position_scores)), 0.5)
        self.assertLess(
            comparison.integrity.soft_detection_accuracy, 0.5
        )
        baseline_score, _ = _formal_v7_score(_frames())
        shifted_score, shifted_components = _formal_v7_score(
            _phase_shifted_frames(0.8)
        )
        self.assertGreater(baseline_score, 0.99)
        self.assertLess(shifted_score, 0.75)
        self.assertLess(shifted_components["orbit_physics"], 0.65)

    def test_condition_frozen_radius_penalizes_platform_scale_hack(
        self,
    ) -> None:
        scaled = _scaled_platform_frames(0.75)
        score, components = _formal_v7_score(scaled)
        self.assertLess(score, 0.9)
        self.assertLess(components["orbit_physics"], 0.85)

    def test_same_hue_dark_extra_is_not_erased_as_apparatus(self) -> None:
        for moving in (False, True):
            with self.subTest(moving=moving):
                frames = _frames()
                for frame_index, frame in enumerate(frames):
                    angle = (
                        2.6 + 0.15 * frame_index if moving else 2.6
                    )
                    x = int(round(64 + 30 * math.cos(angle)))
                    y = int(round(64 + 30 * math.sin(angle)))
                    cv2.rectangle(
                        frame,
                        (x - 4, y - 3),
                        (x + 4, y + 3),
                        (0, 70, 0),
                        -1,
                    )
                comparison, observation = _integrity(
                    frames, config=_V7_COLOR_CONFIG
                )
                self.assertGreaterEqual(
                    len(observation.objects.tracks), 3
                )
                self.assertLess(
                    comparison.integrity.integrity_gate, 1.0
                )
                self.assertTrue(
                    any(
                        row["extra_track_ids"]
                        for row in comparison.per_frame
                    )
                )

    def test_competing_disk_cannot_redefine_frozen_coordinates(self) -> None:
        frames = _frames()
        for frame in frames:
            cv2.circle(frame, (8, 8), 20, (0, 0, 220), -1)
        anchor = freeze_circular_apparatus(
            _frames()[0],
            config=_V7_COLOR_CONFIG,
            source="unit_condition",
        )
        observation = observe_circular_objects(
            frames,
            time_grid=_GRID,
            config=_V7_COLOR_CONFIG,
            apparatus_anchor=anchor,
        )
        np.testing.assert_allclose(
            observation.disk_centers_xy,
            np.tile([64.0, 64.0], (_FRAME_COUNT, 1)),
            atol=1.0,
        )
        np.testing.assert_allclose(
            observation.disk_radii_px,
            np.full(_FRAME_COUNT, 52.0),
            atol=2.0,
        )

    def test_missing_penalty_is_monotonic_with_duration(self) -> None:
        scores = []
        for missing_count in (0, 2, 4):
            visible = set(range(_FRAME_COUNT - missing_count))
            comparison, _ = _integrity(
                _frames(
                    present={
                        0: set(range(_FRAME_COUNT)),
                        1: visible,
                    }
                )
            )
            scores.append(comparison.integrity.integrity_gate)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_extra_penalty_is_monotonic_with_duration(self) -> None:
        scores = []
        for extra_count in (0, 3, 6):
            first = _FRAME_COUNT - extra_count
            windows = (
                ()
                if extra_count == 0
                else ((30, first, _FRAME_COUNT - 1, 2.6),)
            )
            comparison, _ = _integrity(
                _frames(extra_windows=windows)
            )
            scores.append(comparison.integrity.integrity_gate)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_far_replacement_cannot_take_over_without_identity_penalty(
        self,
    ) -> None:
        present = {
            0: set(range(_FRAME_COUNT)),
            1: set(range(5)),
        }
        frames = _frames(
            present=present,
            extra_windows=((38, 5, _FRAME_COUNT - 1, math.pi),),
        )
        comparison, _ = _integrity(
            frames, config=_V7_COLOR_CONFIG
        )
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(
            all(
                row["missing_entity_ids"] and row["extra_track_ids"]
                for row in comparison.per_frame[5:]
            ),
            "a residual replacement must not fill the frozen expected ID",
        )

    def test_appearance_identity_swap_is_split_and_penalized(self) -> None:
        frames = _frames()
        for frame_index in range(5, _FRAME_COUNT):
            angle = 0.15 * frame_index + 1.4
            x = int(round(64 + 38 * math.cos(angle)))
            y = int(round(64 + 38 * math.sin(angle)))
            cv2.rectangle(
                frames[frame_index],
                (x - 4, y - 3),
                (x + 4, y + 3),
                (255, 255, 255),
                -1,
            )
        comparison, observation = _integrity(frames)
        self.assertGreater(
            int(
                observation.diagnostics[
                    "appearance_identity_switch_splits"
                ]
            ),
            0,
        )
        self.assertLess(comparison.integrity.integrity_gate, 1.0)
        self.assertTrue(
            any(
                row["missing_entity_ids"] and row["extra_track_ids"]
                for row in comparison.per_frame[5:]
            )
        )

    def test_short_occlusion_bridges_one_causal_track_id(self) -> None:
        present = {
            0: set(range(_FRAME_COUNT)),
            1: set(range(_FRAME_COUNT)) - {4},
        }
        observation = observe_circular_objects(
            _frames(present=present),
            time_grid=_GRID,
            config=_COLOR_CONFIG,
        )
        outer = max(
            observation.objects.tracks,
            key=lambda track: float(
                np.median(
                    [
                        detection.metadata["normalized_orbit_radius"]
                        for detection in track.detections
                    ]
                )
            ),
        )
        self.assertEqual(_FRAME_COUNT - 1, len(outer.detections))
        self.assertEqual(
            [0, 1, 2, 3, 5, 6, 7, 8],
            [value.frame_index for value in outer.detections],
        )

    def test_center_fallback_and_overflow_are_finite_and_auditable(
        self,
    ) -> None:
        center = polar_position_similarity(
            [0.0, 0.0], [2.0, -1.0]
        )
        self.assertEqual("cartesian_center_fallback", center["mode"])
        self.assertTrue(math.isfinite(float(center["score"])))
        config = {
            **_COLOR_CONFIG,
            "open_world_maximum_candidates_per_frame": 2,
        }
        observation = observe_circular_objects(
            _frames(radii=(13, 23, 33, 43)),
            time_grid=_GRID,
            config=config,
        )
        self.assertTrue(
            np.all(observation.objects.overflow_counts >= 2.0)
        )
        self.assertTrue(
            np.isfinite(observation.objects.overflow_counts).all()
        )


class CircularOpenWorldEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "type": "uniform_circular_motion_state_v6",
            "evaluator_contract": "robust_subject_v3",
            "timeline": {"decode_policy": "sequential_forward"},
            "spatial": {"width": 128, "height": 128, "pad_value": 0},
            "color_observation": copy.deepcopy(_COLOR_CONFIG),
            "scoring": {
                "angular_trajectory_scale_rad": 0.35,
                "angular_velocity_error_scale": 0.25,
                "radial_cv_scale": 0.08,
                "angular_fit_rmse_scale_rad": 0.2,
                "radius_configuration_scale": 0.12,
                "weights": {
                    "angular_trajectory": 0.5,
                    "angular_velocity": 0.25,
                    "orbit_geometry": 0.15,
                    "uniform_motion": 0.1,
                },
            },
            "subject_scoring": {
                "minimum_observed_pixels": 4,
                "position_distance_scale": 0.08,
                "canonical_crop_size": 32,
                "boundary_tolerance_px": 2,
                "weights": {
                    "position": 0.5,
                    "shape": 0.2,
                    "appearance": 0.3,
                },
            },
            "object_centric_scoring": {
                "assignment": {
                    "minimum_match_position_similarity": 0.1
                }
            },
        }

    @classmethod
    def _v7_config(cls) -> dict[str, object]:
        config = cls._config()
        config["type"] = "uniform_circular_motion_state_v7"
        config["reference_observation_policy"] = (
            "apparatus_geometry_symmetric"
        )
        config["color_observation"] = copy.deepcopy(_V7_COLOR_CONFIG)
        config["object_centric_scoring"] = {
            "assignment": {
                "minimum_match_position_similarity": 0.1,
                "maximum_condition_identity_cost": 3.0,
                "initial_window_frames": 3,
            },
            "content_weights": {
                "orbit_physics": 0.55,
                "shape": 0.15,
                "appearance": 0.30,
            },
            "polar_distance": {
                "center_fallback_radius_ratio": 0.08,
                "polar_radial_scale_ratio": 0.15,
                "polar_angular_scale_rad": 0.35,
                "center_cartesian_scale_ratio": 0.15,
            },
        }
        return config

    def test_v7_csti_preserves_independent_orbiter_tubes(self) -> None:
        config = self._v7_config()
        config["general_metrics"] = {"csti": _CSTI_MAPPING}
        evaluator = CircularMotionOpenWorldCaseEvaluatorV7(
            copy.deepcopy(config)
        )
        case = _case()
        manifest = materialize_entity_manifest(case)
        with tempfile.TemporaryDirectory() as temporary:
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v7_csti_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=config,
            )
            video = SimpleNamespace(
                frames=_frames(),
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=video,
                    prediction_video=video,
                )

        self.assertIsNotNone(analysis.csti_input)
        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=tuple(
                (entity.entity_id, entity.role_id)
                for entity in manifest.entities
            ),
            config=_CSTI_CONFIG,
        )
        self.assertAlmostEqual(1.0, metric["score"], places=12)
        self.assertEqual(2, len(metric["objects"]))
        self.assertTrue(all(item["matched"] for item in metric["objects"]))

    def test_normal_result_has_unified_metrics_and_finite_score(self) -> None:
        config = self._config()
        evaluator = CircularMotionOpenWorldCaseEvaluator(
            copy.deepcopy(config)
        )
        case = _case()
        with tempfile.TemporaryDirectory() as temporary:
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v6_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=config,
            )
            video = SimpleNamespace(
                frames=_frames(),
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=video,
                    prediction_video=video,
                )
        self.assertTrue(math.isfinite(analysis.score))
        self.assertGreater(analysis.score, 0.99)
        for name in (
            "scene_subject_state_similarity",
            "uniform_circular_motion_open_world_similarity",
            "object_centric_integrity",
            "entity_manifest",
        ):
            self.assertIn(name, analysis.metrics)
        self.assertEqual(
            analysis.metrics["scene_subject_state_similarity"],
            analysis.metrics[
                "uniform_circular_motion_open_world_similarity"
            ],
        )
        self.assertIn("object_centric_audit", analysis.artifacts)
        self.assertEqual(
            "open_world_v2",
            analysis.provenance["open_world_protocol"]["id"],
        )

    def test_empty_prediction_is_finite_conservative_zero(self) -> None:
        _, frozen = _reference()
        manifest = materialize_entity_manifest(_case())
        reference_by_id = {
            track.matched_entity_id: track for track in frozen.tracks
        }
        timelines = [
            resolve_expected_entity_timeline(
                entity.to_entity_spec(),
                time_grid=_GRID,
                capability=ReferenceCapability.SAME_CASE_GT,
                reference_track=reference_by_id[entity.entity_id],
            )
            for entity in manifest.entities
        ]
        comparison = safe_compare_open_world_v2(
            expected_timelines=timelines,
            prediction_factory=lambda: empty_open_world_observation(
                _FRAME_COUNT,
                code="prediction_observation_failed",
                reason="synthetic empty prediction",
            ),
            time_grid=_GRID,
            frame_diagonal_px=2.0 * math.sqrt(2.0) * 100.0,
            minimum_match_position_similarity=0.1,
        )
        self.assertTrue(math.isfinite(comparison.integrity.integrity_gate))
        self.assertEqual(0.0, comparison.integrity.integrity_gate)

    def test_v7_observer_policies_require_explicit_opt_in(self) -> None:
        config = self._config()
        config["type"] = "uniform_circular_motion_state_v7"
        with self.assertRaisesRegex(ValueError, "explicit opt-in"):
            CircularMotionOpenWorldCaseEvaluatorV7(
                copy.deepcopy(config)
            )
        config["color_observation"].update(
            {
                "disk_colour_policy": "adaptive_dominant_hue",
                "open_world_component_shape_filter": (
                    "compact_participant_v1"
                ),
                "apparatus_coordinate_policy": (
                    "condition_frozen_shared_frame_v1"
                ),
                "apparatus_appearance_policy": (
                    "condition_lab_value_temporal_residual_v1"
                ),
                "apparatus_palette_frame_shift_policy": (
                    "condition_support_robust_median_v1"
                ),
                "coordinate_phase_policy": (
                    "condition_frozen_absolute_phase_v1"
                ),
                "physics_parent_phase_policy": (
                    "separate_condition_relative_dynamics_v1"
                ),
            }
        )
        config["reference_observation_policy"] = (
            "apparatus_geometry_symmetric"
        )
        evaluator = CircularMotionOpenWorldCaseEvaluatorV7(config)
        self.assertEqual("1.0", evaluator.describe()["version"])
        self.assertIn(
            "confidence_only",
            evaluator.describe()["observation"]["shape_policy"],
        )

    def test_v7_prediction_observation_failure_is_finite_fail_closed(
        self,
    ) -> None:
        config = self._v7_config()
        config["general_metrics"] = {"csti": _CSTI_MAPPING}
        evaluator = CircularMotionOpenWorldCaseEvaluatorV7(
            copy.deepcopy(config)
        )
        case = _case()
        manifest = materialize_entity_manifest(case)
        with tempfile.TemporaryDirectory() as temporary:
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v7_fail_closed_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={
                    "status": "complete",
                    "video_path": "mock.mp4",
                },
                asset_root=Path(temporary),
                artifact_dir=Path(temporary) / "artifacts",
                evaluator_config=config,
            )
            reference = SimpleNamespace(
                frames=_frames(),
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            prediction = SimpleNamespace(
                frames=[],
                available=np.zeros(0, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=reference,
                    prediction_video=prediction,
                )
        self.assertEqual(0.0, analysis.score)
        self.assertTrue(analysis.quality["degraded"])
        self.assertIn(
            "prediction_circular_open_world_observation_failed",
            analysis.quality["degradation_codes"],
        )
        self.assertEqual(
            0.0,
            analysis.metrics["object_centric_integrity"][
                "integrity_gate"
            ],
        )
        metric = evaluate_csti(
            analysis.csti_input,
            expected_entities=tuple(
                (entity.entity_id, entity.role_id)
                for entity in manifest.entities
            ),
            config=_CSTI_CONFIG,
        )
        self.assertEqual(0.0, metric["score"])
        self.assertTrue(
            all(not item["matched"] for item in metric["objects"])
        )

    def test_physics_parent_uses_condition_identity_not_future_pixels(
        self,
    ) -> None:
        config = self._config()
        evaluator = CircularMotionOpenWorldCaseEvaluator(
            copy.deepcopy(config)
        )
        case = _case()
        case["has_real_reference_video"] = False
        case["provenance"] = {"parent_case_id": "physics_parent"}
        case["assets"] = {"first_frame": "condition.png"}
        frames = _frames()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(
                cv2.imwrite(str(root / "condition.png"), frames[0])
            )
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v6_parent_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={"status": "complete", "video_path": "mock.mp4"},
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config=config,
            )
            video = SimpleNamespace(
                frames=frames,
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=video,
                    prediction_video=video,
                )
        timelines = analysis.provenance["expected_timelines"]
        self.assertTrue(
            all(
                value["capability"] == "physics_parent"
                for value in timelines
            )
        )
        self.assertTrue(
            all(
                value["localization_supervised"][0]
                and not any(value["localization_supervised"][1:])
                for value in timelines
            )
        )
        self.assertTrue(
            all(
                all(
                    all(coordinate is None for coordinate in position)
                    for position in value["reference_xy"][1:]
                )
                for value in timelines
            )
        )
        self.assertGreater(analysis.score, 0.8)

    def test_v7_physics_parent_aligns_only_condition_relative_dynamics(
        self,
    ) -> None:
        config = self._v7_config()
        evaluator = CircularMotionOpenWorldCaseEvaluatorV7(
            copy.deepcopy(config)
        )
        case = _case()
        case["has_real_reference_video"] = False
        case["provenance"] = {"parent_case_id": "physics_parent"}
        case["assets"] = {"first_frame": "condition.png"}
        reference_frames = _frames()
        prediction_frames = _phase_shifted_frames(0.8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(
                cv2.imwrite(
                    str(root / "condition.png"),
                    prediction_frames[0],
                )
            )
            request = CaseEvaluationRequest(
                job={"job_id": "circular_v7_parent_phase_unit"},
                case=case,
                case_catalog={str(case["case_id"]): case},
                prediction={
                    "status": "complete",
                    "video_path": "mock.mp4",
                },
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config=config,
            )
            reference = SimpleNamespace(
                frames=reference_frames,
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            prediction = SimpleNamespace(
                frames=prediction_frames,
                available=np.ones(_FRAME_COUNT, dtype=bool),
            )
            with (
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_rows_csv"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_iou_curve"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.save_series_comparison"
                ),
                patch(
                    "physbench.evaluation.scenes.circular_motion."
                    "v6_evaluator.write_subject_artifacts",
                    return_value={},
                ),
            ):
                analysis = evaluator.analyze(
                    request,
                    times_s=_TIMES,
                    reference_video=reference,
                    prediction_video=prediction,
                )
        self.assertGreater(analysis.score, 0.8)
        orbit = analysis.metrics[
            "uniform_circular_motion_state_similarity"
        ]
        self.assertEqual(
            "separate_condition_relative_dynamics_v1",
            orbit["phase_policy"],
        )
        timelines = analysis.provenance["expected_timelines"]
        self.assertTrue(
            all(
                value["localization_supervised"][0]
                and not any(value["localization_supervised"][1:])
                for value in timelines
            )
        )


if __name__ == "__main__":
    unittest.main()
