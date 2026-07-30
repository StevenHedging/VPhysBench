from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import cv2
import numpy as np

from ...common.entities.contracts import (
    ObjectTrack,
    VisibilityState,
)
from ...common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
    deduplicate_frame_detections,
    track_open_world_detections,
)
from ...common.entities.timeline import CommonTimeGrid
from ...common.errors import SceneAnalysisError
from ...common.masks.quality import mask_centroid
from ...common.masks.sam2 import MaskPrompt
from .observation import (
    _circle_prompt,
    _hough_candidates,
    _motion_candidates,
    _seed_frame_indices,
)


def _select_entity_set(
    circles: Sequence[tuple[float, float, float]],
    *,
    expected_count: int,
    frame_width: int,
    config: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], float] | None:
    """Select a pre-event N-body seed without assigning striker semantics."""

    if expected_count < 2:
        raise ValueError("collision requires at least two expected entities")
    values = list(circles)
    if len(values) < expected_count:
        return None
    maximum_spread = float(config["maximum_entity_y_spread_px"])
    maximum_radius_ratio = float(config["maximum_radius_ratio"])
    best: tuple[float, list[tuple[float, float, float]]] | None = None
    for combination in itertools.combinations(values, expected_count):
        ordered = sorted(combination, key=lambda item: item[0])
        y = np.asarray([item[1] for item in ordered], dtype=np.float64)
        radii = np.asarray(
            [item[2] for item in ordered], dtype=np.float64
        )
        y_spread = float(np.ptp(y))
        radius_ratio = float(
            np.max(radii) / max(np.min(radii), 1e-6)
        )
        if y_spread > maximum_spread or radius_ratio > maximum_radius_ratio:
            continue
        gaps = np.diff(np.asarray([item[0] for item in ordered]))
        radius_sums = radii[:-1] + radii[1:]
        if np.any(gaps < 0.45 * radius_sums):
            continue
        margin = min(
            ordered[0][0] - ordered[0][2],
            frame_width - ordered[-1][0] - ordered[-1][2],
        )
        visible = float(margin >= 1.0)
        # A compact, level N-body set on one apparatus is preferred, but no
        # assumption is made about which side moves or how many are active.
        normalized_spread = y_spread / max(maximum_spread, 1.0)
        gap_regularizer = float(
            np.mean(
                np.minimum(
                    gaps / np.maximum(4.0 * radius_sums, 1.0),
                    1.0,
                )
            )
        )
        score = (
            0.4 * visible
            + 0.35 * gap_regularizer
            - normalized_spread
            - 0.1 * abs(math.log(max(radius_ratio, 1e-6)))
        )
        if best is None or score > best[0]:
            best = (score, ordered)
    if best is None:
        return None
    return best[1], float(best[0])


def build_multiframe_collision_entity_prompts(
    frames: list[np.ndarray],
    *,
    expected_count: int,
    config: dict[str, Any],
    entity_ids: Sequence[str] | None = None,
    available: Sequence[bool] | None = None,
) -> tuple[list[MaskPrompt], dict[str, Any]]:
    """Build an arbitrary-N shared SAM2 seed from causal visual geometry."""

    if not frames:
        raise SceneAnalysisError(
            "collision_empty_observation",
            "collision video has no frames",
        )
    if expected_count < 2:
        raise SceneAnalysisError(
            "collision_invalid_entity_count",
            f"collision entity manifest declares {expected_count}; expected >=2",
        )
    if entity_ids is None:
        ordered_entity_ids = tuple(
            f"ball_{index + 1}" for index in range(expected_count)
        )
    else:
        ordered_entity_ids = tuple(entity_ids)
        if (
            len(ordered_entity_ids) != expected_count
            or len(set(ordered_entity_ids)) != expected_count
            or any(not value for value in ordered_entity_ids)
        ):
            raise SceneAnalysisError(
                "collision_entity_identity_mismatch",
                "prompt entity IDs must be unique and match expected_count",
            )
    if available is None:
        availability = np.ones(len(frames), dtype=bool)
    else:
        availability = np.asarray(available)
        if (
            availability.shape != (len(frames),)
            or availability.dtype.kind != "b"
        ):
            raise SceneAnalysisError(
                "collision_availability_mismatch",
                "availability must contain one boolean per frame",
            )
    if not availability.any():
        raise SceneAnalysisError(
            "collision_prediction_timeline_unavailable",
            "collision video has no available sampled frames",
        )
    seed_indices = _seed_frame_indices(len(frames), config=config)
    seed_indices = [
        index for index in seed_indices if availability[index]
    ]
    thresholds = [
        float(value) for value in config["hough_accumulator_thresholds"]
    ]
    attempts: list[dict[str, Any]] = []
    viable: list[
        tuple[
            float,
            int,
            list[tuple[float, float, float]],
            str,
            float | None,
        ]
    ] = []
    track_band = (0, frames[0].shape[0])
    for frame_index in seed_indices:
        for threshold in thresholds:
            circles, track_band = _hough_candidates(
                frames[frame_index],
                config=config,
                accumulator_threshold=threshold,
            )
            selected = _select_entity_set(
                circles,
                expected_count=expected_count,
                frame_width=frames[frame_index].shape[1],
                config=config,
            )
            attempts.append(
                {
                    "frame": frame_index,
                    "source": "hough_circle",
                    "accumulator_threshold": threshold,
                    "candidate_count": len(circles),
                    "expected_count": expected_count,
                    "viable": selected is not None,
                }
            )
            if selected is not None:
                objects, geometry_score = selected
                quality = (
                    threshold / max(thresholds)
                    + 0.2 * geometry_score
                    - 0.08 * frame_index / max(len(frames) - 1, 1)
                )
                viable.append(
                    (
                        quality,
                        frame_index,
                        objects,
                        "multiframe_hough_circle",
                        threshold,
                    )
                )
                break
    if not viable:
        background = np.median(
            np.stack(
                [
                    frame
                    for frame, is_available in zip(frames, availability)
                    if is_available
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.uint8)
        for frame_index in seed_indices:
            circles = _motion_candidates(
                frames[frame_index],
                background,
                config=config,
            )
            selected = _select_entity_set(
                circles,
                expected_count=expected_count,
                frame_width=frames[frame_index].shape[1],
                config=config,
            )
            attempts.append(
                {
                    "frame": frame_index,
                    "source": "temporal_median_motion",
                    "candidate_count": len(circles),
                    "expected_count": expected_count,
                    "viable": selected is not None,
                }
            )
            if selected is not None:
                objects, geometry_score = selected
                viable.append(
                    (
                        0.2 * geometry_score,
                        frame_index,
                        objects,
                        "temporal_median_motion",
                        None,
                    )
                )
    if not viable:
        raise SceneAnalysisError(
            "collision_multiframe_objects_missing",
            "no sampled frame contains the manifest-declared collision "
            f"entity count {expected_count}",
        )
    quality, frame_index, ordered, source, threshold = max(
        viable, key=lambda value: value[0]
    )
    prompts = [
        _circle_prompt(
            circle,
            frame_index=frame_index,
            frame_shape=frames[frame_index].shape,
            expand=float(config["box_expand"]),
            minimum_side=int(config["minimum_box_side"]),
            metadata={
                "entity_id": entity_id,
                "source": source,
                "circle_xyr": list(circle),
                "hough_accumulator_threshold": threshold,
            },
        )
        for circle, entity_id in zip(ordered, ordered_entity_ids)
    ]
    return prompts, {
        "backend": (
            "manifest_cardinality_multiframe_circle_motion_plus_bidirectional_sam2"
        ),
        "seed_frame": frame_index,
        "seed_source": source,
        "seed_quality": float(quality),
        "track_band_y": list(track_band),
        "expected_count": expected_count,
        "selected_circles_xyr": [list(value) for value in ordered],
        "selected_entity_ids": list(ordered_entity_ids),
        "candidate_attempts": attempts,
    }


def _valid_mask(
    mask: np.ndarray,
    *,
    minimum_area: int,
    maximum_area: int,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    area = int(np.count_nonzero(binary))
    if area < minimum_area or area > maximum_area:
        return None
    centroid = mask_centroid(binary)
    if centroid is None:
        return None
    return binary, float(area), np.asarray(centroid, dtype=np.float64)


def reference_tracks_from_instance_masks(
    instance_masks: Sequence[Sequence[np.ndarray]],
    *,
    entity_ids: Sequence[str],
    entity_class: str,
    time_grid: CommonTimeGrid,
    minimum_area: int,
    maximum_area_ratio: float,
) -> list[ObjectTrack]:
    """Freeze GT mask channels onto the case's declared physical IDs."""

    if len(instance_masks) != len(entity_ids):
        raise ValueError(
            "reference masks and entity manifest cardinality differ"
        )
    if not instance_masks:
        raise ValueError("reference entity masks are empty")
    frame_count = len(time_grid.times_s)
    height, width = instance_masks[0][0].shape
    maximum_area = int(round(height * width * maximum_area_ratio))
    output: list[ObjectTrack] = []
    for entity_id, masks in zip(entity_ids, instance_masks):
        if len(masks) != frame_count:
            raise ValueError(
                "reference mask timeline differs from common time grid"
            )
        xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
        areas = np.full(frame_count, np.nan, dtype=np.float64)
        observed = np.zeros(frame_count, dtype=bool)
        for frame_index, mask in enumerate(masks):
            value = _valid_mask(
                mask,
                minimum_area=minimum_area,
                maximum_area=maximum_area,
            )
            if value is None:
                continue
            _, area, centroid = value
            xy[frame_index] = centroid
            areas[frame_index] = area
            observed[frame_index] = True
        if not observed.any():
            raise SceneAnalysisError(
                "reference_entity_unobserved",
                f"reference entity {entity_id} has no valid mask",
            )
        output.append(
            ObjectTrack(
                track_id=f"gt_{entity_id}",
                matched_entity_id=entity_id,
                xy=xy,
                observed=observed,
                visibility=tuple(
                    (
                        VisibilityState.VISIBLE
                        if value
                        else VisibilityState.UNKNOWN
                    )
                    for value in observed
                ),
                areas_px2=areas,
                confidence=np.where(observed, 1.0, 0.0),
                existence_observed=observed,
                localization_eligible=observed,
                association_eligible=observed,
                time_weights_s=time_grid.cell_weights_s,
                metadata={
                    "entity_class": entity_class,
                    "formal_exposure_weight": 1.0,
                },
            )
        )
    return output


def _direct_prediction_tracks(
    instance_masks: Sequence[Sequence[np.ndarray]],
    *,
    entity_class: str,
    time_grid: CommonTimeGrid,
    minimum_area: int,
    maximum_area_ratio: float,
) -> tuple[list[OpenWorldTrack], list[list[np.ndarray]]]:
    if not instance_masks:
        return [], []
    frame_count = len(time_grid.times_s)
    height, width = instance_masks[0][0].shape
    maximum_area = int(round(height * width * maximum_area_ratio))
    tracks: list[OpenWorldTrack] = []
    normalized_masks: list[list[np.ndarray]] = []
    for object_index, masks in enumerate(instance_masks):
        if len(masks) != frame_count:
            raise ValueError(
                "prediction mask timeline differs from common time grid"
            )
        detections: list[ObjectDetection] = []
        current_masks: list[np.ndarray] = []
        for frame_index, mask in enumerate(masks):
            binary = np.where(np.asarray(mask) > 0, 255, 0).astype(
                np.uint8
            )
            current_masks.append(binary)
            value = _valid_mask(
                binary,
                minimum_area=minimum_area,
                maximum_area=maximum_area,
            )
            if value is None:
                continue
            valid_mask, area, centroid = value
            detections.append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"direct_{object_index:03d}_{frame_index:05d}"
                    ),
                    xy=centroid,
                    area_px2=area,
                    entity_class=entity_class,
                    mask=valid_mask,
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("expected_directed_sam2",),
                    metadata={"direct_instance": object_index},
                )
            )
        normalized_masks.append(current_masks)
        if detections:
            tracks.append(
                OpenWorldTrack(
                    track_id=f"direct_{object_index:03d}",
                    detections=tuple(detections),
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                )
            )
    return tracks, normalized_masks


def _circle_detection(
    circle: tuple[float, float, float],
    *,
    frame_index: int,
    shape: tuple[int, int],
    source: str,
    tier: EvidenceTier,
    confidence: float,
) -> ObjectDetection:
    x, y, radius = circle
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(x)), int(round(y))),
        max(int(round(radius)), 1),
        255,
        -1,
    )
    return ObjectDetection(
        frame_index=frame_index,
        detection_id=(
            f"{source}_{frame_index:05d}_{int(round(x)):04d}_"
            f"{int(round(y)):04d}"
        ),
        xy=np.asarray([x, y], dtype=np.float64),
        area_px2=float(np.count_nonzero(mask)),
        entity_class="ball",
        mask=mask,
        confidence=confidence,
        evidence_tier=tier,
        sources=(source,),
        metadata={"circle_xyr": [x, y, radius]},
    )


def _matches_direct_mask(
    detection: ObjectDetection,
    direct_masks: Sequence[Sequence[np.ndarray]],
    *,
    center_distance_fraction: float = 0.8,
) -> bool:
    if detection.mask is None:
        return False
    if (
        not math.isfinite(float(center_distance_fraction))
        or center_distance_fraction < 0.0
    ):
        raise ValueError(
            "center_distance_fraction must be finite and non-negative"
        )
    frame_index = detection.frame_index
    x = int(round(float(detection.xy[0])))
    y = int(round(float(detection.xy[1])))
    for instance in direct_masks:
        mask = instance[frame_index]
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue
        if mask[y, x] > 0:
            return True
        direct_area = int(np.count_nonzero(mask))
        direct_centroid = mask_centroid(mask)
        if direct_area > 0 and direct_centroid is not None:
            direct_radius = math.sqrt(direct_area / math.pi)
            candidate_radius = math.sqrt(detection.area_px2 / math.pi)
            if float(
                np.linalg.norm(
                    np.asarray(direct_centroid, dtype=np.float64)
                    - detection.xy
                )
            ) <= center_distance_fraction * max(
                direct_radius,
                candidate_radius,
                1.0,
            ):
                # This rejects a second detector response on the same body.
                # It deliberately stays below one object radius, so two
                # distinct tangent balls are not merged.
                return True
        intersection = int(
            np.logical_and(mask > 0, detection.mask > 0).sum()
        )
        if intersection / max(
            min(
                int(np.count_nonzero(mask)),
                int(np.count_nonzero(detection.mask)),
            ),
            1,
        ) >= 0.45:
            return True
    return False


def _calibrate_residual_track_evidence(
    tracks: Sequence[OpenWorldTrack],
    *,
    hough_confirmation_frames: int,
    motion_confirmation_frames: int,
    motion_minimum_displacement_radius_ratio: float,
) -> tuple[tuple[OpenWorldTrack, ...], dict[str, int]]:
    """Keep static-circle proposals visible without over-trusting them.

    Motion-supported tracks need temporal confirmation before becoming full
    participants; short motion tracks retain a tentative penalty. A
    Hough-only track needs persistence before it contributes even tentative
    exposure, and persistence alone never upgrades a static circle to a full
    participant. This prevents ruler markings, highlights, and median-
    background motion ghosts from becoming physical balls.
    """

    if hough_confirmation_frames < 2:
        raise ValueError("hough_confirmation_frames must be at least two")
    if motion_confirmation_frames < 2:
        raise ValueError("motion_confirmation_frames must be at least two")
    minimum_motion_ratio = float(
        motion_minimum_displacement_radius_ratio
    )
    if (
        not math.isfinite(minimum_motion_ratio)
        or minimum_motion_ratio <= 0.0
    ):
        raise ValueError(
            "motion_minimum_displacement_radius_ratio must be positive"
        )
    output: list[OpenWorldTrack] = []
    ambiguous = 0
    tentative = 0
    participant = 0
    static_motion_rejections = 0
    insufficient_motion_support = 0
    for track in tracks:
        sources = {
            source
            for detection in track.detections
            for source in detection.sources
        }
        distinct_frames = len(
            {detection.frame_index for detection in track.detections}
        )
        centers = np.stack(
            [detection.xy for detection in track.detections],
            axis=0,
        )
        motion_span_px = float(
            np.max(np.linalg.norm(centers - centers[0], axis=1))
        )
        median_radius_px = float(
            np.median(
                [
                    math.sqrt(detection.area_px2 / math.pi)
                    for detection in track.detections
                ]
            )
        )
        motion_span_radius_ratio = (
            motion_span_px / max(median_radius_px, 1e-6)
        )
        if "motion_circle" in sources:
            motion_supported_frames = len(
                {
                    detection.frame_index
                    for detection in track.detections
                    if "motion_circle" in detection.sources
                }
            )
            temporally_confirmed = (
                motion_supported_frames >= motion_confirmation_frames
            )
            physically_moving = (
                motion_span_radius_ratio >= minimum_motion_ratio
            )
            confirmed = temporally_confirmed and physically_moving
            if temporally_confirmed and not physically_moving:
                static_motion_rejections += 1
            elif not temporally_confirmed:
                insufficient_motion_support += 1
            calibrated = OpenWorldTrack(
                track_id=track.track_id,
                detections=track.detections,
                confirmed=confirmed,
                evidence_tier=(
                    EvidenceTier.PARTICIPANT
                    if confirmed
                    else EvidenceTier.TENTATIVE
                ),
            )
        elif sources and sources <= {"hough_circle"}:
            confirmed = distinct_frames >= hough_confirmation_frames
            calibrated = OpenWorldTrack(
                track_id=track.track_id,
                detections=track.detections,
                confirmed=confirmed,
                evidence_tier=(
                    EvidenceTier.TENTATIVE
                    if confirmed
                    else EvidenceTier.AMBIGUOUS
                ),
            )
        else:
            calibrated = track
        output.append(calibrated)
        if calibrated.evidence_tier is EvidenceTier.AMBIGUOUS:
            ambiguous += 1
        elif calibrated.evidence_tier is EvidenceTier.TENTATIVE:
            tentative += 1
        elif calibrated.evidence_tier is EvidenceTier.PARTICIPANT:
            participant += 1
    return tuple(output), {
        "ambiguous_tracks": ambiguous,
        "tentative_tracks": tentative,
        "participant_tracks": participant,
        "static_motion_rejections": static_motion_rejections,
        "insufficient_motion_support_tracks": (
            insufficient_motion_support
        ),
    }


def _apply_availability_to_masks(
    instance_masks: Sequence[Sequence[np.ndarray]],
    *,
    available: np.ndarray,
) -> list[list[np.ndarray]]:
    output: list[list[np.ndarray]] = []
    for instance in instance_masks:
        if len(instance) != len(available):
            raise ValueError(
                "prediction mask timeline differs from availability"
            )
        output.append(
            [
                (
                    np.asarray(mask, dtype=np.uint8)
                    if available[index]
                    else np.zeros_like(mask, dtype=np.uint8)
                )
                for index, mask in enumerate(instance)
            ]
        )
    return output


def prediction_observation_from_instance_masks(
    directed_instance_masks: Sequence[Sequence[np.ndarray]],
    *,
    time_grid: CommonTimeGrid,
    quality_config: dict[str, Any],
    available: Sequence[bool] | None = None,
) -> OpenWorldObservation:
    """Build a direct-only fallback when residual discovery cannot run."""

    frame_count = len(time_grid.times_s)
    availability = (
        np.ones(frame_count, dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if (
        availability.shape != (frame_count,)
        or availability.dtype.kind != "b"
    ):
        raise ValueError(
            "availability must contain one boolean per common time sample"
        )
    normalized = _apply_availability_to_masks(
        directed_instance_masks,
        available=availability,
    )
    direct_tracks, _ = _direct_prediction_tracks(
        normalized,
        entity_class="ball",
        time_grid=time_grid,
        minimum_area=int(quality_config["minimum_mask_pixels"]),
        maximum_area_ratio=float(
            quality_config["maximum_mask_area_ratio"]
        ),
    )
    return OpenWorldObservation(
        tracks=tuple(direct_tracks),
        overflow_counts=np.zeros(frame_count, dtype=np.float64),
        diagnostics={
            "status": "directed_tracks_only",
            "directed_tracks": len(direct_tracks),
            "available_frames": int(np.count_nonzero(availability)),
        },
    )


def discover_prediction_objects(
    frames: list[np.ndarray],
    *,
    directed_instance_masks: Sequence[Sequence[np.ndarray]],
    time_grid: CommonTimeGrid,
    observation_config: dict[str, Any],
    quality_config: dict[str, Any],
    available: Sequence[bool] | None = None,
) -> OpenWorldObservation:
    """Add per-frame residual discovery to expected-directed SAM2 masks."""

    if len(frames) != len(time_grid.times_s):
        raise ValueError("frames and common time grid have different lengths")
    availability = (
        np.ones(len(frames), dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if (
        availability.shape != (len(frames),)
        or availability.dtype.kind != "b"
    ):
        raise ValueError(
            "availability must contain one boolean per common time sample"
        )
    directed_instance_masks = _apply_availability_to_masks(
        directed_instance_masks,
        available=availability,
    )
    direct_tracks, direct_masks = _direct_prediction_tracks(
        directed_instance_masks,
        entity_class="ball",
        time_grid=time_grid,
        minimum_area=int(quality_config["minimum_mask_pixels"]),
        maximum_area_ratio=float(
            quality_config["maximum_mask_area_ratio"]
        ),
    )
    if not availability.any():
        return OpenWorldObservation(
            tracks=tuple(direct_tracks),
            overflow_counts=np.zeros(len(frames), dtype=np.float64),
            diagnostics={
                "status": "no_available_prediction_frames",
                "directed_tracks": len(direct_tracks),
                "residual_tracks": 0,
                "available_frames": 0,
            },
        )
    background = np.median(
        np.stack(
            [
                frame
                for frame, is_available in zip(frames, availability)
                if is_available
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.uint8)
    direct_centroids = [
        mask_centroid(mask)
        for instance in direct_masks
        for mask in instance
        if np.count_nonzero(mask)
    ]
    direct_areas = [
        int(np.count_nonzero(mask))
        for instance in direct_masks
        for mask in instance
        if np.count_nonzero(mask)
    ]
    axis_y = (
        float(np.median([value[1] for value in direct_centroids if value is not None]))
        if any(value is not None for value in direct_centroids)
        else 0.5
        * frames[0].shape[0]
        * (
            float(observation_config["track_top_ratio"])
            + float(observation_config["track_bottom_ratio"])
        )
    )
    median_radius = (
        math.sqrt(float(np.median(direct_areas)) / math.pi)
        if direct_areas
        else float(observation_config["minimum_circle_radius_px"])
    )
    minimum_residual_radius = median_radius * float(
        observation_config.get("residual_minimum_radius_ratio", 0.4)
    )
    maximum_residual_radius = median_radius * float(
        observation_config.get("residual_maximum_radius_ratio", 2.5)
    )
    if (
        not math.isfinite(minimum_residual_radius)
        or not math.isfinite(maximum_residual_radius)
        or minimum_residual_radius <= 0.0
        or maximum_residual_radius < minimum_residual_radius
    ):
        raise ValueError("residual radius ratios define an invalid range")
    duplicate_center_fraction = float(
        observation_config.get(
            "residual_direct_duplicate_center_fraction",
            0.8,
        )
    )
    axis_tolerance = min(
        float(observation_config["maximum_entity_y_spread_px"]),
        max(1.5 * median_radius, 12.0),
    )
    candidates_by_frame: list[list[ObjectDetection]] = []
    thresholds = [
        float(value)
        for value in observation_config["hough_accumulator_thresholds"][:2]
    ]
    scale_rejections = 0
    direct_duplicate_rejections = 0
    for frame_index, frame in enumerate(frames):
        candidates: list[ObjectDetection] = []
        if not availability[frame_index]:
            candidates_by_frame.append(candidates)
            continue
        for circle in _motion_candidates(
            frame,
            background,
            config=observation_config,
        ):
            if abs(circle[1] - axis_y) > axis_tolerance:
                continue
            if not minimum_residual_radius <= circle[2] <= maximum_residual_radius:
                scale_rejections += 1
                continue
            candidates.append(
                _circle_detection(
                    circle,
                    frame_index=frame_index,
                    shape=frame.shape[:2],
                    source="motion_circle",
                    tier=EvidenceTier.PARTICIPANT,
                    confidence=0.95,
                )
            )
        hough_values: list[tuple[float, float, float]] = []
        for threshold in thresholds:
            values, _ = _hough_candidates(
                frame,
                config=observation_config,
                accumulator_threshold=threshold,
            )
            hough_values.extend(values)
        for circle in hough_values:
            if abs(circle[1] - axis_y) > axis_tolerance:
                continue
            if not minimum_residual_radius <= circle[2] <= maximum_residual_radius:
                scale_rejections += 1
                continue
            candidates.append(
                _circle_detection(
                    circle,
                    frame_index=frame_index,
                    shape=frame.shape[:2],
                    source="hough_circle",
                    # Static circle geometry is ambiguous until temporal
                    # calibration; apparatus markings often look circular.
                    tier=EvidenceTier.AMBIGUOUS,
                    confidence=0.8,
                )
            )
        candidates = deduplicate_frame_detections(candidates)
        residual_candidates: list[ObjectDetection] = []
        for value in candidates:
            if _matches_direct_mask(
                value,
                direct_masks,
                center_distance_fraction=duplicate_center_fraction,
            ):
                direct_duplicate_rejections += 1
                continue
            residual_candidates.append(value)
        candidates_by_frame.append(residual_candidates)
    residual = track_open_world_detections(
        candidates_by_frame,
        time_grid=time_grid,
        maximum_gap_s=float(
            observation_config.get("residual_maximum_gap_s", 0.25)
        ),
        maximum_assignment_cost=float(
            observation_config.get(
                "residual_maximum_assignment_cost",
                4.0,
            )
        ),
        minimum_scale_px=float(
            observation_config["minimum_circle_radius_px"]
        ),
        maximum_tracks=int(
            observation_config.get("maximum_residual_tracks", 32)
        ),
    )
    calibrated_tracks, evidence_diagnostics = (
        _calibrate_residual_track_evidence(
            residual.tracks,
            hough_confirmation_frames=int(
                observation_config.get(
                    "residual_hough_confirmation_frames",
                    3,
                )
            ),
            motion_confirmation_frames=int(
                observation_config.get(
                    "residual_motion_confirmation_frames",
                    3,
                )
            ),
            motion_minimum_displacement_radius_ratio=float(
                observation_config.get(
                    "residual_motion_minimum_displacement_radius_ratio",
                    1.0,
                )
            ),
        )
    )
    return OpenWorldObservation(
        tracks=tuple([*direct_tracks, *calibrated_tracks]),
        overflow_counts=residual.overflow_counts,
        diagnostics={
            "directed_tracks": len(direct_tracks),
            "residual_tracks": len(calibrated_tracks),
            "axis_y": axis_y,
            "axis_tolerance_px": axis_tolerance,
            "median_direct_radius_px": median_radius,
            "residual_radius_range_px": [
                minimum_residual_radius,
                maximum_residual_radius,
            ],
            "residual_scale_rejections": scale_rejections,
            "residual_direct_duplicate_rejections": (
                direct_duplicate_rejections
            ),
            "residual_evidence": evidence_diagnostics,
            "residual_track_summaries": [
                {
                    "track_id": track.track_id,
                    "frames": len(track.detections),
                    "first_frame": track.detections[0].frame_index,
                    "last_frame": track.detections[-1].frame_index,
                    "confirmed": track.confirmed,
                    "evidence_tier": track.evidence_tier.value,
                    "formal_exposure_weight": (
                        track.formal_exposure_weight
                    ),
                    "observed_exposure_s": float(
                        sum(
                            time_grid.cell_weights_s[
                                detection.frame_index
                            ]
                            for detection in track.detections
                        )
                    ),
                    "formal_exposure_s": float(
                        track.formal_exposure_weight
                        * sum(
                            time_grid.cell_weights_s[
                                detection.frame_index
                            ]
                            for detection in track.detections
                        )
                    ),
                    "physics_participant": (
                        track.evidence_tier
                        is EvidenceTier.PARTICIPANT
                    ),
                    "motion_supported_frames": len(
                        {
                            detection.frame_index
                            for detection in track.detections
                            if "motion_circle" in detection.sources
                        }
                    ),
                    "sources": sorted(
                        {
                            source
                            for detection in track.detections
                            for source in detection.sources
                        }
                    ),
                    "median_radius_px": float(
                        np.median(
                            [
                                math.sqrt(
                                    detection.area_px2 / math.pi
                                )
                                for detection in track.detections
                            ]
                        )
                    ),
                    "motion_span_px": float(
                        np.max(
                            np.linalg.norm(
                                np.stack(
                                    [
                                        detection.xy
                                        for detection in track.detections
                                    ],
                                    axis=0,
                                )
                                - track.detections[0].xy,
                                axis=1,
                            )
                        )
                    ),
                    "motion_span_radius_ratio": float(
                        np.max(
                            np.linalg.norm(
                                np.stack(
                                    [
                                        detection.xy
                                        for detection in track.detections
                                    ],
                                    axis=0,
                                )
                                - track.detections[0].xy,
                                axis=1,
                            )
                        )
                        / max(
                            np.median(
                                [
                                    math.sqrt(
                                        detection.area_px2 / math.pi
                                    )
                                    for detection in track.detections
                                ]
                            ),
                            1e-6,
                        )
                    ),
                }
                for track in calibrated_tracks
            ],
            "available_frames": int(np.count_nonzero(availability)),
            "residual": residual.diagnostics,
        },
    )


__all__ = [
    "build_multiframe_collision_entity_prompts",
    "discover_prediction_objects",
    "prediction_observation_from_instance_masks",
    "reference_tracks_from_instance_masks",
]
