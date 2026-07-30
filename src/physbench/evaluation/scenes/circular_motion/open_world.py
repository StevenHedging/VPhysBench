from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ...common.entities import (
    CommonTimeGrid,
    EvidenceTier,
    ObjectDetection,
    ObjectTrack,
    OpenWorldObservation,
    OpenWorldTrack,
    VisibilityState,
    freeze_condition_identity,
    track_open_world_detections,
)
from ...common.entities.manifest import EntityDeclaration
from ...common.errors import SceneAnalysisError
from ...common.tracking import InstanceTracks, interpolate_trace
from .scoring import extract_orbit_traces, score_orbits


_CANONICAL_DISK_RADIUS = 100.0


@dataclass(frozen=True)
class CircularOpenWorldObservation:
    """All physical-object proposals found inside the rotating platform."""

    objects: OpenWorldObservation
    disk_centers_xy: np.ndarray
    disk_radii_px: np.ndarray
    union_masks: tuple[np.ndarray, ...]
    disk_masks: tuple[np.ndarray, ...]
    candidate_counts: np.ndarray
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class FrozenCircularReference:
    """Reference identities frozen from the condition/initial reliable window."""

    tracks: tuple[ObjectTrack, ...]
    entity_ids: tuple[str, ...]
    source_track_ids: tuple[str, ...]
    instance_masks: tuple[tuple[np.ndarray, ...], ...]
    anchors: tuple[Mapping[str, object], ...]


def _filled_series(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise SceneAnalysisError(
            "circular_disk_not_found",
            "no reliable rotating-platform geometry was observed",
        )
    output = np.asarray(values, dtype=np.float64).copy()
    indices = np.arange(len(output), dtype=np.float64)
    if output.ndim == 1:
        output[~valid] = np.interp(
            indices[~valid], indices[valid], output[valid]
        )
        return output
    for column in range(output.shape[1]):
        output[~valid, column] = np.interp(
            indices[~valid],
            indices[valid],
            output[valid, column],
        )
    return output


def _disk_geometry(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    lower = np.asarray(config["green_hsv_lower"], dtype=np.uint8)
    upper = np.asarray(config["green_hsv_upper"], dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(
        green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise SceneAnalysisError(
            "circular_disk_not_found",
            "no green rotating-platform proposal",
        )
    contour = max(contours, key=cv2.contourArea)
    disk = np.zeros_like(green)
    cv2.drawContours(disk, [contour], -1, 255, thickness=-1)
    ratio = float(np.mean(disk > 0))
    minimum_ratio = float(config["minimum_disk_area_ratio"])
    if ratio < minimum_ratio:
        raise SceneAnalysisError(
            "circular_disk_not_found",
            f"green disk ratio {ratio:.4f} is below {minimum_ratio:.4f}",
        )
    erosion_size = max(3, int(config["disk_erosion_kernel"]) | 1)
    erosion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )
    interior = cv2.erode(disk, erosion)
    moments = cv2.moments(contour)
    if abs(float(moments["m00"])) <= 1e-12:
        raise SceneAnalysisError(
            "circular_disk_not_found",
            "rotating-platform contour has zero area",
        )
    center = np.asarray(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ],
        dtype=np.float64,
    )
    radius = math.sqrt(max(float(cv2.contourArea(contour)), 1.0) / math.pi)
    return center, radius, interior, green


def _component_detections(
    frame: np.ndarray,
    *,
    frame_index: int,
    center_xy: np.ndarray,
    disk_radius_px: float,
    interior_mask: np.ndarray,
    green_mask: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[ObjectDetection], np.ndarray, int, int]:
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.bitwise_and(
        cv2.bitwise_not(green_mask), interior_mask
    )
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, opening)
    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(foreground, connectivity=8)
    )
    minimum_area = int(config["minimum_component_area"])
    maximum_area = int(
        frame.shape[0]
        * frame.shape[1]
        * float(config["maximum_component_area_ratio"])
    )
    candidates: list[tuple[int, int]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if minimum_area <= area <= maximum_area:
            candidates.append((label, area))
    # The legacy maximum_candidates value used to truncate to an expected-N
    # selector.  The open-world observer deliberately ignores it and instead
    # uses a much larger safety cap.  Everything beyond the cap is conserved
    # as overflow exposure rather than silently disappearing.
    safety_cap = int(config.get("open_world_maximum_candidates_per_frame", 64))
    if safety_cap < 1:
        raise ValueError(
            "open_world_maximum_candidates_per_frame must be positive"
        )
    candidates.sort(key=lambda item: (-item[1], item[0]))
    retained = candidates[:safety_cap]
    overflow = max(len(candidates) - safety_cap, 0)
    output: list[ObjectDetection] = []
    union = np.zeros(frame.shape[:2], dtype=np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    for component_index, (label, area) in enumerate(retained):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        union = cv2.bitwise_or(union, mask)
        actual_xy = np.asarray(centroids[label], dtype=np.float64)
        normalized_xy = (
            (actual_xy - center_xy)
            / max(float(disk_radius_px), 1e-6)
        )
        canonical_xy = normalized_xy * _CANONICAL_DISK_RADIUS
        normalized_area = (
            float(area)
            * (_CANONICAL_DISK_RADIUS / max(disk_radius_px, 1e-6)) ** 2
        )
        pixels = mask > 0
        mean_lab = (
            np.mean(lab[pixels], axis=0).astype(float).tolist()
            if np.any(pixels)
            else [0.0, 0.0, 0.0]
        )
        output.append(
            ObjectDetection(
                frame_index=frame_index,
                detection_id=(
                    f"disk_component_{frame_index:05d}_"
                    f"{component_index:03d}"
                ),
                xy=canonical_xy,
                area_px2=max(normalized_area, 1e-6),
                entity_class="orbiter",
                mask=mask,
                confidence=0.85,
                evidence_tier=EvidenceTier.TENTATIVE,
                sources=("disk_interior_color_contrast",),
                metadata={
                    "actual_xy": actual_xy.tolist(),
                    "actual_area_px2": area,
                    "normalized_disk_xy": normalized_xy.tolist(),
                    "normalized_orbit_radius": float(
                        np.linalg.norm(normalized_xy)
                    ),
                    "absolute_phase_rad": float(
                        math.atan2(normalized_xy[1], normalized_xy[0])
                    ),
                    "mean_lab": mean_lab,
                    "apparatus_exclusion": "eroded_green_disk_interior",
                },
            )
        )
    # Include every accepted component in the audit union, including objects
    # that cannot be assigned a causal track because of the global track cap.
    for label, _ in candidates[safety_cap:]:
        union[labels == label] = 255
    return output, union, len(candidates), overflow


def observe_circular_objects(
    frames: Sequence[np.ndarray],
    *,
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
    available: Sequence[bool] | None = None,
) -> CircularOpenWorldObservation:
    """Discover and causally track every credible object inside the disk."""

    frame_count = len(time_grid.times_s)
    if len(frames) != frame_count or frame_count < 1:
        raise ValueError(
            "frames must be non-empty and aligned to the common time grid"
        )
    availability = (
        np.ones(frame_count, dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if availability.shape != (frame_count,) or availability.dtype.kind != "b":
        raise ValueError("available must contain one boolean per frame")
    shape = np.asarray(frames[0]).shape[:2]
    if any(np.asarray(frame).shape[:2] != shape for frame in frames):
        raise ValueError("all circular-motion frames must share one canvas")

    centers = np.full((frame_count, 2), np.nan, dtype=np.float64)
    radii = np.full(frame_count, np.nan, dtype=np.float64)
    disk_masks = [np.zeros(shape, dtype=np.uint8) for _ in frames]
    union_masks = [np.zeros(shape, dtype=np.uint8) for _ in frames]
    detections_by_frame: list[list[ObjectDetection]] = [
        [] for _ in frames
    ]
    candidate_counts = np.zeros(frame_count, dtype=np.int64)
    candidate_overflow = np.zeros(frame_count, dtype=np.float64)
    disk_valid = np.zeros(frame_count, dtype=bool)
    frame_failures: list[dict[str, object]] = []
    for frame_index, frame in enumerate(frames):
        if not availability[frame_index]:
            continue
        try:
            center, radius, interior, green = _disk_geometry(
                np.asarray(frame), config=config
            )
            (
                detections,
                union,
                candidate_count,
                overflow,
            ) = _component_detections(
                np.asarray(frame),
                frame_index=frame_index,
                center_xy=center,
                disk_radius_px=radius,
                interior_mask=interior,
                green_mask=green,
                config=config,
            )
        except (SceneAnalysisError, ValueError, cv2.error) as exc:
            frame_failures.append(
                {
                    "frame": frame_index,
                    "code": getattr(exc, "code", "circular_frame_observation_failed"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        centers[frame_index] = center
        radii[frame_index] = radius
        disk_masks[frame_index] = interior
        union_masks[frame_index] = union
        detections_by_frame[frame_index] = detections
        candidate_counts[frame_index] = candidate_count
        candidate_overflow[frame_index] = overflow
        disk_valid[frame_index] = True
    centers = _filled_series(centers, disk_valid)
    radii = _filled_series(radii, disk_valid)

    tracking = track_open_world_detections(
        detections_by_frame,
        time_grid=time_grid,
        maximum_gap_s=float(config.get("open_world_maximum_gap_s", 0.4)),
        maximum_assignment_cost=float(
            config.get("open_world_maximum_assignment_cost", 4.0)
        ),
        minimum_scale_px=float(
            config.get("open_world_minimum_scale_px", 2.0)
        ),
        area_weight=float(config.get("open_world_area_weight", 0.4)),
        maximum_tracks=int(config.get("open_world_maximum_tracks", 64)),
    )
    split_tracks, appearance_switches = _split_appearance_switches(
        tracking.tracks,
        threshold_lab=float(
            config.get(
                "open_world_appearance_switch_threshold_lab", 45.0
            )
        ),
    )
    objects = _relative_phase_observation(
        OpenWorldObservation(
            tracks=split_tracks,
            overflow_counts=tracking.overflow_counts + candidate_overflow,
            diagnostics=tracking.diagnostics,
        ),
        center_fallback_radius=float(
            config.get("center_fallback_radius_ratio", 0.08)
        ),
    )
    return CircularOpenWorldObservation(
        objects=objects,
        disk_centers_xy=centers,
        disk_radii_px=radii,
        union_masks=tuple(union_masks),
        disk_masks=tuple(disk_masks),
        candidate_counts=candidate_counts,
        diagnostics={
            "backend": "green_disk_open_world_components_v2",
            "disk_valid_frame_ratio": float(np.mean(disk_valid)),
            "candidate_counts": candidate_counts.tolist(),
            "candidate_safety_cap": int(
                config.get("open_world_maximum_candidates_per_frame", 64)
            ),
            "frame_failures": frame_failures,
            "apparatus_policy": "green_disk_removed_then_eroded_interior",
            "tracking": tracking.diagnostics,
            "appearance_identity_switch_splits": appearance_switches,
            "formal_track_count": sum(
                track.formal_exposure_weight > 0.0
                for track in objects.tracks
            ),
        },
    )


def _split_appearance_switches(
    tracks: Sequence[OpenWorldTrack],
    *,
    threshold_lab: float,
) -> tuple[tuple[OpenWorldTrack, ...], int]:
    """Turn a large within-track appearance jump into an auditable ID break."""

    if not math.isfinite(threshold_lab) or threshold_lab <= 0.0:
        raise ValueError(
            "open_world_appearance_switch_threshold_lab must be positive"
        )
    output: list[OpenWorldTrack] = []
    split_count = 0
    for track in tracks:
        segments: list[list[ObjectDetection]] = [[]]
        previous_lab: np.ndarray | None = None
        for detection in track.detections:
            current_lab = np.asarray(
                detection.metadata["mean_lab"], dtype=np.float64
            )
            if (
                previous_lab is not None
                and float(np.linalg.norm(current_lab - previous_lab))
                > threshold_lab
            ):
                segments.append([])
                split_count += 1
            segments[-1].append(detection)
            previous_lab = current_lab
        for segment_index, detections in enumerate(segments):
            output.append(
                OpenWorldTrack(
                    track_id=(
                        track.track_id
                        if len(segments) == 1
                        else f"{track.track_id}:appearance_{segment_index}"
                    ),
                    detections=tuple(detections),
                    confirmed=track.confirmed,
                    evidence_tier=track.evidence_tier,
                )
            )
    return tuple(output), split_count


def _relative_phase_observation(
    observation: OpenWorldObservation,
    *,
    center_fallback_radius: float,
) -> OpenWorldObservation:
    if not 0.0 <= center_fallback_radius < 1.0:
        raise ValueError("center_fallback_radius must be in [0, 1)")
    tracks: list[OpenWorldTrack] = []
    for track in observation.tracks:
        normalized = np.stack(
            [
                np.asarray(
                    detection.metadata["normalized_disk_xy"],
                    dtype=np.float64,
                )
                for detection in track.detections
            ],
            axis=0,
        )
        radii = np.linalg.norm(normalized, axis=1)
        phases = np.unwrap(np.arctan2(normalized[:, 1], normalized[:, 0]))
        reliable = np.flatnonzero(radii > center_fallback_radius)
        phase_anchor = (
            float(phases[reliable[0]]) if len(reliable) else 0.0
        )
        detections: list[ObjectDetection] = []
        for index, detection in enumerate(track.detections):
            radius = float(radii[index])
            if radius <= center_fallback_radius:
                canonical = normalized[index] * _CANONICAL_DISK_RADIUS
                mode = "cartesian_center_fallback"
            else:
                relative_phase = float(phases[index] - phase_anchor)
                canonical = (
                    _CANONICAL_DISK_RADIUS
                    * radius
                    * np.asarray(
                        [math.cos(relative_phase), math.sin(relative_phase)]
                    )
                )
                mode = "relative_polar"
            metadata = {
                **dict(detection.metadata),
                "phase_anchor_rad": phase_anchor,
                "relative_phase_rad": float(phases[index] - phase_anchor),
                "distance_mode": mode,
            }
            detections.append(
                ObjectDetection(
                    frame_index=detection.frame_index,
                    detection_id=detection.detection_id,
                    xy=canonical,
                    area_px2=detection.area_px2,
                    entity_class=detection.entity_class,
                    mask=detection.mask,
                    confidence=detection.confidence,
                    evidence_tier=detection.evidence_tier,
                    sources=detection.sources,
                    metadata=metadata,
                )
            )
        tracks.append(
            OpenWorldTrack(
                track_id=track.track_id,
                detections=tuple(detections),
                confirmed=track.confirmed,
                evidence_tier=track.evidence_tier,
            )
        )
    return OpenWorldObservation(
        tracks=tuple(tracks),
        overflow_counts=observation.overflow_counts,
        diagnostics={
            **dict(observation.diagnostics),
            "coordinate_system": (
                "disk_normalized_relative_polar_with_cartesian_center_fallback"
            ),
            "canonical_disk_radius": _CANONICAL_DISK_RADIUS,
            "center_fallback_radius_ratio": center_fallback_radius,
        },
    )


def _entity_radius(entity: EntityDeclaration) -> float:
    values = {
        attribute.name: float(attribute.value)
        for attribute in entity.physical_attributes
    }
    radius = values.get("orbit_radius")
    if radius is not None and math.isfinite(radius):
        return radius
    anchor = entity.condition_anchor.get("initial_order_index", 0)
    return float(anchor) if isinstance(anchor, int) else 0.0


def _track_descriptor(track: OpenWorldTrack) -> dict[str, object]:
    detections = track.detections
    initial_window = detections[: min(len(detections), 3)]
    radii = [
        float(value.metadata["normalized_orbit_radius"])
        for value in initial_window
    ]
    labs = np.asarray(
        [value.metadata["mean_lab"] for value in initial_window],
        dtype=np.float64,
    )
    return {
        "track_id": track.track_id,
        "normalized_orbit_radius": float(np.median(radii)),
        "initial_phase_rad": float(
            initial_window[0].metadata["absolute_phase_rad"]
        ),
        "mean_lab": np.mean(labs, axis=0).tolist(),
        "first_frame": int(initial_window[0].frame_index),
        "observed_frames": len(detections),
        "evidence_tier": track.evidence_tier.value,
    }


def identity_anchors_from_observation(
    observation: CircularOpenWorldObservation,
    *,
    entities: Sequence[EntityDeclaration],
) -> tuple[Mapping[str, object], ...]:
    """Map immutable condition descriptors to manifest IDs by orbit radius."""

    entity_values = sorted(
        entities,
        key=lambda entity: (
            _entity_radius(entity),
            int(entity.condition_anchor.get("initial_order_index", 0)),
            entity.entity_id,
        ),
    )
    candidates = list(observation.objects.tracks)
    candidates.sort(
        key=lambda track: (
            -len(track.detections),
            track.track_id,
        )
    )
    candidates = candidates[: len(entity_values)]
    candidates.sort(
        key=lambda track: (
            float(_track_descriptor(track)["normalized_orbit_radius"]),
            float(_track_descriptor(track)["initial_phase_rad"]),
            track.track_id,
        )
    )
    if len(candidates) < len(entity_values):
        raise SceneAnalysisError(
            "insufficient_condition_entity_anchors",
            f"condition exposes {len(candidates)} objects for "
            f"{len(entity_values)} manifest entities",
        )
    return tuple(
        {
            "entity_id": entity.entity_id,
            "appearance_label": entity.condition_anchor.get(
                "appearance_label"
            ),
            **_track_descriptor(track),
        }
        for entity, track in zip(entity_values, candidates)
    )


def freeze_prediction_identity(
    observation: OpenWorldObservation,
    *,
    entities: Sequence[EntityDeclaration],
    condition_anchors: Sequence[Mapping[str, object]],
    initial_window_frames: int = 3,
    maximum_match_cost: float = 3.0,
):
    """Freeze prediction track IDs against condition appearance/radius/phase."""

    if initial_window_frames < 1:
        raise ValueError("initial_window_frames must be positive")
    entity_specs = [entity.to_entity_spec() for entity in entities]
    anchors = {
        str(value["entity_id"]): value for value in condition_anchors
    }
    descriptors = {
        track.track_id: _track_descriptor(track)
        for track in observation.tracks
        if track.detections[0].frame_index < initial_window_frames
        and track.formal_exposure_weight > 0.0
    }

    def identity_cost(entity, track_id: str) -> float:
        anchor = anchors.get(entity.entity_id)
        candidate = descriptors.get(track_id)
        if anchor is None or candidate is None:
            return float("inf")
        radial = abs(
            float(anchor["normalized_orbit_radius"])
            - float(candidate["normalized_orbit_radius"])
        ) / 0.2
        first_lab = np.asarray(anchor["mean_lab"], dtype=np.float64)
        second_lab = np.asarray(candidate["mean_lab"], dtype=np.float64)
        appearance = float(np.linalg.norm(first_lab - second_lab)) / 80.0
        phase_delta = abs(
            math.atan2(
                math.sin(
                    float(candidate["initial_phase_rad"])
                    - float(anchor["initial_phase_rad"])
                ),
                math.cos(
                    float(candidate["initial_phase_rad"])
                    - float(anchor["initial_phase_rad"])
                ),
            )
        )
        return radial + appearance + phase_delta

    return freeze_condition_identity(
        entity_specs,
        sorted(descriptors),
        identity_cost_hook=identity_cost,
        maximum_match_cost=maximum_match_cost,
        unmatched_entity_cost=1.0,
        unmatched_track_cost=1.0,
        maximum_entities=max(16, len(entity_specs) + len(descriptors)),
    )


def freeze_reference_identities(
    observation: CircularOpenWorldObservation,
    *,
    entities: Sequence[EntityDeclaration],
    time_grid: CommonTimeGrid,
    minimum_valid_ratio: float = 0.5,
) -> FrozenCircularReference:
    """Freeze expected IDs by appearance/radius/phase in the initial window."""

    entity_values = list(entities)
    if not entity_values:
        raise ValueError("at least one circular entity is required")
    if any(entity.entity_class != "orbiter" for entity in entity_values):
        raise ValueError("circular entities must use entity_class='orbiter'")
    frame_count = len(time_grid.times_s)
    minimum_frames = max(2, int(math.ceil(minimum_valid_ratio * frame_count)))
    candidates = [
        track
        for track in observation.objects.tracks
        if len(track.detections) >= minimum_frames
        and track.formal_exposure_weight > 0.0
    ]
    if len(candidates) < len(entity_values):
        raise SceneAnalysisError(
            "insufficient_reference_entity_tracks",
            f"reference exposes {len(candidates)} reliable tracks for "
            f"{len(entity_values)} manifest entities",
        )
    # Reliability selection is independent of expected N; only the immutable
    # reference channel is reduced to manifest cardinality. Prediction
    # candidates are never discarded.
    candidates.sort(
        key=lambda track: (
            -len(track.detections),
            -track.formal_exposure_weight,
            track.track_id,
        )
    )
    selected = candidates[: len(entity_values)]
    selected.sort(
        key=lambda track: (
            float(_track_descriptor(track)["normalized_orbit_radius"]),
            float(_track_descriptor(track)["initial_phase_rad"]),
            track.track_id,
        )
    )
    ordered_entities = sorted(
        entity_values,
        key=lambda entity: (
            _entity_radius(entity),
            int(entity.condition_anchor.get("initial_order_index", 0)),
            entity.entity_id,
        ),
    )
    tracks: list[ObjectTrack] = []
    instance_masks: list[tuple[np.ndarray, ...]] = []
    source_ids: list[str] = []
    anchors: list[Mapping[str, object]] = []
    shape = observation.union_masks[0].shape
    for entity, source in zip(ordered_entities, selected):
        xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
        area = np.full(frame_count, np.nan, dtype=np.float64)
        valid = np.zeros(frame_count, dtype=bool)
        masks = [np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)]
        for detection in source.detections:
            index = detection.frame_index
            xy[index] = detection.xy
            area[index] = detection.area_px2
            valid[index] = True
            if detection.mask is not None:
                masks[index] = np.asarray(detection.mask, dtype=np.uint8)
        filled_xy = interpolate_trace(xy, valid)
        indices = np.arange(frame_count)
        filled_area = area.copy()
        filled_area[~valid] = np.interp(
            indices[~valid], indices[valid], area[valid]
        )
        all_expected = np.ones(frame_count, dtype=bool)
        tracks.append(
            ObjectTrack(
                track_id=f"reference:{entity.entity_id}",
                matched_entity_id=entity.entity_id,
                xy=filled_xy,
                observed=all_expected,
                visibility=tuple(
                    VisibilityState.VISIBLE for _ in range(frame_count)
                ),
                areas_px2=filled_area,
                confidence=np.ones(frame_count, dtype=np.float64),
                existence_observed=all_expected,
                localization_eligible=all_expected,
                association_eligible=all_expected,
                time_weights_s=time_grid.cell_weights_s,
                metadata={
                    "entity_class": "orbiter",
                    "source_track_id": source.track_id,
                    "lifecycle": entity.lifecycle.value,
                    "short_reference_gaps_interpolated": int(
                        np.count_nonzero(~valid)
                    ),
                },
            )
        )
        descriptor = _track_descriptor(source)
        anchors.append(
            {
                "entity_id": entity.entity_id,
                "appearance_label": entity.condition_anchor.get(
                    "appearance_label"
                ),
                **descriptor,
            }
        )
        instance_masks.append(tuple(masks))
        source_ids.append(source.track_id)
    return FrozenCircularReference(
        tracks=tuple(tracks),
        entity_ids=tuple(entity.entity_id for entity in ordered_entities),
        source_track_ids=tuple(source_ids),
        instance_masks=tuple(instance_masks),
        anchors=tuple(anchors),
    )


def polar_position_similarity(
    reference_xy: Sequence[float],
    prediction_xy: Sequence[float],
    *,
    center_fallback_radius_ratio: float = 0.08,
    radial_scale_ratio: float = 0.15,
    angular_scale_rad: float = 0.35,
    cartesian_scale_ratio: float = 0.15,
) -> dict[str, float | str]:
    """Continuous scene distance; near the center it avoids undefined phase."""

    reference = np.asarray(reference_xy, dtype=np.float64)
    prediction = np.asarray(prediction_xy, dtype=np.float64)
    if (
        reference.shape != (2,)
        or prediction.shape != (2,)
        or not np.isfinite(reference).all()
        or not np.isfinite(prediction).all()
    ):
        return {
            "score": 0.0,
            "normalized_distance": float("inf"),
            "mode": "invalid",
        }
    reference = reference / _CANONICAL_DISK_RADIUS
    prediction = prediction / _CANONICAL_DISK_RADIUS
    reference_radius = float(np.linalg.norm(reference))
    prediction_radius = float(np.linalg.norm(prediction))
    if min(reference_radius, prediction_radius) <= center_fallback_radius_ratio:
        distance = float(np.linalg.norm(reference - prediction)) / max(
            cartesian_scale_ratio, 1e-9
        )
        mode = "cartesian_center_fallback"
    else:
        radial = abs(reference_radius - prediction_radius) / max(
            radial_scale_ratio, 1e-9
        )
        reference_angle = math.atan2(reference[1], reference[0])
        prediction_angle = math.atan2(prediction[1], prediction[0])
        angular_delta = abs(
            math.atan2(
                math.sin(prediction_angle - reference_angle),
                math.cos(prediction_angle - reference_angle),
            )
        )
        angular = angular_delta / max(angular_scale_rad, 1e-9)
        distance = float(math.hypot(radial, angular))
        mode = "polar"
    return {
        "score": float(1.0 / (1.0 + distance * distance)),
        "normalized_distance": distance,
        "mode": mode,
    }


def matched_prediction_inputs(
    *,
    reference: FrozenCircularReference,
    prediction: OpenWorldObservation,
    matches: Sequence[Any],
    frame_count: int,
) -> tuple[InstanceTracks, list[np.ndarray]]:
    """Build fixed-ID prediction channels without erasing missing evidence."""

    track_lookup = {
        track.track_id: track.to_object_track(
            frame_count=frame_count,
            time_weights_s=np.ones(frame_count, dtype=np.float64),
        )
        for track in prediction.tracks
    }
    detection_masks = {
        (track.track_id, detection.frame_index): detection.mask
        for track in prediction.tracks
        for detection in track.detections
        if detection.mask is not None
    }
    entity_lookup = {
        entity_id: index
        for index, entity_id in enumerate(reference.entity_ids)
    }
    count = len(reference.entity_ids)
    xy = np.full((frame_count, count, 2), np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, count), dtype=bool)
    shape = reference.instance_masks[0][0].shape
    masks = [
        [np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)]
        for _ in range(count)
    ]
    matched_union = [
        np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)
    ]
    for match in matches:
        entity_index = entity_lookup[match.entity_id]
        track = track_lookup[match.track_id]
        frame_index = int(match.frame_index)
        xy[frame_index, entity_index] = track.xy[frame_index]
        valid[frame_index, entity_index] = True
        mask = detection_masks.get((match.track_id, frame_index))
        if mask is not None:
            masks[entity_index][frame_index] = np.asarray(
                mask, dtype=np.uint8
            )
            matched_union[frame_index] = cv2.bitwise_or(
                matched_union[frame_index], masks[entity_index][frame_index]
            )
    filled = xy.copy()
    for object_index in range(count):
        if int(np.count_nonzero(valid[:, object_index])) >= 2:
            filled[:, object_index] = interpolate_trace(
                xy[:, object_index], valid[:, object_index]
            )
    union_masks = []
    for frame_index in range(frame_count):
        union = np.zeros(shape, dtype=np.uint8)
        for object_index in range(count):
            union = cv2.bitwise_or(
                union, masks[object_index][frame_index]
            )
        union_masks.append(union)
    return (
        InstanceTracks(
            xy=filled,
            valid=valid,
            valid_ratio=np.mean(valid, axis=0),
            instance_masks=masks,
            union_masks=union_masks,
        ),
        matched_union,
    )


def reference_instance_tracks(
    reference: FrozenCircularReference,
) -> InstanceTracks:
    frame_count = len(reference.tracks[0].xy)
    xy = np.stack([track.xy for track in reference.tracks], axis=1)
    valid = np.stack(
        [track.observed for track in reference.tracks], axis=1
    )
    masks = [list(value) for value in reference.instance_masks]
    shape = masks[0][0].shape
    union_masks: list[np.ndarray] = []
    for frame_index in range(frame_count):
        union = np.zeros(shape, dtype=np.uint8)
        for instance in masks:
            union = cv2.bitwise_or(union, instance[frame_index])
        union_masks.append(union)
    return InstanceTracks(
        xy=xy,
        valid=valid,
        valid_ratio=np.mean(valid, axis=0),
        instance_masks=masks,
        union_masks=union_masks,
    )


def score_open_world_orbits(
    reference_tracks: InstanceTracks,
    prediction_tracks: InstanceTracks,
    *,
    matches: Sequence[Any],
    reference_object_tracks: Sequence[ObjectTrack],
    prediction_observation: OpenWorldObservation,
    times_s: Sequence[float],
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain legacy orbit physics but charge every missing time cell."""

    expected_exposure = len(reference_object_tracks) * max(
        time_grid.duration_s, 1e-12
    )
    matched_exposure = float(sum(float(match.weight) for match in matches))
    evidence_recall = float(
        np.clip(matched_exposure / expected_exposure, 0.0, 1.0)
    )
    polar_weighted_sum = 0.0
    entity_lookup = {
        track.matched_entity_id: track
        for track in reference_object_tracks
    }
    prediction_lookup = {
        track.track_id: track.to_object_track(
            frame_count=len(times_s),
            time_weights_s=time_grid.cell_weights_s,
        )
        for track in prediction_observation.tracks
    }
    polar_rows = []
    for match in matches:
        reference = entity_lookup[match.entity_id]
        prediction = prediction_lookup[match.track_id]
        value = polar_position_similarity(
            reference.xy[match.frame_index],
            prediction.xy[match.frame_index],
            center_fallback_radius_ratio=float(
                config.get("center_fallback_radius_ratio", 0.08)
            ),
            radial_scale_ratio=float(
                config.get("polar_radial_scale_ratio", 0.15)
            ),
            angular_scale_rad=float(
                config.get("polar_angular_scale_rad", 0.35)
            ),
            cartesian_scale_ratio=float(
                config.get("center_cartesian_scale_ratio", 0.15)
            ),
        )
        polar_weighted_sum += float(match.weight) * float(value["score"])
        polar_rows.append(
            {
                "frame": int(match.frame_index),
                "entity_id": match.entity_id,
                "prediction_track_id": match.track_id,
                **value,
            }
        )
    polar_score = float(
        np.clip(polar_weighted_sum / expected_exposure, 0.0, 1.0)
    )
    raw: dict[str, Any]
    if (
        prediction_tracks.xy.size == 0
        or np.any(np.sum(prediction_tracks.valid, axis=0) < 3)
        or not np.isfinite(prediction_tracks.xy).all()
    ):
        raw = {
            "score": 0.0,
            "components": {
                "angular_trajectory": 0.0,
                "angular_velocity": 0.0,
                "orbit_geometry": 0.0,
                "uniform_motion": 0.0,
            },
            "degraded": True,
            "degradation_code": "insufficient_prediction_orbit_evidence",
        }
    else:
        try:
            reference_orbits = extract_orbit_traces(
                reference_tracks, list(times_s)
            )
            prediction_orbits = extract_orbit_traces(
                prediction_tracks, list(times_s)
            )
            raw = score_orbits(
                reference_orbits,
                prediction_orbits,
                config=dict(config),
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raw = {
                "score": 0.0,
                "components": {
                    "angular_trajectory": 0.0,
                    "angular_velocity": 0.0,
                    "orbit_geometry": 0.0,
                    "uniform_motion": 0.0,
                },
                "degraded": True,
                "degradation_code": "orbit_state_fit_failed",
                "degradation_reason": f"{type(exc).__name__}: {exc}",
            }
    raw_score = float(raw.get("score", 0.0))
    score = float(
        np.clip(
            evidence_recall
            * math.sqrt(max(raw_score, 0.0) * max(polar_score, 0.0)),
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "components": {
            "legacy_orbit_state": raw_score,
            "polar_position": polar_score,
            "matched_evidence_recall": evidence_recall,
        },
        "legacy_orbit_state": raw,
        "polar_matches": polar_rows,
        "formula": (
            "matched_evidence_recall_times_sqrt_"
            "legacy_orbit_state_times_polar_position"
        ),
    }


def empty_open_world_observation(
    frame_count: int,
    *,
    code: str,
    reason: str,
) -> OpenWorldObservation:
    return OpenWorldObservation(
        tracks=(),
        overflow_counts=np.zeros(frame_count, dtype=np.float64),
        diagnostics={
            "status": "failed_as_empty_prediction",
            "code": code,
            "reason": reason,
        },
    )


__all__ = [
    "CircularOpenWorldObservation",
    "FrozenCircularReference",
    "empty_open_world_observation",
    "freeze_reference_identities",
    "freeze_prediction_identity",
    "identity_anchors_from_observation",
    "matched_prediction_inputs",
    "observe_circular_objects",
    "polar_position_similarity",
    "reference_instance_tracks",
    "score_open_world_orbits",
]
