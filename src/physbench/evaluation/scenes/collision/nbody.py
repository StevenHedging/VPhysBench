"""Role-free N-body collision kinematics and scoring.

This module deliberately knows nothing about ``striker`` or ``target`` roles.
Physical bodies are addressed only by persistent entity IDs, and every
unordered pair is treated uniformly.  Observation, entity assignment, and
open-world presence penalties belong to the caller; this module consumes the
resulting trajectories on a common physical timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_EPSILON = 1e-12


def _readonly(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array).copy()
    output.setflags(write=False)
    return output


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class NBodyExtractionConfig:
    """Parameters used to turn trajectories into collision events."""

    contact_tolerance_px: float | None = None
    contact_tolerance_radius_fraction: float = 0.15
    contact_hysteresis_px: float | None = None
    contact_hysteresis_radius_fraction: float = 0.10
    minimum_contact_speed_px_s: float = 0.0
    minimum_event_confidence: float = 0.05
    velocity_window_frames: int = 2

    def __post_init__(self) -> None:
        if self.contact_tolerance_px is not None:
            _positive_finite(
                self.contact_tolerance_px, "contact_tolerance_px"
            )
        _positive_finite(
            self.contact_tolerance_radius_fraction,
            "contact_tolerance_radius_fraction",
        )
        if self.contact_hysteresis_px is not None:
            _positive_finite(
                self.contact_hysteresis_px, "contact_hysteresis_px"
            )
        _positive_finite(
            self.contact_hysteresis_radius_fraction,
            "contact_hysteresis_radius_fraction",
        )
        if (
            not math.isfinite(float(self.minimum_contact_speed_px_s))
            or self.minimum_contact_speed_px_s < 0.0
        ):
            raise ValueError(
                "minimum_contact_speed_px_s must be finite and non-negative"
            )
        if (
            not math.isfinite(float(self.minimum_event_confidence))
            or not 0.0 <= self.minimum_event_confidence <= 1.0
        ):
            raise ValueError(
                "minimum_event_confidence must be finite and in [0, 1]"
            )
        if self.velocity_window_frames < 1:
            raise ValueError("velocity_window_frames must be positive")


@dataclass(frozen=True)
class NBodyScoringConfig:
    """Scale and component weights for reference-relative N-body scoring."""

    along_scale_radius_fraction: float = 2.0
    along_scale_diagonal_fraction: float = 0.02
    normal_scale_radius_fraction: float = 1.0
    normal_scale_diagonal_fraction: float = 0.01
    position_jitter_radius_fraction: float = 0.05
    event_time_scale_s: float | None = None
    event_time_duration_fraction: float = 0.04
    event_gap_scale_px: float | None = None
    event_gap_radius_fraction: float = 0.25
    velocity_error_scale_fraction: float = 0.25
    minimum_velocity_scale_px_s: float = 1.0
    momentum_error_scale_fraction: float = 0.20
    minimum_momentum_scale_kg_px_s: float = 1e-6
    excess_penetration_scale: float = 0.05
    position_weight: float = 0.30
    contact_weight: float = 0.25
    velocity_weight: float = 0.20
    momentum_weight: float = 0.15
    nonpenetration_weight: float = 0.10

    def __post_init__(self) -> None:
        positive = (
            "along_scale_radius_fraction",
            "along_scale_diagonal_fraction",
            "normal_scale_radius_fraction",
            "normal_scale_diagonal_fraction",
            "event_time_duration_fraction",
            "event_gap_radius_fraction",
            "velocity_error_scale_fraction",
            "minimum_velocity_scale_px_s",
            "momentum_error_scale_fraction",
            "minimum_momentum_scale_kg_px_s",
            "excess_penetration_scale",
        )
        for name in positive:
            _positive_finite(getattr(self, name), name)
        if self.event_time_scale_s is not None:
            _positive_finite(self.event_time_scale_s, "event_time_scale_s")
        if self.event_gap_scale_px is not None:
            _positive_finite(self.event_gap_scale_px, "event_gap_scale_px")
        jitter = float(self.position_jitter_radius_fraction)
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError(
                "position_jitter_radius_fraction must be finite and "
                "non-negative"
            )
        weights = self.component_weights()
        if any(
            not math.isfinite(value) or value < 0.0
            for value in weights.values()
        ):
            raise ValueError("component weights must be finite and non-negative")
        if sum(weights.values()) <= 0.0:
            raise ValueError("at least one component weight must be positive")

    def component_weights(self) -> dict[str, float]:
        return {
            "track_position": float(self.position_weight),
            "contact_graph": float(self.contact_weight),
            "velocity": float(self.velocity_weight),
            "momentum": float(self.momentum_weight),
            "nonpenetration": float(self.nonpenetration_weight),
        }


@dataclass(frozen=True)
class ContactEvent:
    """One dynamic near-contact episode for an unordered entity pair."""

    entity_pair: tuple[str, str]
    frame_index: int
    start_frame: int
    end_frame: int
    time_s: float
    minimum_surface_gap_px: float
    closing_speed_px_s: float
    separation_speed_px_s: float
    confidence: float

    def __post_init__(self) -> None:
        if len(self.entity_pair) != 2:
            raise ValueError("entity_pair must contain exactly two IDs")
        first, second = self.entity_pair
        if (
            not isinstance(first, str)
            or not isinstance(second, str)
            or not first.strip()
            or not second.strip()
            or first == second
        ):
            raise ValueError("entity_pair must contain two distinct IDs")
        if self.entity_pair != tuple(sorted(self.entity_pair)):
            raise ValueError("entity_pair must use canonical sorted order")
        if (
            self.start_frame < 0
            or self.frame_index < self.start_frame
            or self.end_frame < self.frame_index
        ):
            raise ValueError("contact event frame bounds are inconsistent")
        for name in (
            "time_s",
            "minimum_surface_gap_px",
            "closing_speed_px_s",
            "separation_speed_px_s",
            "confidence",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.closing_speed_px_s < 0.0 or self.separation_speed_px_s < 0.0:
            raise ValueError("contact speeds must be non-negative")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("contact confidence must be in (0, 1]")


@dataclass(frozen=True)
class NBodyCollisionState:
    """Validated kinematics for an arbitrary number of named bodies."""

    times_s: np.ndarray
    entity_ids: tuple[str, ...]
    centers_xy: np.ndarray
    radii_px: np.ndarray
    valid: np.ndarray
    masses_kg: np.ndarray
    axis_origin_xy: np.ndarray
    axis_direction_xy: np.ndarray
    axis_normal_xy: np.ndarray
    along_px: np.ndarray
    normal_px: np.ndarray
    along_velocity_px_s: np.ndarray
    velocity_valid: np.ndarray
    surface_gap_px: np.ndarray
    contact_events: tuple[ContactEvent, ...]

    def __post_init__(self) -> None:
        frame_count = len(self.times_s)
        body_count = len(self.entity_ids)
        if frame_count < 2 or body_count < 2:
            raise ValueError("an N-body state requires >=2 frames and >=2 bodies")
        if self.centers_xy.shape != (frame_count, body_count, 2):
            raise ValueError("centers_xy has an inconsistent shape")
        for array, expected, name in (
            (self.radii_px, (frame_count, body_count), "radii_px"),
            (self.valid, (frame_count, body_count), "valid"),
            (self.along_px, (frame_count, body_count), "along_px"),
            (self.normal_px, (frame_count, body_count), "normal_px"),
            (
                self.along_velocity_px_s,
                (frame_count, body_count),
                "along_velocity_px_s",
            ),
            (
                self.velocity_valid,
                (frame_count, body_count),
                "velocity_valid",
            ),
            (
                self.surface_gap_px,
                (frame_count, body_count, body_count),
                "surface_gap_px",
            ),
        ):
            if array.shape != expected:
                raise ValueError(f"{name} has an inconsistent shape")
        if self.masses_kg.shape != (body_count,):
            raise ValueError("masses_kg has an inconsistent shape")

    def entity_index(self, entity_id: str) -> int:
        try:
            return self.entity_ids.index(entity_id)
        except ValueError as exc:
            raise KeyError(entity_id) from exc


def _normalize_inputs(
    centers_xy: np.ndarray,
    valid: np.ndarray,
    times_s: Sequence[float],
    entity_ids: Sequence[str],
    radii_px: np.ndarray,
    masses_kg: np.ndarray,
    axis_origin_xy: np.ndarray,
    axis_direction_xy: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    centers = np.asarray(centers_xy, dtype=np.float64)
    observed = np.asarray(valid, dtype=bool)
    times = np.asarray(times_s, dtype=np.float64)
    ids = tuple(entity_ids)
    masses = np.asarray(masses_kg, dtype=np.float64)
    origin = np.asarray(axis_origin_xy, dtype=np.float64)
    direction = np.asarray(axis_direction_xy, dtype=np.float64)

    if centers.ndim != 3 or centers.shape[2] != 2:
        raise ValueError("centers_xy must have shape [frames, bodies, 2]")
    frame_count, body_count, _ = centers.shape
    if frame_count < 2:
        raise ValueError("at least two frames are required")
    if body_count < 2:
        raise ValueError("at least two physical bodies are required")
    if observed.shape != (frame_count, body_count):
        raise ValueError("valid must have shape [frames, bodies]")
    if times.shape != (frame_count,):
        raise ValueError("times_s must have shape [frames]")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be finite and strictly increasing")
    if len(ids) != body_count:
        raise ValueError("entity_ids length must match the body dimension")
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("entity IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("entity IDs must be unique")
    if masses.shape != (body_count,):
        raise ValueError("masses_kg must have shape [bodies]")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_kg must be finite and positive")
    if np.any(np.isinf(centers)):
        raise ValueError("centers_xy must not contain infinity")
    if not np.all(np.isfinite(centers[observed])):
        raise ValueError("observed centers must be finite")

    radii_input = np.asarray(radii_px, dtype=np.float64)
    if radii_input.shape == (body_count,):
        radii = np.broadcast_to(
            radii_input[np.newaxis, :], (frame_count, body_count)
        ).copy()
    elif radii_input.shape == (frame_count, body_count):
        radii = radii_input.copy()
    else:
        raise ValueError(
            "radii_px must have shape [bodies] or [frames, bodies]"
        )
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_px must be finite and positive")
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("axis_origin_xy must be a finite 2-vector")
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("axis_direction_xy must be a finite 2-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= _EPSILON:
        raise ValueError("axis_direction_xy must have non-zero length")
    return (
        centers,
        observed,
        times,
        ids,
        radii,
        masses,
        origin,
        direction / norm,
    )


def _local_slopes(
    times_s: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.full(values.shape, np.nan, dtype=np.float64)
    slope_valid = np.zeros(values.shape, dtype=bool)
    for body_index in range(values.shape[1]):
        for frame_index in range(len(times_s)):
            if not valid[frame_index, body_index]:
                continue
            start = max(0, frame_index - window_frames)
            end = min(len(times_s), frame_index + window_frames + 1)
            indices = np.flatnonzero(valid[start:end, body_index]) + start
            if len(indices) < 2:
                continue
            centered_time = times_s[indices] - np.mean(times_s[indices])
            denominator = float(np.dot(centered_time, centered_time))
            if denominator <= _EPSILON:
                continue
            centered_values = values[indices, body_index] - np.mean(
                values[indices, body_index]
            )
            slopes[frame_index, body_index] = float(
                np.dot(centered_time, centered_values) / denominator
            )
            slope_valid[frame_index, body_index] = True
    return slopes, slope_valid


def _surface_gaps(
    centers_xy: np.ndarray,
    radii_px: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    frame_count, body_count, _ = centers_xy.shape
    gaps = np.full(
        (frame_count, body_count, body_count), np.nan, dtype=np.float64
    )
    for first in range(body_count):
        for second in range(first + 1, body_count):
            pair_valid = valid[:, first] & valid[:, second]
            distance = np.linalg.norm(
                centers_xy[:, first] - centers_xy[:, second], axis=1
            )
            gap = distance - radii_px[:, first] - radii_px[:, second]
            gaps[pair_valid, first, second] = gap[pair_valid]
            gaps[pair_valid, second, first] = gap[pair_valid]
    return gaps


def _line_slope(
    times_s: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
) -> float:
    finite = indices[np.isfinite(values[indices])]
    if len(finite) < 2:
        return 0.0
    centered_time = times_s[finite] - np.mean(times_s[finite])
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= _EPSILON:
        return 0.0
    centered_values = values[finite] - np.mean(values[finite])
    return float(np.dot(centered_time, centered_values) / denominator)


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    regions: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            regions.append((start, previous))
            start = index
        previous = index
    regions.append((start, previous))
    return regions


def _extract_contact_events(
    times_s: np.ndarray,
    entity_ids: tuple[str, ...],
    radii_px: np.ndarray,
    surface_gap_px: np.ndarray,
    *,
    config: NBodyExtractionConfig,
) -> tuple[ContactEvent, ...]:
    median_radius = float(np.median(radii_px))
    tolerance = (
        float(config.contact_tolerance_px)
        if config.contact_tolerance_px is not None
        else median_radius * config.contact_tolerance_radius_fraction
    )
    hysteresis = (
        float(config.contact_hysteresis_px)
        if config.contact_hysteresis_px is not None
        else median_radius * config.contact_hysteresis_radius_fraction
    )
    median_step = float(np.median(np.diff(times_s)))
    dynamic_speed_scale = tolerance / max(median_step, _EPSILON)
    events: list[ContactEvent] = []
    body_count = len(entity_ids)
    window = config.velocity_window_frames

    for first in range(body_count):
        for second in range(first + 1, body_count):
            gaps = surface_gap_px[:, first, second]
            candidate = np.isfinite(gaps) & (gaps <= tolerance)
            for start, end in _contiguous_regions(candidate):
                region = np.arange(start, end + 1)
                event_frame = int(region[np.nanargmin(gaps[region])])
                before = np.arange(max(0, event_frame - window), event_frame + 1)
                after = np.arange(
                    event_frame, min(len(times_s), event_frame + window + 1)
                )
                pre_slope = _line_slope(times_s, gaps, before)
                post_slope = _line_slope(times_s, gaps, after)
                closing_speed = max(-pre_slope, 0.0)
                separation_speed = max(post_slope, 0.0)

                left = gaps[max(0, start - 1)]
                right = gaps[min(len(gaps) - 1, end + 1)]
                entered_or_left = bool(
                    (math.isfinite(float(left)) and left > tolerance + hysteresis)
                    or (
                        math.isfinite(float(right))
                        and right > tolerance + hysteresis
                    )
                )
                dynamic_speed = closing_speed + separation_speed
                if (
                    dynamic_speed <= config.minimum_contact_speed_px_s
                    and not entered_or_left
                ):
                    continue
                proximity = math.exp(
                    -0.5
                    * (
                        max(float(gaps[event_frame]), 0.0)
                        / max(tolerance, _EPSILON)
                    )
                    ** 2
                )
                dynamics = 1.0 - math.exp(
                    -dynamic_speed / max(dynamic_speed_scale, _EPSILON)
                )
                if entered_or_left:
                    dynamics = max(dynamics, 0.5)
                confidence = float(np.clip(proximity * dynamics, 0.0, 1.0))
                if confidence < config.minimum_event_confidence:
                    continue
                pair = tuple(sorted((entity_ids[first], entity_ids[second])))
                events.append(
                    ContactEvent(
                        entity_pair=pair,
                        frame_index=event_frame,
                        start_frame=start,
                        end_frame=end,
                        time_s=float(times_s[event_frame]),
                        minimum_surface_gap_px=float(gaps[event_frame]),
                        closing_speed_px_s=float(closing_speed),
                        separation_speed_px_s=float(separation_speed),
                        confidence=confidence,
                    )
                )
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.time_s,
                event.entity_pair,
                event.frame_index,
            ),
        )
    )


def extract_nbody_collision_state(
    centers_xy: np.ndarray,
    valid: np.ndarray,
    times_s: Sequence[float],
    *,
    entity_ids: Sequence[str],
    radii_px: np.ndarray,
    masses_kg: np.ndarray,
    axis_origin_xy: np.ndarray,
    axis_direction_xy: np.ndarray,
    config: NBodyExtractionConfig | None = None,
) -> NBodyCollisionState:
    """Extract role-free kinematics and pairwise contact events.

    ``axis_origin_xy`` and ``axis_direction_xy`` should be frozen from the
    condition/reference side and reused verbatim for a prediction.
    """

    extraction = config or NBodyExtractionConfig()
    (
        centers,
        observed,
        times,
        ids,
        radii,
        masses,
        origin,
        direction,
    ) = _normalize_inputs(
        centers_xy,
        valid,
        times_s,
        entity_ids,
        radii_px,
        masses_kg,
        axis_origin_xy,
        axis_direction_xy,
    )
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    centered = centers - origin
    along = centered @ direction
    cross_track = centered @ normal
    along[~observed] = np.nan
    cross_track[~observed] = np.nan
    velocity, velocity_valid = _local_slopes(
        times,
        along,
        observed,
        window_frames=extraction.velocity_window_frames,
    )
    gaps = _surface_gaps(centers, radii, observed)
    events = _extract_contact_events(
        times,
        ids,
        radii,
        gaps,
        config=extraction,
    )
    return NBodyCollisionState(
        times_s=_readonly(times),
        entity_ids=ids,
        centers_xy=_readonly(centers),
        radii_px=_readonly(radii),
        valid=_readonly(observed),
        masses_kg=_readonly(masses),
        axis_origin_xy=_readonly(origin),
        axis_direction_xy=_readonly(direction),
        axis_normal_xy=_readonly(normal),
        along_px=_readonly(along),
        normal_px=_readonly(cross_track),
        along_velocity_px_s=_readonly(velocity),
        velocity_valid=_readonly(velocity_valid),
        surface_gap_px=_readonly(gaps),
        contact_events=events,
    )


def infer_initial_active_entities(
    state: NBodyCollisionState,
    *,
    minimum_speed_px_s: float,
    initial_window_frames: int = 3,
) -> tuple[str, ...]:
    """Infer all initially moving bodies without assigning a striker role."""

    threshold = float(minimum_speed_px_s)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum_speed_px_s must be finite and non-negative")
    if initial_window_frames < 2:
        raise ValueError("initial_window_frames must be at least two")
    end = min(len(state.times_s), initial_window_frames)
    active: list[str] = []
    for body_index, entity_id in enumerate(state.entity_ids):
        indices = np.flatnonzero(state.valid[:end, body_index])
        if len(indices) < 2:
            continue
        slope = _line_slope(
            state.times_s,
            state.along_px[:, body_index],
            indices,
        )
        if abs(slope) > threshold:
            active.append(entity_id)
    return tuple(active)


def _validate_comparison(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
) -> None:
    if reference.times_s.shape != prediction.times_s.shape or not np.allclose(
        reference.times_s,
        prediction.times_s,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("reference and prediction must share a common timeline")
    if not np.allclose(
        reference.axis_origin_xy,
        prediction.axis_origin_xy,
        rtol=0.0,
        atol=1e-9,
    ) or not np.allclose(
        reference.axis_direction_xy,
        prediction.axis_direction_xy,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "reference and prediction must share the frozen reference axis"
        )
    prediction_lookup = {
        entity_id: index
        for index, entity_id in enumerate(prediction.entity_ids)
    }
    for reference_index, entity_id in enumerate(reference.entity_ids):
        prediction_index = prediction_lookup.get(entity_id)
        if prediction_index is None:
            continue
        if not math.isclose(
            float(reference.masses_kg[reference_index]),
            float(prediction.masses_kg[prediction_index]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"mass mismatch for matched entity {entity_id!r}"
            )


def _prediction_indices(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
) -> list[int | None]:
    lookup = {
        entity_id: index
        for index, entity_id in enumerate(prediction.entity_ids)
    }
    return [lookup.get(entity_id) for entity_id in reference.entity_ids]


def score_track_coordinates(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
    *,
    frame_diagonal_px: float,
    config: NBodyScoringConfig | None = None,
) -> dict[str, Any]:
    """Score along-track and cross-track displacement with a Cauchy kernel."""

    scoring = config or NBodyScoringConfig()
    diagonal = _positive_finite(frame_diagonal_px, "frame_diagonal_px")
    _validate_comparison(reference, prediction)
    prediction_indices = _prediction_indices(reference, prediction)
    all_qualities: list[float] = []
    reference_exposure = 0
    matched_exposure = 0
    per_entity: dict[str, dict[str, Any]] = {}

    for reference_index, entity_id in enumerate(reference.entity_ids):
        prediction_index = prediction_indices[reference_index]
        expected = reference.valid[:, reference_index]
        reference_count = int(np.count_nonzero(expected))
        reference_exposure += reference_count
        frame_quality: list[float | None] = [None] * len(reference.times_s)
        frame_distance: list[float | None] = [None] * len(reference.times_s)
        frame_along_delta: list[float | None] = [None] * len(
            reference.times_s
        )
        frame_normal_delta: list[float | None] = [None] * len(
            reference.times_s
        )
        entity_qualities: list[float] = []
        if prediction_index is not None:
            common = expected & prediction.valid[:, prediction_index]
            for frame_index in np.flatnonzero(common):
                radius = float(reference.radii_px[frame_index, reference_index])
                along_scale = max(
                    scoring.along_scale_radius_fraction * radius,
                    scoring.along_scale_diagonal_fraction * diagonal,
                )
                normal_scale = max(
                    scoring.normal_scale_radius_fraction * radius,
                    scoring.normal_scale_diagonal_fraction * diagonal,
                )
                jitter = scoring.position_jitter_radius_fraction * radius
                along_delta = max(
                    abs(
                        float(
                            prediction.along_px[frame_index, prediction_index]
                            - reference.along_px[frame_index, reference_index]
                        )
                    )
                    - jitter,
                    0.0,
                )
                normal_delta = max(
                    abs(
                        float(
                            prediction.normal_px[frame_index, prediction_index]
                            - reference.normal_px[frame_index, reference_index]
                        )
                    )
                    - jitter,
                    0.0,
                )
                normalized_distance = math.hypot(
                    along_delta / along_scale,
                    normal_delta / normal_scale,
                )
                quality = 1.0 / (1.0 + normalized_distance**2)
                frame_quality[int(frame_index)] = quality
                frame_distance[int(frame_index)] = normalized_distance
                frame_along_delta[int(frame_index)] = along_delta
                frame_normal_delta[int(frame_index)] = normal_delta
                entity_qualities.append(quality)
                all_qualities.append(quality)
            common_count = int(np.count_nonzero(common))
        else:
            common_count = 0
        matched_exposure += common_count
        per_entity[entity_id] = {
            "score": (
                float(np.mean(entity_qualities))
                if entity_qualities
                else 0.0
            ),
            "reference_exposure_frames": reference_count,
            "matched_exposure_frames": common_count,
            "coverage": common_count / max(reference_count, 1),
            "frame_quality": frame_quality,
            "frame_normalized_distance": frame_distance,
            "frame_along_delta_px": frame_along_delta,
            "frame_normal_delta_px": frame_normal_delta,
        }
    return {
        "score": (
            float(np.mean(all_qualities)) if all_qualities else None
        ),
        "reference_exposure_frames": reference_exposure,
        "matched_exposure_frames": matched_exposure,
        "coverage": matched_exposure / max(reference_exposure, 1),
        "per_entity": per_entity,
    }


def _event_match_quality(
    reference: ContactEvent,
    prediction: ContactEvent,
    *,
    time_scale_s: float,
    gap_scale_px: float,
) -> float:
    time_error = (prediction.time_s - reference.time_s) / time_scale_s
    gap_error = (
        prediction.minimum_surface_gap_px
        - reference.minimum_surface_gap_px
    ) / gap_scale_px
    return float(math.exp(-0.5 * (time_error**2 + gap_error**2)))


def _match_pair_events(
    reference: list[ContactEvent],
    prediction: list[ContactEvent],
    *,
    time_scale_s: float,
    gap_scale_px: float,
) -> tuple[float, list[dict[str, Any]]]:
    """Maximum-weight monotone one-to-one matching for one entity pair."""

    rows, columns = len(reference), len(prediction)
    values = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    decisions = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    raw_quality = np.zeros((rows, columns), dtype=np.float64)
    weighted_quality = np.zeros((rows, columns), dtype=np.float64)
    for row, reference_event in enumerate(reference):
        for column, prediction_event in enumerate(prediction):
            quality = _event_match_quality(
                reference_event,
                prediction_event,
                time_scale_s=time_scale_s,
                gap_scale_px=gap_scale_px,
            )
            raw_quality[row, column] = quality
            weighted_quality[row, column] = quality * min(
                reference_event.confidence,
                prediction_event.confidence,
            )
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = (
                values[row - 1, column],
                values[row, column - 1],
                values[row - 1, column - 1]
                + weighted_quality[row - 1, column - 1],
            )
            decision = int(np.argmax(candidates))
            values[row, column] = candidates[decision]
            decisions[row, column] = decision

    matches: list[dict[str, Any]] = []
    row, column = rows, columns
    while row > 0 and column > 0:
        decision = int(decisions[row, column])
        if decision == 2:
            reference_event = reference[row - 1]
            prediction_event = prediction[column - 1]
            matches.append(
                {
                    "entity_pair": list(reference_event.entity_pair),
                    "reference_time_s": reference_event.time_s,
                    "prediction_time_s": prediction_event.time_s,
                    "quality": float(raw_quality[row - 1, column - 1]),
                    "weighted_quality": float(
                        weighted_quality[row - 1, column - 1]
                    ),
                }
            )
            row -= 1
            column -= 1
        elif decision == 0:
            row -= 1
        else:
            column -= 1
    matches.reverse()
    return float(values[rows, columns]), matches


def score_contact_events(
    reference_events: Sequence[ContactEvent],
    prediction_events: Sequence[ContactEvent],
    *,
    time_scale_s: float,
    gap_scale_px: float,
) -> dict[str, Any]:
    """Soft-Jaccard event score with exact-pair, one-to-one matching."""

    time_scale = _positive_finite(time_scale_s, "time_scale_s")
    gap_scale = _positive_finite(gap_scale_px, "gap_scale_px")
    reference_by_pair: dict[tuple[str, str], list[ContactEvent]] = {}
    prediction_by_pair: dict[tuple[str, str], list[ContactEvent]] = {}
    for event in reference_events:
        reference_by_pair.setdefault(event.entity_pair, []).append(event)
    for event in prediction_events:
        prediction_by_pair.setdefault(event.entity_pair, []).append(event)
    pairs = sorted(set(reference_by_pair) | set(prediction_by_pair))
    matched_quality = 0.0
    matches: list[dict[str, Any]] = []
    for pair in pairs:
        pair_reference = sorted(
            reference_by_pair.get(pair, []), key=lambda event: event.time_s
        )
        pair_prediction = sorted(
            prediction_by_pair.get(pair, []), key=lambda event: event.time_s
        )
        pair_quality, pair_matches = _match_pair_events(
            pair_reference,
            pair_prediction,
            time_scale_s=time_scale,
            gap_scale_px=gap_scale,
        )
        matched_quality += pair_quality
        matches.extend(pair_matches)
    reference_exposure = float(
        sum(event.confidence for event in reference_events)
    )
    prediction_exposure = float(
        sum(event.confidence for event in prediction_events)
    )
    denominator = reference_exposure + prediction_exposure - matched_quality
    score = 1.0 if denominator <= _EPSILON else matched_quality / denominator
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "matched_quality": matched_quality,
        "reference_event_exposure": reference_exposure,
        "prediction_event_exposure": prediction_exposure,
        "reference_event_count": len(reference_events),
        "prediction_event_count": len(prediction_events),
        "matched_event_count": len(matches),
        "extra_event_count": len(prediction_events) - len(matches),
        "missing_event_count": len(reference_events) - len(matches),
        "matches": matches,
    }


def score_velocity(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
    *,
    config: NBodyScoringConfig | None = None,
) -> dict[str, Any]:
    """Compare per-ID along-track velocities on the frozen timeline."""

    scoring = config or NBodyScoringConfig()
    _validate_comparison(reference, prediction)
    prediction_indices = _prediction_indices(reference, prediction)
    qualities: list[float] = []
    reference_exposure = 0
    matched_exposure = 0
    per_entity: dict[str, dict[str, float | int]] = {}
    for reference_index, entity_id in enumerate(reference.entity_ids):
        expected = reference.velocity_valid[:, reference_index]
        expected_count = int(np.count_nonzero(expected))
        reference_exposure += expected_count
        prediction_index = prediction_indices[reference_index]
        entity_qualities: list[float] = []
        common_count = 0
        if prediction_index is not None:
            common = expected & prediction.velocity_valid[:, prediction_index]
            common_count = int(np.count_nonzero(common))
            reference_velocity = reference.along_velocity_px_s[
                common, reference_index
            ]
            prediction_velocity = prediction.along_velocity_px_s[
                common, prediction_index
            ]
            all_reference_velocity = reference.along_velocity_px_s[
                expected, reference_index
            ]
            speed = (
                float(np.percentile(np.abs(all_reference_velocity), 90.0))
                if len(all_reference_velocity)
                else 0.0
            )
            scale = max(
                scoring.minimum_velocity_scale_px_s,
                scoring.velocity_error_scale_fraction * speed,
            )
            normalized_error = (
                prediction_velocity - reference_velocity
            ) / scale
            entity_qualities = (
                1.0 / (1.0 + np.square(normalized_error))
            ).tolist()
            qualities.extend(entity_qualities)
        matched_exposure += common_count
        per_entity[entity_id] = {
            "score": (
                float(np.mean(entity_qualities))
                if entity_qualities
                else 0.0
            ),
            "reference_exposure_frames": expected_count,
            "matched_exposure_frames": common_count,
            "coverage": common_count / max(expected_count, 1),
        }
    return {
        "score": float(np.mean(qualities)) if qualities else None,
        "reference_exposure_frames": reference_exposure,
        "matched_exposure_frames": matched_exposure,
        "coverage": matched_exposure / max(reference_exposure, 1),
        "per_entity": per_entity,
    }


def score_momentum(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
    *,
    config: NBodyScoringConfig | None = None,
) -> dict[str, Any]:
    """Compare total-system momentum change, including simultaneous contacts."""

    scoring = config or NBodyScoringConfig()
    _validate_comparison(reference, prediction)
    prediction_indices = _prediction_indices(reference, prediction)
    frame_count = len(reference.times_s)
    reference_complete = np.all(reference.velocity_valid, axis=1)
    prediction_complete = np.ones(frame_count, dtype=bool)
    if any(index is None for index in prediction_indices):
        prediction_complete[:] = False
    else:
        for reference_index, prediction_index in enumerate(prediction_indices):
            assert prediction_index is not None
            prediction_complete &= prediction.velocity_valid[
                :, prediction_index
            ]
    common = reference_complete & prediction_complete
    common_indices = np.flatnonzero(common)
    reference_count = int(np.count_nonzero(reference_complete))
    if len(common_indices) < 2:
        return {
            "score": None,
            "reference_exposure_frames": reference_count,
            "matched_exposure_frames": int(len(common_indices)),
            "coverage": len(common_indices) / max(reference_count, 1),
            "normalized_change_rmse": None,
        }

    reference_velocity = reference.along_velocity_px_s[common]
    prediction_velocity = np.column_stack(
        [
            prediction.along_velocity_px_s[common, prediction_index]
            for prediction_index in prediction_indices
            if prediction_index is not None
        ]
    )
    masses = reference.masses_kg
    reference_momentum = reference_velocity @ masses
    prediction_momentum = prediction_velocity @ masses
    reference_change = reference_momentum - reference_momentum[0]
    prediction_change = prediction_momentum - prediction_momentum[0]
    absolute_momentum = np.sum(
        np.abs(reference_velocity * masses[np.newaxis, :]), axis=1
    )
    scale = max(
        scoring.minimum_momentum_scale_kg_px_s,
        scoring.momentum_error_scale_fraction
        * float(np.percentile(absolute_momentum, 90.0)),
    )
    normalized_error = (prediction_change - reference_change) / scale
    qualities = 1.0 / (1.0 + np.square(normalized_error))
    return {
        "score": float(np.mean(qualities)),
        "reference_exposure_frames": reference_count,
        "matched_exposure_frames": int(len(common_indices)),
        "coverage": len(common_indices) / max(reference_count, 1),
        "normalized_change_rmse": float(
            np.sqrt(np.mean(np.square(normalized_error)))
        ),
        "reference_momentum_change_kg_px_s": reference_change.tolist(),
        "prediction_momentum_change_kg_px_s": prediction_change.tolist(),
    }


def score_nonpenetration(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
    *,
    config: NBodyScoringConfig | None = None,
) -> dict[str, Any]:
    """Penalize overlap beyond the overlap already present in the reference."""

    scoring = config or NBodyScoringConfig()
    _validate_comparison(reference, prediction)
    prediction_lookup = {
        entity_id: index
        for index, entity_id in enumerate(prediction.entity_ids)
    }
    qualities: list[float] = []
    excess_values: list[float] = []
    pair_exposure = 0
    matched_exposure = 0
    per_pair: dict[str, dict[str, float | int]] = {}
    for first in range(len(reference.entity_ids)):
        for second in range(first + 1, len(reference.entity_ids)):
            first_id = reference.entity_ids[first]
            second_id = reference.entity_ids[second]
            pair_name = f"{first_id}|{second_id}"
            reference_valid = reference.valid[:, first] & reference.valid[:, second]
            reference_count = int(np.count_nonzero(reference_valid))
            pair_exposure += reference_count
            prediction_first = prediction_lookup.get(first_id)
            prediction_second = prediction_lookup.get(second_id)
            pair_quality: list[float] = []
            pair_excess: list[float] = []
            if prediction_first is not None and prediction_second is not None:
                common = (
                    reference_valid
                    & prediction.valid[:, prediction_first]
                    & prediction.valid[:, prediction_second]
                )
                indices = np.flatnonzero(common)
                for frame_index in indices:
                    reference_radius_sum = float(
                        reference.radii_px[frame_index, first]
                        + reference.radii_px[frame_index, second]
                    )
                    prediction_radius_sum = float(
                        prediction.radii_px[frame_index, prediction_first]
                        + prediction.radii_px[frame_index, prediction_second]
                    )
                    reference_penetration = max(
                        -float(
                            reference.surface_gap_px[
                                frame_index, first, second
                            ]
                        )
                        / reference_radius_sum,
                        0.0,
                    )
                    prediction_penetration = max(
                        -float(
                            prediction.surface_gap_px[
                                frame_index,
                                prediction_first,
                                prediction_second,
                            ]
                        )
                        / prediction_radius_sum,
                        0.0,
                    )
                    excess = max(
                        prediction_penetration - reference_penetration, 0.0
                    )
                    quality = 1.0 / (
                        1.0
                        + (
                            excess / scoring.excess_penetration_scale
                        )
                        ** 2
                    )
                    pair_excess.append(excess)
                    pair_quality.append(quality)
                    excess_values.append(excess)
                    qualities.append(quality)
                common_count = int(len(indices))
            else:
                common_count = 0
            matched_exposure += common_count
            per_pair[pair_name] = {
                "score": (
                    float(np.mean(pair_quality)) if pair_quality else 0.0
                ),
                "mean_excess_penetration": (
                    float(np.mean(pair_excess)) if pair_excess else 0.0
                ),
                "reference_exposure_frames": reference_count,
                "matched_exposure_frames": common_count,
            }
    return {
        "score": float(np.mean(qualities)) if qualities else None,
        "mean_excess_penetration": (
            float(np.mean(excess_values)) if excess_values else 0.0
        ),
        "reference_exposure_frames": pair_exposure,
        "matched_exposure_frames": matched_exposure,
        "coverage": matched_exposure / max(pair_exposure, 1),
        "per_pair": per_pair,
    }


def score_nbody_collision(
    reference: NBodyCollisionState,
    prediction: NBodyCollisionState,
    *,
    frame_diagonal_px: float,
    config: NBodyScoringConfig | None = None,
) -> dict[str, Any]:
    """Return a finite, role-free N-body collision physics score."""

    scoring = config or NBodyScoringConfig()
    _validate_comparison(reference, prediction)
    duration = float(reference.times_s[-1] - reference.times_s[0])
    median_radius = float(np.median(reference.radii_px))
    event_time_scale = (
        float(scoring.event_time_scale_s)
        if scoring.event_time_scale_s is not None
        else max(
            scoring.event_time_duration_fraction * duration,
            float(np.median(np.diff(reference.times_s))),
        )
    )
    event_gap_scale = (
        float(scoring.event_gap_scale_px)
        if scoring.event_gap_scale_px is not None
        else scoring.event_gap_radius_fraction * median_radius
    )
    position = score_track_coordinates(
        reference,
        prediction,
        frame_diagonal_px=frame_diagonal_px,
        config=scoring,
    )
    prediction_lookup = {
        entity_id: index
        for index, entity_id in enumerate(prediction.entity_ids)
    }
    reference_lookup = {
        entity_id: index
        for index, entity_id in enumerate(reference.entity_ids)
    }

    def pair_comparable(
        pair: tuple[str, str],
        frame_index: int,
    ) -> bool:
        if not 0 <= frame_index < len(reference.times_s):
            return False
        reference_indices = [
            reference_lookup.get(entity_id) for entity_id in pair
        ]
        prediction_indices = [
            prediction_lookup.get(entity_id) for entity_id in pair
        ]
        if any(index is None for index in reference_indices):
            return True
        if any(index is None for index in prediction_indices):
            return False
        return all(
            reference.valid[frame_index, reference_index]
            and prediction.valid[frame_index, prediction_index]
            for reference_index, prediction_index in zip(
                reference_indices,
                prediction_indices,
            )
            if reference_index is not None and prediction_index is not None
        )

    comparable_reference_events = [
        event
        for event in reference.contact_events
        if pair_comparable(event.entity_pair, event.frame_index)
    ]
    comparable_prediction_events = [
        event
        for event in prediction.contact_events
        if (
            any(
                entity_id not in reference_lookup
                for entity_id in event.entity_pair
            )
            or pair_comparable(event.entity_pair, event.frame_index)
        )
    ]
    contact = score_contact_events(
        comparable_reference_events,
        comparable_prediction_events,
        time_scale_s=event_time_scale,
        gap_scale_px=event_gap_scale,
    )
    comparable_pair_exposure = 0
    for first in range(len(reference.entity_ids)):
        for second in range(first + 1, len(reference.entity_ids)):
            first_prediction = prediction_lookup.get(
                reference.entity_ids[first]
            )
            second_prediction = prediction_lookup.get(
                reference.entity_ids[second]
            )
            if first_prediction is None or second_prediction is None:
                continue
            comparable_pair_exposure += int(
                np.count_nonzero(
                    reference.valid[:, first]
                    & reference.valid[:, second]
                    & prediction.valid[:, first_prediction]
                    & prediction.valid[:, second_prediction]
                )
            )
    contact["omitted_reference_event_count"] = (
        len(reference.contact_events) - len(comparable_reference_events)
    )
    contact["comparable_pair_exposure_frames"] = comparable_pair_exposure
    if (
        comparable_pair_exposure == 0
        and not comparable_reference_events
        and not comparable_prediction_events
    ):
        contact["score"] = None
    velocity = score_velocity(reference, prediction, config=scoring)
    momentum = score_momentum(reference, prediction, config=scoring)
    nonpenetration = score_nonpenetration(
        reference, prediction, config=scoring
    )
    details = {
        "track_position": position,
        "contact_graph": contact,
        "velocity": velocity,
        "momentum": momentum,
        "nonpenetration": nonpenetration,
    }
    components: dict[str, float | None] = {
        name: (
            None
            if result["score"] is None
            else float(result["score"])
        )
        for name, result in details.items()
    }
    weights = scoring.component_weights()
    usable = {
        name: value
        for name, value in components.items()
        if value is not None and weights[name] > 0.0
    }
    denominator = sum(weights[name] for name in usable)
    if any(
        value <= 0.0 for value in usable.values()
    ):
        total = 0.0
    elif not usable:
        total = 0.0
    else:
        total = math.exp(
            sum(
                weights[name] * math.log(value)
                for name, value in usable.items()
            )
            / denominator
        )
    return {
        "score": float(np.clip(total, 0.0, 1.0)),
        "components": components,
        "details": details,
        "weights_used": {
            name: weights[name] / denominator for name in usable
        },
        "omitted_components": sorted(set(components) - set(usable)),
        "reference_body_count": len(reference.entity_ids),
        "prediction_body_count": len(prediction.entity_ids),
        "reference_active_entity_ids": list(
            infer_initial_active_entities(
                reference,
                minimum_speed_px_s=scoring.minimum_velocity_scale_px_s,
            )
        ),
        "prediction_active_entity_ids": list(
            infer_initial_active_entities(
                prediction,
                minimum_speed_px_s=scoring.minimum_velocity_scale_px_s,
            )
        ),
    }


__all__ = [
    "ContactEvent",
    "NBodyCollisionState",
    "NBodyExtractionConfig",
    "NBodyScoringConfig",
    "extract_nbody_collision_state",
    "infer_initial_active_entities",
    "score_contact_events",
    "score_momentum",
    "score_nbody_collision",
    "score_nonpenetration",
    "score_track_coordinates",
    "score_velocity",
]
