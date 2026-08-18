"""Scene-aware diagnostics for frozen reference-observation entities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np


_VISIBLE = 0
_OCCLUDED = 1
_OUT_OF_FRAME = 2
_UNRESOLVED = 3
_AREA_LOG_JUMP_FLOOR = {
    "collision_1d": 0.45,
    "inclined_plane_slide": 0.40,
    "parabolic_motion": 0.70,
    "pendulum": 0.40,
    "push_bottle": 0.70,
    "uniform_circular_motion": 0.40,
    "vertical_spring_oscillator": 0.40,
}


@dataclass(frozen=True)
class EventRange:
    code: str
    index_range: tuple[int, int]
    severity: str = "review"
    statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticIssue:
    code: str
    severity: str
    message: str
    observation_indices: tuple[int, ...] = ()
    statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityDiagnostics:
    events: tuple[EventRange, ...]
    review_reasons: tuple[str, ...]
    statistics: Mapping[str, Any]


def _arrays(
    *,
    area: np.ndarray,
    centroid_xy: np.ndarray,
    state: np.ndarray,
    physical_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    areas = np.asarray(area)
    centroids = np.asarray(centroid_xy)
    states = np.asarray(state)
    times = np.asarray(physical_time, dtype=np.float64)
    count = len(areas)
    if (
        areas.shape != (count,)
        or centroids.shape != (count, 2)
        or states.shape != (count,)
        or times.shape != (count,)
    ):
        raise ValueError("trajectory arrays must share one observation dimension")
    if count == 0:
        raise ValueError("trajectory arrays must not be empty")
    if np.any(areas < 0):
        raise ValueError("area_pixels must be non-negative")
    if not set(np.unique(states).tolist()).issubset({0, 1, 2, 3}):
        raise ValueError("state contains an unknown lifecycle value")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("physical_time must be finite and strictly increasing")
    return areas, centroids, states, times


def _state_runs(states: np.ndarray, value: int) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(states == value)
    if not len(indices):
        return ()
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return tuple(runs)


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def fast_entity_diagnostics(
    *,
    area: np.ndarray,
    centroid_xy: np.ndarray,
    state: np.ndarray,
    physical_time: np.ndarray,
    scene_id: str,
) -> EntityDiagnostics:
    areas, centroids, states, times = _arrays(
        area=area,
        centroid_xy=centroid_xy,
        state=state,
        physical_time=physical_time,
    )
    events: list[EventRange] = []
    reasons: list[str] = []
    for lifecycle_value, code in (
        (_OCCLUDED, "occluded_run"),
        (_OUT_OF_FRAME, "out_of_frame_run"),
        (_UNRESOLVED, "unresolved_run"),
    ):
        for start, end in _state_runs(states, lifecycle_value):
            events.append(
                EventRange(
                    code=code,
                    index_range=(start, end),
                    statistics={"sample_count": end - start + 1},
                )
            )
            reasons.append(code)
    empty_visible = np.flatnonzero((states == _VISIBLE) & (areas == 0))
    for raw_index in empty_visible:
        index = int(raw_index)
        events.append(EventRange(code="empty_visible_mask", index_range=(index, index)))
        reasons.append("empty_visible_mask")

    area_floor = _AREA_LOG_JUMP_FLOOR.get(scene_id, 0.45)
    log_changes: list[float] = []
    for index in range(len(areas) - 1):
        if (
            states[index] != _VISIBLE
            or states[index + 1] != _VISIBLE
            or areas[index] <= 0
            or areas[index + 1] <= 0
        ):
            continue
        change = abs(math.log(float(areas[index + 1]) / float(areas[index])))
        log_changes.append(change)
        if change > area_floor:
            events.append(
                EventRange(
                    code="area_jump",
                    index_range=(index, index + 1),
                    statistics={
                        "absolute_log_area_ratio": change,
                        "threshold": area_floor,
                    },
                )
            )
            reasons.append("area_instability")

    visible = (states == _VISIBLE) & (areas > 0)
    visible_centroids = centroids[visible]
    finite_visible_centroids = bool(
        not visible_centroids.size or np.all(np.isfinite(visible_centroids))
    )
    if not finite_visible_centroids:
        bad = tuple(
            int(index)
            for index in np.flatnonzero(
                visible & ~np.all(np.isfinite(centroids), axis=1)
            )
        )
        events.append(
            EventRange(
                code="nonfinite_visible_centroid",
                index_range=(bad[0], bad[-1]),
                severity="error",
                statistics={"indices": bad},
            )
        )
        reasons.append("nonfinite_visible_centroid")
    visible_areas = areas[visible]
    statistics = {
        "sample_count": int(len(areas)),
        "visible_samples": int(np.count_nonzero(states == _VISIBLE)),
        "occluded_samples": int(np.count_nonzero(states == _OCCLUDED)),
        "out_of_frame_samples": int(np.count_nonzero(states == _OUT_OF_FRAME)),
        "unresolved_samples": int(np.count_nonzero(states == _UNRESOLVED)),
        "visible_area_min": int(visible_areas.min()) if len(visible_areas) else 0,
        "visible_area_max": int(visible_areas.max()) if len(visible_areas) else 0,
        "maximum_absolute_log_area_ratio": max(log_changes, default=0.0),
        "physical_duration_seconds": float(times[-1] - times[0]),
    }
    return EntityDiagnostics(
        events=tuple(events),
        review_reasons=_ordered_unique(reasons),
        statistics=statistics,
    )


def _binary_masks(value: np.ndarray) -> np.ndarray:
    masks = np.asarray(value)
    if masks.dtype != np.uint8 or masks.ndim != 3:
        raise ValueError("masks must use uint8 THW layout")
    if not set(np.unique(masks).tolist()).issubset({0, 1}):
        raise ValueError("masks must be binary")
    return masks


def pixel_entity_diagnostics(
    *,
    masks: np.ndarray,
    state: np.ndarray,
    scene_id: str,
) -> EntityDiagnostics:
    del scene_id
    values = _binary_masks(masks)
    states = np.asarray(state)
    if states.shape != (values.shape[0],):
        raise ValueError("mask and state sample counts differ")
    events: list[EventRange] = []
    reasons: list[str] = []
    border_samples = 0
    for index, (mask, lifecycle) in enumerate(zip(values, states, strict=True)):
        area = int(np.count_nonzero(mask))
        if lifecycle == _VISIBLE and area == 0:
            events.append(EventRange("empty_visible_mask", (index, index), "error"))
            reasons.append("empty_visible_mask")
            continue
        if lifecycle != _VISIBLE and area:
            events.append(
                EventRange(
                    "nonvisible_state_has_mask",
                    (index, index),
                    "review",
                    {"area_pixels": area, "state": int(lifecycle)},
                )
            )
            reasons.append("nonvisible_state_has_mask")
        if not area:
            continue
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        component_areas = sorted(
            (int(value) for value in component_stats[1:, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        if component_count > 2:
            primary = component_areas[0]
            material_secondary = [
                value for value in component_areas[1:] if value >= max(4, 0.05 * primary)
            ]
            if material_secondary:
                events.append(
                    EventRange(
                        "fragmented_mask",
                        (index, index),
                        "review",
                        {"component_areas": tuple(component_areas)},
                    )
                )
                reasons.append("fragmented_mask")
        if (
            np.any(mask[0])
            or np.any(mask[-1])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        ):
            border_samples += 1
            events.append(EventRange("mask_touches_border", (index, index)))
            reasons.append("mask_touches_border")
    return EntityDiagnostics(
        events=tuple(events),
        review_reasons=_ordered_unique(reasons),
        statistics={
            "sample_count": int(values.shape[0]),
            "border_samples": border_samples,
        },
    )


def audit_anchor_tube_zero(
    anchor: np.ndarray,
    tube_zero: np.ndarray,
) -> DiagnosticIssue | None:
    anchor_mask = np.asarray(anchor)
    tube_mask = np.asarray(tube_zero)
    if anchor_mask.ndim == 3 and anchor_mask.shape[0] == 1:
        anchor_mask = anchor_mask[0]
    if anchor_mask.shape != tube_mask.shape:
        return DiagnosticIssue(
            code="anchor_tube_zero_shape_mismatch",
            severity="error",
            message=(
                f"anchor shape {anchor_mask.shape} differs from "
                f"tube observation zero {tube_mask.shape}"
            ),
            statistics={
                "anchor_shape": tuple(anchor_mask.shape),
                "tube_shape": tuple(tube_mask.shape),
            },
        )
    difference = np.logical_xor(anchor_mask > 0, tube_mask > 0)
    xor_pixels = int(np.count_nonzero(difference))
    if not xor_pixels:
        return None
    return DiagnosticIssue(
        code="anchor_tube_zero_mismatch",
        severity="error",
        message="first-frame anchor differs from mask-tube observation zero",
        observation_indices=(0,),
        statistics={"xor_pixels": xor_pixels},
    )


def _mask_reductions(
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = masks.shape[0]
    area = np.zeros(count, dtype=np.int64)
    centroid = np.full((count, 2), np.nan, dtype=np.float32)
    bbox = np.full((count, 4), np.nan, dtype=np.float32)
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        if not len(xs):
            continue
        area[index] = len(xs)
        centroid[index] = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
        bbox[index] = np.asarray(
            [xs.min(), ys.min(), xs.max(), ys.max()],
            dtype=np.float32,
        )
    return area, centroid, bbox


def audit_mask_trajectory_reductions(
    *,
    masks: np.ndarray,
    area_pixels: np.ndarray,
    centroid_xy: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> tuple[DiagnosticIssue, ...]:
    values = _binary_masks(masks)
    expected_area, expected_centroid, expected_bbox = _mask_reductions(values)
    observed_area = np.asarray(area_pixels)
    observed_centroid = np.asarray(centroid_xy)
    observed_bbox = np.asarray(bbox_xyxy)
    expected_shapes = ((values.shape[0],), (values.shape[0], 2), (values.shape[0], 4))
    if (
        observed_area.shape != expected_shapes[0]
        or observed_centroid.shape != expected_shapes[1]
        or observed_bbox.shape != expected_shapes[2]
    ):
        raise ValueError("stored trajectory arrays have invalid shapes")
    issues: list[DiagnosticIssue] = []
    area_bad = tuple(int(value) for value in np.flatnonzero(observed_area != expected_area))
    if area_bad:
        issues.append(
            DiagnosticIssue(
                code="trajectory_area_mismatch",
                severity="error",
                message="area_pixels is not the exact reduction of mask pixels",
                observation_indices=area_bad,
                statistics={"mismatch_count": len(area_bad)},
            )
        )
    for code, observed, expected in (
        ("trajectory_centroid_mismatch", observed_centroid, expected_centroid),
        ("trajectory_bbox_mismatch", observed_bbox, expected_bbox),
    ):
        equal = np.all(np.isclose(observed, expected, rtol=0.0, atol=1e-5, equal_nan=True), axis=1)
        bad = tuple(int(value) for value in np.flatnonzero(~equal))
        if bad:
            issues.append(
                DiagnosticIssue(
                    code=code,
                    severity="error",
                    message=f"{code} is not the exact mask reduction",
                    observation_indices=bad,
                    statistics={"mismatch_count": len(bad)},
                )
            )
    return tuple(issues)


__all__ = [
    "DiagnosticIssue",
    "EntityDiagnostics",
    "EventRange",
    "audit_anchor_tube_zero",
    "audit_mask_trajectory_reductions",
    "fast_entity_diagnostics",
    "pixel_entity_diagnostics",
]
