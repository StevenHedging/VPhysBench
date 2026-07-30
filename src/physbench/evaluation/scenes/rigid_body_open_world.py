from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ...io import canonical_sha256, sha256_file
from ..common.entities import (
    EvidenceTier,
    EntitySpec,
    ExpectedEntityTimeline,
    ExpectedPositionSample,
    FrozenAssignment,
    ObjectDetection,
    ObjectTrack,
    ObjectCentricComparisonV2,
    OpenWorldObservation,
    ReferenceCapability,
    VisibilityState,
    build_common_time_grid,
    compose_object_centric_v2,
    materialize_entity_manifest,
    resolve_expected_entity_timeline,
    safe_compare_open_world_v2,
    track_open_world_detections,
)
from ..common.artifacts import (
    save_iou_curve,
    save_series_comparison,
    write_rows_csv,
)
from ..common.artifacts.open_world_v2 import (
    write_open_world_v2_artifacts,
)
from ..common.base import ReferenceCaseEvaluator, SceneAnalysis
from ..common.entities.timeline import CommonTimeGrid
from ..common.errors import ReferenceAnalysisError, SceneAnalysisError
from ..common.geometry import AxisModel, fit_axis
from ..common.masks.motion import build_motion_prompt
from ..common.masks.quality import (
    component_centroids,
    mask_centroid,
    observed_mask_iou,
    summarize_mask_ious,
)
from ..common.masks.sam2 import MaskPrompt, Sam2VideoSegmenter
from ..common.subject import (
    SubjectComparison,
    compare_subjects,
    infer_reference_mode,
)
from ..contracts import CaseEvaluationRequest


RIGID_BODY_OPEN_WORLD_VERSION = "2.0"


@dataclass(frozen=True)
class FrozenRigidAxis:
    """A condition/reference-side coordinate system never fitted to prediction."""

    origin_xy: np.ndarray
    direction_xy: np.ndarray
    normal_xy: np.ndarray
    span_px: float
    source: str
    explained_ratio: float

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_xy, dtype=np.float64)
        direction = np.asarray(self.direction_xy, dtype=np.float64)
        normal = np.asarray(self.normal_xy, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("axis origin must contain two finite coordinates")
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("axis direction must contain two finite coordinates")
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("axis direction must be non-zero")
        direction = direction / norm
        expected_normal = np.asarray(
            [-direction[1], direction[0]], dtype=np.float64
        )
        if normal.shape != (2,) or not np.isfinite(normal).all():
            raise ValueError("axis normal must contain two finite coordinates")
        if float(np.dot(normal, expected_normal)) < 0.0:
            expected_normal = -expected_normal
        span = float(self.span_px)
        explained = float(self.explained_ratio)
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("axis span must be finite and positive")
        if not math.isfinite(explained) or not 0.0 <= explained <= 1.0:
            raise ValueError("axis explained_ratio must be in [0, 1]")
        origin = np.array(origin, copy=True)
        direction = np.array(direction, copy=True)
        expected_normal = np.array(expected_normal, copy=True)
        for value in (origin, direction, expected_normal):
            value.setflags(write=False)
        object.__setattr__(self, "origin_xy", origin)
        object.__setattr__(self, "direction_xy", direction)
        object.__setattr__(self, "normal_xy", expected_normal)
        object.__setattr__(self, "span_px", span)
        object.__setattr__(self, "explained_ratio", explained)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=np.float64)
        centered = points - self.origin_xy
        return centered @ self.direction_xy, centered @ self.normal_xy

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_xy": self.origin_xy.tolist(),
            "direction_xy": self.direction_xy.tolist(),
            "normal_xy": self.normal_xy.tolist(),
            "span_px": self.span_px,
            "source": self.source,
            "explained_ratio": self.explained_ratio,
            "prediction_refit_allowed": False,
        }


@dataclass(frozen=True)
class RigidBodyReference:
    entity_id: str
    entity_class: str
    masks: tuple[np.ndarray, ...]
    xy: np.ndarray
    areas_px2: np.ndarray
    expected: np.ndarray
    raw_observed: np.ndarray
    axis: FrozenRigidAxis
    normalized_progress: np.ndarray
    legal_exit_frame: int | None
    lifecycle_diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.masks)
        arrays = {
            "xy": np.asarray(self.xy),
            "areas_px2": np.asarray(self.areas_px2),
            "expected": np.asarray(self.expected),
            "raw_observed": np.asarray(self.raw_observed),
            "normalized_progress": np.asarray(self.normalized_progress),
        }
        if arrays["xy"].shape != (count, 2):
            raise ValueError("reference xy must have shape [frames, 2]")
        for name in (
            "areas_px2",
            "expected",
            "raw_observed",
            "normalized_progress",
        ):
            if arrays[name].shape != (count,):
                raise ValueError(f"reference {name} must have one value per frame")

    def to_object_track(self, time_grid: CommonTimeGrid) -> ObjectTrack:
        if len(time_grid.times_s) != len(self.masks):
            raise ValueError("reference timeline differs from common time grid")
        visibility = tuple(
            (
                VisibilityState.VISIBLE
                if self.expected[index]
                else VisibilityState.OUT_OF_FRAME
            )
            for index in range(len(self.masks))
        )
        confidence = np.where(self.raw_observed, 1.0, 0.75)
        return ObjectTrack(
            track_id=f"gt_{self.entity_id}",
            matched_entity_id=self.entity_id,
            xy=self.xy,
            observed=self.expected,
            visibility=visibility,
            areas_px2=self.areas_px2,
            confidence=confidence,
            existence_observed=self.expected,
            localization_eligible=self.expected,
            association_eligible=self.expected,
            time_weights_s=time_grid.cell_weights_s,
            metadata={
                "entity_class": self.entity_class,
                "lifecycle": "may_exit",
                "legal_exit_frame": self.legal_exit_frame,
            },
        )


@dataclass(frozen=True)
class RigidBodyOpenWorldResult:
    observation: OpenWorldObservation
    comparison: ObjectCentricComparisonV2
    matched_prediction_masks: tuple[np.ndarray, ...]
    prediction_union_masks: tuple[np.ndarray, ...]
    prediction_xy: np.ndarray
    prediction_observed: np.ndarray
    state_metric: Mapping[str, Any]
    composition: Mapping[str, Any]
    per_frame: tuple[Mapping[str, Any], ...]


def default_rigid_body_config(scene_kind: str) -> dict[str, Any]:
    if scene_kind not in {"free_fall", "inclined_plane"}:
        raise ValueError(f"unsupported rigid-body scene kind: {scene_kind}")
    return {
        "observer_version": RIGID_BODY_OPEN_WORLD_VERSION,
        "condition_difference_threshold": 18.0,
        "temporal_difference_threshold": 12.0,
        "minimum_area_ratio": 0.18,
        "maximum_area_ratio": 5.0,
        "direct_minimum_area_ratio": 0.16,
        "direct_maximum_area_ratio": 5.5,
        "minimum_anchor_color_similarity": 0.12,
        "minimum_duplicate_color_similarity": 0.45,
        "minimum_weak_residual_color_similarity": 0.25,
        "identity_rejection_minimum_consecutive_frames": 2,
        "outside_roi_minimum_anchor_color_similarity": 0.55,
        "outside_roi_minimum_area_ratio": 0.5,
        "outside_roi_maximum_area_ratio": 2.0,
        "roi_half_width_body_diameters": 7.0,
        "roi_minimum_half_width_fraction": (
            0.48 if scene_kind == "free_fall" else 0.28
        ),
        "residual_dilation_body_fraction": 0.18,
        "minimum_circularity": 0.42,
        "maximum_circle_aspect_ratio": 1.9,
        "minimum_rectangularity": 0.5,
        "maximum_rectangle_aspect_ratio": 4.0,
        "maximum_gap_s": 0.25,
        "maximum_assignment_cost": 4.0,
        "maximum_tracks": 32,
        "minimum_match_position_similarity": 0.1,
        "exit_minimum_trailing_frames": 2,
        "exit_minimum_progress": 0.88,
        "exit_minimum_forward_step_ratio": 0.45,
        "exit_minimum_canvas_span_fraction": 0.18,
        "exit_boundary_margin_body_radii": 3.0,
        "exit_boundary_margin_fraction": 0.06,
        "state_coverage_power": 1.0,
        "content_weights": {
            "physics_state": 0.55,
            "shape": 0.2,
            "appearance": 0.25,
        },
    }


def merged_rigid_body_config(
    config: Mapping[str, Any],
    *,
    scene_kind: str,
) -> dict[str, Any]:
    output = default_rigid_body_config(scene_kind)
    supplied = config.get("rigid_body_observation", {})
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError("rigid_body_observation must be an object")
        output.update(dict(supplied))
    scoring = config.get("object_centric_scoring", {})
    if scoring is not None:
        if not isinstance(scoring, Mapping):
            raise ValueError("object_centric_scoring must be an object")
        assignment = scoring.get("assignment", {})
        if isinstance(assignment, Mapping) and (
            "minimum_match_position_similarity" in assignment
        ):
            output["minimum_match_position_similarity"] = float(
                assignment["minimum_match_position_similarity"]
            )
        if "content_weights" in scoring:
            output["content_weights"] = dict(scoring["content_weights"])
    return output


def rigid_body_fingerprint_payload(
    *,
    evaluator_id: str,
    evaluator_version: str,
    config: Mapping[str, Any],
    scene_kind: str,
) -> str:
    return canonical_sha256(
        {
            "id": evaluator_id,
            "version": evaluator_version,
            "config": config,
            "entity_contract": "manifest_v1",
            "observer": (
                f"open_world_v2+rigid_body_observer_"
                f"v{RIGID_BODY_OPEN_WORLD_VERSION}"
            ),
            "scene_kind": scene_kind,
        }
    )


def freeze_reference_axis(
    xy: np.ndarray,
    observed: np.ndarray,
    *,
    scene_kind: str,
    minimum_span_px: float,
) -> FrozenRigidAxis:
    points = np.asarray(xy, dtype=np.float64)
    valid = np.asarray(observed, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or valid.shape != (len(points),):
        raise ValueError("axis observations have incompatible shapes")
    if int(valid.sum()) < 3:
        raise SceneAnalysisError(
            "insufficient_reference_axis_observations",
            "at least three reference body observations are required",
        )
    valid_points = points[valid]
    if scene_kind == "free_fall":
        direction = np.asarray([0.0, 1.0], dtype=np.float64)
        normal = np.asarray([-1.0, 0.0], dtype=np.float64)
        origin = valid_points[0]
        along = (valid_points - origin) @ direction
        span = float(np.max(along) - np.min(along))
        cross = (valid_points - origin) @ normal
        denominator = float(np.var(along) + np.var(cross))
        explained = (
            float(np.var(along) / denominator) if denominator > 1e-12 else 1.0
        )
        source = "reference_vertical_gravity_axis"
    elif scene_kind == "inclined_plane":
        model: AxisModel = fit_axis(valid_points)
        direction = model.direction_xy
        normal = model.normal_xy
        origin = valid_points[0]
        along = (valid_points - origin) @ direction
        if float(np.nanmedian(np.diff(along))) < 0.0:
            direction = -direction
            normal = -normal
            along = -along
        span = float(np.max(along) - np.min(along))
        explained = float(model.explained_ratio)
        source = "reference_sliding_track_axis"
    else:
        raise ValueError(f"unsupported rigid-body scene kind: {scene_kind}")
    if span < float(minimum_span_px):
        raise SceneAnalysisError(
            "insufficient_reference_rigid_body_motion",
            f"reference body spans only {span:.3f}px",
        )
    return FrozenRigidAxis(
        origin_xy=origin,
        direction_xy=direction,
        normal_xy=normal,
        span_px=span,
        source=source,
        explained_ratio=explained,
    )


def freeze_condition_incline_axis(
    condition_frame: np.ndarray,
    *,
    condition_centroid_xy: np.ndarray,
    fallback: FrozenRigidAxis,
) -> FrozenRigidAxis:
    """Estimate an apparatus line from the condition image, never prediction."""

    frame = np.asarray(condition_frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    minimum_length = max(30, int(round(0.22 * math.hypot(width, height))))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(20, minimum_length // 4),
        minLineLength=minimum_length,
        maxLineGap=max(8, minimum_length // 8),
    )
    centroid = np.asarray(condition_centroid_xy, dtype=np.float64)
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    if lines is not None:
        for raw in lines[:, 0]:
            first = raw[:2].astype(np.float64)
            second = raw[2:].astype(np.float64)
            vector = second - first
            length = float(np.linalg.norm(vector))
            if length <= 1e-12:
                continue
            direction = vector / length
            angle = abs(float(np.rad2deg(math.atan2(direction[1], direction[0]))))
            angle = min(angle, abs(180.0 - angle))
            if angle < 7.0 or angle > 83.0:
                continue
            distance = abs(float(np.cross(direction, centroid - first)))
            if distance > max(0.2 * height, 6.0):
                continue
            endpoint = (
                second
                if second[1] > first[1]
                else first
            )
            downhill = endpoint - centroid
            if float(np.dot(direction, downhill)) < 0.0:
                direction = -direction
            forward_span = max(float(np.dot(first - centroid, direction)), 0.0)
            forward_span = max(
                forward_span,
                float(np.dot(second - centroid, direction)),
                1.0,
            )
            score = length - 1.5 * distance + 0.25 * forward_span
            candidates.append((score, direction, centroid, forward_span))
    if not candidates:
        return FrozenRigidAxis(
            origin_xy=centroid,
            direction_xy=fallback.direction_xy,
            normal_xy=fallback.normal_xy,
            span_px=fallback.span_px,
            source="condition_axis_hough_failed_reference_direction_fallback",
            explained_ratio=fallback.explained_ratio,
        )
    _, direction, origin, span = max(candidates, key=lambda item: item[0])
    return FrozenRigidAxis(
        origin_xy=origin,
        direction_xy=direction,
        normal_xy=np.asarray([-direction[1], direction[0]]),
        span_px=max(span, fallback.span_px * 0.5),
        source="condition_apparatus_hough_axis",
        explained_ratio=1.0,
    )


def _valid_mask_geometry(
    mask: np.ndarray,
    *,
    minimum_area: int,
    maximum_area: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    components = component_centroids(
        binary,
        minimum_area=minimum_area,
    )
    components = [
        value for value in components if value[1] <= maximum_area
    ]
    if not components:
        return None
    centroid, area, component = components[0]
    return component, centroid, float(area)


def _interpolate_positions(
    xy: np.ndarray,
    observed: np.ndarray,
    expected: np.ndarray,
) -> np.ndarray:
    output = np.asarray(xy, dtype=np.float64).copy()
    known = np.asarray(observed, dtype=bool)
    target = np.asarray(expected, dtype=bool)
    if int(known.sum()) < 2:
        raise SceneAnalysisError(
            "insufficient_reference_rigid_body_observations",
            "reference body requires at least two valid positions",
        )
    indices = np.arange(len(output), dtype=np.float64)
    for column in range(2):
        output[target, column] = np.interp(
            indices[target],
            indices[known],
            output[known, column],
        )
    output[~target] = np.nan
    return output


def _translate_mask(
    mask: np.ndarray,
    *,
    from_xy: np.ndarray,
    to_xy: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(to_xy, dtype=np.float64) - np.asarray(
        from_xy, dtype=np.float64
    )
    matrix = np.asarray(
        [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]],
        dtype=np.float32,
    )
    height, width = mask.shape[:2]
    return cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _infer_legal_exit(
    observed: np.ndarray,
    along: np.ndarray,
    *,
    xy: np.ndarray,
    areas_px2: np.ndarray,
    frame_shape: tuple[int, int],
    scene_kind: str,
    minimum_trailing_frames: int,
    minimum_progress: float,
    minimum_forward_step_ratio: float,
    minimum_canvas_span_fraction: float,
    boundary_margin_body_radii: float,
    boundary_margin_fraction: float,
) -> tuple[int | None, dict[str, Any]]:
    valid_indices = np.flatnonzero(observed)
    last = int(valid_indices[-1])
    trailing = len(observed) - last - 1
    diagnostics: dict[str, Any] = {
        "last_observed_frame": last,
        "trailing_unobserved_frames": trailing,
        "minimum_trailing_frames": minimum_trailing_frames,
        "minimum_progress": minimum_progress,
    }
    if trailing < minimum_trailing_frames:
        diagnostics.update(
            {"legal_exit": False, "reason": "insufficient_trailing_absence"}
        )
        return None, diagnostics
    values = along[observed]
    span = max(float(np.ptp(values)), 1e-9)
    progress = float((values[-1] - values[0]) / span)
    steps = np.diff(values)
    forward_ratio = (
        float(np.mean(steps >= -0.02 * span)) if len(steps) else 0.0
    )
    diagnostics.update(
        {
            "terminal_progress_ratio": progress,
            "forward_step_ratio": forward_ratio,
        }
    )
    height, width = frame_shape
    last_xy = np.asarray(xy[last], dtype=np.float64)
    radius = math.sqrt(
        max(float(np.nanmedian(areas_px2[observed])), 1.0) / math.pi
    )
    boundary_margin = max(
        boundary_margin_body_radii * radius,
        boundary_margin_fraction * min(height, width),
    )
    if scene_kind == "free_fall":
        near_exit_boundary = bool(
            height - 1.0 - last_xy[1] <= boundary_margin
        )
        canvas_scale = float(height)
    else:
        edge_distance = min(
            last_xy[0],
            last_xy[1],
            width - 1.0 - last_xy[0],
            height - 1.0 - last_xy[1],
        )
        near_exit_boundary = bool(edge_distance <= boundary_margin)
        canvas_scale = float(math.hypot(width, height))
    sufficient_terminal_travel = bool(
        span >= minimum_canvas_span_fraction * canvas_scale
    )
    terminal_geometry_supported = (
        near_exit_boundary or sufficient_terminal_travel
    )
    diagnostics.update(
        {
            "terminal_position_xy": last_xy.tolist(),
            "terminal_motion_span_px": span,
            "near_exit_boundary": near_exit_boundary,
            "sufficient_terminal_travel": sufficient_terminal_travel,
            "minimum_terminal_travel_px": (
                minimum_canvas_span_fraction * canvas_scale
            ),
        }
    )
    if (
        progress < minimum_progress
        or forward_ratio < minimum_forward_step_ratio
        or not terminal_geometry_supported
    ):
        diagnostics.update(
            {"legal_exit": False, "reason": "reference_end_event_not_reached"}
        )
        return None, diagnostics
    exit_frame = last + 1
    diagnostics.update(
        {
            "legal_exit": True,
            "reason": "reference_terminal_progress_then_persistent_absence",
            "legal_exit_frame": exit_frame,
        }
    )
    return exit_frame, diagnostics


def build_rigid_body_reference(
    masks: Sequence[np.ndarray],
    *,
    entity_id: str,
    entity_class: str,
    scene_kind: str,
    frame_shape: tuple[int, int],
    minimum_area: int,
    maximum_area_ratio: float,
    minimum_span_px: float,
    config: Mapping[str, Any],
) -> RigidBodyReference:
    if not masks:
        raise SceneAnalysisError(
            "empty_reference_rigid_body_masks",
            "reference rigid-body mask timeline is empty",
        )
    height, width = frame_shape
    maximum_area = int(round(height * width * maximum_area_ratio))
    xy = np.full((len(masks), 2), np.nan, dtype=np.float64)
    areas = np.full(len(masks), np.nan, dtype=np.float64)
    observed = np.zeros(len(masks), dtype=bool)
    clean_masks: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        value = _valid_mask_geometry(
            mask,
            minimum_area=minimum_area,
            maximum_area=maximum_area,
        )
        if value is None:
            clean_masks.append(np.zeros((height, width), dtype=np.uint8))
            continue
        component, centroid, area = value
        clean_masks.append(component)
        xy[index] = centroid
        areas[index] = area
        observed[index] = True
    axis = freeze_reference_axis(
        xy,
        observed,
        scene_kind=scene_kind,
        minimum_span_px=minimum_span_px,
    )
    along_raw = np.full(len(masks), np.nan, dtype=np.float64)
    along_raw[observed], _ = axis.project(xy[observed])
    exit_frame, lifecycle = _infer_legal_exit(
        observed,
        along_raw,
        xy=xy,
        areas_px2=areas,
        frame_shape=frame_shape,
        scene_kind=scene_kind,
        minimum_trailing_frames=int(config["exit_minimum_trailing_frames"]),
        minimum_progress=float(config["exit_minimum_progress"]),
        minimum_forward_step_ratio=float(
            config["exit_minimum_forward_step_ratio"]
        ),
        minimum_canvas_span_fraction=float(
            config["exit_minimum_canvas_span_fraction"]
        ),
        boundary_margin_body_radii=float(
            config["exit_boundary_margin_body_radii"]
        ),
        boundary_margin_fraction=float(
            config["exit_boundary_margin_fraction"]
        ),
    )
    expected = np.ones(len(masks), dtype=bool)
    if exit_frame is not None:
        expected[exit_frame:] = False
    filled_xy = _interpolate_positions(xy, observed, expected)
    median_area = float(np.nanmedian(areas[observed]))
    filled_areas = np.where(
        expected,
        np.where(np.isfinite(areas), areas, median_area),
        np.nan,
    )
    observed_indices = np.flatnonzero(observed)
    for index in np.flatnonzero(expected & ~observed):
        nearest = int(
            observed_indices[
                np.argmin(np.abs(observed_indices - index))
            ]
        )
        clean_masks[index] = _translate_mask(
            clean_masks[nearest],
            from_xy=xy[nearest],
            to_xy=filled_xy[index],
        )
    for index in np.flatnonzero(~expected):
        clean_masks[index] = np.zeros((height, width), dtype=np.uint8)
    along, _ = axis.project(filled_xy[expected])
    progress = np.full(len(masks), np.nan, dtype=np.float64)
    progress[expected] = (along - along[0]) / axis.span_px
    return RigidBodyReference(
        entity_id=entity_id,
        entity_class=entity_class,
        masks=tuple(clean_masks),
        xy=filled_xy,
        areas_px2=filled_areas,
        expected=expected,
        raw_observed=observed,
        axis=axis,
        normalized_progress=progress,
        legal_exit_frame=exit_frame,
        lifecycle_diagnostics=lifecycle,
    )


def remap_reference_to_condition(
    reference: RigidBodyReference,
    *,
    condition_axis: FrozenRigidAxis,
    condition_area_px2: float,
) -> RigidBodyReference:
    xy = np.full_like(reference.xy, np.nan)
    expected = reference.expected
    xy[expected] = (
        condition_axis.origin_xy
        + reference.normalized_progress[expected, None]
        * condition_axis.span_px
        * condition_axis.direction_xy
    )
    areas = np.where(expected, float(condition_area_px2), np.nan)
    masks = tuple(
        np.zeros_like(reference.masks[0])
        for _ in reference.masks
    )
    return RigidBodyReference(
        entity_id=reference.entity_id,
        entity_class=reference.entity_class,
        masks=masks,
        xy=xy,
        areas_px2=areas,
        expected=reference.expected,
        raw_observed=reference.raw_observed,
        axis=condition_axis,
        normalized_progress=reference.normalized_progress,
        legal_exit_frame=reference.legal_exit_frame,
        lifecycle_diagnostics={
            **reference.lifecycle_diagnostics,
            "reference_geometry": "normalized_physics_parent_mapped_to_condition",
        },
    )


def build_expected_rigid_body_timeline(
    *,
    reference: RigidBodyReference,
    scoring_reference: RigidBodyReference,
    manifest_entity: Any,
    time_grid: CommonTimeGrid,
    capability: ReferenceCapability,
    condition_mask: np.ndarray,
) -> ExpectedEntityTimeline:
    """Declare legal presence and capability channels from reference data only."""

    entity = EntitySpec(
        entity_id=manifest_entity.entity_id,
        role_id=manifest_entity.role_id,
        entity_class=manifest_entity.entity_class,
        parts=tuple(manifest_entity.parts),
        exchangeability_group=manifest_entity.exchangeability_group,
        lifecycle=manifest_entity.lifecycle,
        anchor=dict(manifest_entity.condition_anchor),
    )
    reference_track = scoring_reference.to_object_track(time_grid)
    if capability is ReferenceCapability.SAME_CASE_GT:
        return resolve_expected_entity_timeline(
            entity,
            time_grid=time_grid,
            capability=capability,
            reference_track=reference_track,
            declared_expected_mask=reference.expected,
            reference_masks=list(reference.masks),
            metadata={
                "legal_exit_frame": reference.legal_exit_frame,
                "legal_exit_evidence": dict(
                    reference.lifecycle_diagnostics
                ),
                "future_geometry_source": "same_case_gt_only",
            },
        )
    condition_xy = mask_centroid(condition_mask)
    if condition_xy is None:
        raise ReferenceAnalysisError(
            "reference_condition_subject_unobserved",
            "condition mask has no physical-subject centroid",
        )
    # A physics parent supplies a normalized dynamics target and a
    # reference-only lifecycle event, never future pixel coordinates.
    condition_sample = ExpectedPositionSample(
        xy=condition_xy,
        area_px2=float(np.count_nonzero(condition_mask)),
        mask=condition_mask,
    )
    return resolve_expected_entity_timeline(
        entity,
        time_grid=time_grid,
        capability=capability,
        declared_expected_mask=reference.expected,
        condition_position=condition_sample,
        condition_identity_supervised=True,
        reference_masks=[
            condition_mask if index == 0 else None
            for index in range(len(time_grid.times_s))
        ],
        metadata={
            "legal_exit_frame": reference.legal_exit_frame,
            "legal_exit_evidence": dict(reference.lifecycle_diagnostics),
            "future_geometry_source": (
                "normalized_physics_state_only_no_parent_pixels"
            ),
        },
    )


def freeze_condition_rigid_body_identity(
    *,
    timeline: ExpectedEntityTimeline,
    observation: OpenWorldObservation,
) -> FrozenAssignment:
    """Reserve the condition-anchored directed track against replacements."""

    candidates: list[tuple[float, str]] = []
    for track in observation.tracks:
        if track.entity_class != timeline.entity.entity_class:
            continue
        for detection in track.detections:
            if detection.frame_index != 0:
                continue
            if (
                "directed_sam2" not in detection.sources
                and "directed" not in detection.sources
            ):
                continue
            if detection.metadata.get("identity_anchor_valid") is False:
                continue
            candidates.append((detection.confidence, track.track_id))
    if not candidates:
        raise SceneAnalysisError(
            "prediction_condition_identity_missing",
            "no condition-anchored directed track can freeze the manifest "
            f"identity {timeline.entity_id}",
        )
    _, selected = max(candidates, key=lambda value: (value[0], value[1]))
    residual = tuple(
        sorted(
            track.track_id
            for track in observation.tracks
            if track.track_id != selected
        )
    )
    return FrozenAssignment(
        entity_to_track={timeline.entity_id: selected},
        residual_track_ids=residual,
        unmatched_entity_ids=(),
        total_cost=0.0,
    )


def safe_compare_rigid_body_open_world(
    *,
    expected_timeline: ExpectedEntityTimeline,
    prediction_factory,
    time_grid: CommonTimeGrid,
    frame_shape: tuple[int, int],
    minimum_match_position_similarity: float,
) -> tuple[ObjectCentricComparisonV2, OpenWorldObservation]:
    """Use the v2 robustness boundary and condition-frozen identity contract."""

    observation_holder: dict[str, OpenWorldObservation] = {}
    error_holder: dict[str, Exception] = {}

    def observed() -> OpenWorldObservation:
        try:
            value = prediction_factory()
        except Exception as exc:
            error_holder["error"] = exc
            raise
        observation_holder["value"] = value
        return value

    # Discovery must run once before comparison because the frozen assignment
    # is itself condition-side evidence over the causal tracks.
    try:
        observation = observed()
        frozen = freeze_condition_rigid_body_identity(
            timeline=expected_timeline,
            observation=observation,
        )
        comparison = safe_compare_open_world_v2(
            expected_timelines=[expected_timeline],
            prediction_factory=lambda: observation,
            time_grid=time_grid,
            frame_diagonal_px=float(math.hypot(*frame_shape)),
            frozen_identity=frozen,
            minimum_match_position_similarity=(
                minimum_match_position_similarity
            ),
        )
        if comparison.failed:
            return (
                comparison,
                empty_fail_closed_observation(
                    len(time_grid.times_s),
                    code="prediction_open_world_v2_comparison_failed",
                    reason=comparison.failure_reason or "unknown v2 failure",
                ),
            )
        return comparison, observation
    except Exception as exc:
        # Re-run no observer work: the callable deterministically raises the
        # captured failure inside the protocol-owned fail-closed boundary.
        def failed_factory() -> OpenWorldObservation:
            raise exc

        comparison = safe_compare_open_world_v2(
            expected_timelines=[expected_timeline],
            prediction_factory=failed_factory,
            time_grid=time_grid,
            frame_diagonal_px=float(math.hypot(*frame_shape)),
            minimum_match_position_similarity=(
                minimum_match_position_similarity
            ),
        )
        return (
            comparison,
            empty_fail_closed_observation(
                len(time_grid.times_s),
                code="prediction_open_world_v2_observer_failed",
                reason=comparison.failure_reason or f"{type(exc).__name__}: {exc}",
            ),
        )


def _mask_histogram(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(64, dtype=np.float64)
    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], binary, [8, 8], [0, 180, 0, 256]
    ).reshape(-1)
    total = float(histogram.sum())
    return histogram.astype(np.float64) / total if total > 0.0 else histogram


def _histogram_intersection(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.minimum(first, second).sum())


def _compact_shape_score(mask: np.ndarray, *, scene_kind: str) -> float:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return 0.0
    x, y, width, height = cv2.boundingRect(contour)
    aspect = max(width, height) / max(min(width, height), 1)
    perimeter = float(cv2.arcLength(contour, True))
    if scene_kind == "free_fall":
        circularity = (
            4.0 * math.pi * area / max(perimeter * perimeter, 1e-9)
        )
        return float(np.clip(circularity / max(aspect, 1.0), 0.0, 1.0))
    rectangularity = area / max(float(width * height), 1.0)
    aspect_penalty = 1.0 / max(1.0, aspect / 2.0)
    return float(np.clip(rectangularity * aspect_penalty, 0.0, 1.0))


def _shape_is_plausible(
    mask: np.ndarray,
    *,
    scene_kind: str,
    config: Mapping[str, Any],
) -> bool:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return False
    x, y, width, height = cv2.boundingRect(contour)
    aspect = max(width, height) / max(min(width, height), 1)
    if scene_kind == "free_fall":
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area / max(
            perimeter * perimeter, 1e-9
        )
        return (
            circularity >= float(config["minimum_circularity"])
            and aspect <= float(config["maximum_circle_aspect_ratio"])
        )
    rectangularity = area / max(float(width * height), 1.0)
    return (
        rectangularity >= float(config["minimum_rectangularity"])
        and aspect <= float(config["maximum_rectangle_aspect_ratio"])
    )


def _threshold_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    delta = cv2.absdiff(
        cv2.GaussianBlur(first_gray, (5, 5), 0),
        cv2.GaussianBlur(second_gray, (5, 5), 0),
    )
    binary = np.where(delta >= threshold, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )


def _scene_roi(
    shape: tuple[int, int],
    *,
    axis: FrozenRigidAxis,
    body_area_px2: float,
    scene_kind: str,
    config: Mapping[str, Any],
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx, yy], axis=-1).astype(np.float64)
    centered = points - axis.origin_xy
    cross = centered @ axis.normal_xy
    along = centered @ axis.direction_xy
    radius = math.sqrt(max(body_area_px2, 1.0) / math.pi)
    half_width = max(
        float(config["roi_half_width_body_diameters"]) * 2.0 * radius,
        float(config["roi_minimum_half_width_fraction"])
        * min(height, width),
    )
    if scene_kind == "free_fall":
        # The full gravity viewport is a valid participant region; restricting
        # it to the expected trajectory would let a model hide extras on the
        # opposite side of the image.
        corners = np.asarray(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [0.0, height - 1.0],
                [width - 1.0, height - 1.0],
            ]
        )
        corner_along, _ = axis.project(corners)
        roi = (
            (along >= float(np.min(corner_along)))
            & (along <= float(np.max(corner_along)))
        )
    else:
        along_margin = max(6.0 * radius, 0.2 * axis.span_px)
        roi = (
            (np.abs(cross) <= half_width)
            & (along >= -along_margin)
            & (along <= axis.span_px + along_margin)
        )
    return roi.astype(np.uint8) * 255


def _component_proposals(
    mask: np.ndarray,
    *,
    minimum_area: int,
    maximum_area: int,
    scene_kind: str,
    config: Mapping[str, Any],
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for _, area, component in component_centroids(
        mask,
        minimum_area=minimum_area,
    ):
        if area > maximum_area:
            continue
        if not _shape_is_plausible(
            component, scene_kind=scene_kind, config=config
        ):
            continue
        output.append(component)
    return output


def _independent_compact_proposals(
    frame: np.ndarray,
    *,
    scene_kind: str,
    anchor_area_px2: float,
    minimum_area: int,
    maximum_area: int,
    config: Mapping[str, Any],
) -> list[np.ndarray]:
    """Find static compact copies without relying on prediction motion."""

    gray = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    proposals: list[np.ndarray] = []
    if scene_kind == "free_fall":
        radius = math.sqrt(max(anchor_area_px2, 1.0) / math.pi)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(2.0 * radius, 6.0),
            param1=90.0,
            param2=max(7.0, 0.75 * radius),
            minRadius=max(2, int(round(0.4 * radius))),
            maxRadius=max(3, int(round(2.4 * radius))),
        )
        if circles is not None:
            for x, y, detected_radius in circles[0]:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(
                    mask,
                    (int(round(x)), int(round(y))),
                    int(round(detected_radius)),
                    255,
                    -1,
                )
                area = int(np.count_nonzero(mask))
                if minimum_area <= area <= maximum_area:
                    proposals.append(mask)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < 0.5 * minimum_area or contour_area > maximum_area:
            continue
        if scene_kind == "inclined_plane":
            rectangle = cv2.minAreaRect(contour)
            (_, _), (width, height), _ = rectangle
            rectangle_area = float(width * height)
            if not minimum_area <= rectangle_area <= maximum_area:
                continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(
                mask,
                cv2.boxPoints(rectangle).astype(np.int32),
                255,
            )
        else:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
        if _shape_is_plausible(
            mask,
            scene_kind=scene_kind,
            config=config,
        ):
            proposals.append(mask)
    return [value[0] for value in _merge_residual_proposals(
        [(mask, "independent_compact_shape") for mask in proposals]
    )]


def _overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a = first > 0
    b = second > 0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    containment = intersection / max(min(int(a.sum()), int(b.sum())), 1)
    return intersection / max(union, 1), containment


def _mask_canvas_edges(mask: np.ndarray) -> tuple[str, ...]:
    """Return the canvas edges touched by a non-empty binary mask."""

    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not binary.any():
        return ()
    edges: list[str] = []
    if binary[0, :].any():
        edges.append("top")
    if binary[-1, :].any():
        edges.append("bottom")
    if binary[:, 0].any():
        edges.append("left")
    if binary[:, -1].any():
        edges.append("right")
    return tuple(edges)


def _merge_residual_proposals(
    proposals: Sequence[tuple[np.ndarray, str]],
) -> list[tuple[np.ndarray, tuple[str, ...]]]:
    merged: list[tuple[np.ndarray, set[str]]] = []
    for raw_mask, source in proposals:
        mask = np.where(raw_mask > 0, 255, 0).astype(np.uint8)
        target = None
        for index, (existing, _) in enumerate(merged):
            iou, containment = _overlap(existing, mask)
            if iou >= 0.35 or containment >= 0.7:
                target = index
                break
        if target is None:
            merged.append((mask, {source}))
        else:
            existing, sources = merged[target]
            merged[target] = (cv2.bitwise_or(existing, mask), sources | {source})
    return [
        (mask, tuple(sorted(sources)))
        for mask, sources in merged
    ]


def discover_rigid_body_objects(
    frames: Sequence[np.ndarray],
    *,
    directed_masks: Sequence[np.ndarray],
    condition_frame: np.ndarray,
    condition_mask: np.ndarray,
    reference_axis: FrozenRigidAxis,
    entity_class: str,
    scene_kind: str,
    time_grid: CommonTimeGrid,
    available: Sequence[bool] | None,
    minimum_area: int,
    maximum_area_ratio: float,
    config: Mapping[str, Any],
) -> OpenWorldObservation:
    """Combine directed identity tracking with unrestricted residual discovery."""

    values = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not values or len(values) != len(directed_masks):
        raise SceneAnalysisError(
            "rigid_body_observation_timeline_mismatch",
            "frames and directed masks must share a non-empty timeline",
        )
    if len(values) != len(time_grid.times_s):
        raise SceneAnalysisError(
            "rigid_body_time_grid_mismatch",
            "rigid-body observations differ from the common time grid",
        )
    shape = values[0].shape[:2]
    if condition_frame.shape[:2] != shape or condition_mask.shape != shape:
        raise SceneAnalysisError(
            "rigid_body_condition_shape_mismatch",
            "condition frame/mask differ from the normalized video canvas",
        )
    availability = (
        np.ones(len(values), dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if availability.shape != (len(values),) or availability.dtype.kind != "b":
        raise SceneAnalysisError(
            "rigid_body_availability_mismatch",
            "availability must contain one boolean per frame",
        )
    if not availability.any():
        raise SceneAnalysisError(
            "rigid_body_prediction_unavailable",
            "prediction has no available common-time samples",
        )
    height, width = shape
    frame_area = height * width
    anchor_area = int(np.count_nonzero(condition_mask))
    if anchor_area < minimum_area:
        raise SceneAnalysisError(
            "rigid_body_condition_subject_unobserved",
            "condition mask does not contain the expected rigid body",
        )
    residual_minimum = max(
        minimum_area,
        int(round(anchor_area * float(config["minimum_area_ratio"]))),
    )
    residual_maximum = min(
        int(round(frame_area * maximum_area_ratio)),
        int(round(anchor_area * float(config["maximum_area_ratio"]))),
    )
    direct_minimum = max(
        minimum_area,
        int(round(anchor_area * float(config["direct_minimum_area_ratio"]))),
    )
    direct_maximum = min(
        int(round(frame_area * maximum_area_ratio)),
        int(round(anchor_area * float(config["direct_maximum_area_ratio"]))),
    )
    if residual_maximum < residual_minimum or direct_maximum < direct_minimum:
        raise SceneAnalysisError(
            "rigid_body_invalid_area_contract",
            "body-relative observation area bounds are empty",
        )
    anchor_histogram = _mask_histogram(condition_frame, condition_mask)
    roi = _scene_roi(
        shape,
        axis=reference_axis,
        body_area_px2=float(anchor_area),
        scene_kind=scene_kind,
        config=config,
    )
    body_radius = math.sqrt(anchor_area / math.pi)
    exclusion_radius = max(
        2,
        int(
            round(
                body_radius
                * float(config["residual_dilation_body_fraction"])
            )
        ),
    )
    exclusion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * exclusion_radius + 1, 2 * exclusion_radius + 1),
    )
    detections_by_frame: list[list[ObjectDetection]] = [
        [] for _ in values
    ]
    source_counts = {
        "directed": 0,
        "condition_difference": 0,
        "temporal_motion": 0,
        "independent_compact_shape": 0,
        "replacement": 0,
        "residual": 0,
        "condition_only_boundary_rejected": 0,
    }
    rejected_candidates: list[dict[str, Any]] = []
    previous_direct = np.zeros(shape, dtype=np.uint8)
    identity_rejection_streak = 0
    for frame_index, (frame, raw_direct) in enumerate(
        zip(values, directed_masks)
    ):
        if not availability[frame_index]:
            previous_direct = np.zeros(shape, dtype=np.uint8)
            identity_rejection_streak = 0
            continue
        direct_components = component_centroids(
            np.where(raw_direct > 0, 255, 0).astype(np.uint8),
            minimum_area=direct_minimum,
        )
        direct_components = [
            value for value in direct_components if value[1] <= direct_maximum
        ]
        selected_direct: np.ndarray | None = None
        if direct_components:
            anchor_xy = mask_centroid(condition_mask)
            assert anchor_xy is not None
            if frame_index > 0 and detections_by_frame[frame_index - 1]:
                participant = [
                    detection
                    for detection in detections_by_frame[frame_index - 1]
                    if "directed_sam2" in detection.sources
                ]
                if participant:
                    anchor_xy = participant[0].xy
            selected = min(
                direct_components,
                key=lambda value: float(np.linalg.norm(value[0] - anchor_xy)),
            )
            centroid, area, selected_direct = selected
            color_similarity = _histogram_intersection(
                anchor_histogram,
                _mask_histogram(frame, selected_direct),
            )
            raw_identity_valid = (
                color_similarity
                >= float(config["minimum_anchor_color_similarity"])
            )
            if raw_identity_valid:
                identity_rejection_streak = 0
            else:
                identity_rejection_streak += 1
            identity_valid = bool(
                raw_identity_valid
                or identity_rejection_streak
                < int(
                    config[
                        "identity_rejection_minimum_consecutive_frames"
                    ]
                )
            )
            detection_class = (
                entity_class
                if identity_valid
                else f"{entity_class}__replacement"
            )
            sources = (
                ("directed_sam2",)
                if identity_valid
                else ("directed_sam2", "identity_anchor_rejected")
            )
            detections_by_frame[frame_index].append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=f"direct_{frame_index:05d}",
                    xy=centroid,
                    area_px2=float(area),
                    entity_class=detection_class,
                    mask=selected_direct,
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=sources,
                    metadata={
                        "anchor_color_similarity": color_similarity,
                        "identity_anchor_valid": identity_valid,
                        "identity_anchor_raw_valid": raw_identity_valid,
                        "identity_rejection_streak": (
                            identity_rejection_streak
                        ),
                    },
                )
            )
            source_counts[
                "directed" if identity_valid else "replacement"
            ] += 1
            for component_index, (
                extra_xy,
                extra_area,
                extra_mask,
            ) in enumerate(direct_components):
                if extra_mask is selected_direct:
                    continue
                detections_by_frame[frame_index].append(
                    ObjectDetection(
                        frame_index=frame_index,
                        detection_id=(
                            f"direct_duplicate_{frame_index:05d}_"
                            f"{component_index:03d}"
                        ),
                        xy=extra_xy,
                        area_px2=float(extra_area),
                        entity_class=entity_class,
                        mask=extra_mask,
                        confidence=1.0,
                        evidence_tier=EvidenceTier.PARTICIPANT,
                        sources=("directed_sam2", "disconnected_duplicate"),
                        metadata={"duplicate_directed_component": True},
                    )
                )
        current_direct = (
            selected_direct
            if selected_direct is not None
            else np.zeros(shape, dtype=np.uint8)
        )
        exclusion = cv2.dilate(
            cv2.bitwise_or(
                current_direct,
                cv2.bitwise_or(previous_direct, condition_mask),
            ),
            exclusion_kernel,
        )
        condition_difference = _threshold_difference(
            frame,
            condition_frame,
            threshold=float(config["condition_difference_threshold"]),
        )
        condition_difference = cv2.bitwise_and(condition_difference, roi)
        condition_difference[exclusion > 0] = 0
        raw_proposals: list[tuple[np.ndarray, str]] = [
            (mask, "condition_difference")
            for mask in _component_proposals(
                condition_difference,
                minimum_area=residual_minimum,
                maximum_area=residual_maximum,
                scene_kind=scene_kind,
                config=config,
            )
        ]
        source_counts["condition_difference"] += len(raw_proposals)
        compact_proposals: list[tuple[np.ndarray, str]] = []
        for mask in _independent_compact_proposals(
            frame,
            scene_kind=scene_kind,
            anchor_area_px2=float(anchor_area),
            minimum_area=residual_minimum,
            maximum_area=residual_maximum,
            config=config,
        ):
            proposal_area = int(np.count_nonzero(mask))
            area_ratio = proposal_area / max(float(anchor_area), 1.0)
            appearance = _histogram_intersection(
                anchor_histogram,
                _mask_histogram(frame, mask),
            )
            if not (
                float(config["outside_roi_minimum_area_ratio"])
                <= area_ratio
                <= float(config["outside_roi_maximum_area_ratio"])
                and appearance
                >= float(config["minimum_duplicate_color_similarity"])
            ):
                continue
            inside = cv2.bitwise_and(mask, roi)
            inside_area = int(np.count_nonzero(inside))
            inside_fraction = inside_area / max(proposal_area, 1)
            if inside_area >= residual_minimum and inside_fraction >= 0.5:
                candidate = inside
            else:
                # A second body remains a physical participant even when it
                # is far from the reference axis.  Outside the scene ROI we
                # require strong condition appearance and size agreement so
                # apparatus rectangles do not become cheap false positives.
                if not (
                    appearance >= float(
                        config[
                            "outside_roi_minimum_anchor_color_similarity"
                        ]
                    )
                ):
                    continue
                candidate = mask
            if np.any(current_direct):
                _, containment = _overlap(candidate, current_direct)
                if containment >= 0.7:
                    continue
            compact_proposals.append(
                (candidate, "independent_compact_shape")
            )
        raw_proposals.extend(compact_proposals)
        source_counts["independent_compact_shape"] += len(
            compact_proposals
        )
        if 0 < frame_index < len(values) - 1:
            if availability[frame_index - 1] and availability[frame_index + 1]:
                previous_delta = _threshold_difference(
                    frame,
                    values[frame_index - 1],
                    threshold=float(config["temporal_difference_threshold"]),
                )
                next_delta = _threshold_difference(
                    frame,
                    values[frame_index + 1],
                    threshold=float(config["temporal_difference_threshold"]),
                )
                temporal = cv2.bitwise_and(previous_delta, next_delta)
                temporal = cv2.bitwise_and(temporal, roi)
                temporal[exclusion > 0] = 0
                temporal_proposals = [
                    (mask, "temporal_motion")
                    for mask in _component_proposals(
                        temporal,
                        minimum_area=residual_minimum,
                        maximum_area=residual_maximum,
                        scene_kind=scene_kind,
                        config=config,
                    )
                ]
                raw_proposals.extend(temporal_proposals)
                source_counts["temporal_motion"] += len(temporal_proposals)
        for residual_index, (mask, sources) in enumerate(
            _merge_residual_proposals(raw_proposals)
        ):
            centroid = mask_centroid(mask)
            if centroid is None:
                continue
            area = int(np.count_nonzero(mask))
            formal_sources = tuple(sorted(set(sources)))
            touched_edges = _mask_canvas_edges(mask)
            if (
                set(formal_sources) == {"condition_difference"}
                and touched_edges
            ):
                # A condition-only change clipped by the canvas is commonly
                # an apparatus/background mismatch at the viewport boundary.
                # It has no independent evidence of being a physical body,
                # so retain it for audit without allowing it to accumulate
                # participant exposure.  Motion or compact-shape support
                # survives this gate because it adds another formal source.
                rejected_candidates.append(
                    {
                        "frame_index": frame_index,
                        "candidate_id": (
                            f"residual_{frame_index:05d}_"
                            f"{residual_index:03d}"
                        ),
                        "rejection_reason": (
                            "condition_difference_only_touches_"
                            "canvas_boundary"
                        ),
                        "sources": list(formal_sources),
                        "touched_edges": list(touched_edges),
                        "xy": centroid.tolist(),
                        "area_px2": float(area),
                        "formal_exposure_weight": 0.0,
                    }
                )
                source_counts["condition_only_boundary_rejected"] += 1
                continue
            anchor_color_similarity = _histogram_intersection(
                anchor_histogram,
                _mask_histogram(frame, mask),
            )
            independently_credible_duplicate = (
                set(sources) == {"independent_compact_shape"}
                and anchor_color_similarity
                >= float(config["minimum_duplicate_color_similarity"])
            )
            multi_channel_residual = len(
                {
                    value
                    for value in sources
                    if value
                    in {
                        "condition_difference",
                        "temporal_motion",
                        "independent_compact_shape",
                    }
                }
            ) >= 2
            if (
                not multi_channel_residual
                and not independently_credible_duplicate
                and anchor_color_similarity
                < float(
                    config["minimum_weak_residual_color_similarity"]
                )
            ):
                continue
            detections_by_frame[frame_index].append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"residual_{frame_index:05d}_{residual_index:03d}"
                    ),
                    xy=centroid,
                    area_px2=float(area),
                    entity_class=entity_class,
                    mask=mask,
                    confidence=0.92,
                    evidence_tier=(
                        EvidenceTier.PARTICIPANT
                        if (
                            multi_channel_residual
                            or independently_credible_duplicate
                        )
                        else EvidenceTier.AMBIGUOUS
                    ),
                    sources=formal_sources,
                    metadata={
                        "compact_shape_score": _compact_shape_score(
                            mask, scene_kind=scene_kind
                        ),
                        "anchor_color_similarity": (
                            anchor_color_similarity
                        ),
                        "independently_credible_duplicate": (
                            independently_credible_duplicate
                        ),
                        "residual": True,
                    },
                )
            )
            source_counts["residual"] += 1
        previous_direct = current_direct
    observation = track_open_world_detections(
        detections_by_frame,
        time_grid=time_grid,
        maximum_gap_s=float(config["maximum_gap_s"]),
        maximum_assignment_cost=float(config["maximum_assignment_cost"]),
        minimum_scale_px=max(3.0, 0.35 * body_radius),
        maximum_tracks=int(config["maximum_tracks"]),
    )
    return OpenWorldObservation(
        tracks=observation.tracks,
        overflow_counts=observation.overflow_counts,
        diagnostics={
            **observation.diagnostics,
            "status": "ok",
            "observer_version": RIGID_BODY_OPEN_WORLD_VERSION,
            "scene_kind": scene_kind,
            "source_counts": source_counts,
            "roi_fraction": float(np.mean(roi > 0)),
            "apparatus_exclusion": (
                "scene_roi_plus_compact_shape_and_body_size_contract"
            ),
            "expected_channel": "condition_anchored_directed_sam2",
            "residual_channels": [
                "condition_difference",
                "bidirectional_temporal_motion",
                f"compact_{entity_class}",
            ],
            "rejected_candidates": rejected_candidates,
        },
    )


def observation_from_mask_channels(
    *,
    directed_masks: Sequence[np.ndarray],
    residual_instance_masks: Sequence[Sequence[np.ndarray]],
    entity_class: str,
    time_grid: CommonTimeGrid,
    available: Sequence[bool] | None = None,
    identity_valid: Sequence[bool] | None = None,
    maximum_tracks: int = 32,
) -> OpenWorldObservation:
    """Deterministic test/audit adapter for already segmented mask channels."""

    if not directed_masks:
        raise ValueError("directed mask channel must be non-empty")
    frame_count = len(directed_masks)
    if len(time_grid.times_s) != frame_count:
        raise ValueError("mask channels differ from the common time grid")
    if any(len(instance) != frame_count for instance in residual_instance_masks):
        raise ValueError("residual mask channels must share the common timeline")
    availability = (
        np.ones(frame_count, dtype=bool)
        if available is None
        else np.asarray(available, dtype=bool)
    )
    identities = (
        np.ones(frame_count, dtype=bool)
        if identity_valid is None
        else np.asarray(identity_valid, dtype=bool)
    )
    if availability.shape != (frame_count,) or identities.shape != (frame_count,):
        raise ValueError("availability/identity flags require one value per frame")
    detections: list[list[ObjectDetection]] = [
        [] for _ in range(frame_count)
    ]
    for frame_index in range(frame_count):
        if not availability[frame_index]:
            continue
        channels = [
            (
                directed_masks[frame_index],
                "directed",
                bool(identities[frame_index]),
            )
        ] + [
            (
                instance[frame_index],
                f"residual_{instance_index}",
                True,
            )
            for instance_index, instance in enumerate(residual_instance_masks)
        ]
        for channel_index, (mask, source, valid_identity) in enumerate(channels):
            binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
            centroid = mask_centroid(binary)
            area = int(np.count_nonzero(binary))
            if centroid is None or area <= 0:
                continue
            detections[frame_index].append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"audit_{frame_index:05d}_{channel_index:03d}"
                    ),
                    xy=centroid,
                    area_px2=float(area),
                    entity_class=(
                        entity_class
                        if valid_identity
                        else f"{entity_class}__replacement"
                    ),
                    mask=binary,
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=(source, "presegmented_audit"),
                )
            )
    return track_open_world_detections(
        detections,
        time_grid=time_grid,
        maximum_gap_s=0.25,
        maximum_assignment_cost=4.0,
        minimum_scale_px=2.0,
        maximum_tracks=maximum_tracks,
    )


def empty_fail_closed_observation(
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
            "policy": "fail_closed_no_directed_only_fallback",
        },
    )


def _track_detection_lookup(
    observation: OpenWorldObservation,
) -> dict[tuple[str, int], ObjectDetection]:
    return {
        (track.track_id, detection.frame_index): detection
        for track in observation.tracks
        for detection in track.detections
    }


def matched_prediction_channels(
    *,
    comparison: ObjectCentricComparisonV2,
    observation: OpenWorldObservation,
    entity_id: str,
    frame_count: int,
    shape: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    lookup = _track_detection_lookup(observation)
    matched_masks = [
        np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)
    ]
    union_masks = [
        np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)
    ]
    prediction_xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
    prediction_observed = np.zeros(frame_count, dtype=bool)
    for track in observation.tracks:
        if track.formal_exposure_weight <= 0.0:
            continue
        for detection in track.detections:
            if detection.mask is not None:
                union_masks[detection.frame_index] = cv2.bitwise_or(
                    union_masks[detection.frame_index],
                    detection.mask,
                )
    for match in comparison.matches:
        if match.entity_id != entity_id:
            continue
        detection = lookup.get((match.track_id, match.frame_index))
        if detection is None:
            continue
        prediction_xy[match.frame_index] = detection.xy
        prediction_observed[match.frame_index] = True
        if detection.mask is not None:
            matched_masks[match.frame_index] = detection.mask
    return matched_masks, union_masks, prediction_xy, prediction_observed


def _weighted_mean(
    values: np.ndarray,
    *,
    eligible: np.ndarray,
    time_grid: CommonTimeGrid,
) -> float:
    mask = np.asarray(eligible, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("weighted values and eligibility must share a shape")
    denominator = float(time_grid.cell_weights_s[mask].sum())
    if denominator <= 1e-12:
        return 0.0
    safe = np.where(np.isfinite(values), values, 0.0)
    return float(
        np.dot(safe[mask], time_grid.cell_weights_s[mask]) / denominator
    )


def score_rigid_body_state(
    *,
    reference: RigidBodyReference,
    prediction_xy: np.ndarray,
    prediction_observed: np.ndarray,
    prediction_masks: Sequence[np.ndarray],
    axis: FrozenRigidAxis,
    time_grid: CommonTimeGrid,
    scene_kind: str,
    scoring_config: Mapping[str, Any],
    coverage_power: float = 1.0,
) -> dict[str, Any]:
    """Score state on the complete reference exposure; absence always contributes zero."""

    expected = np.asarray(reference.expected, dtype=bool)
    observed = np.asarray(prediction_observed, dtype=bool) & expected
    if prediction_xy.shape != reference.xy.shape:
        raise ValueError("prediction and reference rigid-body tracks differ")
    along_prediction = np.full(len(expected), np.nan, dtype=np.float64)
    cross_prediction = np.full(len(expected), np.nan, dtype=np.float64)
    if observed.any():
        along, cross = axis.project(prediction_xy[observed])
        along_prediction[observed] = along / axis.span_px
        cross_prediction[observed] = cross / axis.span_px
    reference_progress = reference.normalized_progress
    trajectory_scale = max(
        float(scoring_config.get("trajectory_error_scale", 0.18)), 1e-6
    )
    trajectory_values = np.zeros(len(expected), dtype=np.float64)
    trajectory_values[observed] = np.exp(
        -np.abs(
            along_prediction[observed] - reference_progress[observed]
        )
        / trajectory_scale
    )
    trajectory_score = _weighted_mean(
        trajectory_values,
        eligible=expected,
        time_grid=time_grid,
    )

    acceleration_values = np.zeros(len(expected), dtype=np.float64)
    acceleration_eligible = np.zeros(len(expected), dtype=bool)
    times = time_grid.times_s
    acceleration_scale = max(
        float(scoring_config.get("acceleration_error_scale", 0.4)), 1e-6
    )
    for index in range(1, len(expected) - 1):
        if not np.all(expected[index - 1 : index + 2]):
            continue
        dt_first = times[index] - times[index - 1]
        dt_second = times[index + 1] - times[index]
        if min(dt_first, dt_second) <= 1e-12:
            continue
        acceleration_eligible[index] = True

        def second_difference(values: np.ndarray) -> float:
            first_velocity = (
                values[index] - values[index - 1]
            ) / dt_first
            second_velocity = (
                values[index + 1] - values[index]
            ) / dt_second
            return 2.0 * (second_velocity - first_velocity) / (
                dt_first + dt_second
            )

        reference_acceleration = second_difference(reference_progress)
        if not np.all(observed[index - 1 : index + 2]):
            acceleration_values[index] = 0.0
            continue
        prediction_acceleration = second_difference(along_prediction)
        denominator = max(abs(reference_acceleration), 0.25)
        relative_error = abs(
            prediction_acceleration - reference_acceleration
        ) / denominator
        acceleration_values[index] = math.exp(
            -relative_error / acceleration_scale
        )
    acceleration_score = _weighted_mean(
        acceleration_values,
        eligible=acceleration_eligible,
        time_grid=time_grid,
    )

    event_threshold = 0.95 * float(
        np.nanmax(reference_progress[expected])
    )
    reference_reached = np.flatnonzero(
        expected & (reference_progress >= event_threshold)
    )
    prediction_reached = np.flatnonzero(
        observed & (along_prediction >= event_threshold)
    )
    reference_event = (
        float(times[reference_reached[0]])
        if len(reference_reached)
        else float(times[np.flatnonzero(expected)[-1]])
    )
    prediction_event = (
        float(times[prediction_reached[0]])
        if len(prediction_reached)
        else None
    )
    event_scale = max(
        float(
            scoring_config.get(
                (
                    "impact_time_error_scale"
                    if scene_kind == "free_fall"
                    else "descent_time_error_scale"
                ),
                0.18,
            )
        ),
        1e-6,
    )
    duration = max(time_grid.duration_s, 1e-9)
    event_score = (
        0.0
        if prediction_event is None
        else math.exp(
            -abs(prediction_event - reference_event)
            / duration
            / event_scale
        )
    )

    constraint_values = np.zeros(len(expected), dtype=np.float64)
    if scene_kind == "free_fall":
        cross_scale = max(
            float(scoring_config.get("horizontal_drift_scale", 0.08)),
            1e-6,
        )
        cross_values = np.zeros(len(expected), dtype=np.float64)
        cross_values[observed] = np.exp(
            -np.abs(cross_prediction[observed]) / cross_scale
        )
        monotonic_values = np.zeros(len(expected), dtype=np.float64)
        for index in range(1, len(expected)):
            if (
                expected[index]
                and expected[index - 1]
                and observed[index]
                and observed[index - 1]
            ):
                monotonic_values[index] = float(
                    along_prediction[index]
                    >= along_prediction[index - 1] - 0.02
                )
        constraint_values = 0.65 * cross_values + 0.35 * monotonic_values
    else:
        cross_scale = max(
            float(scoring_config.get("cross_track_scale", 0.05)), 1e-6
        )
        cross_values = np.zeros(len(expected), dtype=np.float64)
        cross_values[observed] = np.exp(
            -np.abs(cross_prediction[observed]) / cross_scale
        )
        monotonic_values = np.zeros(len(expected), dtype=np.float64)
        orientation_values = np.zeros(len(expected), dtype=np.float64)
        reference_orientations = [
            _mask_orientation(mask) for mask in reference.masks
        ]
        prediction_orientations = [
            _mask_orientation(mask) for mask in prediction_masks
        ]
        orientation_scale = max(
            float(
                scoring_config.get("orientation_std_scale_deg", 12.0)
            ),
            1e-6,
        )
        for index in range(len(expected)):
            if observed[index]:
                difference = _orientation_delta_deg(
                    reference_orientations[index],
                    prediction_orientations[index],
                )
                orientation_values[index] = (
                    0.0
                    if difference is None
                    else math.exp(-difference / orientation_scale)
                )
            if (
                index > 0
                and expected[index]
                and expected[index - 1]
                and observed[index]
                and observed[index - 1]
            ):
                monotonic_values[index] = float(
                    along_prediction[index]
                    >= along_prediction[index - 1] - 0.02
                )
        constraint_values = (
            0.5 * cross_values
            + 0.3 * monotonic_values
            + 0.2 * orientation_values
        )
    constraint_score = _weighted_mean(
        constraint_values,
        eligible=expected,
        time_grid=time_grid,
    )
    coverage = (
        float(time_grid.cell_weights_s[observed].sum())
        / max(float(time_grid.cell_weights_s[expected].sum()), 1e-12)
    )
    components = {
        (
            "vertical_trajectory"
            if scene_kind == "free_fall"
            else "along_plane_trajectory"
        ): trajectory_score,
        "normalized_acceleration": acceleration_score,
        (
            "impact_time"
            if scene_kind == "free_fall"
            else "descent_time"
        ): event_score,
        (
            "motion_constraints"
            if scene_kind == "free_fall"
            else "contact_and_pose_constraints"
        ): constraint_score,
    }
    configured_weights = {
        key: float(value)
        for key, value in scoring_config.get("weights", {}).items()
    }
    weights = {
        name: configured_weights.get(name, 1.0)
        for name in components
    }
    denominator = max(sum(weights.values()), 1e-12)
    raw_score = sum(
        weights[name] * value for name, value in components.items()
    ) / denominator
    score = float(
        np.clip(
            raw_score * coverage ** max(float(coverage_power), 0.0),
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "components": components,
        "raw_complete_exposure_score": float(raw_score),
        "matched_reference_exposure_ratio": coverage,
        "coverage_power": float(coverage_power),
        "reference_event_time_s": reference_event,
        "prediction_event_time_s": prediction_event,
        "weights_used": {
            name: weights[name] / denominator for name in components
        },
        "axis": axis.to_dict(),
        "prediction_axis_refit": False,
        "missing_sample_policy": "zero_on_full_reference_exposure",
    }


def _mask_orientation(mask: np.ndarray) -> float | None:
    points = cv2.findNonZero(
        np.where(np.asarray(mask) > 0, 1, 0).astype(np.uint8)
    )
    if points is None or len(points) < 5:
        return None
    (_, _), (width, height), angle = cv2.minAreaRect(points)
    if width < height:
        angle += 90.0
    return float(angle % 180.0)


def _orientation_delta_deg(
    first: float | None,
    second: float | None,
) -> float | None:
    if first is None or second is None:
        return None
    raw = abs(first - second) % 180.0
    return min(raw, 180.0 - raw)


def resummarize_subject_on_expected(
    subject: SubjectComparison,
    *,
    expected: np.ndarray,
    time_grid: CommonTimeGrid,
) -> SubjectComparison:
    """Exclude lifecycle-valid post-exit cells without hiding pre-exit absence."""

    eligible = np.asarray(expected, dtype=bool)
    if eligible.shape != (len(subject.per_frame),):
        raise ValueError("subject rows differ from expected lifecycle timeline")
    component_names = tuple(subject.components)
    components: dict[str, float | None] = {}
    for name in component_names:
        values = np.asarray(
            [
                (
                    np.nan
                    if row.get(name) is None
                    else float(row[name])
                )
                for row in subject.per_frame
            ],
            dtype=np.float64,
        )
        valid = eligible & np.isfinite(values)
        components[name] = (
            _weighted_mean(values, eligible=valid, time_grid=time_grid)
            if valid.any()
            else None
        )
    score_name = (
        "appearance"
        if subject.comparable_reference
        != "same_case_ground_truth_video"
        else "subject"
    )
    scores = np.asarray(
        [
            (
                np.nan
                if row.get(score_name) is None
                else float(row[score_name])
            )
            for row in subject.per_frame
        ],
        dtype=np.float64,
    )
    valid_scores = eligible & np.isfinite(scores)
    score = (
        _weighted_mean(scores, eligible=valid_scores, time_grid=time_grid)
        if valid_scores.any()
        else 0.0
    )
    return SubjectComparison(
        score=float(np.clip(score, 0.0, 1.0)),
        components=components,
        per_frame=subject.per_frame,
        reference_observed_ratio=subject.reference_observed_ratio,
        prediction_observed_ratio=subject.prediction_observed_ratio,
        comparable_reference=subject.comparable_reference,
    )


def evaluate_rigid_body_open_world(
    *,
    reference: RigidBodyReference,
    scoring_reference: RigidBodyReference,
    observation: OpenWorldObservation,
    comparison: ObjectCentricComparisonV2,
    time_grid: CommonTimeGrid,
    frame_shape: tuple[int, int],
    scene_kind: str,
    scoring_config: Mapping[str, Any],
    observer_config: Mapping[str, Any],
    subject: SubjectComparison | None = None,
) -> RigidBodyOpenWorldResult:
    (
        matched_masks,
        union_masks,
        prediction_xy,
        prediction_observed,
    ) = matched_prediction_channels(
        comparison=comparison,
        observation=observation,
        entity_id=reference.entity_id,
        frame_count=len(time_grid.times_s),
        shape=frame_shape,
    )
    state = score_rigid_body_state(
        reference=reference,
        prediction_xy=prediction_xy,
        prediction_observed=prediction_observed,
        prediction_masks=matched_masks,
        axis=scoring_reference.axis,
        time_grid=time_grid,
        scene_kind=scene_kind,
        scoring_config=scoring_config,
        coverage_power=float(observer_config["state_coverage_power"]),
    )
    if subject is None:
        subject_components = {"shape": 1.0, "appearance": 1.0}
    else:
        subject_components = {
            "shape": subject.components.get("shape"),
            "appearance": subject.components.get("appearance"),
        }
    content_components = {
        "physics_state": float(state["score"]),
        **subject_components,
    }
    composition_value = compose_object_centric_v2(
        comparison,
        content_components=content_components,
        content_weights={
            key: float(value)
            for key, value in observer_config["content_weights"].items()
        },
    ).to_dict()
    composition = {
        **composition_value,
        "score": float(composition_value["score"] or 0.0),
        "content_components": content_components,
    }
    ious = [
        observed_mask_iou(reference.masks[index], union_masks[index])
        for index in range(len(reference.masks))
    ]
    subject_rows = (
        subject.per_frame
        if subject is not None
        else [{} for _ in reference.masks]
    )
    per_frame: list[Mapping[str, Any]] = []
    for index, comparison_row in enumerate(comparison.per_frame):
        row = {
            "frame": index,
            "time_s": float(time_grid.times_s[index]),
            "reference_expected": bool(reference.expected[index]),
            "reference_raw_observed": bool(
                reference.raw_observed[index]
            ),
            "prediction_matched": bool(prediction_observed[index]),
            "reference_progress_normalized": (
                float(reference.normalized_progress[index])
                if reference.expected[index]
                else None
            ),
            "prediction_progress_normalized": (
                float(
                    scoring_reference.axis.project(
                        prediction_xy[index : index + 1]
                    )[0][0]
                    / scoring_reference.axis.span_px
                )
                if prediction_observed[index]
                else None
            ),
            "physical_subject_iou": ious[index],
            "missing_entity_ids": comparison_row["missing_entity_ids"],
            "residual_track_ids": comparison_row["extra_track_ids"],
            "rejected_candidate_matches": comparison_row[
                "rejected_candidate_matches"
            ],
            "formal_prediction_cardinality": comparison_row[
                "formal_prediction_cardinality"
            ],
            "overflow_count": comparison_row["overflow_count"],
            "position_diagnostic_score": float(
                comparison_row["position_score"] or 0.0
            ),
            "matched_track_id": (
                comparison_row["matches"][0]["prediction_track_id"]
                if comparison_row["matches"]
                else None
            ),
            "subject_shape": subject_rows[index].get("shape"),
            "subject_appearance": subject_rows[index].get("appearance"),
        }
        per_frame.append(row)
    return RigidBodyOpenWorldResult(
        observation=observation,
        comparison=comparison,
        matched_prediction_masks=tuple(matched_masks),
        prediction_union_masks=tuple(union_masks),
        prediction_xy=prediction_xy,
        prediction_observed=prediction_observed,
        state_metric=state,
        composition=composition,
        per_frame=tuple(per_frame),
    )


def load_condition_frame(
    request: CaseEvaluationRequest,
    *,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    value = request.case.get("assets", {}).get("first_frame")
    if not isinstance(value, str) or not value:
        raise ReferenceAnalysisError(
            "reference_condition_frame_missing",
            "parent-reference case has no first_frame asset",
        )
    asset_root = request.asset_root.resolve()
    path = (asset_root / value).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            "reference_condition_path_escape",
            f"conditioned first frame escapes dataset root: {value}",
        ) from exc
    source = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if source is None:
        raise ReferenceAnalysisError(
            "reference_condition_frame_unreadable",
            f"cannot read conditioned first frame: {path}",
        )
    target_height, target_width = target_shape[:2]
    source_height, source_width = source.shape[:2]
    scale = min(
        target_width / max(source_width, 1),
        target_height / max(source_height, 1),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        source,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    output = np.full(
        (target_height, target_width, 3),
        int(request.evaluator_config["spatial"].get("pad_value", 0)),
        dtype=np.uint8,
    )
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    output[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return output, {
        "condition_frame": str(path),
        "condition_frame_sha256": sha256_file(path),
        "condition_mask_source": "condition_side_directed_segmentation",
        "condition_spatial_transform": {
            "policy": "preserve_aspect_ratio_letterbox",
            "scale": scale,
            "offset_xy": [offset_x, offset_y],
            "source_size": [source_width, source_height],
            "target_size": [target_width, target_height],
        },
    }


def write_rigid_body_audit_json(
    path: Path,
    *,
    reference: RigidBodyReference,
    expected_timeline: ExpectedEntityTimeline,
    result: RigidBodyOpenWorldResult,
    prediction_failures: Sequence[Mapping[str, str]],
) -> None:
    value = {
        "observer_version": RIGID_BODY_OPEN_WORLD_VERSION,
        "entity_id": reference.entity_id,
        "entity_class": reference.entity_class,
        "axis": reference.axis.to_dict(),
        "lifecycle": dict(reference.lifecycle_diagnostics),
        "expected_timeline": expected_timeline.to_dict(),
        "open_world_v2": result.comparison.to_dict(),
        "state": dict(result.state_metric),
        "composition": dict(result.composition),
        "prediction_failures": list(prediction_failures),
        "observation": {
            "diagnostics": dict(result.observation.diagnostics),
            "overflow_counts": (
                result.observation.overflow_counts.tolist()
            ),
            "tracks": [
                {
                    "track_id": track.track_id,
                    "entity_class": track.entity_class,
                    "evidence_tier": track.evidence_tier.value,
                    "formal_exposure_weight": (
                        track.formal_exposure_weight
                    ),
                    "confirmed": track.confirmed,
                    "detections": [
                        {
                            "frame": detection.frame_index,
                            "detection_id": detection.detection_id,
                            "xy": detection.xy.tolist(),
                            "area_px2": detection.area_px2,
                            "confidence": detection.confidence,
                            "sources": list(detection.sources),
                            "metadata": dict(detection.metadata),
                        }
                        for detection in track.detections
                    ],
                }
                for track in result.observation.tracks
            ],
        },
        "per_frame": list(result.per_frame),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _prompt_from_mask(
    mask: np.ndarray,
    *,
    frame_index: int,
    box_expand: float,
    minimum_box_side: int,
    source: str,
) -> MaskPrompt:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(binary)
    centroid = mask_centroid(binary)
    if points is None or centroid is None:
        raise SceneAnalysisError(
            "rigid_body_anchor_mask_empty",
            "cannot build a directed prompt from an empty anchor mask",
        )
    x, y, width, height = cv2.boundingRect(points)
    frame_height, frame_width = binary.shape
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    expanded_width = max(float(minimum_box_side), width * box_expand)
    expanded_height = max(float(minimum_box_side), height * box_expand)
    box = np.asarray(
        [
            max(0.0, center_x - expanded_width / 2.0),
            max(0.0, center_y - expanded_height / 2.0),
            min(frame_width - 1.0, center_x + expanded_width / 2.0),
            min(frame_height - 1.0, center_y + expanded_height / 2.0),
        ],
        dtype=np.float32,
    )
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=box,
        points_xy=np.asarray(centroid, dtype=np.float32).reshape(1, 2),
        point_labels=np.ones(1, dtype=np.int32),
        metadata={
            "source": source,
            "anchor_box_xywh": [x, y, width, height],
        },
    )


def _degraded_subject(
    frame_count: int,
    *,
    reference_mode: str,
) -> SubjectComparison:
    return SubjectComparison(
        score=0.0,
        components={"position": 0.0, "shape": 0.0, "appearance": 0.0},
        per_frame=[
            {
                "frame": index,
                "position": 0.0,
                "shape": 0.0,
                "appearance": 0.0,
                "subject": 0.0,
            }
            for index in range(frame_count)
        ],
        reference_observed_ratio=1.0,
        prediction_observed_ratio=0.0,
        comparable_reference=(
            "same_case_ground_truth_video"
            if reference_mode == "same_case_reference"
            else "case_condition_first_frame_appearance_only"
        ),
    )


class RigidBodyOpenWorldCaseEvaluatorBase(ReferenceCaseEvaluator):
    """Shared v6 implementation for one-body fall and incline scenes."""

    evaluator_version = "2.0"
    sequential_evaluator_version = "2.0"
    robust_evaluator_version = "2.0"
    allow_partial_prediction = True
    scene_kind: str
    entity_class: str
    open_world_metric: str
    scene_name: str
    temporary_prefix: str
    minimum_span_quality_key: str

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])
        self._observer_config = merged_rigid_body_config(
            config, scene_kind=self.scene_kind
        )
        self.fingerprint = rigid_body_fingerprint_payload(
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            config=config,
            scene_kind=self.scene_kind,
        )

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": f"manifest_declared_{self.entity_class}",
            "segmentation": self._segmenter.describe(),
            "expected_channel": "condition_anchored_directed_sam2",
            "residual_channels": [
                "condition_difference",
                "bidirectional_temporal_motion",
                f"compact_{self.entity_class}",
            ],
            "identity": "causal_open_world_tracks_with_null_assignment",
            "open_world_protocol": "open_world_v2",
            "coordinate_system": (
                "reference_vertical_axis_no_prediction_refit"
                if self.scene_kind == "free_fall"
                else "reference_or_condition_plane_axis_no_prediction_refit"
            ),
            "lifecycle": (
                "may_exit_only_after_reference_terminal_event; "
                "reappearance_is_extra"
            ),
            "case_score": "entity_integrity_gate_times_scene_content",
        }

    def _manifest_entity(self, request: CaseEvaluationRequest):
        try:
            manifest = materialize_entity_manifest(request.case)
        except ReferenceAnalysisError:
            raise
        except ValueError as exc:
            raise ReferenceAnalysisError(
                "reference_entity_manifest_invalid",
                f"{self.scene_id} entity manifest is invalid: {exc}",
            ) from exc
        if manifest.scene_id != self.scene_id:
            raise ReferenceAnalysisError(
                "reference_entity_manifest_scene_mismatch",
                f"{self.scene_id} evaluator received manifest for "
                f"{manifest.scene_id}",
            )
        if len(manifest.entities) != 1:
            raise ReferenceAnalysisError(
                "reference_rigid_body_cardinality_invalid",
                f"{self.scene_id} requires exactly one manifest entity; "
                f"received {len(manifest.entities)}",
            )
        entity = manifest.entities[0]
        if entity.entity_class != self.entity_class:
            raise ReferenceAnalysisError(
                "reference_rigid_body_class_invalid",
                f"{self.scene_id} requires entity class {self.entity_class!r}; "
                f"received {entity.entity_class!r}",
            )
        return manifest, entity

    def _reference_masks(
        self,
        frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        proposal = self.config["motion_proposal"]
        prompt = build_motion_prompt(
            frames,
            threshold=float(proposal["threshold"]),
            minimum_area=int(proposal["minimum_area"]),
            box_expand=float(proposal["box_expand"]),
            minimum_box_side=int(proposal["minimum_box_side"]),
        )
        masks, segmentation = self._segmenter.segment(
            frames,
            prompt=prompt,
            temporary_prefix=f"{self.temporary_prefix}reference_",
        )
        return masks, {
            "prompt_builder": {
                "type": "reference_temporal_motion",
                "prompt_frame": prompt.frame_index,
                "prompt_metadata": prompt.metadata,
            },
            "segmentation": segmentation,
        }

    def _condition_mask(
        self,
        condition_frame: np.ndarray,
        *,
        reference_anchor_mask: np.ndarray,
        same_case_reference: bool,
    ) -> tuple[np.ndarray, MaskPrompt, dict[str, Any]]:
        proposal = self.config["motion_proposal"]
        anchor_prompt = _prompt_from_mask(
            reference_anchor_mask,
            frame_index=0,
            box_expand=float(proposal["box_expand"]),
            minimum_box_side=int(proposal["minimum_box_side"]),
            source=(
                "same_case_reference_frame_zero"
                if same_case_reference
                else "physics_parent_normalized_anchor"
            ),
        )
        if same_case_reference:
            condition_mask = np.where(
                reference_anchor_mask > 0, 255, 0
            ).astype(np.uint8)
            metadata = {
                "status": "reused_reference_frame_zero_mask",
                "prompt": anchor_prompt.metadata,
            }
        else:
            values, segmentation = self._segmenter.segment(
                [condition_frame],
                prompt=anchor_prompt,
                temporary_prefix=f"{self.temporary_prefix}condition_",
            )
            condition_mask = values[0]
            metadata = {
                "status": "condition_side_segmented",
                "segmentation": segmentation,
            }
        if (
            int(np.count_nonzero(condition_mask))
            < int(self.config["quality"]["minimum_mask_pixels"])
        ):
            raise ReferenceAnalysisError(
                "reference_condition_subject_unobserved",
                "condition-side directed segmentation did not recover the "
                f"manifest {self.entity_class}",
            )
        condition_prompt = _prompt_from_mask(
            condition_mask,
            frame_index=0,
            box_expand=float(proposal["box_expand"]),
            minimum_box_side=int(proposal["minimum_box_side"]),
            source="condition_subject_frozen_anchor",
        )
        return condition_mask, condition_prompt, metadata

    def _prediction_masks(
        self,
        frames: list[np.ndarray],
        *,
        condition_prompt: MaskPrompt,
        available: np.ndarray,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        available_indices = np.flatnonzero(available)
        if not len(available_indices):
            raise SceneAnalysisError(
                "rigid_body_prediction_unavailable",
                "prediction has no available common-time frame",
            )
        seed = int(available_indices[0])
        prompt = MaskPrompt(
            frame_index=seed,
            box_xyxy=condition_prompt.box_xyxy,
            points_xy=condition_prompt.points_xy,
            point_labels=condition_prompt.point_labels,
            metadata={
                **condition_prompt.metadata,
                "seed_frame_adjusted_for_availability": seed,
            },
        )
        masks, segmentation = self._segmenter.segment(
            frames,
            prompt=prompt,
            temporary_prefix=f"{self.temporary_prefix}prediction_",
        )
        for index, is_available in enumerate(available):
            if not is_available:
                masks[index] = np.zeros_like(masks[index])
        return masks, segmentation

    def _condition_axis(
        self,
        *,
        reference: RigidBodyReference,
        condition_frame: np.ndarray,
        condition_mask: np.ndarray,
        reference_mode: str,
    ) -> FrozenRigidAxis:
        if reference_mode == "same_case_reference":
            return reference.axis
        centroid = mask_centroid(condition_mask)
        if centroid is None:
            raise ReferenceAnalysisError(
                "reference_condition_subject_unobserved",
                "condition subject mask has no centroid",
            )
        if self.scene_kind == "free_fall":
            radius = math.sqrt(
                max(int(np.count_nonzero(condition_mask)), 1) / math.pi
            )
            span = max(
                condition_frame.shape[0] - centroid[1] - radius,
                0.5 * reference.axis.span_px,
                1.0,
            )
            return FrozenRigidAxis(
                origin_xy=centroid,
                direction_xy=np.asarray([0.0, 1.0]),
                normal_xy=np.asarray([-1.0, 0.0]),
                span_px=span,
                source="condition_vertical_gravity_axis",
                explained_ratio=1.0,
            )
        return freeze_condition_incline_axis(
            condition_frame,
            condition_centroid_xy=centroid,
            fallback=reference.axis,
        )

    def _write_open_world_visualization(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: Sequence[float],
        reference_frames: Sequence[np.ndarray],
        prediction_frames: Sequence[np.ndarray],
        expected_timeline: ExpectedEntityTimeline,
        observation: OpenWorldObservation,
        comparison: ObjectCentricComparisonV2,
        reference_union_masks: Sequence[np.ndarray],
        prediction_union_masks: Sequence[np.ndarray],
        full_subject_ious: Sequence[float | None],
        prediction_available: Sequence[bool],
    ) -> dict[str, Any]:
        """Write diagnostics after scoring without extending its failure surface."""

        try:
            return write_open_world_v2_artifacts(
                request,
                scene_name=self.scene_name,
                times_s=times_s,
                reference_frames=reference_frames,
                prediction_frames=prediction_frames,
                expected_timelines=[expected_timeline],
                prediction_observation=observation,
                comparison=comparison,
                config=self.config.get("visualization", {}),
                reference_union_masks=reference_union_masks,
                prediction_union_masks=prediction_union_masks,
                full_subject_ious=full_subject_ious,
                prediction_available=prediction_available,
            )
        except Exception:
            # Even a local-manifest write failure is an artifact failure, not
            # evidence about generated physics or subject integrity.
            return {}

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        manifest, entity = self._manifest_entity(request)
        time_grid = build_common_time_grid(times_s)
        quality = self.config["quality"]
        prediction_failures: list[dict[str, str]] = []
        try:
            raw_reference_masks, reference_observation = (
                self._reference_masks(reference_video.frames)
            )
            reference = build_rigid_body_reference(
                raw_reference_masks,
                entity_id=entity.entity_id,
                entity_class=entity.entity_class,
                scene_kind=self.scene_kind,
                frame_shape=reference_video.frames[0].shape[:2],
                minimum_area=int(quality["minimum_mask_pixels"]),
                maximum_area_ratio=float(
                    quality["maximum_mask_area_ratio"]
                ),
                minimum_span_px=float(
                    quality[self.minimum_span_quality_key]
                ),
                config=self._observer_config,
            )
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_rigid_body_open_world_observation_failed",
                f"{self.scene_name} reference observation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        reference_mode = infer_reference_mode(request.case)
        capability = manifest.reference_capability
        expected_mode = (
            "same_case_reference"
            if capability is ReferenceCapability.SAME_CASE_GT
            else "parent_physics_reference"
        )
        if reference_mode != expected_mode:
            raise ReferenceAnalysisError(
                "reference_capability_mode_mismatch",
                "case reference mode disagrees with the entity manifest "
                f"capability {capability.value}",
            )
        condition_provenance: dict[str, Any] = {}
        if reference_mode == "same_case_reference":
            condition_frame = reference_video.frames[0]
        else:
            condition_frame, condition_provenance = load_condition_frame(
                request,
                target_shape=reference_video.frames[0].shape,
            )
        condition_mask, condition_prompt, condition_observation = (
            self._condition_mask(
                condition_frame,
                reference_anchor_mask=reference.masks[0],
                same_case_reference=(
                    reference_mode == "same_case_reference"
                ),
            )
        )
        condition_axis = self._condition_axis(
            reference=reference,
            condition_frame=condition_frame,
            condition_mask=condition_mask,
            reference_mode=reference_mode,
        )
        scoring_reference = (
            reference
            if reference_mode == "same_case_reference"
            else remap_reference_to_condition(
                reference,
                condition_axis=condition_axis,
                condition_area_px2=float(np.count_nonzero(condition_mask)),
            )
        )
        expected_timeline = build_expected_rigid_body_timeline(
            reference=reference,
            scoring_reference=scoring_reference,
            manifest_entity=entity,
            time_grid=time_grid,
            capability=capability,
            condition_mask=condition_mask,
        )
        prediction_frames = list(prediction_video.frames)
        available = np.asarray(
            (
                prediction_video.available
                if prediction_video.available is not None
                else [True] * len(prediction_frames)
            ),
            dtype=bool,
        )
        if not bool(np.all(available)):
            prediction_failures.append(
                {
                    "code": "prediction_partial_timeline",
                    "reason": (
                        "prediction supplies "
                        f"{int(available.sum())}/{len(available)} usable "
                        "common-time samples; unavailable cells are scored "
                        "as missing"
                    ),
                }
            )
        try:
            prediction_masks, directed_observation = (
                self._prediction_masks(
                    prediction_frames,
                    condition_prompt=condition_prompt,
                    available=available,
                )
            )
        except Exception as exc:
            failure = {
                "code": getattr(
                    exc, "code", "prediction_directed_observation_failed"
                ),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            prediction_masks = [
                np.zeros(reference_video.frames[0].shape[:2], dtype=np.uint8)
                for _ in times_s
            ]
            directed_observation = {
                "status": "failed_but_residual_discovery_continued",
                **failure,
            }
        def prediction_factory() -> OpenWorldObservation:
            return discover_rigid_body_objects(
                prediction_frames,
                directed_masks=prediction_masks,
                condition_frame=condition_frame,
                condition_mask=condition_mask,
                reference_axis=condition_axis,
                entity_class=entity.entity_class,
                scene_kind=self.scene_kind,
                time_grid=time_grid,
                available=available,
                minimum_area=int(quality["minimum_mask_pixels"]),
                maximum_area_ratio=float(
                    quality["maximum_mask_area_ratio"]
                ),
                config=self._observer_config,
            )

        comparison, observation = safe_compare_rigid_body_open_world(
            expected_timeline=expected_timeline,
            prediction_factory=prediction_factory,
            time_grid=time_grid,
            frame_shape=reference_video.frames[0].shape[:2],
            minimum_match_position_similarity=float(
                self._observer_config[
                    "minimum_match_position_similarity"
                ]
            ),
        )
        if comparison.failed:
            failure = {
                "code": "prediction_open_world_v2_failed_closed",
                "reason": (
                    comparison.failure_reason
                    or "open-world v2 comparison failed closed"
                ),
            }
            prediction_failures.append(failure)
        initial = evaluate_rigid_body_open_world(
            reference=reference,
            scoring_reference=scoring_reference,
            observation=observation,
            comparison=comparison,
            time_grid=time_grid,
            frame_shape=reference_video.frames[0].shape[:2],
            scene_kind=self.scene_kind,
            scoring_config=self.config["scoring"],
            observer_config=self._observer_config,
        )
        try:
            subject = compare_subjects(
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                reference_masks=list(reference.masks),
                prediction_masks=list(initial.matched_prediction_masks),
                reference_mode=reference_mode,
                config=self.config["subject_scoring"],
                condition_frame=(
                    condition_frame
                    if reference_mode == "parent_physics_reference"
                    else None
                ),
                condition_mask=(
                    condition_mask
                    if reference_mode == "parent_physics_reference"
                    else None
                ),
            )
            subject = resummarize_subject_on_expected(
                subject,
                expected=reference.expected,
                time_grid=time_grid,
            )
        except Exception as exc:
            failure = {
                "code": "prediction_rigid_body_subject_comparison_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            subject = _degraded_subject(
                len(times_s), reference_mode=reference_mode
            )
        result = evaluate_rigid_body_open_world(
            reference=reference,
            scoring_reference=scoring_reference,
            observation=observation,
            comparison=comparison,
            time_grid=time_grid,
            frame_shape=reference_video.frames[0].shape[:2],
            scene_kind=self.scene_kind,
            scoring_config=self.config["scoring"],
            observer_config=self._observer_config,
            subject=subject,
        )
        score = float(result.composition["score"])
        if not math.isfinite(score):
            prediction_failures.append(
                {
                    "code": "prediction_non_finite_case_score",
                    "reason": "rigid-body case composition was non-finite",
                }
            )
            score = 0.0
        ious = [
            observed_mask_iou(
                reference.masks[index],
                result.prediction_union_masks[index],
            )
            for index in range(len(times_s))
        ]
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        position_path = request.artifact_dir / "entity_position_curve.png"
        cardinality_path = (
            request.artifact_dir / "object_cardinality_timeline.png"
        )
        trajectory_path = request.artifact_dir / (
            "vertical_trajectory_curve.png"
            if self.scene_kind == "free_fall"
            else "along_plane_trajectory_curve.png"
        )
        audit_path = request.artifact_dir / "open_world_audit.json"
        write_rows_csv(csv_path, [dict(row) for row in result.per_frame])
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=ious,
            case_id=request.case["case_id"],
            scene_name=self.scene_name,
        )
        save_iou_curve(
            position_path,
            times_s=times_s,
            ious=[
                float(row["position_diagnostic_score"])
                for row in result.per_frame
            ],
            case_id=request.case["case_id"],
            scene_name=f"{self.scene_name} entity position",
            series_label="Continuous matched/nearest-rejected position score",
            y_label="Position similarity",
            metric_name="score",
        )
        save_series_comparison(
            cardinality_path,
            times_s=times_s,
            reference=reference.expected.astype(np.float64),
            prediction=np.asarray(
                [
                    float(row["formal_prediction_cardinality"])
                    for row in result.per_frame
                ],
                dtype=np.float64,
            ),
            ylabel="Observed physical objects",
            title=(
                f"{self.scene_name} object cardinality — "
                f"{request.case['case_id']}"
            ),
        )
        prediction_progress = np.asarray(
            [
                (
                    np.nan
                    if row["prediction_progress_normalized"] is None
                    else float(row["prediction_progress_normalized"])
                )
                for row in result.per_frame
            ],
            dtype=np.float64,
        )
        save_series_comparison(
            trajectory_path,
            times_s=times_s,
            reference=reference.normalized_progress,
            prediction=prediction_progress,
            ylabel="Normalized frozen-axis displacement",
            title=(
                f"{self.scene_name} frozen-axis trajectory — "
                f"{request.case['case_id']}"
            ),
        )
        write_rigid_body_audit_json(
            audit_path,
            reference=reference,
            expected_timeline=expected_timeline,
            result=result,
            prediction_failures=prediction_failures,
        )
        visualization_artifacts = self._write_open_world_visualization(
            request,
            times_s=times_s,
            reference_frames=reference_video.frames,
            prediction_frames=prediction_frames,
            expected_timeline=expected_timeline,
            observation=observation,
            comparison=comparison,
            reference_union_masks=reference.masks,
            prediction_union_masks=result.prediction_union_masks,
            full_subject_ious=ious,
            prediction_available=available.tolist(),
        )
        primary = {
            **result.composition,
            "score": score,
        }
        return SceneAnalysis(
            score=score,
            metrics={
                "scene_subject_state_similarity": dict(primary),
                self.open_world_metric: primary,
                "object_centric_integrity": (
                    result.comparison.integrity.to_dict()
                ),
                "object_centric_protocol": {
                    "id": result.comparison.protocol_id,
                    "version": result.comparison.protocol_version,
                    "failed": result.comparison.failed,
                    "expected_timeline": expected_timeline.to_dict(),
                },
                (
                    "free_fall_state_similarity"
                    if self.scene_kind == "free_fall"
                    else "inclined_plane_state_similarity"
                ): dict(result.state_metric),
                "physical_subject_similarity": subject.to_metric(
                    weights={
                        key: float(value)
                        for key, value in self.config["subject_scoring"][
                            "weights"
                        ].items()
                    }
                ),
                "physical_subject_mask_iou": {
                    **summarize_mask_ious(ious),
                    "role": "full_open_world_set_visual_diagnostic",
                },
                "entity_manifest": manifest.to_canonical_dict(),
            },
            quality={
                "degraded": bool(prediction_failures),
                "degradation_codes": [
                    item["code"] for item in prediction_failures
                ],
                "degradation_reason": (
                    "; ".join(
                        item["reason"] for item in prediction_failures
                    )
                    if prediction_failures
                    else None
                ),
                "reference_valid_mask_ratio": float(
                    np.mean(reference.raw_observed)
                ),
                "prediction_matched_exposure_ratio": float(
                    result.state_metric[
                        "matched_reference_exposure_ratio"
                    ]
                ),
                "expected_entity_count": 1,
                "prediction_track_count": len(observation.tracks),
                "available_prediction_frames": int(available.sum()),
                "expected_prediction_frames": len(times_s),
                "legal_exit_frame": reference.legal_exit_frame,
                "prediction_axis_refit": False,
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "entity_position_curve": str(position_path),
                "object_cardinality_timeline": str(cardinality_path),
                "frozen_axis_trajectory_curve": str(trajectory_path),
                "open_world_audit": str(audit_path),
                **visualization_artifacts,
            },
            provenance={
                "entity_manifest": {
                    "materializer_id": manifest.materializer_id,
                    "digest": manifest.digest,
                },
                "observation": {
                    "reference": reference_observation,
                    "condition": condition_observation,
                    "prediction_directed": directed_observation,
                    "prediction_open_world": observation.diagnostics,
                },
                "frozen_axis": condition_axis.to_dict(),
                "reference_lifecycle": dict(
                    reference.lifecycle_diagnostics
                ),
                "identity_policy": (
                    "open_world_v2_condition_frozen_track_with_null_assignment"
                ),
                "residual_failure_policy": (
                    "fail_closed_empty_prediction_not_directed_only"
                ),
                "subject_reference_mode": reference_mode,
                **condition_provenance,
            },
        )


__all__ = [
    "RIGID_BODY_OPEN_WORLD_VERSION",
    "FrozenRigidAxis",
    "RigidBodyOpenWorldResult",
    "RigidBodyOpenWorldCaseEvaluatorBase",
    "RigidBodyReference",
    "build_rigid_body_reference",
    "default_rigid_body_config",
    "discover_rigid_body_objects",
    "empty_fail_closed_observation",
    "evaluate_rigid_body_open_world",
    "freeze_condition_incline_axis",
    "freeze_reference_axis",
    "load_condition_frame",
    "merged_rigid_body_config",
    "observation_from_mask_channels",
    "remap_reference_to_condition",
    "resummarize_subject_on_expected",
    "rigid_body_fingerprint_payload",
    "write_rigid_body_audit_json",
]
