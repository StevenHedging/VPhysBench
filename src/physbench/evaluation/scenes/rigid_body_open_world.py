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
    OpenWorldTrack,
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
from ..common.masks.motion import build_motion_prompt, motion_masks
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


def rigid_body_observer_version(config: Mapping[str, Any]) -> str:
    """Resolve the policy-frozen observer version without changing v6.

    The shared implementation serves both the frozen v6 protocol and the
    condition-causal v7 protocol.  V6 intentionally has no explicit observer
    version and must therefore continue to report ``2.0``; v7 freezes
    ``2.1`` inside ``rigid_body_observation``.  Accepting either a complete
    scene configuration or the already-merged observer configuration keeps
    diagnostics and fingerprints consistent.
    """

    nested = config.get("rigid_body_observation")
    source = nested if isinstance(nested, Mapping) else config
    value = source.get("observer_version", RIGID_BODY_OPEN_WORLD_VERSION)
    version = str(value).strip()
    if not version:
        raise ValueError("rigid-body observer_version must be non-empty")
    return version


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


def _tracking_partition_metadata(
    config: Mapping[str, Any],
    partition: str,
) -> dict[str, str]:
    """Opt into directed/residual identity isolation for v7 only.

    The shared rigid-body module also serves the frozen v6 evaluators.  An
    absent policy must therefore preserve the historical tracker behaviour
    byte-for-byte at the detection-contract level; only the v7 protocol may
    attach the partition marker understood by the common causal tracker.
    """

    if (
        str(config.get("exclusive_tracking_partition_policy", "disabled"))
        != "condition_directed_vs_residual_v1"
    ):
        return {}
    return {"exclusive_tracking_partition": partition}


def rigid_body_fingerprint_payload(
    *,
    evaluator_id: str,
    evaluator_version: str,
    config: Mapping[str, Any],
    scene_kind: str,
) -> str:
    observer_version = rigid_body_observer_version(config)
    return canonical_sha256(
        {
            "id": evaluator_id,
            "version": evaluator_version,
            "config": config,
            "entity_contract": "manifest_v1",
            "observer": (
                f"open_world_v2+rigid_body_observer_"
                f"v{observer_version}"
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


def _condition_incline_line_candidates(
    condition_frame: np.ndarray,
    *,
    condition_centroid_xy: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return long diagonal apparatus lines observed in the condition image.

    OpenCV has shipped both ``[N, 1, 4]`` and ``[N, 4]`` return layouts for
    ``HoughLinesP``.  Normalizing with ``reshape(-1, 4)`` is important here:
    indexing ``lines[:, 0]`` silently iterates scalars with the latter layout
    and used to make the condition-only incline path backend-dependent.
    """

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
    centroid = (
        None
        if condition_centroid_xy is None
        else np.asarray(condition_centroid_xy, dtype=np.float64)
    )
    candidates: list[dict[str, Any]] = []
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):
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
            distance = None
            forward_span = length
            oriented = np.array(direction, copy=True)
            if centroid is not None:
                distance = abs(
                    float(
                        np.cross(
                            np.append(oriented, 0.0),
                            np.append(centroid - first, 0.0),
                        )[2]
                    )
                )
                if distance > max(0.2 * height, 6.0):
                    continue
                endpoint = second if second[1] > first[1] else first
                downhill = endpoint - centroid
                if float(np.dot(oriented, downhill)) < 0.0:
                    oriented = -oriented
                forward_span = max(
                    float(np.dot(first - centroid, oriented)),
                    float(np.dot(second - centroid, oriented)),
                    1.0,
                )
            score = (
                length
                - 1.5 * float(distance or 0.0)
                + 0.25 * forward_span
            )
            candidates.append(
                {
                    "score": float(score),
                    "first_xy": first,
                    "second_xy": second,
                    "direction_xy": oriented,
                    "length_px": length,
                    "angle_deg": angle,
                    "distance_to_subject_px": distance,
                    "forward_span_px": forward_span,
                }
            )
    return candidates


def build_condition_incline_apparatus_seed(
    condition_frame: np.ndarray,
    *,
    minimum_area: int,
    maximum_area: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Propose the initial block from condition-side plane geometry only.

    The native incline captures place the released block on the uphill face
    of the long board.  A small rotated box just above the board is a prompt,
    not a final observation; SAM must still recover a plausible compact body.
    No reference-track or prediction coordinate enters this construction.
    """

    frame = np.asarray(condition_frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    diagonal = math.hypot(width, height)
    minimum_length = float(
        config.get(
            "condition_incline_seed_minimum_line_fraction",
            0.30,
        )
    ) * diagonal
    candidates = [
        item
        for item in _condition_incline_line_candidates(frame)
        if (
            float(item["length_px"]) >= minimum_length
            and 12.0 <= float(item["angle_deg"]) <= 70.0
        )
    ]
    if not candidates:
        return None, {
            "status": "apparatus_line_unobserved",
            "condition_future_pixels_used": False,
        }
    champion = max(candidates, key=lambda item: float(item["length_px"]))
    champion_direction = np.asarray(
        champion["direction_xy"], dtype=np.float64
    )
    if champion_direction[0] < 0.0:
        champion_direction = -champion_direction
    champion_angle = math.atan2(
        champion_direction[1], champion_direction[0]
    )
    tolerance = math.radians(
        float(
            config.get(
                "condition_incline_seed_angle_tolerance_deg",
                7.0,
            )
        )
    )
    parallel: list[dict[str, Any]] = []
    for item in candidates:
        direction = np.asarray(item["direction_xy"], dtype=np.float64)
        if direction[0] < 0.0:
            direction = -direction
        delta = abs(
            math.atan2(
                math.sin(
                    math.atan2(direction[1], direction[0])
                    - champion_angle
                ),
                math.cos(
                    math.atan2(direction[1], direction[0])
                    - champion_angle
                ),
            )
        )
        if delta <= tolerance:
            parallel.append(item)
    endpoints = np.vstack(
        [
            np.asarray(item[key], dtype=np.float64)
            for item in parallel
            for key in ("first_xy", "second_xy")
        ]
    )
    along_values = endpoints @ champion_direction
    lower = float(np.min(along_values))
    upper = float(np.max(along_values))
    span = upper - lower
    if span < minimum_length:
        return None, {
            "status": "apparatus_span_too_short",
            "observed_span_px": span,
            "minimum_span_px": minimum_length,
            "condition_future_pixels_used": False,
        }
    # ``champion_direction`` points uphill (right/up in the native captures).
    # Select the upward-facing normal deterministically in image coordinates.
    normal = np.asarray(
        [-champion_direction[1], champion_direction[0]],
        dtype=np.float64,
    )
    if normal[1] > 0.0:
        normal = -normal
    normal_offsets = endpoints @ normal
    surface_offset = float(np.max(normal_offsets))
    center_along = lower + float(
        config.get("condition_incline_seed_uphill_fraction", 0.82)
    ) * span
    center_normal = surface_offset + float(
        config.get("condition_incline_seed_normal_offset_fraction", 0.04)
    ) * span
    center = (
        center_along * champion_direction + center_normal * normal
    )
    along_size = float(
        config.get("condition_incline_seed_along_fraction", 0.18)
    ) * span
    normal_size = float(
        config.get("condition_incline_seed_normal_fraction", 0.11)
    ) * span
    seed = np.zeros((height, width), dtype=np.uint8)
    box = cv2.boxPoints(
        (
            tuple(center.astype(np.float32)),
            (max(along_size, 8.0), max(normal_size, 8.0)),
            float(
                np.rad2deg(
                    math.atan2(
                        champion_direction[1],
                        champion_direction[0],
                    )
                )
            ),
        )
    ).astype(np.int32)
    cv2.fillConvexPoly(seed, box, 255)
    area = int(np.count_nonzero(seed))
    if area < minimum_area or area > maximum_area:
        return None, {
            "status": "apparatus_seed_area_invalid",
            "seed_area_px2": area,
            "minimum_area_px2": minimum_area,
            "maximum_area_px2": maximum_area,
            "condition_future_pixels_used": False,
        }
    return seed, {
        "status": "ok",
        "source": "condition_incline_apparatus_uphill_seed",
        "seed_center_xy": center.tolist(),
        "seed_area_px2": area,
        "board_span_px": span,
        "board_direction_xy": champion_direction.tolist(),
        "board_normal_xy": normal.tolist(),
        "parallel_line_count": len(parallel),
        "condition_future_pixels_used": False,
    }


def freeze_condition_incline_axis(
    condition_frame: np.ndarray,
    *,
    condition_centroid_xy: np.ndarray,
    fallback: FrozenRigidAxis | None,
    allow_reference_fallback: bool = True,
) -> FrozenRigidAxis:
    """Estimate an apparatus line from the condition image, never prediction."""

    centroid = np.asarray(condition_centroid_xy, dtype=np.float64)
    candidates = _condition_incline_line_candidates(
        condition_frame,
        condition_centroid_xy=centroid,
    )
    if not candidates:
        if not allow_reference_fallback or fallback is None:
            raise ReferenceAnalysisError(
                "reference_condition_axis_unobserved",
                "condition-only incline apparatus geometry did not yield a "
                "causal slide axis; future reference coordinates were not "
                "used as a fallback",
            )
        return FrozenRigidAxis(
            origin_xy=centroid,
            direction_xy=fallback.direction_xy,
            normal_xy=fallback.normal_xy,
            span_px=fallback.span_px,
            source="condition_axis_hough_failed_reference_direction_fallback",
            explained_ratio=fallback.explained_ratio,
        )
    selected = max(candidates, key=lambda item: float(item["score"]))
    direction = np.asarray(selected["direction_xy"], dtype=np.float64)
    span = float(selected["forward_span_px"])
    return FrozenRigidAxis(
        origin_xy=centroid,
        direction_xy=direction,
        normal_xy=np.asarray([-direction[1], direction[0]]),
        span_px=(
            max(span, fallback.span_px * 0.5)
            if allow_reference_fallback and fallback is not None
            else max(span, 1.0)
        ),
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
    terminal_geometry_policy: str = "legacy_boundary_or_travel",
    minimum_observed_tail_frames: int = 1,
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
    required_tail = max(int(minimum_observed_tail_frames), 1)
    tail_start = max(last - required_tail + 1, 0)
    consecutive_observed_tail = bool(
        np.all(observed[tail_start : last + 1])
        and last - tail_start + 1 >= required_tail
    )
    diagnostics.update(
        {
            "minimum_observed_tail_frames": required_tail,
            "consecutive_observed_tail": consecutive_observed_tail,
        }
    )
    if not consecutive_observed_tail:
        diagnostics.update(
            {
                "legal_exit": False,
                "reason": "non_consecutive_terminal_observation_tail",
            }
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
    if terminal_geometry_policy == "scene_terminal_geometry_v2":
        # Travel is useful corroboration but is never itself an exit event.
        # Free fall must leave through the bottom boundary; incline currently
        # accepts a canvas boundary and may add a separately frozen apparatus
        # endpoint in a future protocol without weakening this condition.
        terminal_geometry_supported = near_exit_boundary
    elif terminal_geometry_policy == "legacy_boundary_or_travel":
        terminal_geometry_supported = (
            near_exit_boundary or sufficient_terminal_travel
        )
    else:
        raise ValueError(
            "unsupported rigid-body terminal geometry policy: "
            f"{terminal_geometry_policy}"
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
            "terminal_geometry_policy": terminal_geometry_policy,
            "terminal_geometry_supported": terminal_geometry_supported,
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
        terminal_geometry_policy=str(
            config.get(
                "exit_terminal_geometry_policy",
                "legacy_boundary_or_travel",
            )
        ),
        minimum_observed_tail_frames=int(
            config.get("exit_minimum_observed_tail_frames", 1)
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


def reexpress_same_case_reference_on_condition_axis(
    reference: RigidBodyReference,
    *,
    condition_axis: FrozenRigidAxis,
) -> RigidBodyReference:
    """Express real GT positions on the condition-frozen scene coordinate.

    Unlike physics-parent remapping, same-case GT keeps its observed pixel
    positions, masks, cross-track deviations, and pose.  Only the coordinate
    basis and normalized along-axis progress change, so prediction and GT are
    compared in one causal frame without erasing real off-axis behaviour.
    """

    expected = np.asarray(reference.expected, dtype=bool)
    progress = np.full(len(expected), np.nan, dtype=np.float64)
    if expected.any():
        along, _ = condition_axis.project(reference.xy[expected])
        progress[expected] = (
            along - float(along[0])
        ) / condition_axis.span_px
    return RigidBodyReference(
        entity_id=reference.entity_id,
        entity_class=reference.entity_class,
        masks=reference.masks,
        xy=reference.xy,
        areas_px2=reference.areas_px2,
        expected=reference.expected,
        raw_observed=reference.raw_observed,
        axis=condition_axis,
        normalized_progress=progress,
        legal_exit_frame=reference.legal_exit_frame,
        lifecycle_diagnostics={
            **reference.lifecycle_diagnostics,
            "reference_geometry": (
                "same_case_gt_positions_reexpressed_on_"
                "condition_frozen_axis"
            ),
            "cross_track_and_pose_preserved": True,
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


def condition_free_fall_photometric_evidence(
    frame: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Measure condition-only evidence that a compact region is a real ball.

    Free-fall condition frames commonly contain three locally round regions:
    the held ball, fingertips, and a cast shadow.  Geometry alone cannot
    reliably order them.  This observer deliberately uses only the current
    condition image and records every term used by the v7 selector:

    * a real shaded sphere normally has non-zero luminance structure;
    * a cast shadow or an arbitrary patch of wall is often nearly flat;
    * a strongly red/orange, saturated region is likely the releasing hand;
    * signed local contrast is weak supporting evidence, never a hard colour
      requirement (dark balls must remain representable).

    The values are diagnostics, not semantic labels.  In particular,
    ``skin_fraction`` is only allowed to down-rank the expected condition
    anchor; it is not used to erase independently observed prediction-side
    objects.
    """

    image = np.asarray(frame, dtype=np.uint8)
    binary = np.asarray(mask) > 0
    if image.ndim != 3 or image.shape[:2] != binary.shape or not binary.any():
        return {
            "luminance_mean": 0.0,
            "luminance_std": 0.0,
            "local_luminance_contrast": 0.0,
            "structure_support": 0.0,
            "bright_object_support": 0.5,
            "flat_region_support": 1.0,
            "skin_fraction": 0.0,
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    area = int(binary.sum())
    body_radius = math.sqrt(max(float(area), 1.0) / math.pi)
    ring_radius = max(3, int(round(0.60 * body_radius)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )
    dilated = cv2.dilate(binary.astype(np.uint8), kernel) > 0
    ring = dilated & ~binary
    luminance = gray[binary].astype(np.float64)
    ring_luminance = (
        gray[ring].astype(np.float64)
        if ring.any()
        else luminance
    )
    luminance_mean = float(np.mean(luminance))
    luminance_std = float(np.std(luminance))
    local_contrast = float(
        luminance_mean - np.mean(ring_luminance)
    )
    hue = hsv[..., 0][binary]
    saturation = hsv[..., 1][binary]
    strong_warm_skin = (
        ((hue <= 18) | (hue >= 170))
        & (saturation >= 55)
    )
    structure_support = float(
        1.0 - math.exp(-luminance_std / 12.0)
    )
    bright_support = float(
        1.0 / (1.0 + math.exp(-local_contrast / 14.0))
    )
    flat_support = float(math.exp(-luminance_std / 5.0))
    return {
        "luminance_mean": luminance_mean,
        "luminance_std": luminance_std,
        "local_luminance_contrast": local_contrast,
        "structure_support": structure_support,
        "bright_object_support": bright_support,
        "flat_region_support": flat_support,
        "skin_fraction": float(np.mean(strong_warm_skin)),
    }


def rank_free_fall_directed_components(
    components: Sequence[tuple[np.ndarray, float, np.ndarray]],
    *,
    frame: np.ndarray,
    condition_mask: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_area_px2: float,
    config: Mapping[str, Any],
) -> list[tuple[float, tuple[np.ndarray, float, np.ndarray], dict[str, Any]]]:
    """Rank SAM components after a held ball separates from the hand.

    SAM2 can return one connected hand/ball mask for the first few samples
    and two components after release.  Nearest-centre tracking then keeps the
    stationary hand and turns the real falling ball into an ``extra``.  V7
    retains the frozen condition identity but lets the ball component win
    using only current/past condition geometry: roundness, condition scale,
    distance to the previous directed sample, and bounded downward release
    progress.  No GT trajectory or future frame is consulted.
    """

    condition_centroid = mask_centroid(condition_mask)
    if condition_centroid is None:
        condition_centroid = np.asarray(anchor_xy, dtype=np.float64)
    body_radius = math.sqrt(
        max(float(anchor_area_px2), 1.0) / math.pi
    )
    area_cost_weight = float(
        config.get("directed_component_area_cost_weight", 0.80)
    )
    shape_weight = float(
        config.get("directed_release_shape_reward", 1.20)
    )
    progress_weight = float(
        config.get("directed_release_progress_reward", 0.80)
    )
    skin_weight = float(
        config.get("directed_release_skin_cost", 1.25)
    )
    full_progress_radii = max(
        float(config.get("directed_release_full_progress_radii", 3.0)),
        1e-6,
    )
    ranked: list[
        tuple[
            float,
            tuple[np.ndarray, float, np.ndarray],
            dict[str, Any],
        ]
    ] = []
    for component in components:
        centroid, area, component_mask = component
        distance_radii = float(
            np.linalg.norm(centroid - anchor_xy)
            / max(body_radius, 1.0)
        )
        scale_cost = abs(
            math.log(
                max(float(area), 1.0)
                / max(float(anchor_area_px2), 1.0)
            )
        )
        shape_score = _compact_shape_score(
            component_mask,
            scene_kind="free_fall",
        )
        photometric = condition_free_fall_photometric_evidence(
            frame,
            component_mask,
        )
        release_progress_radii = float(
            (centroid[1] - condition_centroid[1])
            / max(body_radius, 1.0)
        )
        release_progress_support = float(
            np.clip(
                release_progress_radii / full_progress_radii,
                0.0,
                1.0,
            )
        )
        base_cost = (
            distance_radii + area_cost_weight * scale_cost
        )
        cost = float(
            base_cost
            - shape_weight * shape_score
            - progress_weight * release_progress_support
            + skin_weight * photometric["skin_fraction"]
        )
        ranked.append(
            (
                cost,
                component,
                {
                    "cost": cost,
                    "base_distance_scale_cost": base_cost,
                    "distance_radii": distance_radii,
                    "scale_log_error": scale_cost,
                    "shape_score": shape_score,
                    "release_progress_radii": (
                        release_progress_radii
                    ),
                    "release_progress_support": (
                        release_progress_support
                    ),
                    "skin_fraction": photometric["skin_fraction"],
                },
            )
        )
    return sorted(ranked, key=lambda item: item[0])


def free_fall_release_hand_rejection(
    *,
    frame: np.ndarray,
    mask: np.ndarray,
    condition_mask: np.ndarray,
    body_radius_px: float,
    config: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Conservatively reject a disconnected releasing-hand fragment.

    The veto requires strong warm-skin evidence, weak enough circularity, and
    proximity to the condition release site.  A normal second ball, including
    a static copy, therefore remains a participant.  This is intentionally
    stricter than condition-anchor down-ranking because prediction-side
    rejection could otherwise hide a coloured extra object.
    """

    centroid = mask_centroid(mask)
    condition_centroid = mask_centroid(condition_mask)
    shape_score = _compact_shape_score(mask, scene_kind="free_fall")
    photometric = condition_free_fall_photometric_evidence(frame, mask)
    distance_radii = (
        float(
            np.linalg.norm(centroid - condition_centroid)
            / max(float(body_radius_px), 1.0)
        )
        if centroid is not None and condition_centroid is not None
        else float("inf")
    )
    minimum_skin = float(
        config.get("directed_release_hand_minimum_skin_fraction", 0.75)
    )
    maximum_shape = float(
        config.get("directed_release_hand_maximum_shape_score", 0.72)
    )
    maximum_distance = float(
        config.get("directed_release_hand_maximum_distance_radii", 4.0)
    )
    rejected = bool(
        photometric["skin_fraction"] >= minimum_skin
        and shape_score <= maximum_shape
        and distance_radii <= maximum_distance
    )
    return rejected, {
        "decision": (
            "rejected_releasing_hand_fragment"
            if rejected
            else "retained_as_independent_object"
        ),
        "skin_fraction": photometric["skin_fraction"],
        "minimum_skin_fraction": minimum_skin,
        "shape_score": shape_score,
        "maximum_shape_score": maximum_shape,
        "condition_distance_radii": (
            distance_radii if math.isfinite(distance_radii) else None
        ),
        "maximum_condition_distance_radii": maximum_distance,
    }


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


def merge_disconnected_directed_fragments(
    selected_mask: np.ndarray,
    other_components: Sequence[
        tuple[np.ndarray, float, np.ndarray]
    ],
    *,
    anchor_area_px2: float,
    config: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    list[tuple[np.ndarray, float, np.ndarray]],
    list[dict[str, Any]],
]:
    """Merge only small, touching pieces from one directed SAM output.

    A disconnected SAM mask is not automatically a second physical object.
    Conversely, blindly merging nearby components would let a real duplicate
    escape cardinality auditing.  The v2 rule therefore requires every merged
    piece to be small relative to the condition-frozen body, separated by only
    a narrow pixel gap, and contained in a body-sized union envelope.  A second
    complete block/ball remains a residual participant.
    """

    selected = np.where(
        np.asarray(selected_mask) > 0, 255, 0
    ).astype(np.uint8)
    retained = list(other_components)
    if (
        str(
            config.get(
                "disconnected_directed_component_policy",
                "all_components_are_objects_v1",
            )
        )
        != "merge_same_body_fragments_v2"
        or not retained
    ):
        return selected, retained, []
    radius = math.sqrt(max(float(anchor_area_px2), 1.0) / math.pi)
    maximum_fragment_area = (
        float(
            config.get(
                "directed_fragment_maximum_anchor_area_fraction",
                0.30,
            )
        )
        * float(anchor_area_px2)
    )
    maximum_gap = max(
        2.0,
        float(
            config.get(
                "directed_fragment_maximum_gap_body_radii",
                0.30,
            )
        )
        * radius,
    )
    maximum_union_span = (
        float(
            config.get(
                "directed_fragment_maximum_union_span_body_radii",
                4.8,
            )
        )
        * radius
    )
    maximum_union_area = (
        float(
            config.get(
                "directed_fragment_maximum_union_anchor_area_ratio",
                1.55,
            )
        )
        * float(anchor_area_px2)
    )
    merged: list[dict[str, Any]] = []
    output_retained: list[
        tuple[np.ndarray, float, np.ndarray]
    ] = []
    # Iterate smallest first.  Multiple pieces of the same damaged mask can
    # then grow the body envelope, while no single piece may exceed the strict
    # condition-scale veto.
    for xy, area, raw_mask in sorted(
        retained,
        key=lambda item: float(item[1]),
    ):
        candidate = np.where(
            np.asarray(raw_mask) > 0, 255, 0
        ).astype(np.uint8)
        inverse = np.where(selected > 0, 0, 255).astype(np.uint8)
        distances = cv2.distanceTransform(
            inverse,
            cv2.DIST_L2,
            3,
        )
        candidate_pixels = distances[candidate > 0]
        gap = (
            float(np.min(candidate_pixels))
            if candidate_pixels.size
            else float("inf")
        )
        union = cv2.bitwise_or(selected, candidate)
        points = cv2.findNonZero(union)
        if points is None:
            output_retained.append((xy, area, raw_mask))
            continue
        _, _, width, height = cv2.boundingRect(points)
        union_span = float(math.hypot(width, height))
        union_area = float(np.count_nonzero(union))
        accepted = bool(
            float(area) <= maximum_fragment_area
            and gap <= maximum_gap
            and union_span <= maximum_union_span
            and union_area <= maximum_union_area
        )
        record = {
            "xy": np.asarray(xy, dtype=np.float64).tolist(),
            "area_px2": float(area),
            "anchor_area_fraction": float(area)
            / max(float(anchor_area_px2), 1.0),
            "gap_px": gap,
            "maximum_gap_px": maximum_gap,
            "union_span_px": union_span,
            "maximum_union_span_px": maximum_union_span,
            "union_area_px2": union_area,
            "maximum_union_area_px2": maximum_union_area,
            "decision": (
                "merged_same_body_fragment"
                if accepted
                else "retained_as_independent_object"
            ),
        }
        if accepted:
            selected = union
            merged.append(record)
        else:
            output_retained.append((xy, area, raw_mask))
    return selected, output_retained, merged


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


def coalesce_condition_directed_identity(
    observation: OpenWorldObservation,
    *,
    config: Mapping[str, Any],
) -> OpenWorldObservation:
    """Keep one frozen ID for every validated condition-directed sample.

    The generic Hungarian tracker is intentionally conservative for unknown
    objects.  A high-speed fall, however, can move more than its assignment
    radius in one sample and be split into births.  Directed observations
    already carry a condition-frozen identity partition, so v7 coalesces only
    that validated stream; residual and replacement detections remain
    separate and can never take over the expected ID.
    """

    if (
        str(
            config.get(
                "condition_directed_track_policy",
                "generic_hungarian",
            )
        )
        != "frozen_condition_identity_v2"
    ):
        return observation
    directed: list[ObjectDetection] = []
    retained: list[OpenWorldTrack] = []
    source_track_ids: list[str] = []
    for track in observation.tracks:
        selected = [
            detection
            for detection in track.detections
            if (
                detection.metadata.get("exclusive_tracking_partition")
                == "rigid_condition_directed"
                and bool(
                    detection.metadata.get(
                        "identity_anchor_valid",
                        False,
                    )
                )
                and not detection.entity_class.endswith("__replacement")
            )
        ]
        if not selected:
            retained.append(track)
            continue
        source_track_ids.append(track.track_id)
        directed.extend(selected)
        selected_ids = {detection.detection_id for detection in selected}
        leftovers = [
            detection
            for detection in track.detections
            if detection.detection_id not in selected_ids
        ]
        if leftovers:
            retained.append(
                OpenWorldTrack(
                    track_id=f"{track.track_id}__nonidentity",
                    detections=tuple(leftovers),
                    confirmed=track.confirmed,
                    evidence_tier=track.evidence_tier,
                )
            )
    if not directed:
        return observation
    by_frame: dict[int, ObjectDetection] = {}
    for detection in sorted(
        directed,
        key=lambda item: (
            item.frame_index,
            item.confidence,
            "condition_directed_recovery" in item.sources,
        ),
    ):
        by_frame[detection.frame_index] = detection
    retained.insert(
        0,
        OpenWorldTrack(
            track_id="condition_identity",
            detections=tuple(by_frame[index] for index in sorted(by_frame)),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        ),
    )
    diagnostics = dict(observation.diagnostics)
    diagnostics["condition_directed_track_policy"] = (
        "frozen_condition_identity_v2"
    )
    diagnostics["condition_directed_source_track_ids"] = source_track_ids
    diagnostics["condition_directed_observed_frames"] = sorted(by_frame)
    return OpenWorldObservation(
        tracks=tuple(retained),
        overflow_counts=observation.overflow_counts,
        diagnostics=diagnostics,
    )


def classify_coupled_rigid_body_artifacts(
    observation: OpenWorldObservation,
    *,
    body_radius_px: float,
    config: Mapping[str, Any],
) -> OpenWorldObservation:
    """Demote a weak, kinematically coupled residual such as a cast shadow.

    Motion and frame differencing are dependent observations of the same
    pixel change.  A cast shadow can therefore look like a second moving
    participant unless an independent compact-object channel is required.
    This classifier only demotes a residual when all of the following hold:
    it is absent from the condition frame, has little compact-object support,
    follows the condition-anchored track with a stable offset and velocity,
    and is not a disconnected directed component.  A genuine second body
    with persistent compact evidence remains a full participant.
    """

    if (
        str(
            config.get(
                "coupled_optical_artifact_policy",
                "disabled",
            )
        )
        != "motion_correlation_with_compact_veto_v1"
    ):
        return observation
    primary = next(
        (
            track
            for track in observation.tracks
            if any(
                detection.frame_index == 0
                and "directed_sam2" in detection.sources
                and detection.metadata.get(
                    "identity_anchor_valid", True
                )
                for detection in track.detections
            )
        ),
        None,
    )
    if primary is None:
        return observation
    primary_by_frame = {
        detection.frame_index: detection
        for detection in primary.detections
    }
    minimum_overlap = int(
        config.get("coupled_artifact_minimum_overlap_frames", 4)
    )
    maximum_compact_ratio = float(
        config.get("coupled_artifact_maximum_compact_support_ratio", 0.3)
    )
    maximum_offset_std = float(
        config.get("coupled_artifact_maximum_offset_std_radii", 0.8)
    )
    maximum_velocity_error = float(
        config.get(
            "coupled_artifact_maximum_velocity_error_radii",
            0.75,
        )
    )
    minimum_offset = float(
        config.get("coupled_artifact_minimum_offset_radii", 0.6)
    )
    maximum_offset = float(
        config.get("coupled_artifact_maximum_offset_radii", 6.0)
    )
    radius = max(float(body_radius_px), 1.0)
    output: list[OpenWorldTrack] = []
    demoted: list[dict[str, Any]] = []
    for track in observation.tracks:
        if track.track_id == primary.track_id:
            output.append(track)
            continue
        residual = [
            detection
            for detection in track.detections
            if bool(detection.metadata.get("residual", False))
        ]
        if (
            any(
                bool(
                    detection.metadata.get(
                        "duplicate_directed_component", False
                    )
                )
                for detection in track.detections
            )
            or any(detection.frame_index == 0 for detection in residual)
            or len(residual) < minimum_overlap
        ):
            output.append(track)
            continue
        compact_count = sum(
            "independent_compact_shape" in detection.sources
            for detection in residual
        )
        compact_ratio = compact_count / max(len(residual), 1)
        if compact_ratio > maximum_compact_ratio:
            output.append(track)
            continue
        common = sorted(
            detection.frame_index
            for detection in residual
            if detection.frame_index in primary_by_frame
        )
        if len(common) < minimum_overlap:
            output.append(track)
            continue
        residual_by_frame = {
            detection.frame_index: detection
            for detection in residual
        }
        offsets = np.asarray(
            [
                residual_by_frame[index].xy
                - primary_by_frame[index].xy
                for index in common
            ],
            dtype=np.float64,
        )
        offset_norms = np.linalg.norm(offsets, axis=1) / radius
        offset_std = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            offsets
                            - np.median(offsets, axis=0)
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
            / radius
        )
        median_offset = float(np.median(offset_norms))
        velocity_errors: list[float] = []
        for first, second in zip(common, common[1:]):
            if second != first + 1:
                continue
            primary_step = (
                primary_by_frame[second].xy
                - primary_by_frame[first].xy
            )
            residual_step = (
                residual_by_frame[second].xy
                - residual_by_frame[first].xy
            )
            velocity_errors.append(
                float(
                    np.linalg.norm(residual_step - primary_step)
                    / radius
                )
            )
        if len(velocity_errors) < max(2, minimum_overlap - 2):
            output.append(track)
            continue
        median_velocity_error = float(np.median(velocity_errors))
        coupled = bool(
            minimum_offset <= median_offset <= maximum_offset
            and offset_std <= maximum_offset_std
            and median_velocity_error <= maximum_velocity_error
        )
        if not coupled:
            output.append(track)
            continue
        output.append(
            OpenWorldTrack(
                track_id=track.track_id,
                detections=track.detections,
                confirmed=track.confirmed,
                # Kinematic coupling alone is insufficient to erase a real
                # duplicate.  Keep a conservative tentative exposure unless
                # a future protocol adds independent photometric shadow
                # evidence.
                evidence_tier=EvidenceTier.TENTATIVE,
            )
        )
        demoted.append(
            {
                "track_id": track.track_id,
                "reason": (
                    "weak_compact_evidence_plus_stable_offset_and_"
                    "velocity_coupling_to_condition_identity"
                ),
                "compact_support_ratio": compact_ratio,
                "overlap_frames": len(common),
                "median_offset_radii": median_offset,
                "offset_std_radii": offset_std,
                "median_velocity_error_radii": median_velocity_error,
                "formal_exposure_weight_before": (
                    track.formal_exposure_weight
                ),
                "formal_exposure_weight_after": 0.25,
            }
        )
    weak_motion_only_tracks: list[dict[str, Any]] = []
    minimum_motion_only_frames = int(
        config.get(
            "motion_only_participant_minimum_frames",
            4,
        )
    )
    motion_sources = {
        "condition_difference",
        "temporal_motion",
        "kinematic_recovery_change",
    }
    motion_refined: list[OpenWorldTrack] = []
    for track in output:
        if track.track_id == primary.track_id:
            motion_refined.append(track)
            continue
        observed_sources = {
            source
            for detection in track.detections
            for source in detection.sources
        }
        compact_supported = any(
            "independent_compact_shape" in detection.sources
            or bool(
                detection.metadata.get(
                    "duplicate_directed_component",
                    False,
                )
            )
            for detection in track.detections
        )
        short_dependent_motion = bool(
            not compact_supported
            and observed_sources
            and observed_sources <= motion_sources
            and len(track.detections) < minimum_motion_only_frames
            and track.evidence_tier is EvidenceTier.PARTICIPANT
        )
        if not short_dependent_motion:
            motion_refined.append(track)
            continue
        motion_refined.append(
            OpenWorldTrack(
                track_id=track.track_id,
                detections=track.detections,
                confirmed=track.confirmed,
                evidence_tier=EvidenceTier.TENTATIVE,
            )
        )
        weak_motion_only_tracks.append(
            {
                "track_id": track.track_id,
                "reason": (
                    "short_correlated_pixel_change_without_independent_"
                    "compact_object_evidence"
                ),
                "observed_frames": len(track.detections),
                "minimum_participant_frames": minimum_motion_only_frames,
                "sources": sorted(observed_sources),
                "formal_exposure_weight_before": (
                    track.formal_exposure_weight
                ),
                "formal_exposure_weight_after": 0.25,
            }
        )
    output = motion_refined
    condition_present_apparatus: list[dict[str, Any]] = []
    if (
        str(
            config.get(
                "condition_present_apparatus_policy",
                "disabled",
            )
        )
        == "frozen_condition_track_v2"
    ):
        refined_condition_tracks: list[OpenWorldTrack] = []
        for track in output:
            condition_anchors = [
                detection
                for detection in track.detections
                if (
                    bool(
                        detection.metadata.get(
                            "condition_frame_apparatus_supported",
                            False,
                        )
                    )
                    or (
                        detection.frame_index == 0
                        and bool(
                        detection.metadata.get(
                            "suppress_persistence_only_promotion",
                            False,
                        )
                    )
                    )
                )
            ]
            apparatus_radius = max(
                (
                    math.sqrt(
                        max(detection.area_px2, 1.0) / math.pi
                    )
                    for detection in condition_anchors
                ),
                default=1.0,
            )
            maximum_apparatus_span = float(
                config.get(
                    "condition_present_apparatus_maximum_span_radii",
                    3.0,
                )
            )
            apparatus_points = np.asarray(
                [detection.xy for detection in track.detections],
                dtype=np.float64,
            )
            apparatus_span = (
                float(
                    np.linalg.norm(np.ptp(apparatus_points, axis=0))
                    / apparatus_radius
                )
                if len(apparatus_points)
                else float("inf")
            )
            has_independent_motion = any(
                "temporal_motion" in detection.sources
                and not bool(
                    detection.metadata.get(
                        "condition_frame_apparatus_supported",
                        False,
                    )
                )
                for detection in track.detections
            )
            if (
                track.track_id == primary.track_id
                or not condition_anchors
                or apparatus_span > maximum_apparatus_span
                or has_independent_motion
                or any(
                    bool(
                        detection.metadata.get(
                            "duplicate_directed_component",
                            False,
                        )
                    )
                    for detection in track.detections
                )
            ):
                refined_condition_tracks.append(track)
                continue
            refined_condition_tracks.append(
                OpenWorldTrack(
                    track_id=track.track_id,
                    detections=track.detections,
                    confirmed=track.confirmed,
                    evidence_tier=EvidenceTier.AMBIGUOUS,
                )
            )
            condition_present_apparatus.append(
                {
                    "track_id": track.track_id,
                    "reason": (
                        "condition_present_unmatched_compact_track_"
                        "frozen_as_apparatus"
                    ),
                    "condition_anchor_detection_ids": [
                        item.detection_id for item in condition_anchors
                    ],
                    "motion_span_radii": apparatus_span,
                    "has_independent_motion": has_independent_motion,
                    "formal_exposure_weight_before": (
                        track.formal_exposure_weight
                    ),
                    "formal_exposure_weight_after": 0.0,
                }
            )
        output = refined_condition_tracks
    condition_apparatus_anchors = [
        detection
        for track in output
        if track.track_id != primary.track_id
        for detection in track.detections
        if (
            detection.frame_index == 0
            and "independent_compact_shape" in detection.sources
            and bool(
                detection.metadata.get(
                    "suppress_persistence_only_promotion",
                    False,
                )
            )
        )
    ]
    apparatus_fragments: list[dict[str, Any]] = []
    if condition_apparatus_anchors:
        refined: list[OpenWorldTrack] = []
        maximum_fragment_span = float(
            config.get(
                "condition_apparatus_maximum_fragment_span_radii",
                3.0,
            )
        )
        maximum_anchor_distance = float(
            config.get(
                "condition_apparatus_maximum_anchor_distance_radii",
                3.0,
            )
        )
        minimum_compact_ratio = float(
            config.get(
                "condition_apparatus_minimum_compact_support_ratio",
                0.5,
            )
        )
        minimum_anchor_containment = float(
            config.get(
                "condition_apparatus_minimum_mask_containment",
                0.35,
            )
        )
        for track in output:
            if (
                track.track_id == primary.track_id
                or track.evidence_tier is EvidenceTier.AMBIGUOUS
                or any(
                    detection.frame_index == 0
                    for detection in track.detections
                )
                or any(
                    "temporal_motion" in detection.sources
                    for detection in track.detections
                )
            ):
                refined.append(track)
                continue
            compact_ratio = sum(
                "independent_compact_shape" in detection.sources
                for detection in track.detections
            ) / len(track.detections)
            if compact_ratio < minimum_compact_ratio:
                refined.append(track)
                continue
            points = np.asarray(
                [
                    detection.xy
                    for detection in track.detections
                ],
                dtype=np.float64,
            )
            span = float(
                np.linalg.norm(np.ptp(points, axis=0)) / radius
            )
            median_xy = np.median(points, axis=0)
            anchor_distance = min(
                float(
                    np.linalg.norm(median_xy - anchor.xy)
                    / radius
                )
                for anchor in condition_apparatus_anchors
            )
            overlap_values = [
                _overlap(detection.mask, anchor.mask)
                for detection in track.detections
                for anchor in condition_apparatus_anchors
                if detection.mask is not None and anchor.mask is not None
            ]
            maximum_anchor_iou = max(
                (value[0] for value in overlap_values),
                default=0.0,
            )
            maximum_anchor_containment = max(
                (value[1] for value in overlap_values),
                default=0.0,
            )
            if (
                span > maximum_fragment_span
                or anchor_distance > maximum_anchor_distance
                or maximum_anchor_containment
                < minimum_anchor_containment
            ):
                refined.append(track)
                continue
            refined.append(
                OpenWorldTrack(
                    track_id=track.track_id,
                    detections=track.detections,
                    confirmed=track.confirmed,
                    evidence_tier=EvidenceTier.AMBIGUOUS,
                )
            )
            apparatus_fragments.append(
                {
                    "track_id": track.track_id,
                    "reason": (
                        "fragment_of_condition_present_compact_"
                        "apparatus_without_temporal_motion"
                    ),
                    "compact_support_ratio": compact_ratio,
                    "motion_span_radii": span,
                    "condition_anchor_distance_radii": (
                        anchor_distance
                    ),
                    "condition_anchor_maximum_iou": maximum_anchor_iou,
                    "condition_anchor_maximum_containment": (
                        maximum_anchor_containment
                    ),
                    "formal_exposure_weight_before": (
                        track.formal_exposure_weight
                    ),
                    "formal_exposure_weight_after": 0.0,
                }
            )
        output = refined
    release_site_artifacts: list[dict[str, Any]] = []
    if (
        str(
            config.get(
                "free_fall_release_site_apparatus_policy",
                "disabled",
            )
        )
        == "condition_anchor_static_track_v1"
    ):
        primary_anchor = min(
            primary.detections,
            key=lambda detection: detection.frame_index,
        )
        maximum_distance = float(
            config.get(
                "release_site_apparatus_maximum_distance_radii",
                2.0,
            )
        )
        maximum_span = float(
            config.get(
                "release_site_apparatus_maximum_motion_span_radii",
                2.0,
            )
        )
        maximum_temporal_motion_ratio = float(
            config.get(
                "release_site_apparatus_maximum_temporal_motion_ratio",
                0.0,
            )
        )
        minimum_static_support_ratio = float(
            config.get(
                "release_site_apparatus_minimum_static_support_ratio",
                0.75,
            )
        )
        minimum_anchor_containment = float(
            config.get(
                "release_site_apparatus_minimum_anchor_containment",
                0.25,
            )
        )
        maximum_weak_shape = float(
            config.get(
                "release_site_apparatus_maximum_weak_shape_score",
                0.58,
            )
        )
        minimum_stable_area_ratio = float(
            config.get(
                "release_site_apparatus_minimum_stable_area_ratio",
                0.60,
            )
        )
        minimum_skin_fraction = float(
            config.get(
                "release_site_apparatus_minimum_skin_fraction",
                0.65,
            )
        )
        refined_release_site: list[OpenWorldTrack] = []
        for track in output:
            if (
                track.track_id == primary.track_id
                or track.evidence_tier is EvidenceTier.AMBIGUOUS
                or not track.detections
            ):
                refined_release_site.append(track)
                continue
            points = np.asarray(
                [detection.xy for detection in track.detections],
                dtype=np.float64,
            )
            median_xy = np.median(points, axis=0)
            distance_radii = float(
                np.linalg.norm(median_xy - primary_anchor.xy) / radius
            )
            motion_span_radii = float(
                np.linalg.norm(np.ptp(points, axis=0)) / radius
            )
            temporal_motion_ratio = float(
                np.mean(
                    [
                        "temporal_motion" in detection.sources
                        for detection in track.detections
                    ]
                )
            )
            static_support_ratio = float(
                np.mean(
                    [
                        bool(
                            detection.metadata.get(
                                "duplicate_directed_component",
                                False,
                            )
                        )
                        or (
                            "independent_compact_shape"
                            in detection.sources
                            and "temporal_motion"
                            not in detection.sources
                        )
                        for detection in track.detections
                    ]
                )
            )
            shapes = np.asarray(
                [
                    float(
                        detection.metadata.get(
                            "compact_shape_score",
                            _compact_shape_score(
                                detection.mask,
                                scene_kind="free_fall",
                            )
                            if detection.mask is not None
                            else 0.0,
                        )
                    )
                    for detection in track.detections
                ],
                dtype=np.float64,
            )
            areas = np.asarray(
                [detection.area_px2 for detection in track.detections],
                dtype=np.float64,
            )
            stable_area_ratio = float(
                np.min(areas) / max(float(np.max(areas)), 1.0)
            )
            anchor_containments: list[float] = []
            if primary_anchor.mask is not None:
                anchor_containments.extend(
                    _overlap(detection.mask, primary_anchor.mask)[1]
                    for detection in track.detections
                    if detection.mask is not None
                )
            anchor_containments.extend(
                float(
                    detection.metadata.get(
                        "condition_subject_containment",
                        0.0,
                    )
                )
                for detection in track.detections
            )
            maximum_anchor_containment = max(
                anchor_containments,
                default=0.0,
            )
            skin_fractions: list[float] = []
            for detection in track.detections:
                release_diagnostics = detection.metadata.get(
                    "release_hand_diagnostics",
                    {},
                )
                if isinstance(release_diagnostics, Mapping):
                    skin_fractions.append(
                        float(
                            release_diagnostics.get(
                                "skin_fraction",
                                0.0,
                            )
                        )
                    )
                photometric = detection.metadata.get(
                    "free_fall_photometric_evidence",
                    {},
                )
                if isinstance(photometric, Mapping):
                    skin_fractions.append(
                        float(photometric.get("skin_fraction", 0.0))
                    )
            maximum_skin_fraction = max(skin_fractions, default=0.0)
            median_shape = float(np.median(shapes))
            weak_objectness = bool(
                median_shape <= maximum_weak_shape
                or stable_area_ratio < minimum_stable_area_ratio
                or maximum_skin_fraction >= minimum_skin_fraction
            )
            release_site_artifact = bool(
                distance_radii <= maximum_distance
                and motion_span_radii <= maximum_span
                and temporal_motion_ratio
                <= maximum_temporal_motion_ratio
                and static_support_ratio >= minimum_static_support_ratio
                and maximum_anchor_containment
                >= minimum_anchor_containment
                and weak_objectness
            )
            if not release_site_artifact:
                refined_release_site.append(track)
                continue
            refined_release_site.append(
                OpenWorldTrack(
                    track_id=track.track_id,
                    detections=track.detections,
                    confirmed=track.confirmed,
                    evidence_tier=EvidenceTier.AMBIGUOUS,
                )
            )
            release_site_artifacts.append(
                {
                    "track_id": track.track_id,
                    "reason": (
                        "condition_release_site_static_fragment_with_"
                        "weak_objectness"
                    ),
                    "condition_distance_radii": distance_radii,
                    "motion_span_radii": motion_span_radii,
                    "temporal_motion_ratio": temporal_motion_ratio,
                    "static_support_ratio": static_support_ratio,
                    "median_compact_shape_score": median_shape,
                    "stable_area_ratio": stable_area_ratio,
                    "maximum_anchor_containment": (
                        maximum_anchor_containment
                    ),
                    "maximum_skin_fraction": maximum_skin_fraction,
                    "formal_exposure_weight_before": (
                        track.formal_exposure_weight
                    ),
                    "formal_exposure_weight_after": 0.0,
                }
            )
        output = refined_release_site
    diagnostics = dict(observation.diagnostics)
    diagnostics["coupled_optical_artifacts"] = demoted
    diagnostics["weak_motion_only_tracks"] = weak_motion_only_tracks
    diagnostics["condition_apparatus_fragments"] = (
        apparatus_fragments
    )
    diagnostics["condition_present_apparatus_tracks"] = (
        condition_present_apparatus
    )
    diagnostics["free_fall_release_site_apparatus_tracks"] = (
        release_site_artifacts
    )
    diagnostics["coupled_optical_artifact_policy"] = (
        "motion_correlation_with_compact_veto_v1"
    )
    return OpenWorldObservation(
        tracks=tuple(output),
        overflow_counts=observation.overflow_counts,
        diagnostics=diagnostics,
    )


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
    condition_static_compact: list[
        tuple[np.ndarray, np.ndarray, float]
    ] = []
    condition_static_region = np.zeros(shape, dtype=np.uint8)
    if (
        str(
            config.get(
                "condition_static_compact_policy",
                "disabled",
            )
        )
        == "frozen_condition_geometry_v1"
    ):
        for baseline_mask in _independent_compact_proposals(
            condition_frame,
            scene_kind=scene_kind,
            anchor_area_px2=float(anchor_area),
            minimum_area=residual_minimum,
            maximum_area=residual_maximum,
            config=config,
        ):
            baseline_xy = mask_centroid(baseline_mask)
            if baseline_xy is None:
                continue
            _, subject_containment = _overlap(
                baseline_mask,
                condition_mask,
            )
            if subject_containment >= 0.7:
                continue
            condition_static_compact.append(
                (
                    baseline_mask,
                    baseline_xy,
                    float(np.count_nonzero(baseline_mask)),
                )
            )
            condition_static_region = cv2.bitwise_or(
                condition_static_region,
                baseline_mask,
            )
        static_region_radius = max(
            1,
            int(
                round(
                    math.sqrt(anchor_area / math.pi)
                    * float(
                        config.get(
                            "condition_static_compact_dilation_body_radii",
                            1.5,
                        )
                    )
                )
            ),
        )
        condition_static_region = cv2.dilate(
            condition_static_region,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * static_region_radius + 1,
                    2 * static_region_radius + 1,
                ),
            ),
        )
        if (
            scene_kind == "free_fall"
            and str(
                config.get(
                    "condition_static_apparatus_cluster_policy",
                    "disabled",
                )
            )
            == "vertical_compact_density_v1"
        ):
            radius = max(math.sqrt(anchor_area / math.pi), 1.0)
            maximum_cluster_width = (
                float(
                    config.get(
                        "condition_static_apparatus_maximum_cluster_width_radii",
                        3.5,
                    )
                )
                * radius
            )
            minimum_cluster_size = int(
                config.get(
                    "condition_static_apparatus_minimum_cluster_anchors",
                    3,
                )
            )
            minimum_vertical_span = (
                float(
                    config.get(
                        "condition_static_apparatus_minimum_vertical_span_fraction",
                        0.25,
                    )
                )
                * height
            )
            remaining = sorted(
                condition_static_compact,
                key=lambda item: float(item[1][0]),
            )
            clusters: list[
                list[tuple[np.ndarray, np.ndarray, float]]
            ] = []
            for item in remaining:
                target = next(
                    (
                        cluster
                        for cluster in clusters
                        if abs(
                            float(item[1][0])
                            - float(
                                np.median(
                                    [
                                        value[1][0]
                                        for value in cluster
                                    ]
                                )
                            )
                        )
                        <= maximum_cluster_width
                    ),
                    None,
                )
                if target is None:
                    clusters.append([item])
                else:
                    target.append(item)
            apparatus_margin = int(
                round(
                    float(
                        config.get(
                            "condition_static_apparatus_dilation_body_radii",
                            2.0,
                        )
                    )
                    * radius
                )
            )
            for cluster in clusters:
                if len(cluster) < minimum_cluster_size:
                    continue
                xs = np.asarray(
                    [item[1][0] for item in cluster],
                    dtype=np.float64,
                )
                ys = np.asarray(
                    [item[1][1] for item in cluster],
                    dtype=np.float64,
                )
                if float(np.ptp(ys)) < minimum_vertical_span:
                    continue
                cv2.rectangle(
                    condition_static_region,
                    (
                        max(0, int(math.floor(float(xs.min()))) - apparatus_margin),
                        max(0, int(math.floor(float(ys.min()))) - apparatus_margin),
                    ),
                    (
                        min(
                            width - 1,
                            int(math.ceil(float(xs.max()))) + apparatus_margin,
                        ),
                        min(
                            height - 1,
                            int(math.ceil(float(ys.max()))) + apparatus_margin,
                        ),
                    ),
                    255,
                    -1,
                )
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
        "directed_recovery": 0,
        "replacement": 0,
        "residual": 0,
        "condition_only_boundary_rejected": 0,
        "condition_static_compact_rejected": 0,
        "merged_directed_fragments": 0,
        "directed_release_hand_rejected": 0,
    }
    rejected_candidates: list[dict[str, Any]] = []
    merged_directed_fragments: list[dict[str, Any]] = []
    previous_direct = np.zeros(shape, dtype=np.uint8)
    identity_rejection_streak = 0
    direct_history: list[ObjectDetection] = []
    directed_identity_swept = np.where(
        condition_mask > 0, 255, 0
    ).astype(np.uint8)
    swept_exclusion_enabled = bool(
        str(
            config.get(
                "directed_identity_swept_exclusion_policy",
                "disabled",
            )
        )
        == "validated_identity_history_v1"
    )
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
        selected_identity_valid = False
        selected_component_diagnostics: dict[str, Any] = {}
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
            directed_selection_policy = str(
                config.get(
                    "directed_component_selection_policy",
                    "nearest_anchor_v1",
                )
            )
            if (
                directed_selection_policy
                == "release_aware_ball_objectness_v3"
                and scene_kind == "free_fall"
            ):
                component_ranking = rank_free_fall_directed_components(
                    direct_components,
                    frame=frame,
                    condition_mask=condition_mask,
                    anchor_xy=np.asarray(anchor_xy, dtype=np.float64),
                    anchor_area_px2=float(anchor_area),
                    config=config,
                )
                _, selected, selected_component_diagnostics = (
                    component_ranking[0]
                )
                selected_component_diagnostics = {
                    "policy": directed_selection_policy,
                    **selected_component_diagnostics,
                    "candidate_count": len(component_ranking),
                    "candidate_costs": [
                        {
                            "xy": item[1][0].tolist(),
                            "area_px2": float(item[1][1]),
                            **item[2],
                        }
                        for item in component_ranking
                    ],
                }
            elif (
                directed_selection_policy
                == "anchor_distance_plus_condition_scale_v2"
            ):
                selected = min(
                    direct_components,
                    key=lambda value: (
                        float(np.linalg.norm(value[0] - anchor_xy))
                        / max(body_radius, 1.0)
                        + float(
                            config.get(
                                "directed_component_area_cost_weight",
                                0.80,
                            )
                        )
                        * abs(
                            math.log(
                                max(float(value[1]), 1.0)
                                / max(float(anchor_area), 1.0)
                            )
                        )
                    ),
                )
                selected_component_diagnostics = {
                    "policy": directed_selection_policy,
                }
            else:
                selected = min(
                    direct_components,
                    key=lambda value: float(
                        np.linalg.norm(value[0] - anchor_xy)
                    ),
                )
                selected_component_diagnostics = {
                    "policy": directed_selection_policy,
                }
            centroid, area, selected_direct = selected
            remaining_direct = [
                value
                for value in direct_components
                if value is not selected
            ]
            (
                selected_direct,
                remaining_direct,
                merged_fragments,
            ) = merge_disconnected_directed_fragments(
                selected_direct,
                remaining_direct,
                anchor_area_px2=float(anchor_area),
                config=config,
            )
            if merged_fragments:
                merged_directed_fragments.extend(
                    {
                        "frame_index": frame_index,
                        **record,
                    }
                    for record in merged_fragments
                )
                source_counts["merged_directed_fragments"] += len(
                    merged_fragments
                )
                merged_centroid = mask_centroid(selected_direct)
                if merged_centroid is not None:
                    centroid = merged_centroid
                area = float(np.count_nonzero(selected_direct))
            color_similarity = _histogram_intersection(
                anchor_histogram,
                _mask_histogram(frame, selected_direct),
            )
            raw_identity_valid = (
                color_similarity
                >= float(config["minimum_anchor_color_similarity"])
            )
            kinematic_identity_continuity = False
            kinematic_prediction_error_radii: float | None = None
            kinematic_area_ratio: float | None = None
            if (
                direct_history
                and str(
                    config.get(
                        "identity_rejection_policy",
                        "appearance_streak_v1",
                    )
                )
                == "appearance_plus_kinematic_discontinuity_v1"
            ):
                latest = direct_history[-1]
                latest_radius = math.sqrt(
                    max(latest.area_px2, 1.0) / math.pi
                )
                current_radius = math.sqrt(
                    max(float(area), 1.0) / math.pi
                )
                radius_scale = max(
                    body_radius,
                    latest_radius,
                    current_radius,
                    1.0,
                )
                prediction = latest.xy
                if len(direct_history) >= 2:
                    previous = direct_history[-2]
                    history_delta = max(
                        latest.frame_index - previous.frame_index,
                        1,
                    )
                    current_delta = max(
                        frame_index - latest.frame_index,
                        1,
                    )
                    prediction = latest.xy + (
                        (latest.xy - previous.xy)
                        * (current_delta / history_delta)
                    )
                    maximum_error_radii = float(
                        config.get(
                            "identity_continuity_maximum_prediction_error_radii",
                            4.0,
                        )
                    )
                else:
                    maximum_error_radii = float(
                        config.get(
                            "identity_continuity_maximum_initial_jump_radii",
                            10.0,
                        )
                    )
                kinematic_prediction_error_radii = float(
                    np.linalg.norm(centroid - prediction) / radius_scale
                )
                kinematic_area_ratio = float(area) / max(
                    latest.area_px2,
                    1.0,
                )
                maximum_area_change = float(
                    config.get(
                        "identity_continuity_maximum_area_ratio",
                        4.0,
                    )
                )
                kinematic_identity_continuity = bool(
                    kinematic_prediction_error_radii
                    <= maximum_error_radii
                    and 1.0 / maximum_area_change
                    <= kinematic_area_ratio
                    <= maximum_area_change
                )
            if raw_identity_valid or kinematic_identity_continuity:
                identity_rejection_streak = 0
            else:
                identity_rejection_streak += 1
            identity_valid = bool(
                raw_identity_valid
                or kinematic_identity_continuity
                or identity_rejection_streak
                < int(
                    config[
                        "identity_rejection_minimum_consecutive_frames"
                    ]
                )
            )
            selected_identity_valid = identity_valid
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
            direct_detection = ObjectDetection(
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
                    "identity_anchor_kinematic_valid": (
                        kinematic_identity_continuity
                    ),
                    "identity_kinematic_prediction_error_radii": (
                        kinematic_prediction_error_radii
                    ),
                    "identity_kinematic_area_ratio": (
                        kinematic_area_ratio
                    ),
                    **_tracking_partition_metadata(
                        config,
                        "rigid_condition_directed",
                    ),
                    "identity_rejection_streak": (
                        identity_rejection_streak
                    ),
                    "directed_component_selection": (
                        selected_component_diagnostics
                    ),
                },
            )
            detections_by_frame[frame_index].append(direct_detection)
            if identity_valid:
                direct_history.append(direct_detection)
            source_counts[
                "directed" if identity_valid else "replacement"
            ] += 1
            for component_index, (
                extra_xy,
                extra_area,
                extra_mask,
            ) in enumerate(remaining_direct):
                hand_diagnostics: dict[str, Any] = {}
                if (
                    directed_selection_policy
                    == "release_aware_ball_objectness_v3"
                    and scene_kind == "free_fall"
                ):
                    hand_rejected, hand_diagnostics = (
                        free_fall_release_hand_rejection(
                            frame=frame,
                            mask=extra_mask,
                            condition_mask=condition_mask,
                            body_radius_px=body_radius,
                            config=config,
                        )
                    )
                    if hand_rejected:
                        source_counts[
                            "directed_release_hand_rejected"
                        ] += 1
                        rejected_candidates.append(
                            {
                                "frame_index": frame_index,
                                "source": (
                                    "directed_sam2_disconnected_component"
                                ),
                                "xy": extra_xy.tolist(),
                                "area_px2": float(extra_area),
                                **hand_diagnostics,
                            }
                        )
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
                        metadata={
                            "duplicate_directed_component": True,
                            "compact_shape_score": _compact_shape_score(
                                extra_mask,
                                scene_kind=scene_kind,
                            ),
                            "release_hand_diagnostics": (
                                hand_diagnostics
                            ),
                            **_tracking_partition_metadata(
                                config,
                                "rigid_residual",
                            ),
                        },
                    )
                )
        current_direct = (
            selected_direct
            if selected_direct is not None
            else np.zeros(shape, dtype=np.uint8)
        )
        if selected_identity_valid:
            directed_identity_swept = cv2.bitwise_or(
                directed_identity_swept,
                current_direct,
            )
            if swept_exclusion_enabled and len(direct_history) >= 2:
                prior = direct_history[-2]
                current = direct_history[-1]
                bridge_distance = float(
                    np.linalg.norm(current.xy - prior.xy)
                    / max(body_radius, 1.0)
                )
                if bridge_distance <= float(
                    config.get(
                        "directed_identity_swept_maximum_bridge_radii",
                        20.0,
                    )
                ):
                    cv2.line(
                        directed_identity_swept,
                        tuple(np.rint(prior.xy).astype(int).tolist()),
                        tuple(np.rint(current.xy).astype(int).tolist()),
                        255,
                        max(
                            1,
                            int(
                                round(
                                    2.0
                                    * body_radius
                                    * float(
                                        config.get(
                                            "directed_identity_swept_bridge_body_fraction",
                                            0.5,
                                        )
                                    )
                                )
                            ),
                        ),
                    )
        exclusion_seed = cv2.bitwise_or(
            current_direct,
            cv2.bitwise_or(previous_direct, condition_mask),
        )
        if swept_exclusion_enabled:
            exclusion_seed = cv2.bitwise_or(
                exclusion_seed,
                directed_identity_swept,
            )
        exclusion = cv2.dilate(
            exclusion_seed,
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
        if (
            selected_direct is None
            and str(
                config.get(
                    "directed_recovery_policy",
                    "disabled",
                )
            )
            == "one_step_compact_or_change_v1"
        ):
            # Impact or partial boundary clipping may violate the normal
            # compact-shape proposal contract for one frame.  Preserve raw
            # change components solely as gated identity-recovery
            # hypotheses; they are never emitted as free residual objects.
            for _, area, mask in component_centroids(
                condition_difference,
                minimum_area=residual_minimum,
            ):
                if area <= max(direct_maximum, residual_maximum):
                    raw_proposals.append(
                        (mask, "kinematic_recovery_change")
                    )
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
            candidate_xy = mask_centroid(mask)
            condition_static_match: dict[str, float] | None = None
            if candidate_xy is not None:
                candidate_x = int(
                    np.clip(round(float(candidate_xy[0])), 0, width - 1)
                )
                candidate_y = int(
                    np.clip(round(float(candidate_xy[1])), 0, height - 1)
                )
                if condition_static_region[candidate_y, candidate_x] > 0:
                    condition_static_match = {
                        "condition_iou": 0.0,
                        "condition_containment": 0.0,
                        "condition_center_distance_radii": 0.0,
                        "condition_area_ratio": 1.0,
                        "condition_static_region_match": 1.0,
                    }
                for (
                    baseline_mask,
                    baseline_xy,
                    baseline_area,
                ) in condition_static_compact:
                    if condition_static_match is not None:
                        break
                    baseline_iou, baseline_containment = _overlap(
                        mask,
                        baseline_mask,
                    )
                    radius_scale = max(
                        math.sqrt(
                            min(
                                float(proposal_area),
                                baseline_area,
                            )
                            / math.pi
                        ),
                        1.0,
                    )
                    center_distance_radii = float(
                        np.linalg.norm(candidate_xy - baseline_xy)
                        / radius_scale
                    )
                    area_ratio_to_baseline = (
                        float(proposal_area) / max(baseline_area, 1.0)
                    )
                    if (
                        baseline_containment
                        >= float(
                            config.get(
                                "condition_static_compact_minimum_containment",
                                0.55,
                            )
                        )
                        or (
                            center_distance_radii
                            <= float(
                                config.get(
                                    "condition_static_compact_maximum_center_distance_radii",
                                    0.75,
                                )
                            )
                            and 0.4
                            <= area_ratio_to_baseline
                            <= 2.5
                        )
                    ):
                        condition_static_match = {
                            "condition_iou": baseline_iou,
                            "condition_containment": (
                                baseline_containment
                            ),
                            "condition_center_distance_radii": (
                                center_distance_radii
                            ),
                            "condition_area_ratio": (
                                area_ratio_to_baseline
                            ),
                        }
                        break
            if condition_static_match is not None:
                rejected_candidates.append(
                    {
                        "frame_index": frame_index,
                        "candidate_id": (
                            f"condition_static_compact_"
                            f"{frame_index:05d}_"
                            f"{len(rejected_candidates):03d}"
                        ),
                        "rejection_reason": (
                            "frozen_condition_static_compact_geometry"
                        ),
                        "sources": ["independent_compact_shape"],
                        "xy": (
                            candidate_xy.tolist()
                            if candidate_xy is not None
                            else None
                        ),
                        "area_px2": float(proposal_area),
                        "formal_exposure_weight": 0.0,
                        **condition_static_match,
                    }
                )
                source_counts[
                    "condition_static_compact_rejected"
                ] += 1
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
        merged_proposals = _merge_residual_proposals(raw_proposals)
        condition_frame_delta = np.max(
            np.abs(
                frame.astype(np.int16)
                - condition_frame.astype(np.int16)
            ),
            axis=2,
        )
        recovery_index: int | None = None
        recovery_diagnostics: dict[str, Any] | None = None
        if (
            selected_direct is None
            and str(
                config.get(
                    "directed_recovery_policy",
                    "disabled",
                )
            )
            == "one_step_compact_or_change_v1"
            and len(direct_history) >= 2
        ):
            latest = direct_history[-1]
            previous = direct_history[-2]
            gap = frame_index - latest.frame_index
            maximum_gap = int(
                config.get(
                    "directed_recovery_maximum_gap_frames",
                    1,
                )
            )
            if 1 <= gap <= maximum_gap:
                frame_delta = max(
                    latest.frame_index - previous.frame_index,
                    1,
                )
                predicted = latest.xy + (
                    (latest.xy - previous.xy)
                    * (gap / frame_delta)
                )
                maximum_distance = (
                    float(
                        config.get(
                            "directed_recovery_maximum_distance_radii",
                            2.5,
                        )
                    )
                    * body_radius
                )
                recovery_candidates: list[
                    tuple[float, int, float, float]
                ] = []
                for candidate_index, (mask, sources) in enumerate(
                    merged_proposals
                ):
                    centroid = mask_centroid(mask)
                    if centroid is None:
                        continue
                    area = float(np.count_nonzero(mask))
                    area_ratio = area / max(anchor_area, 1)
                    if not (
                        float(
                            config.get(
                                "directed_recovery_minimum_area_ratio",
                                0.35,
                            )
                        )
                        <= area_ratio
                        <= float(
                            config.get(
                                "directed_recovery_maximum_area_ratio",
                                3.0,
                            )
                        )
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(centroid - predicted)
                    )
                    if distance > maximum_distance:
                        continue
                    appearance = _histogram_intersection(
                        anchor_histogram,
                        _mask_histogram(frame, mask),
                    )
                    if appearance < float(
                        config.get(
                            "directed_recovery_minimum_color_similarity",
                            0.12,
                        )
                    ):
                        continue
                    independent = (
                        "independent_compact_shape" in sources
                    )
                    source_penalty = 0.0 if independent else body_radius
                    cost = (
                        distance
                        + source_penalty
                        + 0.25
                        * body_radius
                        * abs(math.log(max(area_ratio, 1e-6)))
                    )
                    recovery_candidates.append(
                        (
                            cost,
                            candidate_index,
                            distance,
                            appearance,
                        )
                    )
                if recovery_candidates:
                    (
                        _,
                        recovery_index,
                        recovery_distance,
                        recovery_appearance,
                    ) = min(recovery_candidates)
                    recovery_diagnostics = {
                        "predicted_xy": predicted.tolist(),
                        "distance_px": recovery_distance,
                        "maximum_distance_px": maximum_distance,
                        "anchor_color_similarity": recovery_appearance,
                        "gap_frames": gap,
                    }
        for residual_index, (mask, sources) in enumerate(
            merged_proposals
        ):
            centroid = mask_centroid(mask)
            if centroid is None:
                continue
            area = int(np.count_nonzero(mask))
            formal_sources = tuple(sorted(set(sources)))
            if residual_index == recovery_index:
                color_similarity = _histogram_intersection(
                    anchor_histogram,
                    _mask_histogram(frame, mask),
                )
                recovered = ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"direct_recovery_{frame_index:05d}"
                    ),
                    xy=centroid,
                    area_px2=float(area),
                    entity_class=entity_class,
                    mask=mask,
                    confidence=0.9,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=(
                        "condition_directed_recovery",
                        *formal_sources,
                    ),
                    metadata={
                        "anchor_color_similarity": color_similarity,
                        "identity_anchor_valid": True,
                        "identity_anchor_raw_valid": True,
                        **_tracking_partition_metadata(
                            config,
                            "rigid_condition_directed",
                        ),
                        "directed_recovery": True,
                        "directed_recovery_diagnostics": (
                            recovery_diagnostics or {}
                        ),
                    },
                )
                detections_by_frame[frame_index].append(recovered)
                direct_history.append(recovered)
                directed_identity_swept = cv2.bitwise_or(
                    directed_identity_swept,
                    mask,
                )
                source_counts["directed_recovery"] += 1
                continue
            if set(formal_sources) == {"kinematic_recovery_change"}:
                continue
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
            compact_shape_score = _compact_shape_score(
                mask,
                scene_kind=scene_kind,
            )
            free_fall_photometric_evidence: dict[str, float] = {}
            if (
                scene_kind == "free_fall"
                and str(
                    config.get(
                        "free_fall_residual_photometric_policy",
                        "disabled",
                    )
                )
                == "condition_only_ball_vs_hand_shadow_v1"
            ):
                free_fall_photometric_evidence = (
                    condition_free_fall_photometric_evidence(
                        frame,
                        mask,
                    )
                )
            independently_credible_duplicate = (
                set(sources) == {"independent_compact_shape"}
                and anchor_color_similarity
                >= float(config["minimum_duplicate_color_similarity"])
            )
            candidate_pixels = mask > 0
            if np.any(candidate_pixels):
                condition_frame_change_fraction = float(
                    np.mean(
                        condition_frame_delta[candidate_pixels]
                        >= float(config["condition_difference_threshold"])
                    )
                )
            else:
                condition_frame_change_fraction = 1.0
            _, condition_subject_containment = _overlap(
                mask,
                condition_mask,
            )
            condition_frame_apparatus_supported = bool(
                condition_frame_change_fraction
                <= float(
                    config.get(
                        "condition_frame_apparatus_maximum_change_fraction",
                        0.25,
                    )
                )
                and condition_subject_containment
                < float(
                    config.get(
                        "condition_frame_apparatus_maximum_subject_containment",
                        0.35,
                    )
                )
            )
            compact_only_audit = bool(
                independently_credible_duplicate
                and str(
                    config.get(
                        "compact_only_evidence_policy",
                        "immediate_participant",
                    )
                )
                in {
                    "audit_only_until_independent_change",
                    "condition_frame_apparatus_only_v1",
                }
                and (
                    str(
                        config.get(
                            "compact_only_evidence_policy",
                            "immediate_participant",
                        )
                    )
                    != "condition_frame_apparatus_only_v1"
                    or condition_frame_apparatus_supported
                )
            )
            physical_sources = {
                value
                for value in sources
                if value != "kinematic_recovery_change"
            }
            low_objectness_change_only_audit = bool(
                str(
                    config.get(
                        "dependent_change_only_policy",
                        "tentative",
                    )
                )
                in {
                    "low_objectness_audit_only",
                    "low_objectness_tentative_v2",
                }
                and physical_sources
                <= {"condition_difference"}
                and compact_shape_score
                < float(
                    config.get(
                        "dependent_change_minimum_objectness",
                        0.45,
                    )
                )
            )
            persistence_only_audit = bool(
                compact_only_audit
                or (
                    low_objectness_change_only_audit
                    and str(
                        config.get(
                            "dependent_change_only_policy",
                            "tentative",
                        )
                    )
                    == "low_objectness_audit_only"
                )
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
                    confidence=float(
                        config.get(
                            "residual_detection_confidence",
                            0.92,
                        )
                    ),
                    evidence_tier=(
                        EvidenceTier.AMBIGUOUS
                        if persistence_only_audit
                        else EvidenceTier.PARTICIPANT
                        if (
                            multi_channel_residual
                            or independently_credible_duplicate
                        )
                        else EvidenceTier.TENTATIVE
                    ),
                    sources=formal_sources,
                    metadata={
                        "compact_shape_score": compact_shape_score,
                        "free_fall_photometric_evidence": (
                            free_fall_photometric_evidence
                        ),
                        "anchor_color_similarity": (
                            anchor_color_similarity
                        ),
                        "independently_credible_duplicate": (
                            independently_credible_duplicate
                        ),
                        "condition_frame_change_fraction": (
                            condition_frame_change_fraction
                        ),
                        "condition_frame_apparatus_supported": (
                            condition_frame_apparatus_supported
                        ),
                        "condition_subject_containment": (
                            condition_subject_containment
                        ),
                        "suppress_persistence_only_promotion": (
                            persistence_only_audit
                        ),
                        "low_objectness_change_only_audit": (
                            low_objectness_change_only_audit
                        ),
                        "residual": True,
                        **_tracking_partition_metadata(
                            config,
                            "rigid_residual",
                        ),
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
    observation = coalesce_condition_directed_identity(
        observation,
        config=config,
    )
    observation = classify_coupled_rigid_body_artifacts(
        observation,
        body_radius_px=body_radius,
        config=config,
    )
    return OpenWorldObservation(
        tracks=observation.tracks,
        overflow_counts=observation.overflow_counts,
        diagnostics={
            **observation.diagnostics,
            "status": "ok",
            "observer_version": rigid_body_observer_version(config),
            "scene_kind": scene_kind,
            "directed_identity_swept_exclusion_policy": str(
                config.get(
                    "directed_identity_swept_exclusion_policy",
                    "disabled",
                )
            ),
            "source_counts": source_counts,
            "roi_fraction": float(np.mean(roi > 0)),
            "apparatus_exclusion": (
                "scene_roi_plus_frozen_condition_geometry_plus_"
                "compact_shape_and_body_size_contract"
            ),
            "expected_channel": "condition_anchored_directed_sam2",
            "residual_channels": [
                "condition_difference",
                "bidirectional_temporal_motion",
                f"compact_{entity_class}",
            ],
            "rejected_candidates": rejected_candidates,
            "merged_directed_fragments": merged_directed_fragments,
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
    reference_cross = np.full(len(expected), np.nan, dtype=np.float64)
    if expected.any():
        _, cross = axis.project(reference.xy[expected])
        reference_cross[expected] = cross / axis.span_px
    empirical_constraints = (
        str(
            scoring_config.get(
                "constraint_reference_mode", "ideal_axis_absolute"
            )
        )
        == "empirical_reference_difference"
    )
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
        cross_error = cross_prediction[observed]
        if empirical_constraints:
            cross_error = (
                cross_error - reference_cross[observed]
            )
        cross_values[observed] = np.exp(
            -np.abs(cross_error) / cross_scale
        )
        monotonic_values = np.zeros(len(expected), dtype=np.float64)
        if empirical_constraints:
            monotonic_values[observed] = 1.0
        for index in range(1, len(expected)):
            if (
                expected[index]
                and expected[index - 1]
                and observed[index]
                and observed[index - 1]
            ):
                if empirical_constraints:
                    step_error = abs(
                        (
                            along_prediction[index]
                            - along_prediction[index - 1]
                        )
                        - (
                            reference_progress[index]
                            - reference_progress[index - 1]
                        )
                    )
                    monotonic_values[index] = math.exp(
                        -step_error
                        / max(
                            float(
                                scoring_config.get(
                                    "empirical_step_error_scale",
                                    0.04,
                                )
                            ),
                            1e-6,
                        )
                    )
                else:
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
        cross_error = cross_prediction[observed]
        if empirical_constraints:
            cross_error = (
                cross_error - reference_cross[observed]
            )
        cross_values[observed] = np.exp(
            -np.abs(cross_error) / cross_scale
        )
        monotonic_values = np.zeros(len(expected), dtype=np.float64)
        if empirical_constraints:
            monotonic_values[observed] = 1.0
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
                if empirical_constraints:
                    step_error = abs(
                        (
                            along_prediction[index]
                            - along_prediction[index - 1]
                        )
                        - (
                            reference_progress[index]
                            - reference_progress[index - 1]
                        )
                    )
                    monotonic_values[index] = math.exp(
                        -step_error
                        / max(
                            float(
                                scoring_config.get(
                                    "empirical_step_error_scale",
                                    0.04,
                                )
                            ),
                            1e-6,
                        )
                    )
                else:
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
        "constraint_reference_mode": (
            "empirical_reference_difference"
            if empirical_constraints
            else "ideal_axis_absolute"
        ),
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
    use_condition_frozen_same_case_state = bool(
        str(
            observer_config.get(
                "state_reference_geometry_policy",
                "legacy_reference_state_v1",
            )
        )
        == "condition_frozen_same_case_state_v2"
        and bool(
            scoring_reference.lifecycle_diagnostics.get(
                "cross_track_and_pose_preserved",
                False,
            )
        )
    )
    state_reference = (
        scoring_reference
        if use_condition_frozen_same_case_state
        else reference
    )
    state = score_rigid_body_state(
        reference=state_reference,
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
                float(state_reference.normalized_progress[index])
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
        "observer_version": result.observation.diagnostics.get(
            "observer_version",
            RIGID_BODY_OPEN_WORLD_VERSION,
        ),
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


def build_condition_rigid_body_seed_hypotheses(
    condition_frame: np.ndarray,
    *,
    scene_kind: str,
    minimum_area: int,
    maximum_area: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rank causal condition-frame prompts before directed segmentation."""

    frame = np.asarray(condition_frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    nominal_area = max(
        float(
            config.get(
                "condition_anchor_nominal_area_fraction",
                0.0015,
            )
        )
        * height
        * width,
        float(minimum_area),
    )
    proposals = _independent_compact_proposals(
        frame,
        scene_kind=scene_kind,
        anchor_area_px2=nominal_area,
        minimum_area=minimum_area,
        maximum_area=maximum_area,
        config=config,
    )
    raw: list[tuple[np.ndarray, str, dict[str, Any]]] = [
        (
            proposal,
            "condition_independent_compact_geometry",
            {},
        )
        for proposal in proposals
    ]
    apparatus_seed: np.ndarray | None = None
    apparatus_diagnostics: dict[str, Any] = {}
    if scene_kind == "inclined_plane":
        apparatus_seed, apparatus_diagnostics = (
            build_condition_incline_apparatus_seed(
                frame,
                minimum_area=minimum_area,
                maximum_area=maximum_area,
                config=config,
            )
        )
        if apparatus_seed is not None:
            raw.insert(
                0,
                (
                    apparatus_seed,
                    "condition_incline_apparatus_uphill_seed",
                    apparatus_diagnostics,
                ),
            )
    ranked: list[dict[str, Any]] = []
    for mask, source, source_diagnostics in raw:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        centroid = mask_centroid(binary)
        area = int(np.count_nonzero(binary))
        if (
            centroid is None
            or area < minimum_area
            or area > maximum_area
        ):
            continue
        shape = _compact_shape_score(binary, scene_kind=scene_kind)
        touches = _mask_canvas_edges(binary)
        interior = 1.0 if not touches else 0.0
        scale_support = math.exp(
            -0.5
            * abs(
                math.log(
                    max(float(area), 1.0)
                    / max(nominal_area, 1.0)
                )
            )
        )
        apparatus_overlap = 0.0
        apparatus_containment = 0.0
        if apparatus_seed is not None:
            apparatus_overlap, apparatus_containment = _overlap(
                binary,
                apparatus_seed,
            )
            if (
                scene_kind == "inclined_plane"
                and source
                != "condition_incline_apparatus_uphill_seed"
                and apparatus_containment
                < float(
                    config.get(
                        "condition_incline_seed_minimum_candidate_"
                        "containment",
                        0.18,
                    )
                )
            ):
                # The protractor and support base contain many excellent
                # rectangles.  They are apparatus, not release candidates:
                # an independent seed must intersect the condition-frozen
                # uphill body envelope before SAM is allowed to promote it.
                continue
        if scene_kind == "free_fall":
            upper_support = float(
                np.clip(
                    1.0
                    - float(centroid[1])
                    / max(0.45 * height, 1.0),
                    0.0,
                    1.0,
                )
            )
            rank_score = (
                0.50 * upper_support
                + 0.32 * shape
                + 0.10 * interior
                + 0.08 * scale_support
            )
        else:
            upper_support = 0.0
            apparatus_source = float(
                source == "condition_incline_apparatus_uphill_seed"
            )
            resolved_area_support = min(
                float(area)
                / max(
                    float(
                        np.count_nonzero(apparatus_seed)
                        if apparatus_seed is not None
                        else nominal_area
                    ),
                    1.0,
                ),
                1.0,
            )
            rank_score = (
                0.35 * apparatus_containment
                + 0.22 * resolved_area_support
                + 0.18 * shape
                + 0.15 * apparatus_source
                + 0.10 * interior
            )
        ranked.append(
            {
                "mask": binary,
                "source": source,
                "source_diagnostics": source_diagnostics,
                "rank_score": float(rank_score),
                "area_px2": area,
                "centroid_xy": centroid.tolist(),
                "shape_score": shape,
                "upper_support": upper_support,
                "scale_support": scale_support,
                "apparatus_iou": apparatus_overlap,
                "apparatus_containment": apparatus_containment,
            }
        )
    ordered = sorted(
        ranked,
        key=lambda item: (
            float(item["rank_score"]),
            float(item["area_px2"]),
        ),
        reverse=True,
    )
    limit = max(
        1,
        int(
            config.get(
                "condition_anchor_maximum_candidates",
                8 if scene_kind == "free_fall" else 5,
            )
        ),
    )
    selected: list[dict[str, Any]] = []
    for item in ordered:
        duplicate = False
        for prior in selected:
            iou, containment = _overlap(
                item["mask"],
                prior["mask"],
            )
            if iou >= 0.45 or containment >= 0.82:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def build_rigid_body_motion_prompts(
    frames: Sequence[np.ndarray],
    *,
    scene_kind: str,
    threshold: float,
    minimum_area: int,
    box_expand: float,
    minimum_box_side: int,
    config: Mapping[str, Any],
    maximum_candidates: int = 4,
) -> list[MaskPrompt]:
    """Build diverse compact-body hypotheses from reference-side motion.

    The historical observer selected the largest temporal-median component.
    That is frequently a hand or apparatus edge.  V7 instead keeps several
    compact hypotheses and lets their complete, forward/backward reference
    tracks compete.  Candidate generation never uses a Case ID, annotation
    pixel coordinate, or prediction frame.
    """

    values = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not values:
        raise SceneAnalysisError(
            "empty_reference_hypothesis_timeline",
            "cannot propose a rigid body from an empty reference",
        )
    if scene_kind not in {"free_fall", "inclined_plane"}:
        raise ValueError(f"unsupported rigid-body scene kind: {scene_kind}")
    limit = max(1, int(maximum_candidates))
    proposals = motion_masks(
        values,
        threshold=float(threshold),
        minimum_area=int(minimum_area),
    )
    height, width = values[0].shape[:2]
    canvas_area = float(height * width)
    margin_x = max(1, int(round(0.01 * width)))
    margin_y = max(1, int(round(0.01 * height)))
    maximum_area = int(
        round(
            canvas_area
            * float(
                config.get(
                    "reference_candidate_maximum_area_ratio",
                    0.08 if scene_kind == "free_fall" else 0.12,
                )
            )
        )
    )
    minimum_compact = float(
        config.get(
            "reference_candidate_minimum_compact_score",
            0.16 if scene_kind == "free_fall" else 0.12,
        )
    )
    maximum_aspect = float(
        config.get(
            "reference_candidate_maximum_aspect_ratio",
            3.5 if scene_kind == "free_fall" else 5.0,
        )
    )
    candidates: list[
        tuple[
            float,
            int,
            np.ndarray,
            np.ndarray,
            float,
            float,
            tuple[int, int, int, int],
        ]
    ] = []
    for frame_index, proposal in enumerate(proposals):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            proposal
        )
        for label in range(1, count):
            x, y, box_width, box_height, area = [
                int(value) for value in stats[label]
            ]
            if (
                area < int(minimum_area)
                or area > maximum_area
                or x <= margin_x
                or y <= margin_y
                or x + box_width >= width - margin_x
                or y + box_height >= height - margin_y
            ):
                continue
            aspect = max(box_width, box_height) / max(
                min(box_width, box_height), 1
            )
            if aspect > maximum_aspect:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            compact = _compact_shape_score(
                component,
                scene_kind=scene_kind,
            )
            if compact < minimum_compact:
                continue
            histogram = _mask_histogram(
                values[frame_index],
                component,
            )
            # Tiny threshold speckles should not outrank a consistently
            # resolved body merely because they are perfectly round.
            area_support = float(
                np.clip(
                    math.log1p(area / max(float(minimum_area), 1.0))
                    / math.log(17.0),
                    0.0,
                    1.0,
                )
            )
            rank_score = 0.78 * compact + 0.22 * area_support
            candidates.append(
                (
                    rank_score,
                    frame_index,
                    np.asarray(centroids[label], dtype=np.float64),
                    histogram,
                    float(area),
                    compact,
                    (x, y, box_width, box_height),
                )
            )
    if not candidates:
        raise SceneAnalysisError(
            "rigid_body_reference_candidates_missing",
            "no compact reference-side motion hypothesis survived geometry "
            "filtering",
        )

    # Add persistence and displacement evidence without collapsing candidates
    # into a single identity.  Similar appearance/scale in other frames is
    # supporting evidence; it is deliberately not enough to choose the final
    # subject before full-track validation.
    rescored: list[
        tuple[
            float,
            int,
            np.ndarray,
            np.ndarray,
            float,
            float,
            tuple[int, int, int, int],
        ]
    ] = []
    for candidate in candidates:
        (
            rank_score,
            frame_index,
            centroid,
            histogram,
            area,
            compact,
            box,
        ) = candidate
        support_frames: set[int] = set()
        supporting_xy: list[np.ndarray] = []
        for other in candidates:
            if other[1] == frame_index:
                continue
            area_ratio = other[4] / max(area, 1.0)
            if not 0.35 <= area_ratio <= 2.85:
                continue
            if _histogram_intersection(histogram, other[3]) < 0.42:
                continue
            support_frames.add(int(other[1]))
            supporting_xy.append(other[2])
        persistence = min(
            len(support_frames)
            / max(0.2 * len(values), 3.0),
            1.0,
        )
        if supporting_xy:
            points = np.vstack([centroid, *supporting_xy])
            if scene_kind == "free_fall":
                displacement = float(np.ptp(points[:, 1]))
            else:
                displacement = float(
                    np.linalg.norm(np.ptp(points, axis=0))
                )
            body_diameter = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
            motion_support = float(
                np.clip(displacement / max(4.0 * body_diameter, 1.0), 0, 1)
            )
        else:
            motion_support = 0.0
        score = (
            0.62 * rank_score
            + 0.23 * persistence
            + 0.15 * motion_support
        )
        rescored.append(
            (
                score,
                frame_index,
                centroid,
                histogram,
                area,
                compact,
                box,
            )
        )

    selected: list[MaskPrompt] = []
    selected_signatures: list[
        tuple[int, np.ndarray, np.ndarray, float]
    ] = []
    ordered = sorted(rescored, key=lambda item: item[0], reverse=True)
    if (
        str(
            config.get(
                "reference_candidate_diversity_policy",
                "rank_only_v1",
            )
        )
        == "rank_plus_area_champion_v2"
    ):
        # A fast body is frequently a large, elongated motion-blur component.
        # Pure compactness ranking otherwise fills every hypothesis slot with
        # tiny hand/ruler highlights.  Reserve one slot for the strongest
        # area-supported candidate, then retain the historical ranked order.
        area_champion = max(
            rescored,
            key=lambda item: item[4] * max(item[5], 0.10),
        )
        ordered = [
            area_champion,
            *[item for item in ordered if item is not area_champion],
        ]
    for (
        score,
        frame_index,
        centroid,
        histogram,
        area,
        compact,
        (x, y, box_width, box_height),
    ) in ordered:
        duplicate = False
        for (
            prior_frame,
            prior_centroid,
            prior_histogram,
            prior_area,
        ) in selected_signatures:
            temporal_distance = abs(frame_index - prior_frame)
            spatial_distance = float(
                np.linalg.norm(centroid - prior_centroid)
            )
            body_scale = 2.0 * math.sqrt(
                max(min(area, prior_area), 1.0) / math.pi
            )
            similar_appearance = (
                _histogram_intersection(histogram, prior_histogram) >= 0.78
            )
            similar_scale = 0.55 <= area / max(prior_area, 1.0) <= 1.8
            if (
                similar_appearance
                and similar_scale
                and (
                    temporal_distance <= max(2, len(values) // 12)
                    or spatial_distance <= 2.5 * body_scale
                )
            ):
                duplicate = True
                break
        if duplicate:
            continue
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        expanded_width = max(
            float(minimum_box_side),
            box_width * float(box_expand),
        )
        expanded_height = max(
            float(minimum_box_side),
            box_height * float(box_expand),
        )
        box_xyxy = np.asarray(
            [
                max(0.0, center_x - expanded_width / 2.0),
                max(0.0, center_y - expanded_height / 2.0),
                min(width - 1.0, center_x + expanded_width / 2.0),
                min(height - 1.0, center_y + expanded_height / 2.0),
            ],
            dtype=np.float32,
        )
        selected.append(
            MaskPrompt(
                frame_index=int(frame_index),
                box_xyxy=box_xyxy,
                points_xy=np.asarray(
                    centroid, dtype=np.float32
                ).reshape(1, 2),
                point_labels=np.ones(1, dtype=np.int32),
                metadata={
                    "source": (
                        "reference_multi_hypothesis_compact_motion"
                    ),
                    "candidate_rank_score": float(score),
                    "candidate_compact_score": float(compact),
                    "candidate_seed_area_px2": float(area),
                    "candidate_diversity_policy": str(
                        config.get(
                            "reference_candidate_diversity_policy",
                            "rank_only_v1",
                        )
                    ),
                    "candidate_area_supported_seed": bool(
                        ordered
                        and (
                            frame_index,
                            float(area),
                        )
                        == (
                            ordered[0][1],
                            float(ordered[0][4]),
                        )
                        and str(
                            config.get(
                                "reference_candidate_diversity_policy",
                                "rank_only_v1",
                            )
                        )
                        == "rank_plus_area_champion_v2"
                    ),
                    "motion_box_xywh": [
                        x,
                        y,
                        box_width,
                        box_height,
                    ],
                },
            )
        )
        selected_signatures.append(
            (frame_index, centroid, histogram, area)
        )
        if len(selected) >= limit:
            break
    if not selected:
        raise SceneAnalysisError(
            "rigid_body_reference_candidates_missing",
            "reference candidate diversity suppression removed every "
            "compact motion hypothesis",
        )
    return selected


def select_rigid_body_reference_hypothesis(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    config: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Choose among already validated tracks without scene/Case identifiers.

    ``validation_plus_consensus_scale_v2`` fixes a subtle failure of plain
    validation maximization: a tiny, almost static apparatus highlight can
    be exceptionally compact and continuous.  Real body hypotheses proposed
    at different times agree on their tracked scale.  Consensus is weighted
    by equivalent diameter, so several tiny speckles cannot outvote one
    well-resolved body merely through proposal multiplicity.
    """

    accepted = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("status") == "accepted"
    ]
    if not accepted:
        raise ValueError("reference selection requires an accepted candidate")
    if policy == "compact_multi_hypothesis_v1":
        selected = max(
            accepted,
            key=lambda item: float(item["validation"]["score"]),
        )
        return int(selected["candidate_index"]), {
            "policy": policy,
            "selection_contract": (
                "maximum_generic_reference_validation_score"
            ),
            "selected_adjusted_score": float(
                selected["validation"]["score"]
            ),
        }
    if policy != "validation_plus_consensus_scale_v2":
        raise ValueError(
            f"unsupported rigid-body reference hypothesis policy: {policy}"
        )
    if len(accepted) == 1:
        selected = accepted[0]
        return int(selected["candidate_index"]), {
            "policy": policy,
            "selection_contract": (
                "single_validated_hypothesis_no_consensus_available"
            ),
            "consensus_candidate_count": 1,
            "selected_adjusted_score": float(
                selected["validation"]["score"]
            ),
        }
    bandwidth = max(
        float(
            config.get(
                "reference_scale_consensus_log_area_bandwidth",
                0.55,
            )
        ),
        1e-6,
    )
    validation_weight = float(
        config.get("reference_scale_validation_weight", 1.0)
    )
    consensus_weight = float(
        config.get("reference_scale_consensus_weight", 0.20)
    )
    resolved_scale_weight = float(
        config.get("reference_scale_resolved_body_weight", 0.08)
    )
    internal_weight = float(
        config.get("reference_scale_internal_consistency_weight", 0.04)
    )
    log_areas = np.asarray(
        [
            float(item["scale_observation"]["robust_log_area"])
            for item in accepted
        ],
        dtype=np.float64,
    )
    internal = np.asarray(
        [
            float(item["scale_observation"]["internal_consistency"])
            for item in accepted
        ],
        dtype=np.float64,
    )
    validation = np.asarray(
        [
            float(item["validation"]["score"])
            for item in accepted
        ],
        dtype=np.float64,
    )
    diameters = np.exp(0.5 * log_areas)
    evidence_weights = (
        np.maximum(validation, 1e-6)
        * np.maximum(diameters, 1e-6)
        * np.maximum(internal, 0.05)
    )
    pairwise = np.exp(
        -0.5
        * (
            (log_areas[:, None] - log_areas[None, :])
            / bandwidth
        )
        ** 2
    )
    consensus = (
        pairwise @ evidence_weights
        / max(float(np.sum(evidence_weights)), 1e-12)
    )
    resolved = diameters / max(float(np.max(diameters)), 1e-12)
    adjusted = (
        validation_weight * validation
        + consensus_weight * consensus
        + resolved_scale_weight * resolved
        + internal_weight * internal
    )
    for index, item in enumerate(accepted):
        item["selection_adjusted_score"] = float(adjusted[index])
        item["selection_consensus_support"] = float(consensus[index])
        item["selection_resolved_scale_support"] = float(resolved[index])
    selected_index = int(np.argmax(adjusted))
    selected = accepted[selected_index]
    # A weighted median is diagnostic only; selection uses the full kernel so
    # multimodal candidates do not collapse into an unobserved average scale.
    order = np.argsort(log_areas)
    cumulative = np.cumsum(evidence_weights[order])
    center_index = int(
        order[
            np.searchsorted(
                cumulative,
                0.5 * float(cumulative[-1]),
                side="left",
            )
        ]
    )
    return int(selected["candidate_index"]), {
        "policy": policy,
        "selection_contract": (
            "validated_track_plus_equivalent_diameter_weighted_"
            "log_scale_consensus_no_case_id_no_prediction_pixels"
        ),
        "consensus_candidate_count": len(accepted),
        "consensus_log_area_center": float(log_areas[center_index]),
        "consensus_equivalent_diameter_px": float(
            diameters[center_index]
        ),
        "consensus_bandwidth_log_area": bandwidth,
        "selected_adjusted_score": float(adjusted[selected_index]),
        "candidate_selection": [
            {
                "candidate_index": int(item["candidate_index"]),
                "validation_score": float(
                    item["validation"]["score"]
                ),
                "robust_log_area": float(
                    item["scale_observation"]["robust_log_area"]
                ),
                "equivalent_diameter_px": float(diameters[index]),
                "internal_consistency": float(internal[index]),
                "consensus_support": float(consensus[index]),
                "resolved_scale_support": float(resolved[index]),
                "adjusted_score": float(adjusted[index]),
            }
            for index, item in enumerate(accepted)
        ],
    }


def score_rigid_body_reference_hypothesis(
    reference: RigidBodyReference,
    *,
    frames: Sequence[np.ndarray],
    scene_kind: str,
    config: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Score a candidate reference track using generic physical continuity."""

    observed = np.asarray(reference.raw_observed, dtype=bool)
    diagnostics: dict[str, Any] = {
        "observed_ratio": float(np.mean(observed)),
        "condition_observed": bool(observed[0]),
        "legal_exit_frame": reference.legal_exit_frame,
    }
    if not observed[0]:
        diagnostics.update(
            {
                "accepted": False,
                "reason": "condition_frame_not_observed",
                "score": 0.0,
            }
        )
        return 0.0, diagnostics
    observed_indices = np.flatnonzero(observed)
    shape_scores = np.asarray(
        [
            _compact_shape_score(
                reference.masks[index],
                scene_kind=scene_kind,
            )
            for index in observed_indices
        ],
        dtype=np.float64,
    )
    condition_shape = float(shape_scores[0])
    median_shape = float(np.median(shape_scores))
    areas = np.asarray(
        reference.areas_px2[observed],
        dtype=np.float64,
    )
    log_areas = np.log(np.maximum(areas, 1.0))
    area_mad = float(
        np.median(np.abs(log_areas - np.median(log_areas)))
    )
    area_stability = math.exp(
        -area_mad
        / max(
            float(
                config.get(
                    "reference_hypothesis_log_area_mad_scale",
                    0.45,
                )
            ),
            1e-6,
        )
    )
    adjacent = observed[:-1] & observed[1:]
    adjacent_observation_steps = int(np.count_nonzero(adjacent))
    step_vectors = (
        reference.xy[1:][adjacent]
        - reference.xy[:-1][adjacent]
    )
    body_diameter = 2.0 * math.sqrt(
        max(float(np.median(areas)), 1.0) / math.pi
    )
    if len(step_vectors):
        step_lengths = np.linalg.norm(step_vectors, axis=1)
        median_step = float(np.median(step_lengths))
        step_mad = float(
            np.median(np.abs(step_lengths - median_step))
        )
        robust_limit = max(
            median_step + 4.0 * step_mad,
            1.5 * body_diameter,
            1.0,
        )
        excess = max(
            float(np.max(step_lengths)) / robust_limit - 1.0,
            0.0,
        )
        continuity = math.exp(-excess)
        along_steps = step_vectors @ reference.axis.direction_xy
        direction_consistency = float(
            np.mean(along_steps >= -0.15 * body_diameter)
        )
    else:
        median_step = 0.0
        step_mad = 0.0
        continuity = 0.0
        direction_consistency = 0.0
    total_displacement = 0.0
    if scene_kind == "free_fall":
        total_displacement = float(
            reference.xy[observed_indices[-1], 1]
            - reference.xy[observed_indices[0], 1]
        )
        if total_displacement <= 0.0:
            direction_consistency *= 0.25
    coverage = float(np.mean(observed))
    observed_span = int(observed_indices[-1] - observed_indices[0] + 1)
    internal_gap_frames = int(
        observed_span - int(np.count_nonzero(observed[observed_indices[0] : observed_indices[-1] + 1]))
    )
    span_support = float(
        np.clip(
            reference.axis.span_px
            / max(4.0 * body_diameter, 1.0),
            0.0,
            1.0,
        )
    )
    explained = float(reference.axis.explained_ratio)
    score = float(
        np.clip(
            0.22 * condition_shape
            + 0.18 * median_shape
            + 0.16 * area_stability
            + 0.16 * continuity
            + 0.12 * direction_consistency
            + 0.10 * coverage
            + 0.04 * span_support
            + 0.02 * explained,
            0.0,
            1.0,
        )
    )
    minimum_raw_coverage = float(
        config.get("reference_hypothesis_minimum_raw_coverage", 0.0)
    )
    minimum_adjacent_steps = int(
        config.get("reference_hypothesis_minimum_adjacent_steps", 0)
    )
    minimum_direction_consistency = float(
        config.get(
            "reference_hypothesis_minimum_direction_consistency",
            0.0,
        )
    )
    maximum_internal_gap_frames = int(
        config.get(
            "reference_hypothesis_maximum_internal_gap_frames",
            len(observed),
        )
    )
    minimum_area_stability = float(
        config.get(
            "reference_hypothesis_minimum_area_stability",
            0.0,
        )
    )
    minimum_score = float(
        config.get("reference_hypothesis_minimum_score", 0.0)
    )
    minimum_free_fall_displacement = (
        float(
            config.get(
                "reference_hypothesis_minimum_free_fall_downward_span_fraction",
                0.0,
            )
        )
        * float(frames[0].shape[0])
        if scene_kind == "free_fall"
        else 0.0
    )
    accepted = bool(
        condition_shape
        >= float(
            config.get(
                "reference_hypothesis_minimum_condition_shape",
                0.12,
            )
        )
        and median_shape
        >= float(
            config.get(
                "reference_hypothesis_minimum_median_shape",
                0.10,
            )
        )
        and continuity
        >= float(
            config.get(
                "reference_hypothesis_minimum_continuity",
                0.2,
            )
        )
        and coverage >= minimum_raw_coverage
        and adjacent_observation_steps >= minimum_adjacent_steps
        and direction_consistency >= minimum_direction_consistency
        and internal_gap_frames <= maximum_internal_gap_frames
        and area_stability >= minimum_area_stability
        and score >= minimum_score
        and (
            scene_kind != "free_fall"
            or total_displacement >= minimum_free_fall_displacement
        )
    )
    diagnostics.update(
        {
            "score": score,
            "accepted": accepted,
            "reason": (
                "generic_shape_motion_continuity_supported"
                if accepted
                else "generic_reference_hypothesis_validation_failed"
            ),
            "condition_shape_score": condition_shape,
            "median_shape_score": median_shape,
            "log_area_mad": area_mad,
            "area_stability": area_stability,
            "body_diameter_px": body_diameter,
            "median_step_px": median_step,
            "step_mad_px": step_mad,
            "continuity_score": continuity,
            "direction_consistency": direction_consistency,
            "total_displacement_px": total_displacement,
            "minimum_free_fall_displacement_px": (
                minimum_free_fall_displacement
            ),
            "adjacent_observation_steps": adjacent_observation_steps,
            "minimum_adjacent_observation_steps": (
                minimum_adjacent_steps
            ),
            "internal_gap_frames": internal_gap_frames,
            "maximum_internal_gap_frames": maximum_internal_gap_frames,
            "minimum_raw_coverage": minimum_raw_coverage,
            "minimum_direction_consistency": (
                minimum_direction_consistency
            ),
            "minimum_area_stability": minimum_area_stability,
            "minimum_score": minimum_score,
            "motion_span_px": reference.axis.span_px,
            "motion_span_support": span_support,
            "axis_explained_ratio": explained,
        }
    )
    return score, diagnostics


def recover_rigid_body_mask_gaps(
    masks: Sequence[np.ndarray],
    *,
    frames: Sequence[np.ndarray],
    scene_kind: str,
    minimum_area: int,
    maximum_area_ratio: float,
    config: Mapping[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Recover a one-frame directed gap from local, predicted pixel change.

    The same bounded rule is used for reference-side formal observation and
    prediction-side identity recovery.  It does not interpolate a
    disappearance: a nearby change component must exist in the current
    pixels, and only the immediately predicted position is eligible.
    """

    values = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    output = [
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        for mask in masks
    ]
    if not values or len(values) != len(output):
        raise SceneAnalysisError(
            "rigid_body_gap_recovery_timeline_mismatch",
            "mask gap recovery requires one frame per mask",
        )
    if (
        str(
            config.get(
                "directed_recovery_policy",
                "disabled",
            )
        )
        != "one_step_compact_or_change_v1"
    ):
        return output, {
            "policy": "disabled",
            "recovered_frames": [],
        }
    height, width = output[0].shape
    maximum_area = int(
        round(height * width * float(maximum_area_ratio))
    )
    anchor_geometry = _valid_mask_geometry(
        output[0],
        minimum_area=int(minimum_area),
        maximum_area=maximum_area,
    )
    if anchor_geometry is None:
        return output, {
            "policy": "one_step_compact_or_change_v1",
            "recovered_frames": [],
            "status": "condition_anchor_invalid",
        }
    anchor_mask, _, anchor_area = anchor_geometry
    anchor_histogram = _mask_histogram(values[0], anchor_mask)
    body_radius = math.sqrt(max(anchor_area, 1.0) / math.pi)
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
    history: list[tuple[int, np.ndarray, float, np.ndarray]] = []
    recovered: list[dict[str, Any]] = []
    maximum_gap = int(
        config.get("directed_recovery_maximum_gap_frames", 1)
    )
    for frame_index in range(len(output)):
        geometry = _valid_mask_geometry(
            output[frame_index],
            minimum_area=int(minimum_area),
            maximum_area=maximum_area,
        )
        if geometry is not None:
            component, centroid, area = geometry
            output[frame_index] = component
            history.append(
                (frame_index, centroid, float(area), component)
            )
            continue
        if len(history) < 2:
            continue
        latest_index, latest_xy, latest_area, latest_mask = history[-1]
        previous_index, previous_xy, _, _ = history[-2]
        gap = frame_index - latest_index
        if (
            gap < 1
            or gap > maximum_gap
            or latest_index <= previous_index
        ):
            continue
        velocity = (
            (latest_xy - previous_xy)
            / (latest_index - previous_index)
        )
        predicted = latest_xy + velocity * gap
        difference = _threshold_difference(
            values[frame_index],
            values[0],
            threshold=float(config["condition_difference_threshold"]),
        )
        exclusion = cv2.dilate(
            cv2.bitwise_or(latest_mask, anchor_mask),
            exclusion_kernel,
        )
        difference[exclusion > 0] = 0
        minimum_ratio = float(
            config.get(
                "directed_recovery_minimum_area_ratio",
                0.35,
            )
        )
        maximum_ratio = float(
            config.get(
                "directed_recovery_maximum_area_ratio",
                3.0,
            )
        )
        maximum_distance = (
            float(
                config.get(
                    "directed_recovery_maximum_distance_radii",
                    2.5,
                )
            )
            * body_radius
        )
        candidates: list[
            tuple[float, np.ndarray, np.ndarray, float, float]
        ] = []
        for centroid, area, component in component_centroids(
            difference,
            minimum_area=max(
                int(minimum_area),
                int(round(anchor_area * minimum_ratio)),
            ),
        ):
            if area > min(
                maximum_area,
                int(round(anchor_area * maximum_ratio)),
            ):
                continue
            distance = float(np.linalg.norm(centroid - predicted))
            if distance > maximum_distance:
                continue
            appearance = _histogram_intersection(
                anchor_histogram,
                _mask_histogram(
                    values[frame_index],
                    component,
                ),
            )
            if appearance < float(
                config.get(
                    "directed_recovery_minimum_color_similarity",
                    0.12,
                )
            ):
                continue
            cost = (
                distance
                + 0.25
                * body_radius
                * abs(math.log(max(area / anchor_area, 1e-6)))
            )
            candidates.append(
                (
                    cost,
                    component,
                    centroid,
                    float(area),
                    appearance,
                )
            )
        if not candidates:
            continue
        _, component, centroid, area, appearance = min(
            candidates,
            key=lambda item: item[0],
        )
        output[frame_index] = component
        history.append((frame_index, centroid, area, component))
        recovered.append(
            {
                "frame_index": frame_index,
                "predicted_xy": predicted.tolist(),
                "recovered_xy": centroid.tolist(),
                "distance_px": float(
                    np.linalg.norm(centroid - predicted)
                ),
                "area_px2": area,
                "anchor_color_similarity": appearance,
                "evidence": "current_pixel_change_at_predicted_position",
            }
        )
    return output, {
        "policy": "one_step_compact_or_change_v1",
        "recovered_frames": recovered,
        "missing_frames_remain": [
            index
            for index, mask in enumerate(output)
            if int(np.count_nonzero(mask)) < int(minimum_area)
        ],
    }


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
    """Shared, policy-gated implementation for v6/v7 rigid-body scenes."""

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
    required_observer_version: str | None = None
    required_observer_policies: Mapping[str, str] = {}

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])
        self._observer_config = merged_rigid_body_config(
            config, scene_kind=self.scene_kind
        )
        if self.required_observer_version is not None:
            actual_version = rigid_body_observer_version(
                self._observer_config
            )
            if actual_version != self.required_observer_version:
                raise ValueError(
                    f"{self.__class__.__name__} requires rigid-body "
                    f"observer {self.required_observer_version}; received "
                    f"{actual_version}"
                )
        mismatches = {
            key: {
                "required": expected,
                "received": self._observer_config.get(key),
            }
            for key, expected in self.required_observer_policies.items()
            if self._observer_config.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}={value['received']!r} "
                f"(required {value['required']!r})"
                for key, value in sorted(mismatches.items())
            )
            raise ValueError(
                f"{self.__class__.__name__} observer policy contract "
                f"is incomplete: {details}"
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
        reference_policy = str(
            self._observer_config.get(
                "reference_hypothesis_policy",
                "largest_motion_component",
            )
        )
        if reference_policy in {
            "compact_multi_hypothesis_v1",
            "validation_plus_consensus_scale_v2",
        }:
            quality = self.config["quality"]
            prompts = build_rigid_body_motion_prompts(
                frames,
                scene_kind=self.scene_kind,
                threshold=float(proposal["threshold"]),
                minimum_area=int(proposal["minimum_area"]),
                box_expand=float(proposal["box_expand"]),
                minimum_box_side=int(proposal["minimum_box_side"]),
                config=self._observer_config,
                maximum_candidates=int(
                    self._observer_config.get(
                        "reference_hypothesis_maximum_candidates",
                        4,
                    )
                ),
            )
            hypotheses: list[
                tuple[
                    float,
                    list[np.ndarray],
                    RigidBodyReference,
                    dict[str, Any],
                    int,
                ]
            ] = []
            diagnostics: list[dict[str, Any]] = []
            for candidate_index, prompt in enumerate(prompts):
                record: dict[str, Any] = {
                    "candidate_index": candidate_index,
                    "prompt_frame": prompt.frame_index,
                    "prompt_metadata": prompt.metadata,
                }
                try:
                    masks, segmentation = self._segmenter.segment(
                        frames,
                        prompt=prompt,
                        temporary_prefix=(
                            f"{self.temporary_prefix}"
                            f"reference_hypothesis_{candidate_index}_"
                        ),
                    )
                    reference = build_rigid_body_reference(
                        masks,
                        entity_id="reference_hypothesis",
                        entity_class=self.entity_class,
                        scene_kind=self.scene_kind,
                        frame_shape=frames[0].shape[:2],
                        minimum_area=int(
                            quality["minimum_mask_pixels"]
                        ),
                        maximum_area_ratio=float(
                            quality["maximum_mask_area_ratio"]
                        ),
                        minimum_span_px=float(
                            quality[self.minimum_span_quality_key]
                        ),
                        config=self._observer_config,
                    )
                    score, validation = (
                        score_rigid_body_reference_hypothesis(
                            reference,
                            frames=frames,
                            scene_kind=self.scene_kind,
                            config=self._observer_config,
                        )
                    )
                    record.update(
                        {
                            "status": (
                                "accepted"
                                if validation["accepted"]
                                else "rejected"
                            ),
                            "validation": validation,
                            "segmentation": segmentation,
                        }
                    )
                    if validation["accepted"]:
                        observed_areas = np.asarray(
                            reference.areas_px2[
                                np.asarray(
                                    reference.raw_observed,
                                    dtype=bool,
                                )
                            ],
                            dtype=np.float64,
                        )
                        seed_area = float(
                            prompt.metadata.get(
                                "candidate_seed_area_px2",
                                max(
                                    (
                                        prompt.metadata.get(
                                            "motion_box_xywh",
                                            [0, 0, 1, 1],
                                        )[2]
                                        * prompt.metadata.get(
                                            "motion_box_xywh",
                                            [0, 0, 1, 1],
                                        )[3]
                                    ),
                                    1.0,
                                ),
                            )
                        )
                        scale_areas = np.asarray(
                            [
                                seed_area,
                                max(
                                    float(reference.areas_px2[0]),
                                    1.0,
                                ),
                                max(
                                    float(np.median(observed_areas)),
                                    1.0,
                                ),
                            ],
                            dtype=np.float64,
                        )
                        scale_logs = np.log(
                            np.maximum(scale_areas, 1.0)
                        )
                        scale_mad = float(
                            np.median(
                                np.abs(
                                    scale_logs
                                    - np.median(scale_logs)
                                )
                            )
                        )
                        scale_observation = {
                            "seed_area_px2": float(scale_areas[0]),
                            "condition_area_px2": float(scale_areas[1]),
                            "median_observed_area_px2": float(
                                scale_areas[2]
                            ),
                            "robust_log_area": float(
                                np.median(scale_logs)
                            ),
                            "log_area_mad": scale_mad,
                            "internal_consistency": float(
                                math.exp(
                                    -scale_mad
                                    / max(
                                        float(
                                            self._observer_config.get(
                                                "reference_scale_internal_"
                                                "log_area_mad_scale",
                                                0.70,
                                            )
                                        ),
                                        1e-6,
                                    )
                                )
                            ),
                        }
                        record["scale_observation"] = scale_observation
                        hypotheses.append(
                            (
                                score,
                                masks,
                                reference,
                                segmentation,
                                candidate_index,
                            )
                        )
                except Exception as exc:
                    record.update(
                        {
                            "status": "rejected",
                            "reason_code": getattr(
                                exc,
                                "code",
                                "reference_hypothesis_failed",
                            ),
                            "reason": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                diagnostics.append(record)
            if not hypotheses:
                reasons = "; ".join(
                    (
                        f"#{item['candidate_index']}:"
                        f"{item.get('reason_code', item.get('status'))}"
                    )
                    for item in diagnostics
                )
                raise ReferenceAnalysisError(
                    "reference_rigid_body_hypotheses_exhausted",
                    "no generic compact-motion reference hypothesis "
                    f"passed validation ({reasons})",
                )
            selected_index, selection = (
                select_rigid_body_reference_hypothesis(
                    diagnostics,
                    policy=reference_policy,
                    config=self._observer_config,
                )
            )
            (
                selected_score,
                selected_masks,
                _,
                selected_segmentation,
                _,
            ) = next(
                item
                for item in hypotheses
                if item[4] == selected_index
            )
            return selected_masks, {
                "policy": reference_policy,
                "selection_contract": selection[
                    "selection_contract"
                ],
                "candidate_count": len(prompts),
                "selected_candidate_index": selected_index,
                "selected_score": selected_score,
                "selection": selection,
                "selected_segmentation": selected_segmentation,
                "candidates": diagnostics,
            }
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
        condition_anchor_policy = str(
            self._observer_config.get(
                "condition_anchor_policy",
                "reference_anchor_coordinates",
            )
        )
        if (
            condition_anchor_policy
            in {
                "condition_only_compact_geometry_all_modes_v2",
                "condition_only_ball_photometric_all_modes_v3",
            }
        ):
            minimum_area = int(
                self.config["quality"]["minimum_mask_pixels"]
            )
            maximum_area = int(
                round(
                    condition_frame.shape[0]
                    * condition_frame.shape[1]
                    * float(
                        self.config["quality"][
                            "maximum_mask_area_ratio"
                        ]
                    )
                )
            )
            seeds = build_condition_rigid_body_seed_hypotheses(
                condition_frame,
                scene_kind=self.scene_kind,
                minimum_area=minimum_area,
                maximum_area=maximum_area,
                config=self._observer_config,
            )
            if not seeds:
                raise ReferenceAnalysisError(
                    "reference_condition_subject_unobserved",
                    "condition-only causal geometry produced no physical-"
                    f"subject seed for {self.entity_class}; future reference "
                    "pixels were not used as a fallback",
                )
            hypotheses: list[
                tuple[float, np.ndarray, MaskPrompt, dict[str, Any]]
            ] = []
            records: list[dict[str, Any]] = []
            for candidate_index, seed in enumerate(seeds):
                record: dict[str, Any] = {
                    "candidate_index": candidate_index,
                    "source": seed["source"],
                    "seed_rank_score": float(seed["rank_score"]),
                    "seed_area_px2": int(seed["area_px2"]),
                    "seed_centroid_xy": list(seed["centroid_xy"]),
                    "seed_shape_score": float(seed["shape_score"]),
                    "seed_upper_support": float(
                        seed["upper_support"]
                    ),
                    "seed_apparatus_containment": float(
                        seed["apparatus_containment"]
                    ),
                    "source_diagnostics": dict(
                        seed["source_diagnostics"]
                    ),
                }
                try:
                    anchor_prompt = _prompt_from_mask(
                        seed["mask"],
                        frame_index=0,
                        box_expand=float(proposal["box_expand"]),
                        minimum_box_side=int(
                            proposal["minimum_box_side"]
                        ),
                        source=str(seed["source"]),
                    )
                    values, segmentation = self._segmenter.segment(
                        [condition_frame],
                        prompt=anchor_prompt,
                        temporary_prefix=(
                            f"{self.temporary_prefix}condition_only_"
                            f"{candidate_index}_"
                        ),
                    )
                    geometry = _valid_mask_geometry(
                        values[0],
                        minimum_area=minimum_area,
                        maximum_area=maximum_area,
                    )
                    if geometry is None:
                        record.update(
                            {
                                "status": "rejected",
                                "reason": (
                                    "segmented_condition_geometry_invalid"
                                ),
                                "segmentation": segmentation,
                            }
                        )
                        records.append(record)
                        continue
                    condition_mask, centroid, area = geometry
                    shape_score = _compact_shape_score(
                        condition_mask,
                        scene_kind=self.scene_kind,
                    )
                    minimum_shape = float(
                        self._observer_config.get(
                            "condition_anchor_minimum_shape_score",
                            0.12,
                        )
                    )
                    if shape_score < minimum_shape:
                        record.update(
                            {
                                "status": "rejected",
                                "reason": (
                                    "segmented_condition_shape_implausible"
                                ),
                                "segmented_shape_score": shape_score,
                                "minimum_shape_score": minimum_shape,
                                "segmentation": segmentation,
                            }
                        )
                        records.append(record)
                        continue
                    seed_iou, seed_containment = _overlap(
                        condition_mask,
                        seed["mask"],
                    )
                    scale_agreement = math.exp(
                        -0.5
                        * abs(
                            math.log(
                                max(float(area), 1.0)
                                / max(
                                    float(seed["area_px2"]),
                                    1.0,
                                )
                            )
                        )
                    )
                    if self.scene_kind == "free_fall":
                        location_support = float(
                            np.clip(
                                1.0
                                - float(centroid[1])
                                / max(
                                    0.45
                                    * condition_frame.shape[0],
                                    1.0,
                                ),
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        location_support = max(
                            float(seed["apparatus_containment"]),
                            float(
                                seed["source"]
                                == "condition_incline_apparatus_uphill_seed"
                            ),
                        )
                    base_score = float(
                        0.42 * shape_score
                        + 0.18 * seed_containment
                        + 0.14 * float(seed["rank_score"])
                        + 0.12 * scale_agreement
                        + 0.14 * location_support
                    )
                    photometric_evidence: dict[str, float] = {}
                    photometric_adjustment = 0.0
                    if (
                        self.scene_kind == "free_fall"
                        and condition_anchor_policy
                        == "condition_only_ball_photometric_all_modes_v3"
                    ):
                        photometric_evidence = (
                            condition_free_fall_photometric_evidence(
                                condition_frame,
                                condition_mask,
                            )
                        )
                        photometric_adjustment = float(
                            float(
                                self._observer_config.get(
                                    "condition_ball_structure_weight",
                                    0.16,
                                )
                            )
                            * photometric_evidence["structure_support"]
                            + float(
                                self._observer_config.get(
                                    "condition_ball_bright_contrast_weight",
                                    0.06,
                                )
                            )
                            * photometric_evidence[
                                "bright_object_support"
                            ]
                            - float(
                                self._observer_config.get(
                                    "condition_ball_flat_region_penalty",
                                    0.10,
                                )
                            )
                            * photometric_evidence["flat_region_support"]
                            - float(
                                self._observer_config.get(
                                    "condition_ball_skin_penalty",
                                    0.25,
                                )
                            )
                            * photometric_evidence["skin_fraction"]
                        )
                    score = float(
                        np.clip(
                            base_score + photometric_adjustment,
                            0.0,
                            1.0,
                        )
                    )
                    record.update(
                        {
                            "status": "accepted",
                            "score": score,
                            "geometry_score": base_score,
                            "photometric_adjustment": (
                                photometric_adjustment
                            ),
                            "photometric_evidence": (
                                photometric_evidence
                            ),
                            "segmented_area_px2": float(area),
                            "segmented_centroid_xy": centroid.tolist(),
                            "segmented_shape_score": shape_score,
                            "seed_iou": seed_iou,
                            "seed_containment": seed_containment,
                            "scale_agreement": scale_agreement,
                            "location_support": location_support,
                            "segmentation": segmentation,
                        }
                    )
                    hypotheses.append(
                        (
                            score,
                            condition_mask,
                            anchor_prompt,
                            record,
                        )
                    )
                except Exception as exc:
                    record.update(
                        {
                            "status": "rejected",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                records.append(record)
            if not hypotheses:
                raise ReferenceAnalysisError(
                    "reference_condition_subject_unobserved",
                    "every condition-only causal segmentation hypothesis "
                    f"for {self.entity_class} was rejected; future reference "
                    "pixels were not used as a fallback",
                )
            (
                _,
                condition_mask,
                _,
                selected_record,
            ) = max(
                hypotheses,
                key=lambda item: (
                    item[0],
                    float(np.count_nonzero(item[1])),
                ),
            )
            condition_prompt = _prompt_from_mask(
                condition_mask,
                frame_index=0,
                box_expand=float(proposal["box_expand"]),
                minimum_box_side=int(proposal["minimum_box_side"]),
                source="condition_subject_frozen_causal_anchor",
            )
            return condition_mask, condition_prompt, {
                "status": "condition_only_causal_hypothesis_selected",
                "policy": condition_anchor_policy,
                "same_case_reference": same_case_reference,
                "candidate_count": len(seeds),
                "selected_candidate_index": selected_record[
                    "candidate_index"
                ],
                "selected_source": selected_record["source"],
                "selected_score": selected_record["score"],
                "candidates": records,
                "future_reference_mask_used": False,
                "parent_future_pixel_coordinates_used": False,
                "prediction_pixels_used": False,
            }
        if (
            not same_case_reference
            and condition_anchor_policy == "condition_only_compact_geometry"
        ):
            minimum_area = int(
                self.config["quality"]["minimum_mask_pixels"]
            )
            maximum_area = int(
                round(
                    condition_frame.shape[0]
                    * condition_frame.shape[1]
                    * float(
                        self.config["quality"][
                            "maximum_mask_area_ratio"
                        ]
                    )
                )
            )
            proposals = _independent_compact_proposals(
                condition_frame,
                scene_kind=self.scene_kind,
                anchor_area_px2=float(
                    max(
                        0.0015
                        * condition_frame.shape[0]
                        * condition_frame.shape[1],
                        minimum_area,
                    )
                ),
                minimum_area=minimum_area,
                maximum_area=maximum_area,
                config=self._observer_config,
            )
            if not proposals:
                raise ReferenceAnalysisError(
                    "reference_condition_subject_unobserved",
                    "condition-only compact geometry did not recover a "
                    f"candidate {self.entity_class}; physics-parent pixel "
                    "coordinates were not used as a fallback",
                )
            ranked = sorted(
                proposals,
                key=lambda mask: _compact_shape_score(
                    mask,
                    scene_kind=self.scene_kind,
                ),
                reverse=True,
            )
            anchor_prompt = _prompt_from_mask(
                ranked[0],
                frame_index=0,
                box_expand=float(proposal["box_expand"]),
                minimum_box_side=int(proposal["minimum_box_side"]),
                source="condition_only_compact_geometry",
            )
            values, segmentation = self._segmenter.segment(
                [condition_frame],
                prompt=anchor_prompt,
                temporary_prefix=(
                    f"{self.temporary_prefix}condition_only_"
                ),
            )
            condition_mask = values[0]
            metadata = {
                "status": "condition_only_compact_geometry_segmented",
                "candidate_count": len(proposals),
                "parent_future_pixel_coordinates_used": False,
                "segmentation": segmentation,
            }
            if (
                int(np.count_nonzero(condition_mask))
                < minimum_area
            ):
                raise ReferenceAnalysisError(
                    "reference_condition_subject_unobserved",
                    "condition-only directed segmentation did not recover "
                    f"the manifest {self.entity_class}",
                )
            condition_prompt = _prompt_from_mask(
                condition_mask,
                frame_index=0,
                box_expand=float(proposal["box_expand"]),
                minimum_box_side=int(proposal["minimum_box_side"]),
                source="condition_subject_frozen_anchor",
            )
            return condition_mask, condition_prompt, metadata
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
        condition_mask: np.ndarray,
        available: np.ndarray,
        temporary_role: str = "prediction",
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
            temporary_prefix=(
                f"{self.temporary_prefix}{temporary_role}_"
            ),
        )
        for index, is_available in enumerate(available):
            if not is_available:
                masks[index] = np.zeros_like(masks[index])
        diagnostics: dict[str, Any] = {
            "primary_condition_forward": segmentation,
            "recovery_policy": str(
                self._observer_config.get(
                    "condition_directed_recovery_policy",
                    "disabled",
                )
            ),
        }
        recovery_policy = diagnostics["recovery_policy"]
        if recovery_policy not in {
            "anchor_validated_bidirectional_hypotheses_v1",
            "anchor_or_release_geometry_bidirectional_v2",
        }:
            return masks, segmentation
        release_geometry_recovery = bool(
            recovery_policy
            == "anchor_or_release_geometry_bidirectional_v2"
            and self.scene_kind == "free_fall"
        )

        quality = self.config["quality"]
        validation_config = dict(self._observer_config)
        # Observation recovery must not assume that prediction obeys the
        # target physics.  It may require visibility/shape continuity, but a
        # stationary or upward-moving generated body must remain observable
        # and receive a low scene-specific physics score rather than vanish.
        validation_config[
            "reference_hypothesis_minimum_direction_consistency"
        ] = 0.0
        validation_config[
            "reference_hypothesis_minimum_free_fall_downward_span_fraction"
        ] = 0.0
        validation_config["reference_hypothesis_minimum_score"] = 0.0

        def validate(
            candidate_masks: Sequence[np.ndarray],
        ) -> tuple[float, dict[str, Any]]:
            candidate_reference = build_rigid_body_reference(
                candidate_masks,
                entity_id="condition_directed_hypothesis",
                entity_class=self.entity_class,
                scene_kind=self.scene_kind,
                frame_shape=frames[0].shape[:2],
                minimum_area=int(quality["minimum_mask_pixels"]),
                maximum_area_ratio=float(
                    quality["maximum_mask_area_ratio"]
                ),
                minimum_span_px=float(
                    quality[self.minimum_span_quality_key]
                ),
                config=validation_config,
            )
            return score_rigid_body_reference_hypothesis(
                candidate_reference,
                frames=frames,
                scene_kind=self.scene_kind,
                config=validation_config,
            )

        try:
            primary_score, primary_validation = validate(masks)
        except Exception as exc:
            primary_score = 0.0
            primary_validation = {
                "accepted": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        diagnostics["primary_validation"] = primary_validation
        if bool(primary_validation.get("accepted", False)):
            diagnostics["selected_source"] = "primary_condition_forward"
            return masks, diagnostics

        proposal = self.config["motion_proposal"]
        hypotheses: list[
            tuple[float, list[np.ndarray], dict[str, Any]]
        ] = []
        candidates: list[dict[str, Any]] = []
        try:
            prompts = build_rigid_body_motion_prompts(
                frames,
                scene_kind=self.scene_kind,
                threshold=float(proposal["threshold"]),
                minimum_area=int(proposal["minimum_area"]),
                box_expand=float(proposal["box_expand"]),
                minimum_box_side=int(proposal["minimum_box_side"]),
                config=self._observer_config,
                maximum_candidates=int(
                    self._observer_config.get(
                        "condition_directed_recovery_maximum_candidates",
                        2,
                    )
                ),
            )
        except Exception as exc:
            diagnostics["recovery_failure"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return masks, diagnostics
        minimum_anchor_containment = float(
            self._observer_config.get(
                "condition_directed_recovery_minimum_anchor_containment",
                0.25,
            )
        )
        for candidate_index, prompt_candidate in enumerate(prompts):
            record: dict[str, Any] = {
                "candidate_index": candidate_index,
                "prompt_frame": prompt_candidate.frame_index,
                "prompt_metadata": prompt_candidate.metadata,
            }
            try:
                candidate_masks, candidate_segmentation = (
                    self._segmenter.segment(
                        frames,
                        prompt=prompt_candidate,
                        temporary_prefix=(
                            f"{self.temporary_prefix}{temporary_role}_"
                            f"anchor_recovery_{candidate_index}_"
                        ),
                    )
                )
                for index, is_available in enumerate(available):
                    if not is_available:
                        candidate_masks[index] = np.zeros_like(
                            candidate_masks[index]
                        )
                anchor_iou, anchor_containment = _overlap(
                    candidate_masks[seed],
                    condition_mask,
                )
                anchor_intersection = int(
                    np.logical_and(
                        candidate_masks[seed] > 0,
                        condition_mask > 0,
                    ).sum()
                )
                condition_anchor_coverage = anchor_intersection / max(
                    int(np.count_nonzero(condition_mask)),
                    1,
                )
                condition_area_ratio = int(
                    np.count_nonzero(candidate_masks[seed])
                ) / max(int(np.count_nonzero(condition_mask)), 1)
                record.update(
                    {
                        "anchor_iou": anchor_iou,
                        "anchor_containment": anchor_containment,
                        "condition_anchor_coverage": (
                            condition_anchor_coverage
                        ),
                        "condition_area_ratio": condition_area_ratio,
                        "segmentation": candidate_segmentation,
                    }
                )
                strict_anchor_failed = bool(
                    anchor_containment < minimum_anchor_containment
                    or condition_anchor_coverage
                    < minimum_anchor_containment
                    or not 0.35 <= condition_area_ratio <= 3.0
                )
                candidate_score, candidate_validation = validate(
                    candidate_masks
                )
                record["validation"] = candidate_validation
                if not bool(candidate_validation.get("accepted", False)):
                    record.update(
                        {
                            "status": "rejected",
                            "reason": (
                                "condition_directed_candidate_validation_"
                                "failed"
                            ),
                        }
                    )
                    candidates.append(record)
                    continue
                release_valid = False
                release_score = 0.0
                if release_geometry_recovery and strict_anchor_failed:
                    condition_centroid = mask_centroid(condition_mask)
                    condition_radius = math.sqrt(
                        max(
                            float(np.count_nonzero(condition_mask)),
                            1.0,
                        )
                        / math.pi
                    )
                    observed_geometry: list[
                        tuple[int, np.ndarray, float, float]
                    ] = []
                    for frame_index, candidate_mask in enumerate(
                        candidate_masks
                    ):
                        geometry = _valid_mask_geometry(
                            candidate_mask,
                            minimum_area=int(
                                quality["minimum_mask_pixels"]
                            ),
                            maximum_area=int(
                                round(
                                    frames[0].shape[0]
                                    * frames[0].shape[1]
                                    * float(
                                        quality[
                                            "maximum_mask_area_ratio"
                                        ]
                                    )
                                )
                            ),
                        )
                        if geometry is None:
                            continue
                        _, candidate_centroid, candidate_area = geometry
                        observed_geometry.append(
                            (
                                frame_index,
                                candidate_centroid,
                                float(candidate_area),
                                _compact_shape_score(
                                    candidate_mask,
                                    scene_kind=self.scene_kind,
                                ),
                            )
                        )
                    earliest_observed_frame = (
                        observed_geometry[0][0]
                        if observed_geometry
                        else len(frames)
                    )
                    early_limit = max(
                        earliest_observed_frame,
                        min(
                            len(frames) - 1,
                            int(
                                self._observer_config.get(
                                    "condition_release_recovery_"
                                    "maximum_early_frame",
                                    3,
                                )
                            ),
                        ),
                    )
                    early_geometry = [
                        item
                        for item in observed_geometry
                        if item[0] <= early_limit
                    ]
                    minimum_distance_radii = float("inf")
                    if condition_centroid is not None:
                        for _, candidate_centroid, candidate_area, _ in (
                            early_geometry
                        ):
                            candidate_radius = math.sqrt(
                                max(candidate_area, 1.0) / math.pi
                            )
                            minimum_distance_radii = min(
                                minimum_distance_radii,
                                float(
                                    np.linalg.norm(
                                        candidate_centroid
                                        - condition_centroid
                                    )
                                    / max(
                                        condition_radius,
                                        candidate_radius,
                                        1.0,
                                    )
                                ),
                            )
                    observed_areas = np.asarray(
                        [item[2] for item in observed_geometry],
                        dtype=np.float64,
                    )
                    median_area_ratio = (
                        float(np.median(observed_areas))
                        / max(
                            float(np.count_nonzero(condition_mask)),
                            1.0,
                        )
                        if observed_areas.size
                        else float("inf")
                    )
                    median_shape = (
                        float(
                            np.median(
                                [item[3] for item in observed_geometry]
                            )
                        )
                        if observed_geometry
                        else 0.0
                    )
                    prompt_compact = float(
                        prompt_candidate.metadata.get(
                            "candidate_compact_score",
                            0.0,
                        )
                    )
                    maximum_distance = float(
                        self._observer_config.get(
                            "condition_release_recovery_maximum_"
                            "distance_radii",
                            8.0,
                        )
                    )
                    minimum_area_ratio = float(
                        self._observer_config.get(
                            "condition_release_recovery_minimum_"
                            "median_area_ratio",
                            0.20,
                        )
                    )
                    maximum_area_ratio = float(
                        self._observer_config.get(
                            "condition_release_recovery_maximum_"
                            "median_area_ratio",
                            4.5,
                        )
                    )
                    minimum_shape = float(
                        self._observer_config.get(
                            "condition_release_recovery_minimum_"
                            "median_shape",
                            0.12,
                        )
                    )
                    release_valid = bool(
                        observed_geometry
                        and earliest_observed_frame
                        <= int(
                            self._observer_config.get(
                                "condition_release_recovery_maximum_"
                                "first_observed_frame",
                                2,
                            )
                        )
                        and minimum_distance_radii <= maximum_distance
                        and minimum_area_ratio
                        <= median_area_ratio
                        <= maximum_area_ratio
                        and median_shape >= minimum_shape
                    )
                    proximity_support = (
                        math.exp(
                            -minimum_distance_radii
                            / max(maximum_distance, 1e-6)
                        )
                        if math.isfinite(minimum_distance_radii)
                        else 0.0
                    )
                    area_support = math.exp(
                        -abs(
                            math.log(max(median_area_ratio, 1e-6))
                        )
                    )
                    release_score = float(
                        candidate_score
                        + 0.12 * proximity_support
                        + 0.08 * prompt_compact
                        + 0.05 * area_support
                    )
                    record["release_geometry"] = {
                        "accepted": release_valid,
                        "earliest_observed_frame": (
                            earliest_observed_frame
                        ),
                        "minimum_condition_distance_radii": (
                            minimum_distance_radii
                            if math.isfinite(minimum_distance_radii)
                            else None
                        ),
                        "maximum_condition_distance_radii": (
                            maximum_distance
                        ),
                        "median_area_ratio": median_area_ratio,
                        "median_shape_score": median_shape,
                        "prompt_compact_score": prompt_compact,
                        "selection_score": release_score,
                        "condition_overlap_required": False,
                    }
                accepted = bool(
                    not strict_anchor_failed or release_valid
                )
                record["status"] = "accepted" if accepted else "rejected"
                if accepted:
                    selection_score = (
                        release_score
                        if release_valid
                        else candidate_score + 0.1 * anchor_containment
                    )
                    record["selection_mode"] = (
                        "condition_release_geometry"
                        if release_valid
                        else "condition_anchor_overlap"
                    )
                    hypotheses.append(
                        (
                            selection_score,
                            candidate_masks,
                            record,
                        )
                    )
                else:
                    record["reason"] = (
                        "condition_anchor_and_release_geometry_failed"
                        if release_geometry_recovery
                        else "condition_anchor_overlap_failed"
                    )
            except Exception as exc:
                record.update(
                    {
                        "status": "rejected",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            candidates.append(record)
        diagnostics["recovery_candidates"] = candidates
        if not hypotheses:
            diagnostics["selected_source"] = (
                "primary_condition_forward_fail_closed"
            )
            return masks, diagnostics
        _, selected_masks, selected_record = max(
            hypotheses,
            key=lambda item: item[0],
        )
        for index, is_available in enumerate(available):
            if not is_available:
                selected_masks[index] = np.zeros_like(
                    selected_masks[index]
                )
        diagnostics["selected_source"] = (
            "release_geometry_bidirectional_hypothesis"
            if selected_record.get("selection_mode")
            == "condition_release_geometry"
            else "anchor_validated_bidirectional_hypothesis"
        )
        diagnostics["selected_candidate_index"] = selected_record[
            "candidate_index"
        ]
        diagnostics["primary_score"] = primary_score
        return selected_masks, diagnostics

    def _condition_axis(
        self,
        *,
        reference: RigidBodyReference,
        condition_frame: np.ndarray,
        condition_mask: np.ndarray,
        reference_mode: str,
    ) -> FrozenRigidAxis:
        axis_policy = str(
            self._observer_config.get(
                "condition_axis_policy",
                "reference_axis_for_same_case_v1",
            )
        )
        causal_axis = (
            axis_policy == "condition_scene_geometry_causal_v2"
        )
        if reference_mode == "same_case_reference" and not causal_axis:
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
                1.0,
            )
            if not causal_axis:
                span = max(span, 0.5 * reference.axis.span_px)
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
            fallback=(None if causal_axis else reference.axis),
            allow_reference_fallback=not causal_axis,
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
        if (
            reference_mode == "same_case_reference"
            and str(
                self.config.get(
                    "reference_observation_policy",
                    "motion_seeded_bidirectional",
                )
            )
            == "condition_causal_symmetric"
        ):
            try:
                formal_reference_masks, formal_reference_observation = (
                    self._prediction_masks(
                        list(reference_video.frames),
                        condition_prompt=condition_prompt,
                        condition_mask=condition_mask,
                        available=np.ones(len(times_s), dtype=bool),
                        temporary_role="reference_condition_causal",
                    )
                )
                (
                    formal_reference_masks,
                    formal_reference_recovery,
                ) = recover_rigid_body_mask_gaps(
                    formal_reference_masks,
                    frames=list(reference_video.frames),
                    scene_kind=self.scene_kind,
                    minimum_area=int(
                        quality["minimum_mask_pixels"]
                    ),
                    maximum_area_ratio=float(
                        quality["maximum_mask_area_ratio"]
                    ),
                    config=self._observer_config,
                )
                reference = build_rigid_body_reference(
                    formal_reference_masks,
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
                reference_observation = {
                    "policy": "condition_causal_symmetric",
                    "bootstrap": reference_observation,
                    "formal_condition_directed": (
                        formal_reference_observation
                    ),
                    "formal_gap_recovery": (
                        formal_reference_recovery
                    ),
                }
            except ReferenceAnalysisError:
                raise
            except Exception as exc:
                raise ReferenceAnalysisError(
                    "reference_condition_causal_observation_failed",
                    f"{self.scene_name} condition-causal reference "
                    f"observation failed: {type(exc).__name__}: {exc}",
                ) from exc
        condition_axis = self._condition_axis(
            reference=reference,
            condition_frame=condition_frame,
            condition_mask=condition_mask,
            reference_mode=reference_mode,
        )
        reference_geometry_policy = str(
            self._observer_config.get(
                "reference_geometry_policy",
                "legacy_reference_mode_split_v1",
            )
        )
        if (
            reference_mode == "same_case_reference"
            and reference_geometry_policy
            == "condition_frozen_same_case_coordinates_v2"
        ):
            scoring_reference = (
                reexpress_same_case_reference_on_condition_axis(
                    reference,
                    condition_axis=condition_axis,
                )
            )
        elif reference_mode == "same_case_reference":
            scoring_reference = reference
        else:
            scoring_reference = remap_reference_to_condition(
                reference,
                condition_axis=condition_axis,
                condition_area_px2=float(np.count_nonzero(condition_mask)),
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
                    condition_mask=condition_mask,
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
            reference=np.asarray(
                [
                    (
                        np.nan
                        if row["reference_progress_normalized"] is None
                        else float(
                            row["reference_progress_normalized"]
                        )
                    )
                    for row in result.per_frame
                ],
                dtype=np.float64,
            ),
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
    "build_rigid_body_motion_prompts",
    "build_rigid_body_reference",
    "classify_coupled_rigid_body_artifacts",
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
    "recover_rigid_body_mask_gaps",
    "rigid_body_observer_version",
    "resummarize_subject_on_expected",
    "rigid_body_fingerprint_payload",
    "score_rigid_body_reference_hypothesis",
    "write_rigid_body_audit_json",
]
