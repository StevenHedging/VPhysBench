"""Open-world pendulum observation and topology helpers.

The legacy pendulum evaluator intentionally remains a single SAM2 channel.
This module is used only by ``pendulum_state_v6``.  It separates the bob
(the counted physical participant) from the string/support (apparatus and
topology), preserves all credible residual bobs, and never derives the
condition identity from prediction pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ...common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    track_open_world_detections,
)
from ...common.entities.timeline import CommonTimeGrid
from ...common.errors import ReferenceAnalysisError
from ...common.masks.quality import mask_centroid
from .segmentation import MotionPrompt


PENDULUM_STRUCTURE_SPEC_VERSION = "1.0"


def _point(value: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _binary_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.where(np.asarray(value) > 0, 255, 0).astype(np.uint8)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PendulumStructureSpec:
    """Condition-frozen pivot–string–bob identity."""

    pivot_xy: np.ndarray
    bob_xy: np.ndarray
    bob_radius_px: float
    bob_mask: np.ndarray
    subject_mask: np.ndarray
    confidence: float
    source: str
    version: str = PENDULUM_STRUCTURE_SPEC_VERSION

    def __post_init__(self) -> None:
        pivot = _point(self.pivot_xy, name="pivot_xy")
        bob = _point(self.bob_xy, name="bob_xy")
        radius = float(self.bob_radius_px)
        confidence = float(self.confidence)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("bob_radius_px must be finite and positive")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        bob_mask = _binary_mask(self.bob_mask, name="bob_mask")
        subject_mask = _binary_mask(self.subject_mask, name="subject_mask")
        if bob_mask.shape != subject_mask.shape:
            raise ValueError("pendulum structure masks must share one canvas")
        if not str(self.source).strip():
            raise ValueError("pendulum structure source must be non-empty")
        if self.version != PENDULUM_STRUCTURE_SPEC_VERSION:
            raise ValueError(
                "unsupported PendulumStructureSpec version "
                f"{self.version!r}"
            )
        object.__setattr__(self, "pivot_xy", pivot)
        object.__setattr__(self, "bob_xy", bob)
        object.__setattr__(self, "bob_radius_px", radius)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bob_mask", bob_mask)
        object.__setattr__(self, "subject_mask", subject_mask)

    @property
    def length_px(self) -> float:
        return float(np.linalg.norm(self.bob_xy - self.pivot_xy))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pivot_xy": self.pivot_xy.tolist(),
            "bob_xy": self.bob_xy.tolist(),
            "bob_radius_px": self.bob_radius_px,
            "length_px": self.length_px,
            "bob_mask_area_px2": int(np.count_nonzero(self.bob_mask)),
            "subject_mask_area_px2": int(
                np.count_nonzero(self.subject_mask)
            ),
            "confidence": self.confidence,
            "source": self.source,
        }


def letterbox_condition_image(
    path: Path,
    *,
    width: int,
    height: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a Case condition image onto the evaluator's normalized canvas."""

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ReferenceAnalysisError(
            "reference_condition_frame_unreadable",
            f"cannot decode condition frame: {path}",
        )
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas, {
        "policy": "preserve_aspect_ratio_letterbox",
        "source_size": [source_width, source_height],
        "target_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "offset_xy": [offset_x, offset_y],
        "scale": scale,
    }


def _circle_candidates(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
    radius_hint_px: float | None = None,
    maximum_candidates: int | None = None,
) -> list[tuple[float, float, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_size = int(config.get("blur_kernel", 7))
    if blur_size % 2 == 0:
        blur_size += 1
    gray = cv2.medianBlur(gray, max(3, blur_size))
    height, width = gray.shape
    if radius_hint_px is None:
        minimum_radius = max(
            3, int(round(min(height, width) * 0.008))
        )
        maximum_radius = max(
            minimum_radius + 1,
            int(round(min(height, width) * 0.10)),
        )
    else:
        minimum_radius = max(
            3,
            int(
                round(
                    radius_hint_px
                    * float(config.get("minimum_radius_ratio", 0.55))
                )
            ),
        )
        maximum_radius = max(
            minimum_radius + 1,
            int(
                round(
                    radius_hint_px
                    * float(config.get("maximum_radius_ratio", 1.8))
                )
            ),
        )
    thresholds = config.get(
        "hough_accumulator_thresholds", [32, 27, 23, 19, 15]
    )
    limit = (
        int(maximum_candidates)
        if maximum_candidates is not None
        else int(config.get("maximum_circle_candidates", 96))
    )
    if limit < 1:
        raise ValueError("maximum circle candidates must be positive")
    output: list[tuple[float, float, float]] = []
    for raw_threshold in thresholds:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(config.get("hough_dp", 1.2)),
            minDist=max(
                2.0 * minimum_radius,
                float(config.get("minimum_center_distance_px", 12.0)),
            ),
            param1=float(config.get("edge_threshold", 100.0)),
            param2=float(raw_threshold),
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )
        if circles is None:
            continue
        for x, y, radius in circles[0]:
            candidate = (float(x), float(y), float(radius))
            if any(
                math.hypot(x - old_x, y - old_y)
                <= 0.65 * max(radius, old_radius)
                for old_x, old_y, old_radius in output
            ):
                continue
            output.append(candidate)
        if len(output) >= limit:
            break
    return output[:limit]


def _line_segments(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        float(config.get("line_canny_low", 35.0)),
        float(config.get("line_canny_high", 120.0)),
    )
    raw = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 360.0,
        threshold=int(config.get("line_hough_threshold", 24)),
        minLineLength=int(
            config.get("minimum_string_segment_length_px", 18)
        ),
        maxLineGap=int(config.get("maximum_string_line_gap_px", 12)),
    )
    if raw is None:
        return []
    return [
        (
            np.asarray([x1, y1], dtype=np.float64),
            np.asarray([x2, y2], dtype=np.float64),
        )
        for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4)
    ]


def _circle_string_evidence(
    circle: tuple[float, float, float],
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    frame_shape: tuple[int, ...],
) -> tuple[float, np.ndarray | None]:
    center = np.asarray(circle[:2], dtype=np.float64)
    radius = float(circle[2])
    height = frame_shape[0]
    best_score = 0.0
    best_pivot: np.ndarray | None = None
    for first, second in segments:
        upper, lower = (
            (first, second) if first[1] <= second[1] else (second, first)
        )
        vertical_span = float(lower[1] - upper[1])
        length = float(np.linalg.norm(lower - upper))
        if vertical_span < 1.5 * radius or length < 2.0 * radius:
            continue
        endpoint_distance = float(np.linalg.norm(lower - center))
        # A visible string normally ends at the bob boundary.  Hough may end
        # slightly inside or outside the circle, hence the generous band.
        endpoint_error = abs(endpoint_distance - radius) / max(radius, 1.0)
        if endpoint_error > 2.0:
            continue
        if upper[1] >= center[1] - 1.5 * radius:
            continue
        length_score = min(length / max(0.45 * height, 1.0), 1.0)
        endpoint_score = math.exp(-endpoint_error)
        score = 0.55 * endpoint_score + 0.45 * length_score
        if score > best_score:
            best_score = score
            best_pivot = np.array(upper, dtype=np.float64)
    return float(best_score), best_pivot


def _structure_masks(
    shape: tuple[int, int],
    *,
    pivot_xy: np.ndarray,
    bob_xy: np.ndarray,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    bob = np.zeros(shape, dtype=np.uint8)
    subject = np.zeros(shape, dtype=np.uint8)
    center = tuple(int(round(value)) for value in bob_xy)
    pivot = tuple(int(round(value)) for value in pivot_xy)
    radius = max(2, int(round(radius_px)))
    cv2.circle(bob, center, radius, 255, -1)
    cv2.line(
        subject,
        pivot,
        center,
        255,
        max(1, int(round(radius_px * 0.12))),
        cv2.LINE_AA,
    )
    cv2.circle(subject, center, radius, 255, -1)
    return bob, subject


def detect_condition_structure(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
    expected_radius_length_ratio: float | None = None,
) -> PendulumStructureSpec:
    """Detect the bob and its string from the current Case condition frame."""

    circles = _circle_candidates(
        frame,
        config=config,
        maximum_candidates=int(
            config.get("maximum_condition_circle_candidates", 192)
        ),
    )
    segments = _line_segments(frame, config=config)
    height, width = frame.shape[:2]
    candidates: list[
        tuple[float, tuple[float, float, float], np.ndarray]
    ] = []
    for circle in circles:
        x, y, radius = circle
        if (
            x - radius <= 1
            or x + radius >= width - 1
            or y - radius <= 1
            or y + radius >= height - 1
        ):
            continue
        line_score, pivot = _circle_string_evidence(
            circle,
            segments,
            frame_shape=frame.shape,
        )
        if pivot is None:
            continue
        length = float(
            np.linalg.norm(np.asarray([x, y], dtype=np.float64) - pivot)
        )
        if length < float(config.get("minimum_length_radius_ratio", 3.0)) * radius:
            continue
        if length > float(config.get("maximum_length_canvas_ratio", 0.8)) * math.hypot(
            width, height
        ):
            continue
        angle_deg = abs(
            math.degrees(
                math.atan2(x - float(pivot[0]), y - float(pivot[1]))
            )
        )
        # The current OOD conditions all visibly displace the bob from the
        # vertical support.  Near-vertical circles are overwhelmingly knobs,
        # gauge marks, or the support rod itself.  Same-Case small-angle
        # conditions use their trusted frame-zero reference mask instead.
        if angle_deg < float(
            config.get("minimum_condition_bob_angle_deg", 7.0)
        ):
            continue
        support_penalty = (
            0.15
            if abs(x - pivot[0])
            < float(config.get("support_exclusion_radius_ratio", 1.2))
            * radius
            else 0.0
        )
        score = line_score - support_penalty
        if expected_radius_length_ratio is not None:
            expected_ratio = float(expected_radius_length_ratio)
            if not math.isfinite(expected_ratio) or expected_ratio <= 0.0:
                raise ValueError(
                    "expected_radius_length_ratio must be positive"
                )
            observed_ratio = radius / max(length, 1.0)
            ratio_score = math.exp(
                -abs(math.log(observed_ratio / expected_ratio))
            )
            score = 0.65 * score + 0.35 * ratio_score
        candidates.append((score, circle, pivot))
    if not candidates:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_structure_missing",
            "could not locate a circle connected to an upward pendulum string "
            "in the current Case condition frame",
        )
    score, (x, y, radius), pivot = max(candidates, key=lambda value: value[0])
    minimum_score = float(config.get("minimum_condition_structure_score", 0.25))
    if score < minimum_score:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_structure_ambiguous",
            f"best condition pendulum structure confidence {score:.3f} is "
            f"below {minimum_score:.3f}",
        )
    bob_xy = np.asarray([x, y], dtype=np.float64)
    bob_mask, subject_mask = _structure_masks(
        (height, width),
        pivot_xy=pivot,
        bob_xy=bob_xy,
        radius_px=radius,
    )
    return PendulumStructureSpec(
        pivot_xy=pivot,
        bob_xy=bob_xy,
        bob_radius_px=radius,
        bob_mask=bob_mask,
        subject_mask=subject_mask,
        confidence=float(np.clip(score, 0.0, 1.0)),
        source="condition_frame_hough_circle_plus_string",
    )


def structure_from_reference_trace(
    *,
    pivot_xy: Sequence[float],
    bob_xy: Sequence[float],
    assembly_mask: np.ndarray,
    source: str = "same_case_condition_reference_trace",
) -> PendulumStructureSpec:
    """Build a same-Case condition spec from its frame-zero reference mask."""

    pivot = _point(pivot_xy, name="pivot_xy")
    bob = _point(bob_xy, name="bob_xy")
    binary = np.where(np.asarray(assembly_mask) > 0, 255, 0).astype(
        np.uint8
    )
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    y0 = max(0, int(round(bob[1] - 0.2 * np.linalg.norm(bob - pivot))))
    local = np.array(distance, copy=True)
    local[:y0] = 0.0
    radius = float(local.max())
    if radius < 2.0:
        radius = max(
            2.0,
            math.sqrt(max(int(np.count_nonzero(binary)), 1) / math.pi)
            * 0.18,
        )
    bob_mask, subject_mask = _structure_masks(
        binary.shape,
        pivot_xy=pivot,
        bob_xy=bob,
        radius_px=radius,
    )
    bob_mask = cv2.bitwise_and(bob_mask, binary)
    if np.count_nonzero(bob_mask) < 4:
        bob_mask, subject_mask = _structure_masks(
            binary.shape,
            pivot_xy=pivot,
            bob_xy=bob,
            radius_px=radius,
        )
    return PendulumStructureSpec(
        pivot_xy=pivot,
        bob_xy=bob,
        bob_radius_px=radius,
        bob_mask=bob_mask,
        subject_mask=subject_mask,
        confidence=1.0,
        source=source,
    )


def prompt_from_structure(
    structure: PendulumStructureSpec,
    *,
    box_expand_radius: float = 1.8,
) -> MotionPrompt:
    """Create a directed SAM2 prompt anchored only in the condition image."""

    height, width = structure.bob_mask.shape
    radius = structure.bob_radius_px
    margin = box_expand_radius * radius
    minimum = np.minimum(structure.pivot_xy, structure.bob_xy) - margin
    maximum = np.maximum(structure.pivot_xy, structure.bob_xy) + margin
    box = np.asarray(
        [
            np.clip(minimum[0], 0.0, width - 1.0),
            np.clip(minimum[1], 0.0, height - 1.0),
            np.clip(maximum[0], 0.0, width - 1.0),
            np.clip(maximum[1], 0.0, height - 1.0),
        ],
        dtype=np.float32,
    )
    fractions = np.asarray([0.12, 0.38, 0.68, 1.0], dtype=np.float64)
    points = (
        structure.pivot_xy[None, :]
        + fractions[:, None]
        * (structure.bob_xy - structure.pivot_xy)[None, :]
    ).astype(np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    return MotionPrompt(
        frame_index=0,
        box_xyxy=box,
        motion_box_xyxy=box,
        points_xy=points,
        point_labels=labels,
        proposal_score=structure.confidence,
    )


def extract_bob_masks(
    assembly_masks: Sequence[np.ndarray],
    *,
    expected_radius_px: float | None = None,
    minimum_pixels: int = 4,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Separate the terminal bob from a connected string+bob SAM2 mask."""

    output: list[np.ndarray] = []
    centers = np.full((len(assembly_masks), 2), np.nan, dtype=np.float64)
    radii = np.full(len(assembly_masks), np.nan, dtype=np.float64)
    for index, raw_mask in enumerate(assembly_masks):
        binary = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(
            np.uint8
        )
        if int(np.count_nonzero(binary)) < minimum_pixels:
            output.append(np.zeros_like(binary))
            continue
        ys, xs = np.where(binary > 0)
        bottom_cut = np.quantile(ys, 0.72)
        bottom = ys >= bottom_cut
        if int(np.count_nonzero(bottom)) < minimum_pixels:
            output.append(np.zeros_like(binary))
            continue
        center = np.asarray(
            [float(np.mean(xs[bottom])), float(np.mean(ys[bottom]))],
            dtype=np.float64,
        )
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        search = np.zeros_like(binary)
        search[max(0, int(round(bottom_cut))) :, :] = 1
        local_radius = float(np.max(distance * search))
        if expected_radius_px is not None:
            radius = float(
                np.clip(
                    local_radius if local_radius >= 1.0 else expected_radius_px,
                    0.55 * expected_radius_px,
                    1.8 * expected_radius_px,
                )
            )
        else:
            radius = max(local_radius, 2.0)
        candidate = np.zeros_like(binary)
        cv2.circle(
            candidate,
            tuple(int(round(value)) for value in center),
            max(2, int(round(1.35 * radius))),
            255,
            -1,
        )
        candidate = cv2.bitwise_and(candidate, binary)
        if int(np.count_nonzero(candidate)) < minimum_pixels:
            output.append(np.zeros_like(binary))
            continue
        centroid = mask_centroid(candidate)
        if centroid is not None:
            center = np.asarray(centroid, dtype=np.float64)
        output.append(candidate)
        centers[index] = center
        radii[index] = radius
    return output, centers, radii


def _mask_overlap_fraction(first: np.ndarray, second: np.ndarray) -> float:
    first_binary = np.asarray(first) > 0
    count = int(np.count_nonzero(first_binary))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(first_binary & (np.asarray(second) > 0)) / count)


def _condition_change_mask(
    frame: np.ndarray,
    condition_frame: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    difference = cv2.absdiff(frame, condition_frame)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _mask_histogram(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a compact condition-frozen HSV appearance descriptor."""

    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(64, dtype=np.float64)
    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        binary,
        [8, 8],
        [0, 180, 0, 256],
    ).reshape(-1)
    total = float(histogram.sum())
    if total <= 0.0:
        return np.zeros(64, dtype=np.float64)
    return histogram.astype(np.float64) / total


def _histogram_intersection(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(np.minimum(first, second).sum())


def _weak_directed_string_circle(
    center_xy: Sequence[float],
    *,
    radius_px: float,
    pivot_xy: Sequence[float],
    directed_bob_xy: Sequence[float] | None,
    directed_bob_radius_px: float,
    change_overlap: float,
    minimum_body_change_overlap: float,
    bob_clearance_radius_ratio: float,
) -> tuple[bool, dict[str, float | bool | None]]:
    """Identify a weak Hough fit on the frozen pivot-to-bob string.

    Hough occasionally fits a bob-sized circle to texture on the middle of
    the directed string.  Such a fit is geometrically different from both the
    directed bob and a branch bob: its center lies inside the pivot-to-bob
    segment, close to that segment's centerline, and well clear of the
    directed bob.  Geometry alone is not sufficient to reject a real inline
    second body, so a candidate with strong condition-change support remains
    observable.
    """

    center = np.asarray(center_xy, dtype=np.float64)
    pivot = np.asarray(pivot_xy, dtype=np.float64)
    bob = (
        None
        if directed_bob_xy is None
        else np.asarray(directed_bob_xy, dtype=np.float64)
    )
    radius = float(radius_px)
    bob_radius = float(directed_bob_radius_px)
    changed = float(change_overlap)
    minimum_change = float(minimum_body_change_overlap)
    clearance_ratio = float(bob_clearance_radius_ratio)
    values = (radius, bob_radius, changed, minimum_change, clearance_ratio)
    if (
        center.shape != (2,)
        or pivot.shape != (2,)
        or not np.isfinite(center).all()
        or not np.isfinite(pivot).all()
        or any(not math.isfinite(value) for value in values)
        or radius <= 0.0
        or bob_radius < 0.0
        or not 0.0 <= changed <= 1.0
        or not 0.0 <= minimum_change <= 1.0
        or clearance_ratio < 0.0
    ):
        raise ValueError("directed-string relation inputs are invalid")
    empty = {
        "directed_string_projection_fraction": None,
        "directed_string_centerline_distance_px": None,
        "directed_bob_distance_px": None,
        "directed_string_interior": False,
        "weak_directed_string_circle": False,
    }
    if (
        bob is None
        or bob.shape != (2,)
        or not np.isfinite(bob).all()
    ):
        return False, empty
    string = bob - pivot
    length_squared = float(np.dot(string, string))
    if length_squared <= 1e-9:
        return False, empty
    projection = float(np.dot(center - pivot, string) / length_squared)
    closest = pivot + np.clip(projection, 0.0, 1.0) * string
    centerline_distance = float(np.linalg.norm(center - closest))
    bob_distance = float(np.linalg.norm(center - bob))
    # A true branch bob is off the main centerline.  A real inline extra bob
    # still survives when its body creates strong condition-change evidence.
    centerline_tolerance = max(2.0, 0.35 * radius)
    bob_clearance = clearance_ratio * max(radius, bob_radius, 1.0)
    interior = bool(
        0.0 < projection < 1.0
        and centerline_distance <= centerline_tolerance
        and bob_distance > bob_clearance
    )
    weak_fit = bool(interior and changed < minimum_change)
    return weak_fit, {
        "directed_string_projection_fraction": projection,
        "directed_string_centerline_distance_px": centerline_distance,
        "directed_bob_distance_px": bob_distance,
        "directed_string_interior": interior,
        "weak_directed_string_circle": weak_fit,
    }


def _directed_bob_detections(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    availability: np.ndarray,
    structure: PendulumStructureSpec,
    condition_frame: np.ndarray,
    config: Mapping[str, Any],
) -> list[list[ObjectDetection]]:
    """Validate the directed SAM2 channel against the immutable condition.

    A propagated mask is not automatically the declared bob.  Large changes
    in appearance, area, or pivot distance split it into a formal replacement
    track.  The replacement therefore remains visible to cardinality scoring,
    but cannot silently continue the condition-frozen identity.
    """

    if len(frames) != len(masks) or len(frames) != len(availability):
        raise ValueError("directed pendulum channels must be aligned")
    anchor_histogram = _mask_histogram(
        condition_frame,
        structure.bob_mask,
    )
    anchor_area = max(
        float(np.count_nonzero(structure.bob_mask)),
        1.0,
    )
    minimum_pixels = int(config.get("minimum_bob_pixels", 8))
    minimum_area_ratio = float(
        config.get("direct_minimum_area_ratio", 0.20)
    )
    maximum_area_ratio = float(
        config.get("direct_maximum_area_ratio", 4.0)
    )
    minimum_color = float(
        config.get("minimum_anchor_color_similarity", 0.10)
    )
    minimum_length_ratio = float(
        config.get("minimum_residual_length_ratio", 0.35)
    )
    maximum_length_ratio = float(
        config.get("maximum_residual_length_ratio", 2.0)
    )
    detections: list[list[ObjectDetection]] = [
        [] for _ in frames
    ]
    for frame_index, (frame, raw_mask) in enumerate(zip(frames, masks)):
        if not bool(availability[frame_index]):
            continue
        mask = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(
            np.uint8
        )
        area = int(np.count_nonzero(mask))
        centroid = mask_centroid(mask)
        if area < minimum_pixels or centroid is None:
            continue
        area_ratio = float(area / anchor_area)
        color_similarity = _histogram_intersection(
            anchor_histogram,
            _mask_histogram(frame, mask),
        )
        pivot_distance_ratio = float(
            np.linalg.norm(centroid - structure.pivot_xy)
            / max(structure.length_px, 1.0)
        )
        identity_valid = bool(
            minimum_area_ratio <= area_ratio <= maximum_area_ratio
            and color_similarity >= minimum_color
            and minimum_length_ratio
            <= pivot_distance_ratio
            <= maximum_length_ratio
        )
        detections[frame_index].append(
            ObjectDetection(
                frame_index=frame_index,
                detection_id=f"pendulum_direct_{frame_index:05d}",
                xy=centroid,
                area_px2=float(area),
                entity_class=(
                    "pendulum_bob"
                    if identity_valid
                    else "pendulum_bob__replacement"
                ),
                mask=mask,
                confidence=1.0,
                evidence_tier=EvidenceTier.PARTICIPANT,
                sources=(
                    ("condition_directed_sam2",)
                    if identity_valid
                    else (
                        "condition_directed_sam2",
                        "identity_anchor_rejected",
                    )
                ),
                metadata={
                    "identity_anchor_valid": identity_valid,
                    "anchor_color_similarity": color_similarity,
                    "condition_area_ratio": area_ratio,
                    "pivot_distance_ratio": pivot_distance_ratio,
                },
            )
        )
    return detections


def discover_pendulum_objects(
    frames: Sequence[np.ndarray],
    *,
    directed_bob_masks: Sequence[np.ndarray],
    condition_frame: np.ndarray,
    structure: PendulumStructureSpec,
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
    available: Sequence[bool] | None = None,
) -> OpenWorldObservation:
    """Track the directed bob and every credible circle/round residual."""

    frame_count = len(time_grid.times_s)
    if len(frames) != frame_count or len(directed_bob_masks) != frame_count:
        raise ValueError("pendulum observation timelines must be aligned")
    if condition_frame.shape[:2] != frames[0].shape[:2]:
        raise ValueError("condition and prediction canvases must match")
    availability = (
        np.ones(frame_count, dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if availability.shape != (frame_count,) or availability.dtype.kind != "b":
        raise ValueError("available must contain one boolean per frame")
    detections = _directed_bob_detections(
        frames,
        directed_bob_masks,
        availability=availability,
        structure=structure,
        condition_frame=condition_frame,
        config=config,
    )
    difference_threshold = float(
        config.get("condition_change_threshold", 24.0)
    )
    apparatus_half_width = float(
        config.get("support_exclusion_radius_ratio", 1.4)
    ) * structure.bob_radius_px
    condition_histogram = _mask_histogram(
        condition_frame,
        structure.bob_mask,
    )
    maximum_length_ratio = float(
        config.get("maximum_residual_length_ratio", 2.0)
    )
    minimum_length_ratio = float(
        config.get("minimum_residual_length_ratio", 0.35)
    )
    rejected_directed_string_circles = 0
    for frame_index, frame in enumerate(frames):
        if not bool(availability[frame_index]):
            detections[frame_index] = []
            continue
        change = _condition_change_mask(
            frame,
            condition_frame,
            threshold=difference_threshold,
        )
        circles = _circle_candidates(
            frame,
            config=config,
            radius_hint_px=structure.bob_radius_px,
        )
        segments = _line_segments(frame, config=config)
        directed_center = mask_centroid(
            directed_bob_masks[frame_index]
        )
        directed_area = float(
            np.count_nonzero(directed_bob_masks[frame_index])
        )
        directed_radius = math.sqrt(
            max(directed_area, 1.0) / math.pi
        )
        for candidate_index, circle in enumerate(circles):
            x, y, radius = circle
            center = np.asarray([x, y], dtype=np.float64)
            distance_from_pivot = float(
                np.linalg.norm(center - structure.pivot_xy)
            )
            if not (
                minimum_length_ratio * structure.length_px
                <= distance_from_pivot
                <= maximum_length_ratio * structure.length_px
            ):
                continue
            candidate_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
            cv2.circle(
                candidate_mask,
                (int(round(x)), int(round(y))),
                max(2, int(round(radius))),
                255,
                -1,
            )
            if _mask_overlap_fraction(
                candidate_mask, directed_bob_masks[frame_index]
            ) >= float(config.get("direct_duplicate_overlap", 0.45)):
                continue
            if (
                directed_center is not None
                and float(np.linalg.norm(center - directed_center))
                <= float(
                    config.get(
                        "direct_duplicate_center_radius_ratio",
                        2.25,
                    )
                )
                * max(float(radius), directed_radius, 1.0)
            ):
                # Hough often fits an offset circle to the bob/string junction.
                # Pixel overlap alone is too brittle for that duplicate.
                continue
            line_score, line_pivot = _circle_string_evidence(
                circle,
                segments,
                frame_shape=frame.shape,
            )
            changed = _mask_overlap_fraction(candidate_mask, change)
            color_similarity = _histogram_intersection(
                condition_histogram,
                _mask_histogram(frame, candidate_mask),
            )
            weak_string_circle, string_relation = (
                _weak_directed_string_circle(
                    center,
                    radius_px=radius,
                    pivot_xy=structure.pivot_xy,
                    directed_bob_xy=directed_center,
                    directed_bob_radius_px=directed_radius,
                    change_overlap=changed,
                    minimum_body_change_overlap=float(
                        config.get(
                            "minimum_string_candidate_change_overlap",
                            0.08,
                        )
                    ),
                    bob_clearance_radius_ratio=float(
                        config.get(
                            "direct_duplicate_center_radius_ratio",
                            2.25,
                        )
                    ),
                )
            )
            if weak_string_circle:
                rejected_directed_string_circles += 1
                continue
            pivot_error_ratio = (
                float(
                    np.linalg.norm(line_pivot - structure.pivot_xy)
                    / max(structure.length_px, 1.0)
                )
                if line_pivot is not None
                else math.inf
            )
            # The support rod is apparatus, not a second bob.  A residual
            # circle in its corridor needs strong independent evidence.
            in_support_corridor = (
                abs(x - structure.pivot_xy[0]) <= apparatus_half_width
                and y > structure.pivot_xy[1]
            )
            if in_support_corridor and line_score < 0.55 and changed < 0.55:
                continue
            anchored_string = bool(
                line_score
                >= float(config.get("participant_string_score", 0.55))
                and pivot_error_ratio
                <= float(
                    config.get(
                        "maximum_residual_pivot_error_ratio",
                        0.25,
                    )
                )
            )
            change_threshold = float(
                config.get("minimum_change_overlap", 0.25)
            )
            minimum_color = float(
                config.get("minimum_residual_color_similarity", 0.05)
            )
            if (
                anchored_string
                and changed
                >= float(
                    config.get(
                        "minimum_string_candidate_change_overlap",
                        0.08,
                    )
                )
            ):
                tier = EvidenceTier.PARTICIPANT
                confidence = min(1.0, 0.7 + 0.3 * line_score)
            elif (
                changed >= change_threshold
                and (
                    anchored_string
                    or color_similarity >= minimum_color
                )
            ):
                tier = EvidenceTier.TENTATIVE
                confidence = min(0.89, 0.55 + 0.35 * changed)
            else:
                # Stable apparatus circles and single-source Hough fits are
                # rejected before global tracking.  Letting them consume the
                # safety cap would turn ambiguous detector clutter into
                # formal overflow exposure.
                continue
            sources = ["hough_circle"]
            if line_score > 0.0:
                sources.append("string_segment")
            if changed > 0.0:
                sources.append("condition_frame_change")
            proposal_sources = tuple(sources)
            formal_sources = (
                proposal_sources
                if tier is EvidenceTier.PARTICIPANT
                else ("pendulum_residual_candidate",)
            )
            detections[frame_index].append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"pendulum_residual_{frame_index:05d}_"
                        f"{candidate_index:03d}"
                    ),
                    xy=center,
                    area_px2=float(np.count_nonzero(candidate_mask)),
                    entity_class="pendulum_bob",
                    mask=candidate_mask,
                    confidence=confidence,
                    evidence_tier=tier,
                    sources=formal_sources,
                    metadata={
                        "circle_xyr": [x, y, radius],
                        "proposal_sources": list(proposal_sources),
                        "string_score": line_score,
                        "condition_change_overlap": changed,
                        "anchor_color_similarity": color_similarity,
                        **string_relation,
                        "string_pivot_error_ratio": (
                            pivot_error_ratio
                            if math.isfinite(pivot_error_ratio)
                            else None
                        ),
                        "anchored_string": anchored_string,
                    },
                )
            )
    observation = track_open_world_detections(
        detections,
        time_grid=time_grid,
        maximum_gap_s=float(config.get("maximum_tracking_gap_s", 0.25)),
        maximum_assignment_cost=float(
            config.get("maximum_tracking_assignment_cost", 4.0)
        ),
        minimum_scale_px=max(2.0, 0.5 * structure.bob_radius_px),
        maximum_tracks=int(config.get("maximum_tracks", 24)),
    )
    return OpenWorldObservation(
        tracks=observation.tracks,
        overflow_counts=observation.overflow_counts,
        diagnostics={
            **dict(observation.diagnostics),
            "observer": "pendulum_directed_plus_circle_change_v1",
            "condition_structure": structure.to_dict(),
            "available_frames": int(np.count_nonzero(availability)),
            "identity_rejected_frames": int(
                sum(
                    detection.metadata.get("identity_anchor_valid") is False
                    for values in detections
                    for detection in values
                    if "condition_directed_sam2" in detection.sources
                )
            ),
            "rejected_directed_string_circles": (
                rejected_directed_string_circles
            ),
        },
    )


def audit_pendulum_topology(
    assembly_masks: Sequence[np.ndarray],
    *,
    pivot_xy: Sequence[float],
    bob_xy: np.ndarray,
    bob_observed: Sequence[bool],
    bob_masks: Sequence[np.ndarray] | None = None,
    frame_weights: Sequence[float] | None = None,
    string_half_width_px: int = 2,
    minimum_string_occupancy: float = 0.30,
    branch_penalty_weight: float = 0.65,
    branch_minimum_length_ratio: float = 0.12,
    branch_minimum_elongation: float = 3.0,
    branch_maximum_thickness_radius_ratio: float = 0.75,
    branch_bob_exclusion_margin_ratio: float = 0.40,
    branch_support_half_width_radius_ratio: float = 1.25,
    branch_support_verticality_threshold: float = 0.90,
    branch_pivot_support_verticality_threshold: float = 0.65,
) -> dict[str, object]:
    """Score pivot–string–bob connectivity and unsupported string branches.

    The expected topology is frozen by the condition pivot and the bob that
    open-world assignment matched in the current frame.  A branch is a
    sufficiently long, thin component of the prediction assembly that:

    * lies outside a generous corridor around the expected main string;
    * is pixel-connected to the main string or pivot;
    * is not part of the matched bob or the fixed support apparatus.

    This deliberately does not look for another circular bob.  A generated
    fork therefore remains observable even when the extra string has no body
    at its endpoint.  Per-frame penalties are averaged with real-time cell
    weights when supplied, so longer-lived forks reduce the topology score
    smoothly instead of triggering a binary Case failure.
    """

    pivot = _point(pivot_xy, name="pivot_xy")
    centers = np.asarray(bob_xy, dtype=np.float64)
    observed = np.asarray(bob_observed, dtype=bool)
    frame_count = len(assembly_masks)
    if centers.shape != (frame_count, 2):
        raise ValueError("bob_xy must contain one 2-D point per frame")
    if observed.shape != (frame_count,):
        raise ValueError("bob_observed must contain one value per frame")
    if bob_masks is None:
        normalized_bob_masks: list[np.ndarray | None] = [
            None for _ in assembly_masks
        ]
    else:
        if len(bob_masks) != frame_count:
            raise ValueError(
                "bob_masks must contain one mask per assembly frame"
            )
        normalized_bob_masks = [
            _binary_mask(mask, name=f"bob_masks[{index}]")
            for index, mask in enumerate(bob_masks)
        ]
    if frame_weights is None:
        weights = np.ones(frame_count, dtype=np.float64)
    else:
        weights = np.asarray(frame_weights, dtype=np.float64)
        if (
            weights.shape != (frame_count,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "frame_weights must contain one finite non-negative value "
                "per frame"
            )
    if frame_count and float(weights.sum()) <= 0.0:
        raise ValueError("frame_weights must have positive total weight")
    parameters = (
        float(minimum_string_occupancy),
        float(branch_penalty_weight),
        float(branch_minimum_length_ratio),
        float(branch_minimum_elongation),
        float(branch_maximum_thickness_radius_ratio),
        float(branch_bob_exclusion_margin_ratio),
        float(branch_support_half_width_radius_ratio),
        float(branch_support_verticality_threshold),
        float(branch_pivot_support_verticality_threshold),
    )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("pendulum topology thresholds must be finite")
    if int(string_half_width_px) < 0:
        raise ValueError("string_half_width_px must be non-negative")
    if not 0.0 <= minimum_string_occupancy <= 1.0:
        raise ValueError("minimum_string_occupancy must be in [0, 1]")
    if not 0.0 <= branch_penalty_weight <= 1.0:
        raise ValueError("branch_penalty_weight must be in [0, 1]")
    if (
        branch_minimum_length_ratio <= 0.0
        or branch_minimum_elongation <= 1.0
        or branch_maximum_thickness_radius_ratio <= 0.0
        or branch_bob_exclusion_margin_ratio < 0.0
        or branch_support_half_width_radius_ratio < 0.0
        or not (0.0 <= branch_support_verticality_threshold <= 1.0)
        or not (
            0.0
            <= branch_pivot_support_verticality_threshold
            <= 1.0
        )
    ):
        raise ValueError("pendulum branch thresholds are invalid")
    per_frame: list[dict[str, object]] = []
    string_scores: list[float] = []
    topology_scores: list[float] = []
    branch_evidence_values: list[float] = []
    broken_frames: list[int] = []
    branch_frames: list[int] = []
    for frame_index, raw_mask in enumerate(assembly_masks):
        if not observed[frame_index] or not np.isfinite(
            centers[frame_index]
        ).all():
            string_score = 0.0
            occupancy = 0.0
            broken = True
            branch = {
                "evidence": 0.0,
                "detected": False,
                "component_count": 0,
                "candidate_pixel_count": 0,
                "main_exclusion_half_width_px": None,
                "bob_radius_px": None,
                "components": [],
                "rejected_components": [],
            }
        else:
            mask = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(
                np.uint8
            )
            corridor = np.zeros_like(mask)
            cv2.line(
                corridor,
                tuple(int(round(value)) for value in pivot),
                tuple(int(round(value)) for value in centers[frame_index]),
                255,
                max(1, 2 * int(string_half_width_px) + 1),
                cv2.LINE_AA,
            )
            # Ignore the end caps: the bob and pivot themselves must not make
            # a broken middle string look connected.
            length = float(np.linalg.norm(centers[frame_index] - pivot))
            if length > 8.0:
                direction = (centers[frame_index] - pivot) / length
                start = pivot + 0.08 * length * direction
                end = pivot + 0.82 * length * direction
                corridor[:] = 0
                cv2.line(
                    corridor,
                    tuple(int(round(value)) for value in start),
                    tuple(int(round(value)) for value in end),
                    255,
                    max(1, 2 * int(string_half_width_px) + 1),
                    cv2.LINE_AA,
                )
            selected = corridor > 0
            occupancy = (
                float(np.count_nonzero((mask > 0) & selected))
                / max(int(np.count_nonzero(selected)), 1)
            )
            string_score = float(
                np.clip(
                    occupancy / max(minimum_string_occupancy, 1e-6),
                    0.0,
                    1.0,
                )
            )
            broken = occupancy < minimum_string_occupancy
            branch = _pendulum_branch_evidence(
                mask,
                pivot_xy=pivot,
                bob_xy=centers[frame_index],
                bob_mask=normalized_bob_masks[frame_index],
                string_half_width_px=int(string_half_width_px),
                minimum_length_ratio=float(
                    branch_minimum_length_ratio
                ),
                minimum_elongation=float(branch_minimum_elongation),
                maximum_thickness_radius_ratio=float(
                    branch_maximum_thickness_radius_ratio
                ),
                bob_exclusion_margin_ratio=float(
                    branch_bob_exclusion_margin_ratio
                ),
                support_half_width_radius_ratio=float(
                    branch_support_half_width_radius_ratio
                ),
                support_verticality_threshold=float(
                    branch_support_verticality_threshold
                ),
                pivot_support_verticality_threshold=float(
                    branch_pivot_support_verticality_threshold
                ),
            )
        branch_evidence = float(branch["evidence"])
        topology_score = string_score * max(
            0.0,
            1.0 - branch_penalty_weight * branch_evidence,
        )
        string_scores.append(string_score)
        branch_evidence_values.append(branch_evidence)
        topology_scores.append(topology_score)
        if broken:
            broken_frames.append(frame_index)
        if bool(branch["detected"]):
            branch_frames.append(frame_index)
        per_frame.append(
            {
                "frame": frame_index,
                "bob_observed": bool(observed[frame_index]),
                "string_occupancy": occupancy,
                "string_intact_score": string_score,
                "broken_string": broken,
                "branch_string_evidence": branch_evidence,
                "branch_string_detected": bool(branch["detected"]),
                "branch_string_component_count": int(
                    branch["component_count"]
                ),
                "branch_string_candidate_pixel_count": int(
                    branch["candidate_pixel_count"]
                ),
                "branch_main_exclusion_half_width_px": branch[
                    "main_exclusion_half_width_px"
                ],
                "branch_bob_radius_px": branch["bob_radius_px"],
                "branch_components": branch["components"],
                "branch_rejected_components": branch[
                    "rejected_components"
                ],
                "branch_string_penalty": (
                    branch_penalty_weight * branch_evidence
                ),
                "topology_frame_score": topology_score,
            }
        )
    score = _weighted_mean(topology_scores, weights)
    string_score = _weighted_mean(string_scores, weights)
    branch_exposure = _weighted_mean(branch_evidence_values, weights)
    return {
        "version": PENDULUM_STRUCTURE_SPEC_VERSION,
        "score": score,
        "string_intact_score": string_score,
        "score_before_branch_string_penalty": string_score,
        "branch_string_exposure_ratio": branch_exposure,
        "branch_string_penalty_weight": float(branch_penalty_weight),
        "branch_frame_count": len(branch_frames),
        "branch_frames": branch_frames,
        "broken_frame_count": len(broken_frames),
        "broken_frames": broken_frames,
        "per_frame": per_frame,
    }


def _weighted_mean(
    values: Sequence[float],
    weights: np.ndarray,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(np.dot(array, weights) / total)


def _pendulum_branch_evidence(
    assembly_mask: np.ndarray,
    *,
    pivot_xy: np.ndarray,
    bob_xy: np.ndarray,
    bob_mask: np.ndarray | None,
    string_half_width_px: int,
    minimum_length_ratio: float,
    minimum_elongation: float,
    maximum_thickness_radius_ratio: float,
    bob_exclusion_margin_ratio: float,
    support_half_width_radius_ratio: float,
    support_verticality_threshold: float,
    pivot_support_verticality_threshold: float,
) -> dict[str, object]:
    """Find thin assembly components attached outside the frozen main string."""

    mask = np.where(np.asarray(assembly_mask) > 0, 255, 0).astype(
        np.uint8
    )
    height, _ = mask.shape
    pivot = np.asarray(pivot_xy, dtype=np.float64)
    bob = np.asarray(bob_xy, dtype=np.float64)
    string_vector = bob - pivot
    string_length = float(np.linalg.norm(string_vector))
    empty = {
        "evidence": 0.0,
        "detected": False,
        "component_count": 0,
        "candidate_pixel_count": 0,
        "main_exclusion_half_width_px": None,
        "bob_radius_px": None,
        "components": [],
        "rejected_components": [],
    }
    if (
        string_length <= 8.0
        or not np.isfinite(string_length)
        or not np.any(mask)
    ):
        return empty

    normalized_bob = (
        np.zeros_like(mask)
        if bob_mask is None
        else np.where(np.asarray(bob_mask) > 0, 255, 0).astype(np.uint8)
    )
    bob_area = int(np.count_nonzero(normalized_bob))
    if bob_area > 0:
        bob_radius = math.sqrt(bob_area / math.pi)
    else:
        # The topology audit is still fail-closed when the caller has no
        # usable bob mask.  This conservative geometric disk prevents the bob
        # itself from being reinterpreted as a string branch.
        bob_radius = max(
            2.0 * string_half_width_px + 1.0,
            0.045 * string_length,
        )
        cv2.circle(
            normalized_bob,
            tuple(int(round(value)) for value in bob),
            max(2, int(round(bob_radius))),
            255,
            -1,
        )

    # Estimate the actual half-thickness of the main string from foreground
    # distance values along its middle.  The exclusion is deliberately wider
    # than the occupancy corridor so SAM boundary jitter cannot masquerade as
    # two parallel branches.
    unit = string_vector / string_length
    sample_start = pivot + 0.12 * string_length * unit
    sample_end = pivot + 0.78 * string_length * unit
    sample_line = np.zeros_like(mask)
    cv2.line(
        sample_line,
        tuple(int(round(value)) for value in sample_start),
        tuple(int(round(value)) for value in sample_end),
        255,
        1,
        cv2.LINE_8,
    )
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    samples = distance[(sample_line > 0) & (mask > 0)]
    observed_half_width = (
        float(np.quantile(samples, 0.75))
        if samples.size
        else float(string_half_width_px + 1)
    )
    main_exclusion_half_width = max(
        string_half_width_px + 2,
        int(math.ceil(observed_half_width + 1.5)),
    )
    # A pathological thick foreground blob should not make the exclusion eat
    # the whole scene and thereby hide a real fork.
    main_exclusion_half_width = min(
        main_exclusion_half_width,
        max(string_half_width_px + 2, int(round(0.7 * bob_radius))),
    )
    main_exclusion = np.zeros_like(mask)
    cv2.line(
        main_exclusion,
        tuple(int(round(value)) for value in pivot),
        tuple(int(round(value)) for value in bob),
        255,
        max(1, 2 * main_exclusion_half_width + 1),
        cv2.LINE_AA,
    )
    pivot_exclusion = np.zeros_like(mask)
    cv2.circle(
        pivot_exclusion,
        tuple(int(round(value)) for value in pivot),
        main_exclusion_half_width,
        255,
        -1,
    )
    bob_margin = max(
        1,
        int(round(bob_exclusion_margin_ratio * bob_radius)),
    )
    bob_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * bob_margin + 1, 2 * bob_margin + 1),
    )
    bob_exclusion = cv2.dilate(normalized_bob, bob_kernel)

    residual = np.where(
        (mask > 0)
        & (main_exclusion == 0)
        & (pivot_exclusion == 0)
        & (bob_exclusion == 0),
        255,
        0,
    ).astype(np.uint8)
    # Pendulum branches extend from the pivot into the lower half-plane.
    # Crossbars and suspension hardware above the frozen pivot are apparatus.
    row_limit = max(
        0,
        min(height, int(math.floor(pivot[1] - main_exclusion_half_width))),
    )
    residual[:row_limit, :] = 0

    # Real attachment must remain 8-connected to foreground inside the main
    # exclusion.  A nearby but detached scratch or texture is not a branch.
    main_foreground = np.where(
        (mask > 0) & ((main_exclusion > 0) | (pivot_exclusion > 0)),
        255,
        0,
    ).astype(np.uint8)
    attachment_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    attachment_zone = cv2.dilate(main_foreground, attachment_kernel)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual,
        connectivity=8,
    )
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    minimum_span_px = minimum_length_ratio * string_length
    minimum_pixels = max(6, int(round(0.45 * minimum_span_px)))
    maximum_thickness_px = max(
        2.0 * string_half_width_px + 3.0,
        maximum_thickness_radius_ratio * bob_radius,
    )
    support_half_width = max(
        main_exclusion_half_width + 1.0,
        support_half_width_radius_ratio * bob_radius,
    )
    pivot_bar_half_height = max(
        main_exclusion_half_width + 1.0,
        0.55 * bob_radius,
    )

    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_mask = labels == label
        ys, xs = np.where(component_mask)
        if not len(xs):
            continue
        points = np.column_stack((xs, ys)).astype(np.float64)
        centered = points - points.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(points), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = points @ major_axis
        span = float(projection.max() - projection.min() + 1.0)
        thickness = float(area / max(span, 1.0))
        elongation = float(span / max(thickness, 1.0))
        relative_length = float(span / string_length)
        attached = bool(
            np.any(component_mask & (attachment_zone > 0))
        )

        offsets = points - pivot[None, :]
        main_projection = (
            offsets @ string_vector / max(string_length**2, 1e-9)
        )
        lateral = np.abs(
            offsets[:, 0] * unit[1] - offsets[:, 1] * unit[0]
        )
        nearest = int(np.argmin(lateral))
        attachment_fraction = float(main_projection[nearest])
        verticality = abs(float(major_axis[1]))
        support_fraction = float(
            np.mean(
                (np.abs(points[:, 0] - pivot[0]) <= support_half_width)
                & (points[:, 1] >= pivot[1] - pivot_bar_half_height)
            )
        )
        pivot_bar_fraction = float(
            np.mean(
                np.abs(points[:, 1] - pivot[1])
                <= pivot_bar_half_height
            )
        )
        rejection_reason: str | None = None
        if not attached:
            rejection_reason = "not_connected_to_main_string"
        elif area < minimum_pixels or span < minimum_span_px:
            rejection_reason = "too_short_or_sparse"
        elif elongation < minimum_elongation:
            rejection_reason = "not_elongated"
        elif thickness > maximum_thickness_px:
            rejection_reason = "too_thick_for_string"
        elif not -0.08 <= attachment_fraction <= 0.92:
            rejection_reason = "attachment_outside_pivot_main_string"
        elif (
            verticality >= support_verticality_threshold
            or (
                attachment_fraction <= 0.06
                and verticality
                >= pivot_support_verticality_threshold
            )
            or (
                support_fraction >= 0.55
                and verticality
                >= pivot_support_verticality_threshold
            )
        ):
            rejection_reason = "vertical_support_apparatus"
        elif pivot_bar_fraction >= 0.72 and verticality <= 0.30:
            rejection_reason = "pivot_crossbar_apparatus"

        diagnostic = {
            "area_px2": area,
            "span_px": span,
            "relative_length": relative_length,
            "thickness_px": thickness,
            "elongation": elongation,
            "attachment_fraction": attachment_fraction,
            "support_overlap_fraction": support_fraction,
            "pivot_bar_overlap_fraction": pivot_bar_fraction,
            "major_axis_verticality": verticality,
        }
        if rejection_reason is not None:
            rejected.append(
                {**diagnostic, "reason": rejection_reason}
            )
            continue
        length_strength = float(
            np.clip(
                relative_length / max(2.0 * minimum_length_ratio, 1e-6),
                0.0,
                1.0,
            )
        )
        elongation_strength = float(
            np.clip(
                elongation / max(2.0 * minimum_elongation, 1e-6),
                0.0,
                1.0,
            )
        )
        evidence = math.sqrt(length_strength * elongation_strength)
        accepted.append({**diagnostic, "evidence": evidence})

    combined_evidence = float(
        1.0
        - math.prod(
            1.0 - float(component["evidence"])
            for component in accepted
        )
    )
    combined_evidence = float(np.clip(combined_evidence, 0.0, 1.0))
    return {
        "evidence": combined_evidence,
        "detected": bool(accepted),
        "component_count": len(accepted),
        "candidate_pixel_count": int(
            sum(int(component["area_px2"]) for component in accepted)
        ),
        "main_exclusion_half_width_px": main_exclusion_half_width,
        "bob_radius_px": bob_radius,
        "components": accepted,
        "rejected_components": rejected,
    }


__all__ = [
    "PENDULUM_STRUCTURE_SPEC_VERSION",
    "PendulumStructureSpec",
    "audit_pendulum_topology",
    "detect_condition_structure",
    "discover_pendulum_objects",
    "extract_bob_masks",
    "letterbox_condition_image",
    "prompt_from_structure",
    "structure_from_reference_trace",
]
