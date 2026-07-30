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


@dataclass(frozen=True)
class FrozenCircularApparatus:
    """Condition-side disk geometry and appearance used by V7 observation.

    The arrays remain evaluator-internal.  In particular, prediction pixels
    are never allowed to redefine the coordinate origin, scale, or apparatus
    palette after this value has been materialized.
    """

    center_xy: np.ndarray
    radius_px: float
    interior_mask: np.ndarray
    disk_mask: np.ndarray
    condition_lab: np.ndarray
    condition_apparatus_mask: np.ndarray
    palette_lab: np.ndarray
    selected_hue: int
    source: str


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


def _legacy_disk_geometry(
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


def _adaptive_disk_geometry(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Find the circular platform without assuming a particular hue.

    The circular-motion captures use otherwise equivalent green, blue, and
    pink platforms.  The legacy observer encoded the green interval directly,
    which made platform colour an accidental reference-availability switch.
    Here several saturated hue modes propose apparatus pixels, while circular
    support and fill quality select the platform component.  The procedure is
    frame local and does not use the manifest cardinality or prediction
    trajectory.
    """

    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    minimum_saturation = int(
        config.get("adaptive_disk_minimum_saturation", 35)
    )
    minimum_value = int(config.get("adaptive_disk_minimum_value", 25))
    eligible = (
        (hsv[:, :, 1] >= minimum_saturation)
        & (hsv[:, :, 2] >= minimum_value)
    )
    if not np.any(eligible):
        raise SceneAnalysisError(
            "circular_disk_not_found",
            "no saturated rotating-platform colour support",
        )

    histogram = np.bincount(hue[eligible], minlength=180).astype(
        np.float64
    )
    smoothing_radius = max(
        0, int(config.get("adaptive_disk_hue_smoothing_radius", 4))
    )
    smoothed = np.zeros_like(histogram)
    for offset in range(-smoothing_radius, smoothing_radius + 1):
        smoothed += np.roll(histogram, offset)
    hue_tolerance = int(config.get("adaptive_disk_hue_tolerance", 8))
    if not 0 <= hue_tolerance < 90:
        raise ValueError(
            "adaptive_disk_hue_tolerance must be in [0, 90)"
        )

    height, width = frame.shape[:2]
    frame_area = float(height * width)
    minimum_ratio = float(config["minimum_disk_area_ratio"])
    maximum_ratio = float(
        config.get("adaptive_disk_maximum_area_ratio", 0.75)
    )
    minimum_radius = float(
        config.get("adaptive_disk_minimum_radius_ratio", 0.08)
    ) * min(height, width)
    maximum_radius = float(
        config.get("adaptive_disk_maximum_radius_ratio", 0.48)
    ) * min(height, width)
    close_size = max(
        3, int(config.get("adaptive_disk_close_kernel", 9)) | 1
    )
    open_size = max(
        3, int(config.get("adaptive_disk_open_kernel", 5)) | 1
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    canvas_center = np.asarray(
        [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
        dtype=np.float64,
    )
    canvas_scale = np.asarray(
        [max(width / 2.0, 1.0), max(height / 2.0, 1.0)],
        dtype=np.float64,
    )
    # A saturated scene background can legitimately contain more pixels than
    # the platform.  Treating the global histogram maximum as the only hue
    # hypothesis therefore makes background colour an availability switch.
    # Instead, test several separated hue modes and let circular platform
    # geometry choose between them.  This policy is reached only through the
    # explicit ``adaptive_dominant_hue`` opt-in; the v6 default remains the
    # legacy green interval.
    maximum_hypotheses = int(
        config.get("adaptive_disk_maximum_hue_hypotheses", 24)
    )
    if maximum_hypotheses < 1:
        raise ValueError(
            "adaptive_disk_maximum_hue_hypotheses must be positive"
        )
    ranked_hues = np.argsort(-smoothed, kind="stable").tolist()
    hue_hypotheses: list[int] = []
    for value in ranked_hues:
        if smoothed[value] <= 0.0:
            break
        if any(
            min((value - prior) % 180, (prior - value) % 180)
            <= hue_tolerance
            for prior in hue_hypotheses
        ):
            continue
        hue_hypotheses.append(int(value))
        if len(hue_hypotheses) >= maximum_hypotheses:
            break

    candidates: list[tuple[float, np.ndarray, np.ndarray, int]] = []
    for hue_hypothesis in hue_hypotheses:
        forward = np.mod(hue - hue_hypothesis, 180)
        backward = np.mod(hue_hypothesis - hue, 180)
        hue_distance = np.minimum(forward, backward)
        raw_colour = np.where(
            eligible & (hue_distance <= hue_tolerance), 255, 0
        ).astype(np.uint8)
        apparatus_colour = cv2.morphologyEx(
            raw_colour,
            cv2.MORPH_CLOSE,
            close_kernel,
        )
        apparatus_colour = cv2.morphologyEx(
            apparatus_colour,
            cv2.MORPH_OPEN,
            open_kernel,
        )
        contours, _ = cv2.findContours(
            apparatus_colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / max(frame_area, 1.0)
            if not minimum_ratio <= area_ratio <= maximum_ratio:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-9:
                continue
            circularity = float(
                np.clip(
                    4.0 * math.pi * area / (perimeter * perimeter),
                    0.0,
                    1.0,
                )
            )
            (center_x, center_y), enclosing_radius = (
                cv2.minEnclosingCircle(contour)
            )
            if not minimum_radius <= enclosing_radius <= maximum_radius:
                continue
            circle_area = math.pi * enclosing_radius * enclosing_radius
            fill_ratio = float(
                np.clip(area / max(circle_area, 1.0), 0.0, 1.0)
            )
            center_distance = float(
                np.linalg.norm(
                    (
                        np.asarray(
                            [center_x, center_y], dtype=np.float64
                        )
                        - canvas_center
                    )
                    / canvas_scale
                )
            )
            # Area is only an eligibility constraint above.  Making it a
            # positive ranking term lets a saturated square tabletop around
            # the platform beat the actual disk after letterboxing: its
            # enclosing circle is larger even though that circle is poorly
            # filled.  Rank the surviving hypotheses by circular support
            # instead.  The fourth powers deliberately separate a genuinely
            # filled disk from a centred rectangle (whose enclosing-circle
            # fill is about 2 / pi) without introducing another threshold.
            # This branch is v7-only because v6 uses ``legacy_green_hsv``.
            score = (
                max(circularity, 1e-3) ** 4
                * max(fill_ratio, 1e-3) ** 4
                / (1.0 + center_distance)
            )
            candidates.append(
                (score, contour, raw_colour, hue_hypothesis)
            )
    if not candidates:
        raise SceneAnalysisError(
            "circular_disk_not_found",
            "no adaptive-hue component satisfies circular platform geometry",
        )

    _, contour, apparatus_pixels, selected_hue = max(
        candidates, key=lambda value: value[0]
    )
    (center_x, center_y), enclosing_radius = cv2.minEnclosingCircle(
        contour
    )
    disk = np.zeros(frame.shape[:2], dtype=np.uint8)
    # Use the scene's circular apparatus model rather than the raw colour
    # contour.  A participant at the rim cuts a notch out of that contour;
    # filling the notch-shaped contour would then erase the very object the
    # observer is meant to audit.
    cv2.circle(
        disk,
        (int(round(center_x)), int(round(center_y))),
        max(1, int(round(enclosing_radius))),
        255,
        thickness=-1,
    )
    erosion_size = max(
        1,
        int(
            config.get(
                "adaptive_disk_erosion_kernel",
                config["disk_erosion_kernel"],
            )
        )
        | 1,
    )
    erosion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )
    interior = cv2.erode(disk, erosion)
    center = np.asarray(
        [center_x, center_y],
        dtype=np.float64,
    )
    radius = float(enclosing_radius)
    # The narrow hue interval is useful for geometry selection, but platform
    # highlights and shadows can sit just outside it and otherwise appear as
    # short-lived "objects".  Extend only the high-saturation support after
    # the disk geometry has been frozen.  Low-saturation metal participants
    # remain foreground even when their unstable HSV hue happens to resemble
    # the platform.  Both extensions are derived from the already-frozen v7
    # colour thresholds rather than introducing hidden protocol constants.
    selected_forward = np.mod(hue - selected_hue, 180)
    selected_backward = np.mod(selected_hue - hue, 180)
    selected_distance = np.minimum(selected_forward, selected_backward)
    cleanup_tolerance = min(
        89, hue_tolerance + max(2, hue_tolerance // 2)
    )
    cleanup_saturation = min(255, 2 * minimum_saturation)
    saturated_apparatus = np.where(
        (hsv[:, :, 1] >= cleanup_saturation)
        & (hsv[:, :, 2] >= minimum_value)
        & (selected_distance <= cleanup_tolerance),
        255,
        0,
    ).astype(np.uint8)
    apparatus_pixels = cv2.bitwise_or(
        apparatus_pixels, saturated_apparatus
    )
    # Restrict the colour mask to the selected apparatus geometry.  Similar
    # hues elsewhere in the image must not erase participant proposals.
    apparatus_pixels = cv2.morphologyEx(
        apparatus_pixels,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    apparatus_pixels = cv2.bitwise_and(apparatus_pixels, disk)
    return center, radius, interior, apparatus_pixels


def _disk_geometry(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    policy = str(config.get("disk_colour_policy", "legacy_green_hsv"))
    if policy == "legacy_green_hsv":
        return _legacy_disk_geometry(frame, config=config)
    if policy == "adaptive_dominant_hue":
        return _adaptive_disk_geometry(frame, config=config)
    raise ValueError(f"unsupported circular disk colour policy: {policy!r}")


def _circular_hue_distance(
    hue: np.ndarray,
    selected_hue: int,
) -> np.ndarray:
    forward = np.mod(hue.astype(np.int16) - int(selected_hue), 180)
    backward = np.mod(int(selected_hue) - hue.astype(np.int16), 180)
    return np.minimum(forward, backward)


def _apparatus_palette(
    lab: np.ndarray,
    mask: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> np.ndarray:
    pixels = np.asarray(lab, dtype=np.float64)[np.asarray(mask) > 0]
    if len(pixels) < 1:
        raise SceneAnalysisError(
            "circular_condition_apparatus_empty",
            "condition disk has no apparatus appearance pixels",
        )
    quantization = int(
        config.get("apparatus_palette_lab_quantization", 8)
    )
    maximum_colours = int(
        config.get("apparatus_palette_maximum_colours", 24)
    )
    minimum_fraction = float(
        config.get("apparatus_palette_minimum_fraction", 0.001)
    )
    if quantization < 1:
        raise ValueError(
            "apparatus_palette_lab_quantization must be positive"
        )
    if maximum_colours < 1:
        raise ValueError(
            "apparatus_palette_maximum_colours must be positive"
        )
    if (
        not math.isfinite(minimum_fraction)
        or not 0.0 <= minimum_fraction <= 1.0
    ):
        raise ValueError(
            "apparatus_palette_minimum_fraction must be in [0, 1]"
        )
    bins = np.floor_divide(
        np.asarray(pixels, dtype=np.int16),
        quantization,
    )
    unique, inverse, counts = np.unique(
        bins, axis=0, return_inverse=True, return_counts=True
    )
    minimum_count = max(
        1, int(math.ceil(minimum_fraction * len(pixels)))
    )
    eligible = np.flatnonzero(counts >= minimum_count)
    if len(eligible) < 1:
        eligible = np.asarray([int(np.argmax(counts))], dtype=np.int64)
    order = sorted(
        eligible.tolist(),
        key=lambda index: (
            -int(counts[index]),
            tuple(int(value) for value in unique[index].tolist()),
        ),
    )[:maximum_colours]
    centres = [
        np.mean(pixels[inverse == index], axis=0)
        for index in order
    ]
    return np.asarray(centres, dtype=np.float64)


def freeze_circular_apparatus(
    condition_frame: np.ndarray,
    *,
    config: Mapping[str, Any],
    source: str,
) -> FrozenCircularApparatus:
    """Freeze the V7 disk frame from condition/reference pixels only."""

    if str(
        config.get(
            "apparatus_coordinate_policy",
            "per_frame_prediction_geometry_v1",
        )
    ) != "condition_frozen_shared_frame_v1":
        raise ValueError(
            "frozen circular apparatus requires the explicit V7 "
            "condition_frozen_shared_frame_v1 policy"
        )
    frame = np.asarray(condition_frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("condition_frame must be one BGR image")
    center, radius, interior, apparatus_pixels = _disk_geometry(
        frame, config=config
    )
    disk = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(
        disk,
        (int(round(float(center[0]))), int(round(float(center[1])))),
        max(1, int(round(float(radius)))),
        255,
        thickness=-1,
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    support = cv2.bitwise_and(apparatus_pixels, interior)
    hues = hsv[:, :, 0][support > 0]
    if len(hues) < 1:
        raise SceneAnalysisError(
            "circular_condition_apparatus_empty",
            "condition disk has no hue support for apparatus freezing",
        )
    selected_hue = int(
        np.argmax(np.bincount(hues.astype(np.int64), minlength=180))
    )
    palette = _apparatus_palette(lab, support, config=config)
    values = {
        "center_xy": np.asarray(center, dtype=np.float64),
        "interior_mask": np.asarray(interior, dtype=np.uint8),
        "disk_mask": disk,
        "condition_lab": np.asarray(lab, dtype=np.float64),
        "condition_apparatus_mask": np.asarray(support, dtype=np.uint8),
        "palette_lab": palette,
    }
    for value in values.values():
        value.setflags(write=False)
    return FrozenCircularApparatus(
        center_xy=values["center_xy"],
        radius_px=float(radius),
        interior_mask=values["interior_mask"],
        disk_mask=values["disk_mask"],
        condition_lab=values["condition_lab"],
        condition_apparatus_mask=values[
            "condition_apparatus_mask"
        ],
        palette_lab=values["palette_lab"],
        selected_hue=selected_hue,
        source=str(source),
    )


def _frozen_apparatus_geometry(
    frame: np.ndarray,
    *,
    apparatus: FrozenCircularApparatus,
    previous_frame: np.ndarray | None,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, Mapping[str, float]]:
    """Apply a condition-frozen disk without trusting prediction geometry."""

    value = np.asarray(frame, dtype=np.uint8)
    if value.shape[:2] != apparatus.interior_mask.shape:
        raise ValueError(
            "frame shape differs from frozen circular apparatus"
        )
    appearance_policy = str(
        config.get("apparatus_appearance_policy", "condition_hue_only_v1")
    )
    if appearance_policy != "condition_lab_value_temporal_residual_v1":
        raise ValueError(
            "condition-frozen circular apparatus requires explicit "
            "condition_lab_value_temporal_residual_v1 appearance policy"
        )
    hsv = cv2.cvtColor(value, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(value, cv2.COLOR_BGR2LAB).astype(np.float64)
    minimum_saturation = int(
        config.get("adaptive_disk_minimum_saturation", 35)
    )
    minimum_value = int(config.get("adaptive_disk_minimum_value", 25))
    hue_tolerance = int(config.get("adaptive_disk_hue_tolerance", 8))
    cleanup_tolerance = min(
        89, hue_tolerance + max(2, hue_tolerance // 2)
    )
    hue_support = (
        _circular_hue_distance(
            hsv[:, :, 0], apparatus.selected_hue
        )
        <= cleanup_tolerance
    )
    chromatic_support = (
        (hsv[:, :, 1] >= minimum_saturation)
        & (hsv[:, :, 2] >= minimum_value)
        & hue_support
    )
    frame_shift_policy = str(
        config.get(
            "apparatus_palette_frame_shift_policy",
            "disabled",
        )
    )
    if frame_shift_policy not in {
        "disabled",
        "condition_support_robust_median_v1",
    }:
        raise ValueError(
            "unsupported apparatus palette frame-shift policy: "
            f"{frame_shift_policy!r}"
        )
    condition_support_mask = (
        apparatus.condition_apparatus_mask > 0
    )
    palette_shift = np.zeros(3, dtype=np.float64)
    if frame_shift_policy == "condition_support_robust_median_v1":
        current_pixels = lab[condition_support_mask]
        condition_pixels = apparatus.condition_lab[
            condition_support_mask
        ]
        if len(current_pixels) < 1 or len(condition_pixels) < 1:
            raise SceneAnalysisError(
                "circular_condition_apparatus_empty",
                "frozen apparatus has no pixels for illumination alignment",
            )
        palette_shift = (
            np.median(current_pixels, axis=0)
            - np.median(condition_pixels, axis=0)
        )
        maximum_shift = float(
            config.get(
                "apparatus_palette_maximum_frame_shift_lab", 45.0
            )
        )
        if not math.isfinite(maximum_shift) or maximum_shift < 0.0:
            raise ValueError(
                "apparatus_palette_maximum_frame_shift_lab must be "
                "finite and non-negative"
            )
        shift_norm = float(np.linalg.norm(palette_shift))
        if shift_norm > maximum_shift and shift_norm > 1e-12:
            palette_shift *= maximum_shift / shift_norm
    shifted_palette = apparatus.palette_lab + palette_shift[None, :]
    shifted_condition = apparatus.condition_lab + palette_shift[None, None, :]
    palette_threshold = float(
        config.get("apparatus_palette_maximum_lab_distance", 18.0)
    )
    condition_threshold = float(
        config.get("apparatus_condition_maximum_lab_distance", 22.0)
    )
    temporal_threshold = float(
        config.get("apparatus_temporal_minimum_lab_distance", 30.0)
    )
    temporal_condition_threshold = float(
        config.get(
            "apparatus_temporal_condition_minimum_lab_distance",
            30.0,
        )
    )
    thresholds = {
        "apparatus_palette_maximum_lab_distance": palette_threshold,
        "apparatus_condition_maximum_lab_distance": condition_threshold,
        "apparatus_temporal_minimum_lab_distance": temporal_threshold,
        "apparatus_temporal_condition_minimum_lab_distance": (
            temporal_condition_threshold
        ),
    }
    if any(
        not math.isfinite(raw) or raw < 0.0
        for raw in thresholds.values()
    ):
        raise ValueError(
            "frozen apparatus Lab thresholds must be finite and "
            "non-negative"
        )
    palette_distance = np.min(
        np.linalg.norm(
            lab[:, :, None, :] - shifted_palette[None, None, :, :],
            axis=3,
        ),
        axis=2,
    )
    condition_distance = np.linalg.norm(
        lab - shifted_condition, axis=2
    )
    palette_support = palette_distance <= palette_threshold
    coordinate_support = (
        (apparatus.condition_apparatus_mask > 0)
        & (condition_distance <= condition_threshold)
    )
    apparatus_pixels = (
        chromatic_support & (palette_support | coordinate_support)
    )
    temporal_override = np.zeros(
        apparatus.interior_mask.shape, dtype=bool
    )
    if previous_frame is not None:
        previous = cv2.cvtColor(
            np.asarray(previous_frame, dtype=np.uint8),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float64)
        temporal_distance = np.linalg.norm(lab - previous, axis=2)
        temporal_override = (
            (apparatus.condition_apparatus_mask > 0)
            & (condition_distance >= temporal_condition_threshold)
            & (temporal_distance >= temporal_threshold)
        )
        apparatus_pixels &= ~temporal_override
    apparatus_mask = np.where(
        apparatus_pixels & (apparatus.disk_mask > 0), 255, 0
    ).astype(np.uint8)
    apparatus_mask = cv2.morphologyEx(
        apparatus_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    condition_support = apparatus.condition_apparatus_mask > 0
    condition_support_count = int(np.count_nonzero(condition_support))
    consistency = (
        float(
            np.count_nonzero(
                (apparatus_mask > 0) & condition_support
            )
        )
        / max(condition_support_count, 1)
    )
    return (
        np.asarray(apparatus.center_xy, dtype=np.float64),
        float(apparatus.radius_px),
        np.asarray(apparatus.interior_mask, dtype=np.uint8),
        apparatus_mask,
        {
            "apparatus_condition_consistency": consistency,
            "apparatus_temporal_override_fraction": float(
                np.mean(
                    temporal_override
                    & (apparatus.interior_mask > 0)
                )
            ),
            "apparatus_palette_shift_lab": float(
                np.linalg.norm(palette_shift)
            ),
        },
    )


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
    minimum_area = max(
        int(config["minimum_component_area"]),
        int(
            math.ceil(
                math.pi
                * float(disk_radius_px)
                * float(disk_radius_px)
                * float(
                    config.get(
                        "open_world_minimum_component_disk_area_ratio",
                        0.0,
                    )
                )
            )
        ),
    )
    maximum_area = int(
        frame.shape[0]
        * frame.shape[1]
        * float(config["maximum_component_area_ratio"])
    )
    shape_filter = str(
        config.get("open_world_component_shape_filter", "disabled")
    )
    if shape_filter not in {"disabled", "compact_participant_v1"}:
        raise ValueError(
            "unsupported circular component shape filter: "
            f"{shape_filter!r}"
        )
    candidates: list[tuple[int, int, bool, Mapping[str, float]]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= maximum_area:
            continue
        shape_accepted = True
        shape_metrics: dict[str, float] = {}
        if shape_filter == "compact_participant_v1":
            component = np.where(labels == label, 255, 0).astype(
                np.uint8
            )
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                shape_accepted = False
            else:
                contour = max(contours, key=cv2.contourArea)
                contour_area = float(cv2.contourArea(contour))
                perimeter = float(cv2.arcLength(contour, True))
                _, _, box_width, box_height = cv2.boundingRect(contour)
                hull_area = float(
                    cv2.contourArea(cv2.convexHull(contour))
                )
                rectangularity = contour_area / max(
                    float(box_width * box_height), 1.0
                )
                solidity = contour_area / max(hull_area, 1.0)
                aspect_ratio = max(box_width, box_height) / max(
                    min(box_width, box_height), 1
                )
                circularity = (
                    4.0
                    * math.pi
                    * contour_area
                    / max(perimeter * perimeter, 1e-9)
                )
                shape_metrics = {
                    "rectangularity": float(rectangularity),
                    "solidity": float(solidity),
                    "aspect_ratio": float(aspect_ratio),
                    "circularity": float(circularity),
                }
                shape_accepted = not (
                    rectangularity
                    < float(
                        config.get(
                            "open_world_minimum_component_rectangularity",
                            0.32,
                        )
                    )
                    or solidity
                    < float(
                        config.get(
                            "open_world_minimum_component_solidity",
                            0.75,
                        )
                    )
                    or aspect_ratio
                    > float(
                        config.get(
                            "open_world_maximum_component_aspect_ratio",
                            5.0,
                        )
                    )
                    or circularity
                    < float(
                        config.get(
                            "open_world_minimum_component_circularity",
                            0.25,
                        )
                    )
                )
        # The compactness check is a confidence feature, not a cardinality
        # deletion rule.  Thin or partially occluded generated participants
        # are precisely the cases an open-world integrity audit must retain.
        # Repeated tentative detections are promoted by the causal tracker;
        # one-frame fragments still pay only tentative exposure.
        candidates.append(
            (label, area, shape_accepted, shape_metrics)
        )
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
    for component_index, (
        label,
        area,
        shape_accepted,
        shape_metrics,
    ) in enumerate(retained):
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
                confidence=0.85 if shape_accepted else 0.45,
                evidence_tier=EvidenceTier.TENTATIVE,
                sources=(
                    ("disk_interior_color_contrast",)
                    if shape_filter == "disabled"
                    else (
                        ("disk_interior_compact_participant",)
                        if shape_accepted
                        else (
                            "disk_interior_shape_rejected_audit_candidate",
                        )
                    )
                ),
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
                    "apparatus_exclusion": (
                        "eroded_adaptive_hue_disk_interior"
                        if config.get(
                            "disk_colour_policy", "legacy_green_hsv"
                        )
                        == "adaptive_dominant_hue"
                        else "eroded_green_disk_interior"
                    ),
                    **(
                        {}
                        if shape_filter == "disabled"
                        else {
                            "shape_filter": shape_filter,
                            "shape_filter_accepted": shape_accepted,
                            "shape_metrics": shape_metrics,
                        }
                    ),
                },
            )
        )
    # Include every accepted component in the audit union, including objects
    # that cannot be assigned a causal track because of the global track cap.
    for label, _, _, _ in candidates[safety_cap:]:
        union[labels == label] = 255
    return output, union, len(candidates), overflow


def observe_circular_objects(
    frames: Sequence[np.ndarray],
    *,
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
    available: Sequence[bool] | None = None,
    apparatus_anchor: FrozenCircularApparatus | None = None,
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
    apparatus_consistency = np.full(
        frame_count, np.nan, dtype=np.float64
    )
    temporal_override_fraction = np.full(
        frame_count, np.nan, dtype=np.float64
    )
    palette_shift_lab = np.full(
        frame_count, np.nan, dtype=np.float64
    )
    previous_available_frame: np.ndarray | None = None
    for frame_index, frame in enumerate(frames):
        if not availability[frame_index]:
            continue
        try:
            if apparatus_anchor is None:
                center, radius, interior, green = _disk_geometry(
                    np.asarray(frame), config=config
                )
                apparatus_diagnostics: Mapping[str, float] = {}
            else:
                (
                    center,
                    radius,
                    interior,
                    green,
                    apparatus_diagnostics,
                ) = _frozen_apparatus_geometry(
                    np.asarray(frame),
                    apparatus=apparatus_anchor,
                    previous_frame=previous_available_frame,
                    config=config,
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
        if apparatus_diagnostics:
            apparatus_consistency[frame_index] = float(
                apparatus_diagnostics[
                    "apparatus_condition_consistency"
                ]
            )
            temporal_override_fraction[frame_index] = float(
                apparatus_diagnostics[
                    "apparatus_temporal_override_fraction"
                ]
            )
            palette_shift_lab[frame_index] = float(
                apparatus_diagnostics["apparatus_palette_shift_lab"]
            )
        previous_available_frame = np.asarray(frame)
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
        coordinate_policy=str(
            config.get(
                "coordinate_phase_policy",
                "per_track_relative_phase_v1",
            )
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
            "backend": (
                "adaptive_hue_disk_open_world_components_v2_1"
                if config.get(
                    "disk_colour_policy", "legacy_green_hsv"
                )
                == "adaptive_dominant_hue"
                else "green_disk_open_world_components_v2"
            ),
            "disk_valid_frame_ratio": float(np.mean(disk_valid)),
            "candidate_counts": candidate_counts.tolist(),
            "candidate_safety_cap": int(
                config.get("open_world_maximum_candidates_per_frame", 64)
            ),
            "frame_failures": frame_failures,
            "apparatus_policy": (
                "condition_frozen_lab_value_temporal_apparatus"
                if apparatus_anchor is not None
                else (
                    "adaptive_hue_disk_removed_then_eroded_interior"
                    if config.get(
                        "disk_colour_policy", "legacy_green_hsv"
                    )
                    == "adaptive_dominant_hue"
                    else "green_disk_removed_then_eroded_interior"
                )
            ),
            "apparatus_coordinate_policy": (
                "per_frame_prediction_geometry_v1"
                if apparatus_anchor is None
                else "condition_frozen_shared_frame_v1"
            ),
            "apparatus_anchor_source": (
                None
                if apparatus_anchor is None
                else apparatus_anchor.source
            ),
            "apparatus_condition_consistency": [
                None if not math.isfinite(value) else float(value)
                for value in apparatus_consistency.tolist()
            ],
            "apparatus_temporal_override_fraction": [
                None if not math.isfinite(value) else float(value)
                for value in temporal_override_fraction.tolist()
            ],
            "apparatus_palette_shift_lab": [
                None if not math.isfinite(value) else float(value)
                for value in palette_shift_lab.tolist()
            ],
            "coordinate_phase_policy": str(
                config.get(
                    "coordinate_phase_policy",
                    "per_track_relative_phase_v1",
                )
            ),
            "legacy_apparatus_colour_policy": (
                "adaptive_hue"
                if config.get(
                    "disk_colour_policy", "legacy_green_hsv"
                )
                == "adaptive_dominant_hue"
                else "green_hsv"
            ),
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
    coordinate_policy: str = "per_track_relative_phase_v1",
) -> OpenWorldObservation:
    if not 0.0 <= center_fallback_radius < 1.0:
        raise ValueError("center_fallback_radius must be in [0, 1)")
    if coordinate_policy not in {
        "per_track_relative_phase_v1",
        "condition_frozen_absolute_phase_v1",
    }:
        raise ValueError(
            "unsupported circular coordinate phase policy: "
            f"{coordinate_policy!r}"
        )
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
            if coordinate_policy == "condition_frozen_absolute_phase_v1":
                canonical = (
                    normalized[index] * _CANONICAL_DISK_RADIUS
                )
                mode = (
                    "condition_frozen_cartesian_center"
                    if radius <= center_fallback_radius
                    else "condition_frozen_absolute_polar"
                )
                relative_phase = float(phases[index] - phase_anchor)
            elif radius <= center_fallback_radius:
                canonical = normalized[index] * _CANONICAL_DISK_RADIUS
                mode = "cartesian_center_fallback"
                relative_phase = float(phases[index] - phase_anchor)
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
                "relative_phase_rad": relative_phase,
                "distance_mode": mode,
                "coordinate_phase_policy": coordinate_policy,
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
                (
                    "condition_frozen_disk_absolute_polar_with_"
                    "cartesian_center_fallback"
                )
                if coordinate_policy
                == "condition_frozen_absolute_phase_v1"
                else (
                    "disk_normalized_relative_polar_with_"
                    "cartesian_center_fallback"
                )
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


def _relative_dynamics_tracks(
    tracks: InstanceTracks,
) -> InstanceTracks:
    """Remove only the environment-specific initial phase for parent physics."""

    xy = np.asarray(tracks.xy, dtype=np.float64).copy()
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError("circular instance tracks must use (T, N, 2) xy")
    for object_index in range(xy.shape[1]):
        valid = (
            np.asarray(tracks.valid[:, object_index], dtype=bool)
            & np.isfinite(xy[:, object_index]).all(axis=1)
        )
        indices = np.flatnonzero(valid)
        if len(indices) < 1:
            continue
        first = xy[int(indices[0]), object_index]
        angle = math.atan2(float(first[1]), float(first[0]))
        cosine = math.cos(-angle)
        sine = math.sin(-angle)
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
        finite = np.isfinite(xy[:, object_index]).all(axis=1)
        xy[finite, object_index] = (
            xy[finite, object_index] @ rotation.T
        )
    return InstanceTracks(
        xy=xy,
        valid=np.asarray(tracks.valid, dtype=bool),
        valid_ratio=np.asarray(tracks.valid_ratio, dtype=np.float64),
        instance_masks=tracks.instance_masks,
        union_masks=tracks.union_masks,
    )


def _phase_aligned_xy(
    xy: np.ndarray,
    *,
    anchor_xy: np.ndarray,
) -> np.ndarray:
    value = np.asarray(xy, dtype=np.float64)
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    if (
        value.shape != (2,)
        or anchor.shape != (2,)
        or not np.isfinite(value).all()
        or not np.isfinite(anchor).all()
    ):
        return value
    angle = math.atan2(float(anchor[1]), float(anchor[0]))
    cosine = math.cos(-angle)
    sine = math.sin(-angle)
    return np.asarray(
        [
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1],
        ],
        dtype=np.float64,
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

    parent_relative = (
        str(config.get("reference_mode", "same_case_reference"))
        == "parent_physics_reference"
        and str(
            config.get(
                "physics_parent_phase_policy",
                "legacy_per_track_relative_observation_v1",
            )
        )
        == "separate_condition_relative_dynamics_v1"
    )
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
    reference_phase_anchors = {
        entity_id: track.xy[
            int(np.flatnonzero(track.observed)[0])
        ]
        for entity_id, track in entity_lookup.items()
        if np.any(track.observed)
    }
    prediction_phase_anchors = {
        track_id: track.xy[
            int(np.flatnonzero(track.observed)[0])
        ]
        for track_id, track in prediction_lookup.items()
        if np.any(track.observed)
    }
    polar_rows = []
    for match in matches:
        reference = entity_lookup[match.entity_id]
        prediction = prediction_lookup[match.track_id]
        reference_xy = reference.xy[match.frame_index]
        prediction_xy = prediction.xy[match.frame_index]
        if parent_relative:
            reference_xy = _phase_aligned_xy(
                reference_xy,
                anchor_xy=reference_phase_anchors[match.entity_id],
            )
            prediction_xy = _phase_aligned_xy(
                prediction_xy,
                anchor_xy=prediction_phase_anchors[match.track_id],
            )
        value = polar_position_similarity(
            reference_xy,
            prediction_xy,
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
            raw_reference_tracks = (
                _relative_dynamics_tracks(reference_tracks)
                if parent_relative
                else reference_tracks
            )
            raw_prediction_tracks = (
                _relative_dynamics_tracks(prediction_tracks)
                if parent_relative
                else prediction_tracks
            )
            reference_orbits = extract_orbit_traces(
                raw_reference_tracks, list(times_s)
            )
            prediction_orbits = extract_orbit_traces(
                raw_prediction_tracks, list(times_s)
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
        "phase_policy": (
            "separate_condition_relative_dynamics_v1"
            if parent_relative
            else "condition_frozen_absolute_phase_v1"
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
    "FrozenCircularApparatus",
    "FrozenCircularReference",
    "empty_open_world_observation",
    "freeze_circular_apparatus",
    "freeze_reference_identities",
    "freeze_prediction_identity",
    "identity_anchors_from_observation",
    "matched_prediction_inputs",
    "observe_circular_objects",
    "polar_position_similarity",
    "reference_instance_tracks",
    "score_open_world_orbits",
]
