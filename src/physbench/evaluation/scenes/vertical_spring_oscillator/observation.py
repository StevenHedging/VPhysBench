"""Condition-causal ball identity and spring-topology observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ...common.frozen_subject import FrozenSubjectAnchor
from ...common.masks.sam2 import MaskPrompt


_PROMPT_EXPANSION_RATIO = 1.50


class _ImmutableDict(dict[str, Any]):
    """A JSON-compatible dict whose published audit values cannot drift."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("observation audit mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


@dataclass(frozen=True)
class SpringTopology:
    row_coverage: np.ndarray
    endpoint_support: np.ndarray
    valid: np.ndarray
    score: float


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.flags.writeable = False
    return result


def _finite_config(
    config: Mapping[str, Any], name: str, *, minimum: float | None = None
) -> float:
    try:
        value = float(config[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"config requires a finite {name!r}") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        qualifier = f" at least {minimum}" if minimum is not None else ""
        raise ValueError(f"config {name!r} must be finite{qualifier}")
    return value


def _unit_interval_config(config: Mapping[str, Any], name: str) -> float:
    value = _finite_config(config, name, minimum=0.0)
    if value > 1.0:
        raise ValueError(f"config {name!r} must be in [0, 1]")
    return value


def _binary_mask_or_none(
    mask: object, *, expected_shape: tuple[int, int] | None = None
) -> np.ndarray | None:
    try:
        values = np.asarray(mask)
    except Exception:
        return None
    if (
        values.ndim != 2
        or values.size == 0
        or (expected_shape is not None and values.shape != expected_shape)
        or not (
            np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        )
    ):
        return None
    try:
        if not np.all(np.isfinite(values)):
            return None
    except TypeError:
        return None
    return np.where(values > 0, 255, 0).astype(np.uint8)


def _required_anchor_mask(anchor: FrozenSubjectAnchor) -> np.ndarray:
    mask = _binary_mask_or_none(anchor.mask)
    if mask is None or not np.any(mask):
        raise ValueError(
            "anchor mask must be finite, non-empty, and two-dimensional"
        )
    centroid = np.asarray(anchor.centroid_xy, dtype=np.float64)
    if centroid.shape != (2,) or not np.all(np.isfinite(centroid)):
        raise ValueError("anchor centroid must contain two finite coordinates")
    if (
        not math.isfinite(float(anchor.area_px2))
        or float(anchor.area_px2) <= 0.0
        or not math.isfinite(float(anchor.equivalent_radius_px))
        or float(anchor.equivalent_radius_px) <= 0.0
    ):
        raise ValueError("anchor geometry must be finite and positive")
    return mask


def prompt_from_anchor(anchor: FrozenSubjectAnchor) -> MaskPrompt:
    """Build one frame-zero prompt solely from the frozen Dataset anchor."""
    mask = _required_anchor_mask(anchor)
    height, width = mask.shape
    x, y, box_width, box_height = cv2.boundingRect(mask)
    center_x = x + (box_width - 1.0) / 2.0
    center_y = y + (box_height - 1.0) / 2.0
    expanded_width = box_width * _PROMPT_EXPANSION_RATIO
    expanded_height = box_height * _PROMPT_EXPANSION_RATIO
    box = np.asarray(
        [
            max(0.0, center_x - expanded_width / 2.0),
            max(0.0, center_y - expanded_height / 2.0),
            min(width - 1.0, center_x + expanded_width / 2.0),
            min(height - 1.0, center_y + expanded_height / 2.0),
        ],
        dtype=np.float32,
    )
    points = np.asarray(anchor.centroid_xy, dtype=np.float32).reshape(1, 2)
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
    return MaskPrompt(
        frame_index=0,
        box_xyxy=_readonly(box),
        points_xy=_readonly(points),
        point_labels=_readonly(np.ones(1, dtype=np.int32)),
        metadata=_ImmutableDict(
            {
                "source": "frozen_dataset_subject_anchor",
                "logical_entity_id": str(anchor.logical_entity_id),
                "dataset_object_id": str(anchor.dataset_object_id),
                "box_expansion_ratio": _PROMPT_EXPANSION_RATIO,
            }
        ),
    )


def _mask_quality_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    thresholds = {
        "minimum_mask_pixels": _finite_config(
            config, "minimum_mask_pixels", minimum=0.0
        ),
        "maximum_mask_area_ratio": _unit_interval_config(
            config, "maximum_mask_area_ratio"
        ),
        "minimum_anchor_area_ratio": _finite_config(
            config, "minimum_anchor_area_ratio", minimum=0.0
        ),
        "maximum_anchor_area_ratio": _finite_config(
            config, "maximum_anchor_area_ratio", minimum=0.0
        ),
    }
    if thresholds["minimum_anchor_area_ratio"] > thresholds[
        "maximum_anchor_area_ratio"
    ]:
        raise ValueError(
            "minimum_anchor_area_ratio exceeds maximum_anchor_area_ratio"
        )
    return thresholds


def validate_mask_tube(
    masks: Sequence[np.ndarray],
    *,
    availability: Sequence[bool],
    anchor: FrozenSubjectAnchor,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, ...]:
    """Normalize a ball tube and blank each frame that fails causal quality gates."""
    if len(masks) != len(availability):
        raise ValueError("mask and availability counts differ")
    anchor_mask = _required_anchor_mask(anchor)
    thresholds = _mask_quality_thresholds(config)
    frame_area = float(anchor_mask.size)
    anchor_area = float(np.count_nonzero(anchor_mask))
    output: list[np.ndarray] = []
    for available, candidate in zip(availability, masks):
        normalized = (
            _binary_mask_or_none(candidate, expected_shape=anchor_mask.shape)
            if bool(available)
            else None
        )
        if normalized is not None:
            area = float(np.count_nonzero(normalized))
            anchor_ratio = area / anchor_area
            valid = (
                area >= thresholds["minimum_mask_pixels"]
                and area / frame_area <= thresholds["maximum_mask_area_ratio"]
                and anchor_ratio >= thresholds["minimum_anchor_area_ratio"]
                and anchor_ratio <= thresholds["maximum_anchor_area_ratio"]
            )
        else:
            valid = False
        if not valid:
            normalized = np.zeros(anchor_mask.shape, dtype=np.uint8)
        output.append(_readonly(normalized))
    return tuple(output)


def _identity_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    thresholds = {
        "minimum_anchor_iou": _unit_interval_config(
            config, "minimum_anchor_iou"
        ),
        "maximum_centroid_distance_radii": _finite_config(
            config, "maximum_centroid_distance_radii", minimum=0.0
        ),
        "minimum_anchor_area_ratio": _finite_config(
            config, "minimum_anchor_area_ratio", minimum=0.0
        ),
        "maximum_anchor_area_ratio": _finite_config(
            config, "maximum_anchor_area_ratio", minimum=0.0
        ),
    }
    if thresholds["minimum_anchor_area_ratio"] > thresholds[
        "maximum_anchor_area_ratio"
    ]:
        raise ValueError(
            "minimum_anchor_area_ratio exceeds maximum_anchor_area_ratio"
        )
    return thresholds


def validate_prediction_identity(
    *,
    anchor_mask: np.ndarray,
    prediction_mask: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit frame-zero prediction identity against only the frozen anchor mask."""
    thresholds = _identity_thresholds(config)
    anchor = _binary_mask_or_none(anchor_mask)
    if anchor is None or not np.any(anchor):
        raise ValueError(
            "anchor_mask must be finite, non-empty, and two-dimensional"
        )
    prediction = _binary_mask_or_none(
        prediction_mask, expected_shape=anchor.shape
    )
    if prediction is None:
        prediction = np.zeros_like(anchor)

    anchor_binary = anchor > 0
    prediction_binary = prediction > 0
    anchor_area = float(np.count_nonzero(anchor_binary))
    prediction_area = float(np.count_nonzero(prediction_binary))
    union = float(np.count_nonzero(anchor_binary | prediction_binary))
    intersection = float(np.count_nonzero(anchor_binary & prediction_binary))
    iou = intersection / union if union > 0.0 else 0.0
    area_ratio = prediction_area / anchor_area

    anchor_y, anchor_x = np.where(anchor_binary)
    anchor_centroid = np.asarray(
        [float(anchor_x.mean()), float(anchor_y.mean())], dtype=np.float64
    )
    equivalent_radius = math.sqrt(anchor_area / math.pi)
    if prediction_area > 0.0:
        prediction_y, prediction_x = np.where(prediction_binary)
        prediction_centroid = np.asarray(
            [float(prediction_x.mean()), float(prediction_y.mean())],
            dtype=np.float64,
        )
        distance_radii = float(
            np.linalg.norm(prediction_centroid - anchor_centroid)
            / equivalent_radius
        )
    else:
        maximum_distance = thresholds["maximum_centroid_distance_radii"]
        distance_radii = min(
            np.finfo(np.float64).max,
            maximum_distance + max(1.0, abs(maximum_distance)),
        )

    accepted = bool(
        prediction_area > 0.0
        and iou >= thresholds["minimum_anchor_iou"]
        and distance_radii
        <= thresholds["maximum_centroid_distance_radii"]
        and area_ratio >= thresholds["minimum_anchor_area_ratio"]
        and area_ratio <= thresholds["maximum_anchor_area_ratio"]
    )
    return _ImmutableDict(
        {
            "accepted": accepted,
            "anchor_iou": float(iou),
            "centroid_distance_radii": float(distance_radii),
            "area_ratio": float(area_ratio),
            "thresholds": _ImmutableDict(thresholds),
        }
    )


def _topology_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    thresholds = {
        "canny_low_threshold": _finite_config(
            config, "canny_low_threshold", minimum=0.0
        ),
        "canny_high_threshold": _finite_config(
            config, "canny_high_threshold", minimum=0.0
        ),
        "corridor_half_width_radius_ratio": _finite_config(
            config, "corridor_half_width_radius_ratio", minimum=0.0
        ),
        "minimum_edge_pixels_per_row": _finite_config(
            config, "minimum_edge_pixels_per_row", minimum=1.0
        ),
        "endpoint_height_radius_ratio": _finite_config(
            config, "endpoint_height_radius_ratio", minimum=0.0
        ),
    }
    if thresholds["canny_low_threshold"] > thresholds["canny_high_threshold"]:
        raise ValueError("canny_low_threshold exceeds canny_high_threshold")
    if thresholds["corridor_half_width_radius_ratio"] <= 0.0:
        raise ValueError("corridor_half_width_radius_ratio must be positive")
    if thresholds["endpoint_height_radius_ratio"] <= 0.0:
        raise ValueError("endpoint_height_radius_ratio must be positive")
    minimum_edges = thresholds["minimum_edge_pixels_per_row"]
    if not minimum_edges.is_integer():
        raise ValueError("minimum_edge_pixels_per_row must be an integer")
    return thresholds


def observe_spring_topology(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    availability: Sequence[bool],
    config: Mapping[str, Any],
) -> SpringTopology:
    """Measure spring edges in a corridor derived from each current ball mask."""
    if len(frames) != len(masks) or len(masks) != len(availability):
        raise ValueError("frame, mask, and availability counts differ")
    thresholds = _topology_thresholds(config)
    row_coverage = np.zeros(len(masks), dtype=np.float64)
    endpoint_support = np.zeros(len(masks), dtype=np.float64)
    valid = np.zeros(len(masks), dtype=bool)
    minimum_edges = int(thresholds["minimum_edge_pixels_per_row"])

    for index, (frame_value, mask_value, available) in enumerate(
        zip(frames, masks, availability)
    ):
        if not bool(available):
            continue
        frame = np.asarray(frame_value)
        mask = _binary_mask_or_none(mask_value)
        if (
            mask is None
            or not np.any(mask)
            or frame.ndim != 3
            or frame.shape[:2] != mask.shape
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
            or not np.all(np.isfinite(frame))
        ):
            continue

        ys, xs = np.where(mask > 0)
        center_x = float(xs.mean())
        equivalent_radius = math.sqrt(float(xs.size) / math.pi)
        half_width = (
            thresholds["corridor_half_width_radius_ratio"]
            * equivalent_radius
        )
        x0 = max(0, int(math.floor(center_x - half_width)))
        x1 = min(mask.shape[1], int(math.ceil(center_x + half_width)) + 1)
        spring_bottom = int(ys.min())
        if x1 <= x0 or spring_bottom <= 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            thresholds["canny_low_threshold"],
            thresholds["canny_high_threshold"],
        )
        ball_guard = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        edges[ball_guard > 0] = 0
        corridor = edges[:spring_bottom, x0:x1]
        if corridor.shape[0] == 0:
            continue
        supported_rows = np.count_nonzero(corridor, axis=1) >= minimum_edges
        row_coverage[index] = float(np.mean(supported_rows))
        endpoint_height = max(
            1,
            int(
                math.ceil(
                    thresholds["endpoint_height_radius_ratio"]
                    * equivalent_radius
                )
            ),
        )
        endpoint_start = max(0, spring_bottom - endpoint_height)
        endpoint = edges[endpoint_start:spring_bottom, x0:x1]
        endpoint_support[index] = float(
            np.any(np.count_nonzero(endpoint, axis=1) >= minimum_edges)
        )
        valid[index] = True

    if np.any(valid):
        per_frame = 0.5 * (row_coverage[valid] + endpoint_support[valid])
        score = float(np.mean(per_frame))
    else:
        score = 0.0
    return SpringTopology(
        row_coverage=_readonly(row_coverage),
        endpoint_support=_readonly(endpoint_support),
        valid=_readonly(valid),
        score=float(np.clip(score, 0.0, 1.0)),
    )
