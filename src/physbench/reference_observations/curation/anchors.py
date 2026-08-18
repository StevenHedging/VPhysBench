from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import median
from typing import Mapping, Sequence

import cv2
import numpy as np


def _binary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("anchor masks must be two-dimensional")
    return (value > 0).astype(np.uint8)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a = _binary(left).astype(bool)
    b = _binary(right).astype(bool)
    if a.shape != b.shape:
        raise ValueError("comparison mask shape does not match anchor candidate")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


@dataclass(frozen=True)
class AnchorCandidate:
    object_id: str
    mask: np.ndarray
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    area_pixels: int
    comparison_iou: float | None = None
    persistence_score: float | None = None

    @classmethod
    def from_mask(
        cls,
        object_id: str,
        mask: np.ndarray,
        *,
        comparison_mask: np.ndarray | None = None,
        persistence_score: float | None = None,
    ) -> "AnchorCandidate":
        binary = _binary(mask)
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            raise ValueError(f"anchor candidate {object_id} is empty")
        comparison_iou = (
            None if comparison_mask is None else _mask_iou(binary, comparison_mask)
        )
        return cls(
            object_id=object_id,
            mask=binary,
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            bbox_xyxy=(
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            ),
            area_pixels=int(binary.sum()),
            comparison_iou=comparison_iou,
            persistence_score=persistence_score,
        )


def _validate_frames(frames: Sequence[np.ndarray]) -> tuple[int, int]:
    if len(frames) < 2:
        raise ValueError("independent anchor candidates require at least two frames")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("independent anchor candidates require HWC color frames")
    height, width = first.shape[:2]
    if any(np.asarray(frame).shape != first.shape for frame in frames):
        raise ValueError("independent anchor candidate frames must have equal shapes")
    return height, width


def _motion_components(frames: Sequence[np.ndarray]) -> list[tuple[np.ndarray, float]]:
    height, width = _validate_frames(frames)
    first = cv2.cvtColor(np.asarray(frames[0]), cv2.COLOR_BGR2GRAY)
    changed = []
    for frame in frames[1:]:
        gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2GRAY)
        changed.append(cv2.absdiff(first, gray) >= 20)
    persistence = np.stack(changed).sum(axis=0)
    required = max(1, ceil(0.6 * len(changed)))
    foreground = (persistence >= required).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    minimum_area = max(9, int(round(height * width * 0.0001)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    components: list[tuple[np.ndarray, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component = (labels == label).astype(np.uint8)
        score = float(persistence[component.astype(bool)].mean() / len(changed))
        components.append((component, score))
    return components


def _circular_geometry_components(frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturated = (
        (hsv[:, :, 0] >= 80)
        & (hsv[:, :, 0] <= 115)
        & (hsv[:, :, 1] >= 55)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(saturated, 8)
    if count <= 1:
        return []
    disk_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = (labels == disk_label).astype(np.uint8)
    disk_contours, _ = cv2.findContours(
        disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not disk_contours:
        return []
    disk = np.zeros_like(disk)
    disk_hull = cv2.convexHull(max(disk_contours, key=cv2.contourArea))
    cv2.drawContours(
        disk,
        [disk_hull],
        -1,
        1,
        thickness=cv2.FILLED,
    )
    scale = max(3, int(round(min(height, width) * 0.015)))
    kernel = np.ones((scale | 1, scale | 1), np.uint8)
    disk = cv2.morphologyEx(disk, cv2.MORPH_CLOSE, kernel)
    interior = cv2.erode(disk, kernel)
    channel_edges = [cv2.Canny(frame[:, :, channel], 20, 70) for channel in range(3)]
    edges = cv2.dilate(
        np.maximum.reduce(channel_edges), np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = height * width
    proposals: list[tuple[tuple[int, int, int, int], float]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        contour_area = float(cv2.contourArea(contour))
        center_x = min(width - 1, x + box_width // 2)
        center_y = min(height - 1, y + box_height // 2)
        aspect = box_width / max(1, box_height)
        rectangularity = contour_area / max(1, box_area)
        if not (
            0.0015 * image_area <= box_area <= 0.08 * image_area
            and 0.2 <= aspect <= 5.0
            and rectangularity >= 0.02
            and interior[center_y, center_x]
        ):
            continue
        score = float(box_area * (0.2 + rectangularity))
        proposals.append(((x, y, box_width, box_height), score))
    proposals.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[tuple[int, int, int, int], float]] = []
    for box, score in proposals:
        x, y, box_width, box_height = box
        box_area = box_width * box_height
        box_mask = np.zeros((height, width), np.uint8)
        box_mask[y : y + box_height, x : x + box_width] = 1
        duplicate = False
        for selected_box, _ in selected:
            sx, sy, sw, sh = selected_box
            intersection = max(0, min(x + box_width, sx + sw) - max(x, sx)) * max(
                0, min(y + box_height, sy + sh) - max(y, sy)
            )
            union = box_area + sw * sh - intersection
            if union and intersection / union >= 0.45:
                duplicate = True
                break
        if not duplicate:
            selected.append((box, score))
    result = []
    for (x, y, box_width, box_height), score in selected:
        mask = np.zeros((height, width), np.uint8)
        mask[y : y + box_height, x : x + box_width] = 1
        result.append((mask, score / image_area))
    return result


def _visual_order(
    candidates: list[AnchorCandidate], scene_id: str
) -> list[AnchorCandidate]:
    if scene_id != "uniform_circular_motion":
        return sorted(candidates, key=lambda item: item.centroid_xy)
    tolerance = max(
        1.0,
        0.5
        * median(item.bbox_xyxy[3] - item.bbox_xyxy[1] for item in candidates),
    )
    rows: list[list[AnchorCandidate]] = []
    centers: list[float] = []
    for candidate in sorted(candidates, key=lambda item: item.centroid_xy[1]):
        y = candidate.centroid_xy[1]
        if not rows or abs(y - centers[-1]) > tolerance:
            rows.append([candidate])
            centers.append(y)
        else:
            rows[-1].append(candidate)
            centers[-1] = sum(item.centroid_xy[1] for item in rows[-1]) / len(rows[-1])
    return [item for row in rows for item in sorted(row, key=lambda value: value.centroid_xy[0])]


def build_independent_anchor_candidates(
    frames: Sequence[np.ndarray],
    *,
    scene_id: str,
    expected_count: int,
    comparison_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[AnchorCandidate, ...]:
    """Infer first-frame subjects without consulting existing dataset masks.

    Existing masks are deliberately read only after candidate extraction and
    ordering; they contribute diagnostic IoU values, never pixels or ranking.
    """
    if expected_count <= 0:
        raise ValueError("independent anchor candidates require a positive count")
    if scene_id == "uniform_circular_motion":
        _validate_frames(frames)
        components = _circular_geometry_components(np.asarray(frames[0]))
    else:
        components = _motion_components(frames)
    if len(components) < expected_count:
        raise ValueError(
            "independent anchor candidates could not explain the expected "
            f"subject count: found {len(components)}, expected {expected_count}"
        )
    selected = sorted(
        components,
        key=lambda item: (item[1], int(item[0].sum())),
        reverse=True,
    )[:expected_count]
    provisional = [
        AnchorCandidate.from_mask("unassigned", mask, persistence_score=score)
        for mask, score in selected
    ]
    provisional = _visual_order(provisional, scene_id)
    result = []
    comparisons = comparison_masks or {}
    for index, candidate in enumerate(provisional, start=1):
        object_id = f"object_{index}"
        result.append(
            AnchorCandidate.from_mask(
                object_id,
                candidate.mask,
                comparison_mask=comparisons.get(object_id),
                persistence_score=candidate.persistence_score,
            )
        )
    return tuple(result)
