from __future__ import annotations

from dataclasses import dataclass
from math import ceil
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


def _visual_sort_key(candidate: AnchorCandidate, scene_id: str) -> tuple[float, ...]:
    x, y = candidate.centroid_xy
    if scene_id.startswith("circular"):
        return (y, x)
    return (x, y)


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
    provisional.sort(key=lambda item: _visual_sort_key(item, scene_id))
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
