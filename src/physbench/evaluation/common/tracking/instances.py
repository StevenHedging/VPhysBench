from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from ..errors import SceneAnalysisError
from ..masks.quality import component_centroids
from .centroid import interpolate_trace


@dataclass(frozen=True)
class InstanceTracks:
    xy: np.ndarray
    valid: np.ndarray
    valid_ratio: np.ndarray
    instance_masks: list[list[np.ndarray]]
    union_masks: list[np.ndarray]


def _best_assignment(
    previous: np.ndarray,
    candidates: list[tuple[np.ndarray, int, np.ndarray]],
) -> tuple[int, ...] | None:
    count = len(previous)
    if len(candidates) < count:
        return None
    best_cost = float("inf")
    best: tuple[int, ...] | None = None
    candidate_xy = np.asarray([item[0] for item in candidates])
    for selected in itertools.permutations(range(len(candidates)), count):
        cost = float(
            np.linalg.norm(previous - candidate_xy[list(selected)], axis=1).sum()
        )
        if cost < best_cost:
            best_cost = cost
            best = selected
    return best


def extract_instance_tracks(
    masks: list[np.ndarray],
    *,
    expected_count: int,
    minimum_component_area: int,
    maximum_component_area_ratio: float,
    minimum_valid_ratio: float,
    maximum_candidates: int = 8,
) -> InstanceTracks:
    if not masks or expected_count < 1:
        raise ValueError("non-empty masks and a positive object count are required")
    height, width = masks[0].shape
    maximum_area = int(height * width * maximum_component_area_ratio)
    candidates_by_frame = []
    for mask in masks:
        candidates = [
            item
            for item in component_centroids(
                mask,
                minimum_area=minimum_component_area,
                maximum_components=maximum_candidates,
            )
            if item[1] <= maximum_area
        ]
        candidates_by_frame.append(candidates)

    frame_count = len(masks)
    xy = np.full((frame_count, expected_count, 2), np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, expected_count), dtype=bool)
    instance_masks = [
        [np.zeros((height, width), dtype=np.uint8) for _ in range(frame_count)]
        for _ in range(expected_count)
    ]
    previous: np.ndarray | None = None
    for frame_index, candidates in enumerate(candidates_by_frame):
        if len(candidates) < expected_count:
            continue
        if previous is None:
            chosen = list(range(expected_count))
            chosen.sort(key=lambda index: candidates[index][0][0])
        else:
            assignment = _best_assignment(previous, candidates)
            if assignment is None:
                continue
            chosen = list(assignment)
        for object_index, candidate_index in enumerate(chosen):
            centroid, _, component = candidates[candidate_index]
            xy[frame_index, object_index] = centroid
            valid[frame_index, object_index] = True
            instance_masks[object_index][frame_index] = component
        previous = xy[frame_index].copy()

    ratios = np.mean(valid, axis=0)
    if np.any(ratios < minimum_valid_ratio):
        raise SceneAnalysisError(
            "insufficient_instance_tracks",
            "valid instance-track ratios "
            f"{ratios.tolist()} below required {minimum_valid_ratio:g}",
        )
    for object_index in range(expected_count):
        xy[:, object_index] = interpolate_trace(
            xy[:, object_index], valid[:, object_index]
        )
    union_masks: list[np.ndarray] = []
    for frame_index in range(frame_count):
        union = np.zeros((height, width), dtype=np.uint8)
        for object_index in range(expected_count):
            union = cv2.bitwise_or(
                union, instance_masks[object_index][frame_index]
            )
        union_masks.append(union)
    return InstanceTracks(
        xy=xy,
        valid=valid,
        valid_ratio=ratios,
        instance_masks=instance_masks,
        union_masks=union_masks,
    )
