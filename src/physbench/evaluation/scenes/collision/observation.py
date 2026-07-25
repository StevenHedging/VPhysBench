from __future__ import annotations

import itertools
from typing import Any

import cv2
import numpy as np

from ...common.errors import SceneAnalysisError
from ...common.masks.quality import component_centroids, mask_centroid
from ...common.masks.sam2 import MaskPrompt


def _expanded_prompt(
    component: np.ndarray,
    centroid: np.ndarray,
    *,
    expand: float,
    minimum_side: int,
    metadata: dict[str, Any],
) -> MaskPrompt:
    ys, xs = np.where(component > 0)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    width = max(float(minimum_side), (x1 - x0 + 1.0) * expand)
    height = max(float(minimum_side), (y1 - y0 + 1.0) * expand)
    frame_height, frame_width = component.shape
    box = np.asarray(
        [
            max(0.0, center_x - width / 2.0),
            max(0.0, center_y - height / 2.0),
            min(frame_width - 1.0, center_x + width / 2.0),
            min(frame_height - 1.0, center_y + height / 2.0),
        ],
        dtype=np.float32,
    )
    return MaskPrompt(
        frame_index=0,
        box_xyxy=box,
        points_xy=np.asarray(centroid, dtype=np.float32).reshape(1, 2),
        point_labels=np.ones(1, dtype=np.int32),
        metadata=metadata,
    )


def build_collision_prompts(
    frame: np.ndarray, *, config: dict[str, Any]
) -> tuple[list[MaskPrompt], dict[str, Any]]:
    """Find the entering striker and the closest stationary target pair."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (
        (saturation >= int(config["minimum_saturation"]))
        & (value >= int(config["minimum_value"]))
    ).astype(np.uint8) * 255
    top = int(round(height * float(config["track_top_ratio"])))
    bottom = int(round(height * float(config["track_bottom_ratio"])))
    mask[:top] = 0
    mask[bottom:] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidates = [
        item
        for item in component_centroids(
            mask,
            minimum_area=int(config["minimum_component_area"]),
            maximum_components=int(config["maximum_candidates"]),
        )
        if item[1] <= int(config["maximum_component_area"])
    ]
    expanded_candidates = []
    for centroid, area, component in candidates:
        ys, xs = np.where(component > 0)
        box_width = int(xs.max() - xs.min() + 1)
        box_height = int(ys.max() - ys.min() + 1)
        if (
            area >= int(config["split_merged_targets_minimum_area"])
            and box_width
            / max(float(box_height), 1.0)
            >= float(config["split_merged_targets_aspect_ratio"])
        ):
            midpoint = (int(xs.min()) + int(xs.max())) // 2
            left = component.copy()
            left[:, midpoint + 1 :] = 0
            right = component.copy()
            right[:, : midpoint + 1] = 0
            halves = [left, right]
            if all(
                np.count_nonzero(half)
                >= int(config["minimum_component_area"])
                and mask_centroid(half) is not None
                for half in halves
            ):
                expanded_candidates.extend(
                    [
                        (
                            mask_centroid(half),
                            int(np.count_nonzero(half)),
                            half,
                        )
                        for half in halves
                    ]
                )
                continue
        expanded_candidates.append((centroid, area, component))
    candidates = expanded_candidates
    if len(candidates) < 3:
        raise SceneAnalysisError(
            "collision_frame_zero_objects_missing",
            f"found {len(candidates)} ball candidates in frame zero; expected three",
        )
    minimum_pair = float(config["minimum_target_separation_px"])
    maximum_pair = float(config["maximum_target_separation_px"])
    maximum_y_difference = float(config["maximum_target_y_difference_px"])
    viable_pairs = []
    for first, second in itertools.combinations(range(len(candidates)), 2):
        xy1, area1, _ = candidates[first]
        xy2, area2, _ = candidates[second]
        separation = abs(float(xy1[0] - xy2[0]))
        y_difference = abs(float(xy1[1] - xy2[1]))
        if (
            minimum_pair <= separation <= maximum_pair
            and y_difference <= maximum_y_difference
        ):
            viable_pairs.append(
                (
                    separation
                    - float(config["target_area_reward"]) * (area1 + area2),
                    first,
                    second,
                )
            )
    if not viable_pairs:
        raise SceneAnalysisError(
            "collision_target_pair_missing",
            "could not identify the adjacent stationary target pair",
        )
    _, first, second = min(viable_pairs)
    target_indices = sorted(
        [first, second], key=lambda index: candidates[index][0][0]
    )
    target_left_x = float(candidates[target_indices[0]][0][0])
    target_y = float(
        np.mean([candidates[index][0][1] for index in target_indices])
    )
    striker_candidates = [
        index
        for index, (xy, _, _) in enumerate(candidates)
        if index not in target_indices
        and float(xy[0]) < target_left_x - minimum_pair
        and abs(float(xy[1]) - target_y)
        <= float(config["maximum_striker_y_difference_px"])
    ]
    if not striker_candidates:
        raise SceneAnalysisError(
            "collision_striker_missing",
            "could not identify an entering striker left of the targets",
        )
    striker_index = min(
        striker_candidates, key=lambda index: candidates[index][0][0]
    )
    ordered = [striker_index, *target_indices]
    prompts = [
        _expanded_prompt(
            candidates[index][2],
            candidates[index][0],
            expand=float(config["box_expand"]),
            minimum_side=int(config["minimum_box_side"]),
            metadata={
                "role": role,
                "component_area": candidates[index][1],
                "source": "frame_zero_track_band_color_component",
            },
        )
        for index, role in zip(
            ordered, ["striker", "target_1", "target_2"]
        )
    ]
    return prompts, {
        "backend": "track_band_color_components_plus_sam2",
        "track_band_y": [top, bottom],
        "candidate_count": len(candidates),
        "selected_centroids_xy": [
            candidates[index][0].tolist() for index in ordered
        ],
        "selected_component_areas": [
            candidates[index][1] for index in ordered
        ],
    }
