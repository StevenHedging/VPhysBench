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


def _circle_prompt(
    circle: tuple[float, float, float],
    *,
    frame_index: int,
    frame_shape: tuple[int, ...],
    expand: float,
    minimum_side: int,
    metadata: dict[str, Any],
) -> MaskPrompt:
    center_x, center_y, radius = circle
    height, width = frame_shape[:2]
    side = max(float(minimum_side), 2.0 * radius * expand)
    box = np.asarray(
        [
            max(0.0, center_x - side / 2.0),
            max(0.0, center_y - side / 2.0),
            min(width - 1.0, center_x + side / 2.0),
            min(height - 1.0, center_y + side / 2.0),
        ],
        dtype=np.float32,
    )
    offset = 0.28 * radius
    points = np.asarray(
        [
            [center_x, center_y],
            [center_x - offset, center_y],
            [center_x + offset, center_y],
            [center_x, center_y - offset],
            [center_x, center_y + offset],
        ],
        dtype=np.float32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=box,
        points_xy=points,
        point_labels=np.ones(len(points), dtype=np.int32),
        metadata=metadata,
    )


def _nms_circles(
    circles: list[tuple[float, float, float]],
    *,
    minimum_center_distance: float,
    overlap_distance_factor: float,
    maximum_candidates: int,
) -> list[tuple[float, float, float]]:
    retained: list[tuple[float, float, float]] = []
    for candidate in sorted(circles, key=lambda value: value[2], reverse=True):
        if any(
            np.hypot(candidate[0] - other[0], candidate[1] - other[1])
            < max(
                minimum_center_distance,
                overlap_distance_factor * (candidate[2] + other[2]),
            )
            for other in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= maximum_candidates:
            break
    return sorted(retained, key=lambda value: value[0])


def _hough_candidates(
    frame: np.ndarray,
    *,
    config: dict[str, Any],
    accumulator_threshold: float,
) -> tuple[list[tuple[float, float, float]], tuple[int, int]]:
    height, _ = frame.shape[:2]
    top = int(round(height * float(config["track_top_ratio"])))
    bottom = int(round(height * float(config["track_bottom_ratio"])))
    gray = cv2.cvtColor(frame[top:bottom], cv2.COLOR_BGR2GRAY)
    blur_size = int(config.get("blur_kernel", 5))
    if blur_size % 2 == 0:
        blur_size += 1
    gray = cv2.GaussianBlur(
        gray,
        (blur_size, blur_size),
        float(config.get("blur_sigma", 1.2)),
    )
    detected = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(config.get("hough_dp", 1.2)),
        minDist=float(config["minimum_circle_center_distance_px"]),
        param1=float(config.get("edge_threshold", 100.0)),
        param2=float(accumulator_threshold),
        minRadius=int(config["minimum_circle_radius_px"]),
        maxRadius=int(config["maximum_circle_radius_px"]),
    )
    if detected is None:
        return [], (top, bottom)
    circles = [
        (float(x), float(y + top), float(radius))
        for x, y, radius in detected[0]
    ]
    return (
        _nms_circles(
            circles,
            minimum_center_distance=float(
                config["minimum_circle_center_distance_px"]
            ),
            overlap_distance_factor=float(
                config.get("circle_nms_overlap_distance_factor", 0.55)
            ),
            maximum_candidates=int(config["maximum_candidates"]),
        ),
        (top, bottom),
    )


def _select_role_triple(
    circles: list[tuple[float, float, float]],
    *,
    frame_width: int,
    config: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], float] | None:
    best: tuple[float, list[tuple[float, float, float]]] | None = None
    for combination in itertools.combinations(circles, 3):
        ordered = sorted(combination, key=lambda value: value[0])
        striker, target_1, target_2 = ordered
        target_gap = target_2[0] - target_1[0]
        striker_gap = target_1[0] - striker[0]
        target_radius_sum = target_1[2] + target_2[2]
        striker_radius_sum = striker[2] + target_1[2]
        if target_gap < max(
            float(config["minimum_target_separation_px"]),
            float(config["minimum_target_gap_radius_factor"])
            * target_radius_sum,
        ):
            continue
        if target_gap > min(
            float(config["maximum_target_separation_px"]),
            float(config["maximum_target_gap_radius_factor"])
            * target_radius_sum,
        ):
            continue
        if striker_gap < max(
            float(config["minimum_striker_separation_px"]),
            float(config["minimum_striker_gap_radius_factor"])
            * striker_radius_sum,
        ):
            continue
        y_values = np.asarray([value[1] for value in ordered])
        y_spread = float(np.ptp(y_values))
        if y_spread > float(config["maximum_role_y_spread_px"]):
            continue
        radii = np.asarray([value[2] for value in ordered])
        if float(np.max(radii) / max(np.min(radii), 1e-6)) > float(
            config["maximum_radius_ratio"]
        ):
            continue

        target_contact_error = abs(
            target_gap
            - float(config.get("preferred_target_gap_radius_factor", 1.1))
            * target_radius_sum
        ) / max(target_radius_sum, 1.0)
        separation_reward = min(
            striker_gap / max(4.0 * striker_radius_sum, 1.0), 1.0
        )
        visibility_reward = float(
            striker[0] - striker[2] >= 1.0
            and target_2[0] + target_2[2] <= frame_width - 2.0
        )
        score = (
            2.0 * separation_reward
            + 0.35 * visibility_reward
            - target_contact_error
            - y_spread / max(
                float(config["maximum_role_y_spread_px"]), 1.0
            )
        )
        if best is None or score > best[0]:
            best = (score, ordered)
    if best is None:
        return None
    return best[1], float(best[0])


def _seed_frame_indices(
    frame_count: int, *, config: dict[str, Any]
) -> list[int]:
    maximum_index = max(
        0,
        min(
            frame_count - 1,
            int(
                round(
                    (frame_count - 1)
                    * float(config.get("maximum_seed_fraction", 0.65))
                )
            ),
        ),
    )
    count = min(
        maximum_index + 1,
        int(config.get("maximum_seed_frames", 15)),
    )
    values = np.linspace(0, maximum_index, count)
    return sorted({int(round(value)) for value in values} | {0})


def _motion_candidates(
    frame: np.ndarray,
    background: np.ndarray,
    *,
    config: dict[str, Any],
) -> list[tuple[float, float, float]]:
    height, _ = frame.shape[:2]
    top = int(round(height * float(config["track_top_ratio"])))
    bottom = int(round(height * float(config["track_bottom_ratio"])))
    current = cv2.cvtColor(
        cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2GRAY
    )
    median = cv2.cvtColor(
        cv2.GaussianBlur(background, (5, 5), 0), cv2.COLOR_BGR2GRAY
    )
    delta = cv2.absdiff(current, median)
    binary = (
        delta >= int(config.get("motion_difference_threshold", 18))
    ).astype(np.uint8) * 255
    binary[:top] = 0
    binary[bottom:] = 0
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    circles = []
    for _, area, component in component_centroids(
        binary,
        minimum_area=int(config.get("motion_minimum_area", 20)),
        maximum_components=int(config["maximum_candidates"]),
    ):
        points = cv2.findNonZero((component > 0).astype(np.uint8))
        if points is None:
            continue
        (_, _), radius = cv2.minEnclosingCircle(points)
        equivalent_radius = float(np.sqrt(area / np.pi))
        radius = max(equivalent_radius, min(float(radius), 1.8 * equivalent_radius))
        if not (
            float(config["minimum_circle_radius_px"])
            <= radius
            <= float(config["maximum_circle_radius_px"])
        ):
            continue
        centroid = mask_centroid(component)
        if centroid is not None:
            circles.append(
                (float(centroid[0]), float(centroid[1]), float(radius))
            )
    return _nms_circles(
        circles,
        minimum_center_distance=float(
            config["minimum_circle_center_distance_px"]
        ),
        overlap_distance_factor=float(
            config.get("circle_nms_overlap_distance_factor", 0.55)
        ),
        maximum_candidates=int(config["maximum_candidates"]),
    )


def build_multiframe_collision_prompts(
    frames: list[np.ndarray], *, config: dict[str, Any]
) -> tuple[list[MaskPrompt], dict[str, Any]]:
    """Find three collision roles without assuming colorful frame-zero balls.

    Hough evidence is intentionally evaluated at several frames and several
    confidence thresholds.  The shared seed must contain a left-hand striker
    and an adjacent target pair on one horizontal track.  A temporal-median
    motion proposal is a last-resort fallback, not the primary detector.
    """
    if not frames:
        raise SceneAnalysisError(
            "collision_empty_observation", "collision video has no frames"
        )
    seed_indices = _seed_frame_indices(len(frames), config=config)
    thresholds = [
        float(value) for value in config["hough_accumulator_thresholds"]
    ]
    attempts: list[dict[str, Any]] = []
    viable: list[
        tuple[
            float,
            int,
            list[tuple[float, float, float]],
            str,
            float | None,
        ]
    ] = []
    track_band = (0, frames[0].shape[0])
    for frame_index in seed_indices:
        for threshold in thresholds:
            circles, track_band = _hough_candidates(
                frames[frame_index],
                config=config,
                accumulator_threshold=threshold,
            )
            selected = _select_role_triple(
                circles,
                frame_width=frames[frame_index].shape[1],
                config=config,
            )
            attempts.append(
                {
                    "frame": frame_index,
                    "source": "hough_circle",
                    "accumulator_threshold": threshold,
                    "candidate_count": len(circles),
                    "viable": selected is not None,
                }
            )
            if selected is not None:
                triple, geometry_score = selected
                # Prefer strong circle evidence, then complete subjects and an
                # early pre-contact frame.  Geometry still decides among
                # several plausible triples at one threshold.
                quality = (
                    threshold / max(thresholds)
                    + 0.2 * geometry_score
                    - 0.08
                    * frame_index
                    / max(len(frames) - 1, 1)
                )
                viable.append(
                    (
                        quality,
                        frame_index,
                        triple,
                        "multiframe_hough_circle",
                        threshold,
                    )
                )
                break

    if not viable:
        background = np.median(
            np.stack(frames, axis=0), axis=0
        ).astype(np.uint8)
        for frame_index in seed_indices:
            circles = _motion_candidates(
                frames[frame_index], background, config=config
            )
            selected = _select_role_triple(
                circles,
                frame_width=frames[frame_index].shape[1],
                config=config,
            )
            attempts.append(
                {
                    "frame": frame_index,
                    "source": "temporal_median_motion",
                    "candidate_count": len(circles),
                    "viable": selected is not None,
                }
            )
            if selected is not None:
                triple, geometry_score = selected
                viable.append(
                    (
                        0.2 * geometry_score,
                        frame_index,
                        triple,
                        "temporal_median_motion",
                        None,
                    )
                )

    if not viable:
        raise SceneAnalysisError(
            "collision_multiframe_objects_missing",
            "no sampled frame contains a reliable striker and target pair",
        )
    quality, frame_index, ordered, source, threshold = max(
        viable, key=lambda value: value[0]
    )
    roles = ["striker", "target_1", "target_2"]
    prompts = [
        _circle_prompt(
            circle,
            frame_index=frame_index,
            frame_shape=frames[frame_index].shape,
            expand=float(config["box_expand"]),
            minimum_side=int(config["minimum_box_side"]),
            metadata={
                "role": role,
                "source": source,
                "circle_xyr": list(circle),
                "hough_accumulator_threshold": threshold,
            },
        )
        for circle, role in zip(ordered, roles)
    ]
    return prompts, {
        "backend": "multiframe_circle_motion_proposals_plus_bidirectional_sam2",
        "seed_frame": frame_index,
        "seed_source": source,
        "seed_quality": float(quality),
        "track_band_y": list(track_band),
        "selected_circles_xyr": [list(value) for value in ordered],
        "selected_roles": roles,
        "candidate_attempts": attempts,
    }


def stabilize_collision_instance_masks(
    masks: list[list[np.ndarray]],
    prompts: list[MaskPrompt],
    *,
    config: dict[str, Any],
) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """Keep one temporally continuous component for every semantic role."""
    if len(masks) != len(prompts):
        raise ValueError("collision masks and prompts must have equal counts")
    if not masks:
        return masks, {"policy": "empty"}
    frame_area = masks[0][0].shape[0] * masks[0][0].shape[1]
    minimum_area = int(config["minimum_mask_pixels"])
    maximum_area = int(
        round(frame_area * float(config["maximum_mask_area_ratio"]))
    )
    maximum_jump = float(config.get("maximum_centroid_jump_px", 120.0))
    output = [
        [np.zeros_like(mask) for mask in instance] for instance in masks
    ]
    kept_counts: list[int] = []
    rejected_counts: list[int] = []

    for object_index, (instance, prompt) in enumerate(zip(masks, prompts)):
        seed = int(prompt.frame_index)
        seed_anchor = np.asarray(
            prompt.points_xy[0], dtype=np.float64
        )
        radius = float(
            prompt.metadata.get("circle_xyr", [0.0, 0.0, 8.0])[2]
        )
        expected_area = max(np.pi * radius * radius, minimum_area)
        kept = 0
        rejected = 0

        def consume(indices: list[int], anchor: np.ndarray) -> None:
            nonlocal kept, rejected
            for frame_index in indices:
                candidates = [
                    item
                    for item in component_centroids(
                        instance[frame_index],
                        minimum_area=minimum_area,
                        maximum_components=8,
                    )
                    if item[1] <= maximum_area
                ]
                if not candidates:
                    rejected += 1
                    continue
                ranked = sorted(
                    candidates,
                    key=lambda item: (
                        np.linalg.norm(item[0] - anchor)
                        / max(maximum_jump, 1.0)
                        + 0.2
                        * abs(
                            np.log(
                                max(float(item[1]), 1.0)
                                / expected_area
                            )
                        )
                    ),
                )
                centroid, _, component = ranked[0]
                distance = float(np.linalg.norm(centroid - anchor))
                # At a missing frame the next valid component may legitimately
                # be farther away, so the gate is deliberately soft.
                if distance > maximum_jump and frame_index != seed:
                    rejected += 1
                    continue
                output[object_index][frame_index] = component
                anchor[:] = centroid
                kept += 1

        consume(list(range(seed, len(instance))), seed_anchor.copy())
        if seed > 0:
            consume(list(range(seed - 1, -1, -1)), seed_anchor.copy())
        kept_counts.append(kept)
        rejected_counts.append(rejected)
    return output, {
        "policy": "role_seeded_bidirectional_continuous_component",
        "minimum_area": minimum_area,
        "maximum_area": maximum_area,
        "maximum_centroid_jump_px": maximum_jump,
        "kept_frames_per_role": kept_counts,
        "rejected_frames_per_role": rejected_counts,
    }
