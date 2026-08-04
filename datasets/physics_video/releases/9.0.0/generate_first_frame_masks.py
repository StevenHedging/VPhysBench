#!/usr/bin/env python3
"""Generate one binary first-frame mask per declared physical subject.

This builder is intentionally release-local.  It reads the frozen 8.0.0
Cases and media, writes only derived ``canonical/masks`` directories, and
never rewrites source/canonical images or videos.  Subject cardinality comes
from Case metadata; visual models are not allowed to invent extra subjects.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parent
PHYSICS_VIDEO_ROOT = RELEASE_ROOT.parents[1]
REPOSITORY_ROOT = PHYSICS_VIDEO_ROOT.parents[1]
BASE_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "8.0.0"
SAM2_ROOT = Path("/root/Nico/third_party/sam2")
DEFAULT_CHECKPOINT = Path(
    "/mnt/nvme1/NicoCache/checkpoints/sam2/sam2.1_hiera_tiny.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
GENERATOR_ID = "sam2.1_hiera_tiny_first_frame_physical_subject_v1"

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(SAM2_ROOT))

from physbench.evaluation.scenes.collision.observation import (  # noqa: E402
    _hough_candidates as collision_hough_candidates,
)
from physbench.evaluation.scenes.parabolic_motion.evaluator import (  # noqa: E402
    _hough_candidates as parabolic_hough_candidates,
)
from physbench.evaluation.scenes.pendulum.open_world import (  # noqa: E402
    _circle_candidates as pendulum_circle_candidates,
)


SINGLE_SUBJECTS = {
    "inclined_plane_slide": ("sliding_block", "block"),
    "parabolic_motion": ("projectile_ball", "ball"),
    "pendulum": ("bob", "pendulum_bob"),
    "push_bottle": ("bottle", "bottle"),
}


def read_cases() -> list[dict[str, Any]]:
    path = BASE_RELEASE_ROOT / "cases.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def expected_subject_count(case: dict[str, Any]) -> int:
    scene = case["scene_id"]
    if scene == "collision_1d":
        indices = {
            int(key.split("_")[1])
            for key in case["physics"]
            if key.startswith("ball_") and key.endswith("_mass")
        }
        if not indices or indices != set(range(1, max(indices) + 1)):
            raise ValueError("collision ball indices are not contiguous")
        return max(indices)
    if scene == "uniform_circular_motion":
        count = case.get("appearance", {}).get("object_count")
        if count not in (1, 2):
            raise ValueError("circular object_count must be 1 or 2")
        return int(count)
    if scene in SINGLE_SUBJECTS:
        return 1
    raise ValueError(f"unsupported scene {scene!r}")


def entity_declarations(
    case: dict[str, Any], count: int
) -> list[dict[str, Any]]:
    scene = case["scene_id"]
    physics = case["physics"]
    if scene == "collision_1d":
        return [
            {
                "object_id": f"ball_{index}",
                "entity_class": "ball",
                "physics_keys": sorted(
                    key
                    for key in physics
                    if key.startswith(f"ball_{index}_")
                ),
            }
            for index in range(1, count + 1)
        ]
    if scene == "uniform_circular_motion":
        shared = ["angular_velocity"]
        appearance_labels = case.get("appearance", {}).get("moving_objects", [])
        return [
            {
                "object_id": f"object_{index}",
                "entity_class": "orbiting_block",
                "appearance_label": (
                    appearance_labels[index - 1]
                    if index - 1 < len(appearance_labels)
                    else None
                ),
                "physics_keys": [
                    *[key for key in shared if key in physics],
                    f"object_{index}_orbit_radius",
                ],
            }
            for index in range(1, count + 1)
        ]
    object_id, entity_class = SINGLE_SUBJECTS[scene]
    return [
        {
            "object_id": object_id,
            "entity_class": entity_class,
            "physics_keys": sorted(
                key
                for key, quantity in physics.items()
                if isinstance(quantity, dict) and quantity.get("annotated") is True
            ),
        }
    ]


def sampled_change_map(
    first_frame: np.ndarray,
    video_path: Path,
    *,
    samples: int = 18,
    first_seconds: float | None = None,
) -> np.ndarray:
    height, width = first_frame.shape[:2]
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if frame_count <= 1:
        capture.release()
        raise ValueError("reference video has fewer than two frames")
    last = frame_count - 1
    if first_seconds is not None:
        last = min(last, max(1, int(round(first_seconds * fps))))
    indices = sorted({int(value) for value in np.linspace(1, last, samples)})
    differences: list[np.ndarray] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        delta = cv2.absdiff(frame, first_frame)
        differences.append(cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY))
    capture.release()
    if not differences:
        raise ValueError("reference video yielded no sampled frames")
    change = np.percentile(np.stack(differences), 75, axis=0).astype(np.float32)
    return cv2.GaussianBlur(change, (3, 3), 0)


def _circle_change_metrics(
    change: np.ndarray, circle: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    x, y, radius = circle
    height, width = change.shape
    y0 = max(0, int(y - 2.0 * radius))
    y1 = min(height, int(y + 2.0 * radius + 1))
    x0 = max(0, int(x - 2.0 * radius))
    x1 = min(width, int(x + 2.0 * radius + 1))
    patch = change[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = (xx - x) ** 2 + (yy - y) ** 2
    core = distance <= 0.85 * radius * radius
    ring = (distance >= 1.25 * radius * radius) & (
        distance <= 1.90 * radius * radius
    )
    if not np.any(core) or not np.any(ring):
        return 0.0, 0.0, 0.0, 0.0
    inside = float(np.mean(patch[core]))
    changed_fraction = float(np.mean(patch[core] >= 12.0))
    ring_change = float(np.mean(patch[ring]))
    score = inside + 35.0 * changed_fraction - 0.25 * ring_change
    return score, inside, changed_fraction, ring_change


def localize_pendulum(
    case: dict[str, Any], frame: np.ndarray
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    config = json.loads(
        (REPOSITORY_ROOT / "configs/evaluation/protocols/scene_default_v10.json")
        .read_text()
    )["scenes"]["pendulum"]["open_world_observation"]
    config = dict(config)
    config["hough_accumulator_thresholds"] = [
        35,
        32,
        29,
        26,
        23,
        20,
        17,
        14,
        11,
        8,
        6,
    ]
    change = sampled_change_map(
        frame,
        PHYSICS_VIDEO_ROOT / case["assets"]["reference_video"],
        first_seconds=2.0,
    )
    height, width = frame.shape[:2]
    candidates = []
    for circle in pendulum_circle_candidates(
        frame, config=config, maximum_candidates=240
    ):
        x, y, radius = circle
        if (
            x - radius <= 1
            or x + radius >= width - 1
            or y - radius <= 1
            or y + radius >= height - 1
        ):
            continue
        score, inside, fraction, ring = _circle_change_metrics(change, circle)
        candidates.append(
            {
                "circle": tuple(float(value) for value in circle),
                "score": score,
                "inside_change": inside,
                "changed_fraction": fraction,
                "ring_change": ring,
            }
        )
    if not candidates:
        raise ValueError("no pendulum circle candidates")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    if best["changed_fraction"] < 0.18 or best["inside_change"] < 8.0:
        raise ValueError("no high-motion pendulum circle candidate")
    bx, by, br = best["circle"]
    cluster = [
        item
        for item in candidates[:24]
        if item["score"] >= 0.65 * best["score"]
        and math.hypot(item["circle"][0] - bx, item["circle"][1] - by)
        <= 0.90 * (item["circle"][2] + br)
        and item["circle"][2] <= 0.075 * width
    ]
    selected = max(cluster or [best], key=lambda item: item["circle"][2])
    return [selected["circle"]], {
        "localizer": "hough_circle_plus_reference_departure",
        "selected": selected,
        "top_candidates": candidates[:5],
    }


def localize_parabolic(
    frame: np.ndarray,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    config = {
        "blur_kernel": 7,
        "hough_dp": 1.2,
        "edge_threshold": 90,
        "initial_hough_thresholds": [30, 27, 24, 20, 16, 13, 10, 7],
        "minimum_center_distance_px": 12,
        "minimum_radius_px": 3,
        "maximum_radius_px": 30,
        "initial_minimum_x_ratio": 0.65,
        "initial_maximum_y_ratio": 0.45,
        "minimum_local_contrast": 5,
    }
    candidates = parabolic_hough_candidates(frame, config=config, initial=True)
    candidates.sort(key=lambda item: (item.contrast, item.radius), reverse=True)
    if len(candidates) != 1:
        raise ValueError(f"expected one projectile proposal, got {len(candidates)}")
    value = candidates[0]
    circle = (float(value.xy[0]), float(value.xy[1]), float(value.radius))
    return [circle], {
        "localizer": "initial_region_hough_circle",
        "contrast": float(value.contrast),
        "source": value.source,
    }


def _local_circle_contrast(
    gray: np.ndarray, circle: tuple[float, float, float]
) -> float:
    x, y, radius = circle
    height, width = gray.shape
    y0 = max(0, int(y - 2.2 * radius))
    y1 = min(height, int(y + 2.2 * radius + 1))
    x0 = max(0, int(x - 2.2 * radius))
    x1 = min(width, int(x + 2.2 * radius + 1))
    patch = gray[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = (xx - x) ** 2 + (yy - y) ** 2
    core = distance <= 0.70 * radius * radius
    ring = (distance >= 1.35 * radius * radius) & (
        distance <= 2.10 * radius * radius
    )
    if not np.any(core) or not np.any(ring):
        return 0.0
    return abs(float(np.mean(patch[core])) - float(np.median(patch[ring])))


def localize_collision(
    case: dict[str, Any], frame: np.ndarray, count: int
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    height, width = frame.shape[:2]
    supplemental = height < 700
    config = {
        "track_top_ratio": 0.42 if supplemental else 0.62,
        "track_bottom_ratio": 0.79 if supplemental else 0.94,
        "blur_kernel": 5,
        "blur_sigma": 1.2,
        "hough_dp": 1.2,
        "edge_threshold": 100,
        "minimum_circle_radius_px": 4,
        "maximum_circle_radius_px": 42,
        "minimum_circle_center_distance_px": 8,
        "circle_nms_overlap_distance_factor": 0.45,
        "maximum_candidates": 40,
    }
    raw: list[tuple[float, float, float, float]] = []
    for threshold in (42, 38, 34, 30, 26, 22, 18, 14, 10, 7):
        circles, _ = collision_hough_candidates(
            frame, config=config, accumulator_threshold=threshold
        )
        raw.extend((*circle, float(threshold)) for circle in circles)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[float, float, float, float, float]] = []
    for x, y, radius, threshold in sorted(
        raw, key=lambda item: (item[3], item[2]), reverse=True
    ):
        if any(
            math.hypot(x - old_x, y - old_y) < 0.55 * max(radius, old_radius)
            and abs(math.log(max(radius, 1.0) / max(old_radius, 1.0))) < 0.35
            for old_x, old_y, old_radius, _, _ in candidates
        ):
            continue
        contrast = _local_circle_contrast(gray, (x, y, radius))
        candidates.append((x, y, radius, threshold, contrast))
    candidates = candidates[:40]
    expected_radii = np.asarray(
        [
            float(case["physics"][f"ball_{index}_radius"]["value"])
            for index in range(1, count + 1)
        ]
    )
    viable = []
    for combination in itertools.combinations(candidates, count):
        ordered = sorted(combination, key=lambda item: item[0])
        x = np.asarray([item[0] for item in ordered])
        y = np.asarray([item[1] for item in ordered])
        radius = np.asarray([item[2] for item in ordered])
        threshold = np.asarray([item[3] for item in ordered])
        contrast = np.asarray([item[4] for item in ordered])
        y_spread = float(np.ptp(y))
        gaps = np.diff(x)
        if y_spread > max(30.0, 0.05 * height):
            continue
        if np.any(gaps < 0.50 * (radius[:-1] + radius[1:])):
            continue
        scales = radius / (width * expected_radii)
        scale = float(np.median(scales))
        radius_fit = float(np.mean(np.abs(np.log(scales / scale))))
        scale_target = 2.0 if supplemental else 1.15
        scale_prior = abs(math.log(scale / scale_target))
        if radius_fit > 0.42 or not 0.65 <= scale <= 3.60:
            continue
        structure = 0.0
        if count == 3:
            contact = gaps[1] / max(radius[1] + radius[2], 1.0)
            structure = -5.0 * abs(math.log(max(contact, 0.1) / 1.15))
            structure += min(gaps[0] / max(radius[0] + radius[1], 1.0), 5.0)
        target_y = 0.67 if supplemental else 0.87
        y_penalty = abs(float(np.mean(y)) / height - target_y) * 20.0
        score = (
            float(np.mean(contrast))
            + 0.35 * float(np.mean(threshold))
            - 12.0 * radius_fit
            - 4.0 * scale_prior
            - 0.45 * y_spread
            + structure
            - y_penalty
        )
        viable.append(
            (
                score,
                ordered,
                {
                    "score": score,
                    "radius_fit": radius_fit,
                    "pixel_scale": scale,
                    "mean_contrast": float(np.mean(contrast)),
                    "y_spread": y_spread,
                },
            )
        )
    if not viable:
        raise ValueError("no collision circle set satisfies Case cardinality")
    _, selected, diagnostics = max(viable, key=lambda item: item[0])
    circles = [tuple(float(value) for value in item[:3]) for item in selected]
    return circles, {
        "localizer": "track_hough_plus_case_radius_and_order",
        **diagnostics,
    }


def _filled_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask, 1, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return binary
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == index, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # RLE-decoded automatic masks can be Fortran ordered.  OpenCV drawing
    # requires a C-contiguous destination.
    output = np.zeros(binary.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(output, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return output


def _mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return {
        "area_pixels": int(len(xs)),
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _prompt_circle_mask(
    predictor: Any,
    circle: tuple[float, float, float],
    frame_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    x, y, radius = circle
    height, width = frame_shape[:2]
    side = max(28.0, 3.2 * radius)
    box = np.asarray(
        [
            max(0.0, x - side / 2.0),
            max(0.0, y - side / 2.0),
            min(width - 1.0, x + side / 2.0),
            min(height - 1.0, y + side / 2.0),
        ],
        dtype=np.float32,
    )
    offset = 0.30 * radius
    points = np.asarray(
        [
            [x, y],
            [x - offset, y],
            [x + offset, y],
            [x, y - offset],
            [x, y + offset],
        ],
        dtype=np.float32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    masks, qualities, _ = predictor.predict(
        point_coords=points,
        point_labels=np.ones(len(points), dtype=np.int32),
        box=box,
        multimask_output=True,
    )
    candidates = []
    nominal_area = math.pi * radius * radius
    for raw, quality in zip(masks, qualities):
        mask = _filled_largest_component(raw > 0)
        geometry = _mask_geometry(mask)
        cx, cy = geometry["centroid_xy"]
        center_error = math.hypot(cx - x, cy - y) / max(radius, 1.0)
        area_error = abs(
            math.log(max(geometry["area_pixels"], 1) / max(nominal_area, 1.0))
        )
        score = float(quality) - 0.45 * center_error - 0.15 * area_error
        candidates.append((score, mask, float(quality), geometry))
    score, mask, quality, geometry = max(candidates, key=lambda item: item[0])
    ratio = geometry["area_pixels"] / max(nominal_area, 1.0)
    if not 0.12 <= ratio <= 10.0:
        raise ValueError(f"circle mask/anchor area ratio is implausible: {ratio:.3f}")
    return mask, {
        "sam_predicted_iou": quality,
        "prompt_box_xyxy": box.tolist(),
        "prompt_points_xy": points.tolist(),
        "anchor_circle_xyr": list(circle),
        "selection_score": score,
    }


def _inclined_block_hole_pair(
    frame: np.ndarray, annotated_angle_degrees: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, Any]]:
    """Locate the two round recesses unique to the sliding block."""

    height, width = frame.shape[:2]
    scale = 0.5
    small = cv2.resize(
        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.GaussianBlur(gray_small, (7, 7), 1.5)
    raw: list[tuple[float, float, float, float]] = []
    for threshold in (42, 36, 30, 25, 20, 16):
        circles = cv2.HoughCircles(
            gray_small,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=100,
            param2=threshold,
            minRadius=8,
            maxRadius=50,
        )
        if circles is not None:
            raw.extend(
                (
                    float(x / scale),
                    float(y / scale),
                    float(radius / scale),
                    float(threshold),
                )
                for x, y, radius in circles[0]
            )
    candidates: list[tuple[float, float, float, float]] = []
    for x, y, radius, threshold in sorted(
        raw, key=lambda item: item[3], reverse=True
    ):
        if not (0.52 * width < x < 0.97 * width and 0 < y < 0.40 * height):
            continue
        if any(
            math.hypot(x - old_x, y - old_y)
            < 0.45 * max(radius, old_radius)
            for old_x, old_y, old_radius, _ in candidates
        ):
            continue
        candidates.append((x, y, radius, threshold))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def center_mean(circle: tuple[float, float, float, float]) -> float:
        x, y, radius, _ = circle
        extent = max(2, int(math.ceil(0.60 * radius)))
        x0, x1 = max(0, int(x) - extent), min(width, int(x) + extent + 1)
        y0, y1 = max(0, int(y) - extent), min(height, int(y) + extent + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        selected = (xx - x) ** 2 + (yy - y) ** 2 <= (0.55 * radius) ** 2
        return float(np.mean(gray[y0:y1, x0:x1][selected]))

    pairs = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            left, right = sorted((first, second), key=lambda item: item[0])
            dx = right[0] - left[0]
            dy = left[1] - right[1]
            distance = math.hypot(dx, dy)
            if not 0.045 * width < distance < 0.16 * width or dx <= 0:
                continue
            observed_angle = math.degrees(math.atan2(dy, dx))
            angle_error = abs(observed_angle - annotated_angle_degrees)
            if angle_error > 14:
                continue
            radius_error = abs(
                math.log(max(left[2], 1.0) / max(right[2], 1.0))
            )
            if radius_error > 0.35:
                continue
            brightness = [center_mean(left), center_mean(right)]
            brightness_error = abs(brightness[0] - brightness[1])
            if min(brightness) < 45 or brightness_error > 55:
                continue
            center_x = 0.5 * (left[0] + right[0])
            center_y = 0.5 * (left[1] + right[1])
            center_prior = math.hypot(
                (center_x / width - 0.79) / 0.25,
                (center_y / height - 0.16) / 0.25,
            )
            score = (
                0.08 * (left[3] + right[3])
                - 0.28 * angle_error
                - 3.0 * radius_error
                - 2.0 * abs(distance / width - 0.075)
                - 1.2 * center_prior
                - 0.015 * brightness_error
            )
            pairs.append(
                {
                    "score": score,
                    "left_xyr": [float(value) for value in left[:3]],
                    "right_xyr": [float(value) for value in right[:3]],
                    "observed_angle_degrees": observed_angle,
                    "annotated_angle_degrees": annotated_angle_degrees,
                    "center_brightness": brightness,
                    "center_distance_pixels": distance,
                }
            )
    if not pairs:
        raise ValueError("could not localize the inclined block's two recesses")
    pairs.sort(key=lambda item: item["score"], reverse=True)
    selected = pairs[0]
    left = tuple(selected["left_xyr"])
    right = tuple(selected["right_xyr"])
    return left, right, {
        "localizer": "paired_recess_hough_plus_annotated_incline_angle",
        "selected_pair": selected,
        "top_pairs": pairs[:5],
        "circle_candidate_count": len(candidates),
    }


def _prompt_inclined_block_mask(
    predictor: Any,
    frame: np.ndarray,
    change: np.ndarray,
    annotated_angle_degrees: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the block's paired recesses to prompt and constrain SAM2."""

    height, width = frame.shape[:2]
    left, right, localization = _inclined_block_hole_pair(
        frame, annotated_angle_degrees
    )
    left_xy = np.asarray(left[:2], dtype=np.float64)
    right_xy = np.asarray(right[:2], dtype=np.float64)
    center = 0.5 * (left_xy + right_xy)
    vector = right_xy - left_xy
    distance = float(np.linalg.norm(vector))
    if distance <= 0:
        raise ValueError("inclined block recess pair has zero separation")
    long_axis = vector / distance
    track_normal = np.asarray([-long_axis[1], long_axis[0]])
    half_length = 1.42 * distance
    half_width = 0.56 * distance
    corners = np.asarray(
        [
            center - half_length * long_axis - half_width * track_normal,
            center + half_length * long_axis - half_width * track_normal,
            center + half_length * long_axis + half_width * track_normal,
            center - half_length * long_axis + half_width * track_normal,
        ]
    )
    support = np.zeros((height, width), dtype=np.uint8)
    clipped_corners = corners.copy()
    clipped_corners[:, 0] = np.clip(clipped_corners[:, 0], 0, width - 1)
    clipped_corners[:, 1] = np.clip(clipped_corners[:, 1], 0, height - 1)
    cv2.fillConvexPoly(
        support, np.round(clipped_corners).astype(np.int32), 1
    )
    pad = 0.08 * distance
    box = np.asarray(
        [
            max(0.0, float(corners[:, 0].min() - pad)),
            max(0.0, float(corners[:, 1].min() - pad)),
            min(width - 1.0, float(corners[:, 0].max() + pad)),
            min(height - 1.0, float(corners[:, 1].max() + pad)),
        ],
        dtype=np.float32,
    )
    points = np.asarray(
        [
            center,
            center - 0.28 * distance * long_axis,
            center + 0.28 * distance * long_axis,
            center + 0.78 * distance * track_normal,
            center - 1.70 * distance * long_axis,
            center + 1.70 * distance * long_axis,
        ],
        dtype=np.float32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int32)
    masks, qualities, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )
    candidates = []
    support_area = int(np.count_nonzero(support))
    for raw, quality in zip(masks, qualities):
        constrained = (raw > 0) & (support > 0)
        if not np.any(constrained):
            continue
        mask = _filled_largest_component(constrained)
        geometry = _mask_geometry(mask)
        area_ratio = geometry["area_pixels"] / float(height * width)
        support_coverage = geometry["area_pixels"] / max(support_area, 1)
        values = change[mask > 0]
        changed_fraction = float(np.mean(values >= 12.0))
        mean_change = float(np.mean(values))
        centroid_distance = float(
            np.linalg.norm(np.asarray(geometry["centroid_xy"]) - center)
        )
        size_error = abs(math.log(max(area_ratio, 1e-6) / 0.015))
        score = (
            float(quality)
            + 0.50 * support_coverage
            + 0.20 * changed_fraction
            + 0.004 * mean_change
            - 0.55 * size_error
            - 0.50 * centroid_distance / distance
        )
        candidates.append(
            {
                "score": score,
                "mask": mask,
                "sam_predicted_iou": float(quality),
                "area_ratio": area_ratio,
                "support_coverage": support_coverage,
                "changed_fraction": changed_fraction,
                "mean_change": mean_change,
                "centroid_distance_recess_units": centroid_distance / distance,
            }
        )
    if not candidates:
        raise ValueError("SAM2 returned no inclined-block mask inside recess support")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0]
    if not 0.004 <= selected["area_ratio"] <= 0.040:
        raise ValueError(
            "inclined block SAM mask has implausible area ratio: "
            f"{selected['area_ratio']:.4f}"
        )
    if selected["support_coverage"] < 0.20:
        raise ValueError(
            "inclined block SAM mask has insufficient geometric support: "
            f"{selected['support_coverage']:.3f}"
        )
    if selected["centroid_distance_recess_units"] > 0.65:
        raise ValueError("inclined block SAM mask is not centered on its recess pair")
    return selected["mask"], {
        **localization,
        "prompt_box_xyxy": box.tolist(),
        "prompt_points_xy": points.tolist(),
        "prompt_point_labels": labels.tolist(),
        "oriented_support_corners_xy": corners.tolist(),
        "sam_predicted_iou": selected["sam_predicted_iou"],
        "area_ratio": selected["area_ratio"],
        "support_coverage": selected["support_coverage"],
        "changed_fraction": selected["changed_fraction"],
        "mean_change": selected["mean_change"],
        "centroid_distance_recess_units": selected[
            "centroid_distance_recess_units"
        ],
        "selection_score": selected["score"],
    }


def _proposal_metrics(
    proposal: dict[str, Any], change: np.ndarray
) -> dict[str, float]:
    mask = np.asarray(proposal["segmentation"], dtype=bool)
    area = int(np.count_nonzero(mask))
    height, width = mask.shape
    x, y, box_width, box_height = [float(value) for value in proposal["bbox"]]
    values = change[mask]
    return {
        "area_ratio": area / float(height * width),
        "mean_change": float(np.mean(values)) if values.size else 0.0,
        "changed_fraction": float(np.mean(values >= 12.0)) if values.size else 0.0,
        "bbox_aspect_h_over_w": box_height / max(box_width, 1.0),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": box_width,
        "bbox_height": box_height,
        "centroid_x": x + box_width / 2.0,
        "centroid_y": y + box_height / 2.0,
        "touches_border": float(
            x <= 1.0
            or y <= 1.0
            or x + box_width >= width - 2.0
            or y + box_height >= height - 2.0
        ),
    }


def select_motion_proposals(
    scene: str,
    proposals: list[dict[str, Any]],
    change: np.ndarray,
    count: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    height, width = change.shape
    diagonal = math.hypot(width, height)
    priors = {
        "inclined_plane_slide": (0.015, 0.0025, 0.055),
        "push_bottle": (0.045, 0.0060, 0.140),
        "uniform_circular_motion": (0.006, 0.0005, 0.035),
    }
    target_area, minimum_area, maximum_area = priors[scene]
    platform_center: list[float] | None = None
    if scene == "uniform_circular_motion":
        platform_candidates = []
        for proposal in proposals:
            metrics = _proposal_metrics(proposal, change)
            area_ratio = metrics["area_ratio"]
            aspect = metrics["bbox_aspect_h_over_w"]
            if (
                0.20 <= area_ratio <= 0.55
                and 0.75 <= aspect <= 1.25
                and 0.15 * width <= metrics["centroid_x"] <= 0.85 * width
                and 0.25 * height <= metrics["centroid_y"] <= 0.75 * height
            ):
                platform_candidates.append(
                    (
                        float(proposal["predicted_iou"])
                        + float(proposal["stability_score"]),
                        metrics,
                    )
                )
        if not platform_candidates:
            raise ValueError("could not locate circular-motion platform")
        _, platform = max(platform_candidates, key=lambda item: item[0])
        platform_center = [platform["centroid_x"], platform["centroid_y"]]
    ranked = []
    for proposal_index, proposal in enumerate(proposals):
        metrics = _proposal_metrics(proposal, change)
        area_ratio = metrics["area_ratio"]
        if not minimum_area <= area_ratio <= maximum_area:
            continue
        if metrics["touches_border"]:
            continue
        cx = metrics["centroid_x"]
        cy = metrics["centroid_y"]
        aspect = metrics["bbox_aspect_h_over_w"]
        if scene == "inclined_plane_slide":
            if cy > 0.70 * height or cx < 0.45 * width:
                continue
            shape_bonus = min(max(aspect, 1.0 / max(aspect, 1e-6)), 4.0) / 4.0
        elif scene == "push_bottle":
            if not 0.25 * height <= cy <= 0.88 * height or aspect < 1.15:
                continue
            shape_bonus = min(aspect / 3.0, 1.0)
        else:
            if not (0.08 * width <= cx <= 0.92 * width and 0.20 * height <= cy <= 0.80 * height):
                continue
            shape_bonus = 0.5
        size_penalty = abs(math.log(area_ratio / target_area))
        # For transparent bottles SAM2 often returns nested upper/lower liquid
        # regions in addition to the complete silhouette.  A stronger size
        # prior favors the full bottle while motion evidence still rejects
        # static mats and apparatus.
        size_weight = 0.95 if scene == "push_bottle" else 0.35
        score = (
            1.70 * metrics["changed_fraction"]
            + 0.025 * metrics["mean_change"]
            + float(proposal["predicted_iou"])
            + 0.50 * float(proposal["stability_score"])
            + 0.35 * shape_bonus
            - size_weight * size_penalty
        )
        ranked.append(
            {
                "proposal_index": proposal_index,
                "score": score,
                "metrics": metrics,
                "predicted_iou": float(proposal["predicted_iou"]),
                "stability_score": float(proposal["stability_score"]),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    selected_masks: list[np.ndarray] = []
    for item in ranked:
        raw = proposals[item["proposal_index"]]["segmentation"]
        mask = _filled_largest_component(raw)
        geometry = _mask_geometry(mask)
        center = np.asarray(geometry["centroid_xy"])
        duplicate = False
        for old_mask, old_item in zip(selected_masks, selected):
            intersection = int(np.count_nonzero((mask > 0) & (old_mask > 0)))
            union = int(np.count_nonzero((mask > 0) | (old_mask > 0)))
            iou = intersection / max(union, 1)
            old_center = np.asarray(old_item["geometry"]["centroid_xy"])
            if iou > 0.12 or np.linalg.norm(center - old_center) < 0.025 * diagonal:
                duplicate = True
                break
        if duplicate:
            continue
        item = {**item, "geometry": geometry}
        selected.append(item)
        selected_masks.append(mask)
        if len(selected_masks) == count:
            break
    if len(selected_masks) != count:
        raise ValueError(
            f"motion proposal selection found {len(selected_masks)} of {count} subjects"
        )
    minimum_departure_fraction = (
        0.0 if scene == "inclined_plane_slide" else 0.12
    )
    if any(
        item["metrics"]["changed_fraction"] < minimum_departure_fraction
        for item in selected
    ):
        raise ValueError("selected motion proposal has insufficient departure evidence")
    order = sorted(
        range(len(selected_masks)),
        key=lambda index: (
            selected[index]["geometry"]["centroid_xy"][1],
            selected[index]["geometry"]["centroid_xy"][0],
        ),
    )
    return [selected_masks[index] for index in order], {
        "localizer": "reference_departure_plus_sam2_automatic_proposals",
        "selected": [selected[index] for index in order],
        "top_candidates": ranked[:8],
        **(
            {"rotation_platform_center_xy": platform_center}
            if platform_center is not None
            else {}
        ),
    }


def make_exclusive(
    masks: list[np.ndarray], anchors: list[tuple[float, float, float]] | None
) -> list[np.ndarray]:
    if len(masks) <= 1:
        return masks
    stack = np.stack([mask > 0 for mask in masks])
    overlap = np.sum(stack, axis=0) > 1
    if not np.any(overlap):
        return masks
    if anchors is None:
        output = np.zeros_like(stack)
        winner = np.argmax(stack, axis=0)
    else:
        yy, xx = np.indices(stack.shape[1:])
        distances = np.stack(
            [(xx - x) ** 2 + (yy - y) ** 2 for x, y, _ in anchors]
        ).astype(np.float64)
        distances[~stack] = np.inf
        winner = np.argmin(distances, axis=0)
        output = np.zeros_like(stack)
    for index in range(len(masks)):
        output[index] = stack[index] & (~overlap | (winner == index))
    return [value.astype(np.uint8) for value in output]


def write_preview(
    path: Path,
    frame: np.ndarray,
    masks: list[np.ndarray],
    labels: list[str],
) -> None:
    colors = [(40, 60, 255), (60, 220, 60), (255, 120, 40)]
    output = frame.copy()
    for index, (mask, label) in enumerate(zip(masks, labels)):
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        selected = mask > 0
        output[selected] = (
            0.55 * output[selected].astype(np.float32) + 0.45 * color
        ).astype(np.uint8)
        geometry = _mask_geometry(mask)
        x, y = [int(round(value)) for value in geometry["centroid_xy"]]
        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.6, frame.shape[1] / 1600.0),
            (0, 0, 0),
            max(2, frame.shape[1] // 700),
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output, [cv2.IMWRITE_JPEG_QUALITY, 92])


def process_case(
    case: dict[str, Any],
    *,
    predictor: Any,
    automatic_generator: Any,
    materialize: bool,
    preview_root: Path | None,
) -> dict[str, Any]:
    count = expected_subject_count(case)
    declarations = entity_declarations(case, count)
    first_frame_relative = Path(case["assets"]["first_frame"])
    first_frame_path = PHYSICS_VIDEO_ROOT / first_frame_relative
    frame = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read first frame {first_frame_relative}")
    scene = case["scene_id"]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    anchors: list[tuple[float, float, float]] | None = None
    segmentation_records: list[dict[str, Any]] = []
    if scene == "collision_1d":
        anchors, localization = localize_collision(case, frame, count)
    elif scene == "parabolic_motion":
        anchors, localization = localize_parabolic(frame)
    elif scene == "pendulum":
        anchors, localization = localize_pendulum(case, frame)
    elif scene == "inclined_plane_slide":
        change = sampled_change_map(
            frame, PHYSICS_VIDEO_ROOT / case["assets"]["reference_video"]
        )
        predictor.set_image(rgb)
        mask, localization = _prompt_inclined_block_mask(
            predictor,
            frame,
            change,
            float(case["physics"]["incline_angle"]["value"]),
        )
        masks = [mask]
        segmentation_records = [localization]
    else:
        localization = {}
    if anchors is not None:
        predictor.set_image(rgb)
        masks = []
        for anchor in anchors:
            mask, record = _prompt_circle_mask(predictor, anchor, frame.shape)
            masks.append(mask)
            segmentation_records.append(record)
    elif scene != "inclined_plane_slide":
        change = sampled_change_map(
            frame, PHYSICS_VIDEO_ROOT / case["assets"]["reference_video"]
        )
        proposals = automatic_generator.generate(rgb)
        masks, localization = select_motion_proposals(
            scene, proposals, change, count
        )
        segmentation_records = localization["selected"]
    masks = make_exclusive(masks, anchors)
    if len(masks) != count or any(not np.any(mask) for mask in masks):
        raise ValueError("non-empty mask cardinality does not match Case contract")
    geometries = [_mask_geometry(mask) for mask in masks]
    if scene != "collision_1d" and count > 1:
        order = sorted(
            range(count),
            key=lambda index: (
                geometries[index]["centroid_xy"][1],
                geometries[index]["centroid_xy"][0],
            ),
        )
        masks = [masks[index] for index in order]
        geometries = [geometries[index] for index in order]
        segmentation_records = [segmentation_records[index] for index in order]
    if scene == "uniform_circular_motion" and count > 1:
        platform_center = np.asarray(
            localization["rotation_platform_center_xy"], dtype=np.float64
        )
        visual_radii = [
            float(np.linalg.norm(np.asarray(item["centroid_xy"]) - platform_center))
            for item in geometries
        ]
        physical_radii = [
            float(case["physics"][f"object_{index}_orbit_radius"]["value"])
            for index in range(1, count + 1)
        ]
        visual_order = sorted(range(count), key=lambda index: visual_radii[index])
        physical_order = sorted(range(count), key=lambda index: physical_radii[index])
        aligned: list[dict[str, Any] | None] = [None] * count
        for mask_index, declaration_index in zip(visual_order, physical_order):
            aligned[mask_index] = declarations[declaration_index]
        if any(item is None for item in aligned):
            raise ValueError("circular mask-to-physics assignment is incomplete")
        declarations = [item for item in aligned if item is not None]
        localization["mask_to_object_assignment"] = [
            {
                "mask_id": f"{index + 1:02d}",
                "object_id": declarations[index]["object_id"],
                "visual_orbit_radius_pixels": visual_radii[index],
                "annotated_orbit_radius_m": physical_radii[
                    int(declarations[index]["object_id"].split("_")[1]) - 1
                ],
            }
            for index in range(count)
        ]
    mask_directory_relative = first_frame_relative.parent / "masks"
    instances = []
    for index, (declaration, geometry, segmentation) in enumerate(
        zip(declarations, geometries, segmentation_records), start=1
    ):
        filename = f"{index:02d}.png"
        mask_relative = mask_directory_relative / filename
        instances.append(
            {
                "mask_id": f"{index:02d}",
                **declaration,
                "asset": mask_relative.as_posix(),
                **geometry,
                "segmentation": segmentation,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "scene_id": scene,
        "source_first_frame": first_frame_relative.as_posix(),
        "frame_index": 0,
        "frame_scope": "first_frame_only",
        "image_shape_hw": list(frame.shape[:2]),
        "dtype": "uint8",
        "values": [0, 1],
        "ordering": "row_major_top_to_bottom_then_left_to_right; collision preserves declared left-to-right ball order",
        "generator": {
            "id": GENERATOR_ID,
            "model": "SAM2.1 Hiera Tiny",
            "checkpoint": str(DEFAULT_CHECKPOINT),
            "checkpoint_sha256": DEFAULT_CHECKPOINT_SHA256,
            "sam2_code": str(SAM2_ROOT),
        },
        "localization": localization,
        "instances": instances,
    }
    if materialize:
        mask_directory = PHYSICS_VIDEO_ROOT / mask_directory_relative
        mask_directory.mkdir(parents=True, exist_ok=True)
        for index, mask in enumerate(masks, start=1):
            path = mask_directory / f"{index:02d}.png"
            if not cv2.imwrite(str(path), mask.astype(np.uint8)):
                raise ValueError(f"failed to write {path}")
        (mask_directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    if preview_root is not None:
        write_preview(
            preview_root / scene / f"{case['case_id']}.jpg",
            frame,
            masks,
            [item["mask_id"] for item in instances],
        )
    return {
        "case_id": case["case_id"],
        "scene_id": scene,
        "status": "complete",
        "mask_directory": mask_directory_relative.as_posix(),
        "instances": instances,
        "localization": localization,
    }


def worker_main(
    worker_index: int,
    device_index: int,
    cases: list[dict[str, Any]],
    checkpoint: str,
    materialize: bool,
    preview_root: str | None,
    queue: Any,
) -> None:
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        torch.cuda.set_device(device_index)
        model = build_sam2(
            SAM2_CONFIG,
            checkpoint,
            device=f"cuda:{device_index}",
            apply_postprocessing=False,
        )
        automatic = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=24,
            points_per_batch=128,
            pred_iou_thresh=0.65,
            stability_score_thresh=0.75,
            crop_n_layers=0,
            min_mask_region_area=10,
        )
        predictor = automatic.predictor
        results = []
        for case_index, case in enumerate(cases, start=1):
            try:
                result = process_case(
                    case,
                    predictor=predictor,
                    automatic_generator=automatic,
                    materialize=materialize,
                    preview_root=(Path(preview_root) if preview_root else None),
                )
            except Exception as error:
                result = {
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                    "status": "skipped",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            if case_index % 10 == 0 or case_index == len(cases):
                print(
                    f"worker={worker_index} device={device_index} "
                    f"processed={case_index}/{len(cases)}",
                    flush=True,
                )
        queue.put({"worker": worker_index, "results": results})
    except Exception:
        queue.put(
            {
                "worker": worker_index,
                "fatal_error": traceback.format_exc(),
            }
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    cases = read_cases()
    if arguments.scene:
        allowed = set(arguments.scene)
        cases = [case for case in cases if case["scene_id"] in allowed]
    if arguments.case_id:
        allowed_cases = set(arguments.case_id)
        cases = [case for case in cases if case["case_id"] in allowed_cases]
    if not cases:
        raise ValueError("no Cases selected")
    worker_count = min(arguments.workers, len(cases))
    shards = [cases[index::worker_count] for index in range(worker_count)]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = []
    for worker_index, shard in enumerate(shards):
        process = context.Process(
            target=worker_main,
            args=(
                worker_index,
                worker_index,
                shard,
                str(arguments.checkpoint),
                bool(arguments.materialize),
                str(arguments.preview_root) if arguments.preview_root else None,
                queue,
            ),
        )
        process.start()
        processes.append(process)
    messages = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    fatal = [message for message in messages if "fatal_error" in message]
    if fatal:
        raise RuntimeError("\n".join(message["fatal_error"] for message in fatal))
    results = [
        result
        for message in messages
        for result in message["results"]
    ]
    order = {case["case_id"]: index for index, case in enumerate(cases)}
    results.sort(key=lambda item: order[item["case_id"]])
    report = {
        "schema_version": "1.0",
        "generator_id": GENERATOR_ID,
        "base_release": "8.0.0",
        "target_release": "9.0.0",
        "materialized": bool(arguments.materialize),
        "selected_cases": len(cases),
        "complete_cases": sum(item["status"] == "complete" for item in results),
        "skipped_cases": sum(item["status"] != "complete" for item in results),
        "results": results,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"selected={report['selected_cases']} complete={report['complete_cases']} "
        f"skipped={report['skipped_cases']} report={arguments.report}"
    )
    return 0 if report["skipped_cases"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
