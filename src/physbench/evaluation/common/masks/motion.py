from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..errors import SceneAnalysisError
from .sam2 import MaskPrompt


def estimate_background(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise SceneAnalysisError("empty_timeline", "cannot estimate an empty video")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def motion_masks(
    frames: list[np.ndarray],
    *,
    threshold: float,
    minimum_area: int,
    close_kernel: int = 5,
) -> list[np.ndarray]:
    background = cv2.cvtColor(
        cv2.GaussianBlur(estimate_background(frames), (5, 5), 0),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_size = max(3, int(close_kernel) | 1)
    closing = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    output: list[np.ndarray] = []
    for frame in frames:
        gray = cv2.cvtColor(
            cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        difference = cv2.absdiff(gray, background)
        mask = (difference >= float(threshold)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= int(minimum_area):
                filtered[labels == label] = 255
        output.append(filtered)
    return output


def build_motion_prompt(
    frames: list[np.ndarray],
    *,
    threshold: float,
    minimum_area: int,
    box_expand: float,
    minimum_box_side: int,
    border_margin_ratio: float = 0.01,
) -> MaskPrompt:
    proposals = motion_masks(
        frames,
        threshold=threshold,
        minimum_area=minimum_area,
    )
    height, width = frames[0].shape[:2]
    margin_x = max(1, int(round(width * border_margin_ratio)))
    margin_y = max(1, int(round(height * border_margin_ratio)))
    best: tuple[float, int, int, int, int, int, np.ndarray] | None = None
    for frame_index, mask in enumerate(proposals):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[label]]
            if (
                area < minimum_area
                or x <= margin_x
                or y <= margin_y
                or x + w >= width - margin_x
                or y + h >= height - margin_y
            ):
                continue
            compactness = area / max(float(w * h), 1.0)
            score = float(area * (0.5 + compactness))
            if best is None or score > best[0]:
                best = (
                    score,
                    frame_index,
                    x,
                    y,
                    w,
                    h,
                    np.asarray(centroids[label], dtype=np.float32),
                )
    if best is None:
        raise SceneAnalysisError(
            "motion_prompt_failed", "could not locate a moving subject proposal"
        )
    score, frame_index, x, y, box_width, box_height, centroid = best
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    expanded_width = max(float(minimum_box_side), box_width * box_expand)
    expanded_height = max(float(minimum_box_side), box_height * box_expand)
    box = np.asarray(
        [
            max(0.0, center_x - expanded_width / 2),
            max(0.0, center_y - expanded_height / 2),
            min(width - 1.0, center_x + expanded_width / 2),
            min(height - 1.0, center_y + expanded_height / 2),
        ],
        dtype=np.float32,
    )
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=box,
        points_xy=centroid.reshape(1, 2),
        point_labels=np.ones(1, dtype=np.int32),
        metadata={
            "source": "temporal_median_motion",
            "proposal_score": score,
            "motion_box_xywh": [x, y, box_width, box_height],
        },
    )
