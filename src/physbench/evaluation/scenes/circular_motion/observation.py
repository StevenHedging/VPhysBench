from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...common.errors import SceneAnalysisError


def green_disk_object_masks(
    frames: list[np.ndarray], *, config: dict[str, Any]
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Extract non-green moving bodies inside the scene's annotated green disk."""
    lower = np.asarray(config["green_hsv_lower"], dtype=np.uint8)
    upper = np.asarray(config["green_hsv_upper"], dtype=np.uint8)
    erosion_size = max(3, int(config["disk_erosion_kernel"]) | 1)
    erosion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minimum_disk_ratio = float(config["minimum_disk_area_ratio"])
    masks: list[np.ndarray] = []
    disk_ratios: list[float] = []
    for frame_index, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise SceneAnalysisError(
                "circular_disk_not_found",
                f"no green disk proposal in frame {frame_index}",
            )
        contour = max(contours, key=cv2.contourArea)
        disk = np.zeros_like(green)
        cv2.drawContours(disk, [contour], -1, 255, thickness=-1)
        disk = cv2.erode(disk, erosion)
        ratio = float(np.mean(disk > 0))
        if ratio < minimum_disk_ratio:
            raise SceneAnalysisError(
                "circular_disk_not_found",
                f"green disk ratio {ratio:.4f} in frame {frame_index} "
                f"is below {minimum_disk_ratio:.4f}",
            )
        foreground = cv2.bitwise_and(cv2.bitwise_not(green), disk)
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, opening
        )
        masks.append(foreground)
        disk_ratios.append(ratio)
    return masks, {
        "backend": "green_disk_interior_color_contrast",
        "green_hsv_lower": lower.tolist(),
        "green_hsv_upper": upper.tolist(),
        "mean_disk_area_ratio": float(np.mean(disk_ratios)),
        "minimum_disk_area_ratio": float(np.min(disk_ratios)),
    }
