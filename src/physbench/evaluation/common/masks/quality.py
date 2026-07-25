from __future__ import annotations

import cv2
import numpy as np


def mask_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference_binary = reference > 0
    prediction_binary = prediction > 0
    union = int(np.logical_or(reference_binary, prediction_binary).sum())
    if union == 0:
        return 0.0
    intersection = int(
        np.logical_and(reference_binary, prediction_binary).sum()
    )
    return float(intersection / union)


def largest_component(mask: np.ndarray, *, minimum_area: int = 1) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area:
        return np.zeros_like(binary, dtype=np.uint8)
    return ((labels == label).astype(np.uint8) * 255)


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"] <= 0:
        return None
    return np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def component_centroids(
    mask: np.ndarray,
    *,
    minimum_area: int,
    maximum_components: int | None = None,
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    components = [
        (
            np.asarray(centroids[label], dtype=np.float64),
            int(stats[label, cv2.CC_STAT_AREA]),
            ((labels == label).astype(np.uint8) * 255),
        )
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    components.sort(key=lambda item: item[1], reverse=True)
    return (
        components[:maximum_components]
        if maximum_components is not None
        else components
    )
