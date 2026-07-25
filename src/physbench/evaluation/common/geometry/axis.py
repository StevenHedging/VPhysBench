from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisModel:
    origin_xy: np.ndarray
    direction_xy: np.ndarray
    normal_xy: np.ndarray
    explained_ratio: float

    def project(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centered = np.asarray(points_xy, dtype=np.float64) - self.origin_xy
        return centered @ self.direction_xy, centered @ self.normal_xy


def fit_axis(points_xy: np.ndarray, *, orient_with_time: bool = True) -> AxisModel:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("axis fitting requires at least two 2D points")
    origin = np.mean(points, axis=0)
    centered = points - origin
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if orient_with_time and np.dot(points[-1] - points[0], direction) < 0:
        direction = -direction
    normal = np.asarray([-direction[1], direction[0]])
    denominator = float(np.square(singular).sum())
    explained = (
        float(singular[0] ** 2 / denominator) if denominator > 0 else 0.0
    )
    return AxisModel(origin, direction, normal, explained)
