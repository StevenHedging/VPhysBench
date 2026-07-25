from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleModel:
    center_xy: np.ndarray
    radius_px: float
    radial_cv: float
    residual_rmse_px: float


def fit_circle(points_xy: np.ndarray) -> CircleModel:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("circle fitting requires at least three 2D points")
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points))])
    target = np.square(x) + np.square(y)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:2]
    radius_sq = float(solution[2] + np.dot(center, center))
    radius = float(np.sqrt(max(radius_sq, 0.0)))
    distances = np.linalg.norm(points - center, axis=1)
    residual = distances - radius
    return CircleModel(
        center_xy=center,
        radius_px=radius,
        radial_cv=float(np.std(distances) / max(np.mean(distances), 1e-12)),
        residual_rmse_px=float(np.sqrt(np.mean(np.square(residual)))),
    )
