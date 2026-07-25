from __future__ import annotations

import cv2
import numpy as np

from .axis import AxisModel


def rectify_axis_masks(
    masks: list[np.ndarray],
    *,
    axis: AxisModel,
    span_px: float,
    width: int = 512,
    height: int = 160,
    along_fraction: float = 0.8,
) -> list[np.ndarray]:
    """Map an independently observed motion axis to a canonical strip."""
    scale = float(width) * float(along_fraction) / max(float(span_px), 1e-9)
    direction = axis.direction_xy
    normal = axis.normal_xy
    origin = axis.origin_xy
    matrix = np.asarray(
        [
            [
                scale * direction[0],
                scale * direction[1],
                width / 2.0 - scale * float(np.dot(direction, origin)),
            ],
            [
                scale * normal[0],
                scale * normal[1],
                height / 2.0 - scale * float(np.dot(normal, origin)),
            ],
        ],
        dtype=np.float64,
    )
    return [
        cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for mask in masks
    ]


def rectify_circle_masks(
    masks: list[np.ndarray],
    *,
    center_xy: np.ndarray,
    radius_px: float,
    size: int = 320,
    canonical_radius_px: float = 120.0,
) -> list[np.ndarray]:
    scale = float(canonical_radius_px) / max(float(radius_px), 1e-9)
    center = np.asarray(center_xy, dtype=np.float64)
    matrix = np.asarray(
        [
            [scale, 0.0, size / 2.0 - scale * center[0]],
            [0.0, scale, size / 2.0 - scale * center[1]],
        ],
        dtype=np.float64,
    )
    return [
        cv2.warpAffine(
            mask,
            matrix,
            (size, size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for mask in masks
    ]
