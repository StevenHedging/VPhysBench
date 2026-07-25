from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import SceneAnalysisError
from ..masks.quality import mask_centroid


@dataclass(frozen=True)
class CentroidTrace:
    xy: np.ndarray
    area: np.ndarray
    valid: np.ndarray
    valid_ratio: float


def interpolate_trace(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if int(valid.sum()) < 2:
        raise SceneAnalysisError(
            "insufficient_valid_observations",
            "at least two valid observations are required for interpolation",
        )
    output = values.astype(np.float64, copy=True)
    indices = np.arange(len(values))
    if output.ndim == 1:
        output[~valid] = np.interp(indices[~valid], indices[valid], output[valid])
        return output
    for column in range(output.shape[1]):
        output[~valid, column] = np.interp(
            indices[~valid], indices[valid], output[valid, column]
        )
    return output


def extract_centroid_trace(
    masks: list[np.ndarray],
    *,
    minimum_area: int,
    maximum_area_ratio: float,
    minimum_valid_ratio: float,
) -> CentroidTrace:
    if not masks:
        raise SceneAnalysisError("empty_masks", "mask sequence is empty")
    frame_area = masks[0].shape[0] * masks[0].shape[1]
    maximum_area = int(round(frame_area * maximum_area_ratio))
    xy = np.full((len(masks), 2), np.nan, dtype=np.float64)
    area = np.zeros(len(masks), dtype=np.float64)
    valid = np.zeros(len(masks), dtype=bool)
    for index, mask in enumerate(masks):
        current_area = int(np.count_nonzero(mask))
        area[index] = current_area
        if current_area < minimum_area or current_area > maximum_area:
            continue
        centroid = mask_centroid(mask)
        if centroid is None:
            continue
        xy[index] = centroid
        valid[index] = True
    valid_ratio = float(np.mean(valid))
    if valid_ratio < minimum_valid_ratio or int(valid.sum()) < 3:
        raise SceneAnalysisError(
            "insufficient_valid_masks",
            f"valid masks {int(valid.sum())}/{len(valid)} ({valid_ratio:.3f}) "
            f"below required {minimum_valid_ratio:.3f}",
        )
    return CentroidTrace(
        xy=interpolate_trace(xy, valid),
        area=area,
        valid=valid,
        valid_ratio=valid_ratio,
    )
