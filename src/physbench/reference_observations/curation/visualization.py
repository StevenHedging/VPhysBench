"""Deterministic visual evidence for anchor and full-tube review."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


_EXISTING_COLOR = (0, 255, 0)
_CANDIDATE_COLOR = (255, 0, 255)
_OBJECT_COLORS = (
    (0, 255, 255),
    (255, 128, 0),
    (0, 128, 255),
    (255, 255, 0),
)


def _bgr_image(value: np.ndarray, *, label: str) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} must use uint8 HWC BGR layout")
    return np.array(image, copy=True)


def _binary_mask(value: np.ndarray, shape: tuple[int, int], *, label: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{label} shape {mask.shape} differs from image {shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _draw_boundary(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, lineType=cv2.LINE_8)


def _header(image: np.ndarray, lines: Sequence[str], *, height: int = 44) -> np.ndarray:
    canvas = np.full((image.shape[0] + height, image.shape[1], 3), 20, np.uint8)
    canvas[height:] = image
    for index, line in enumerate(lines[:2]):
        cv2.putText(
            canvas,
            line,
            (4, 16 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_8,
        )
    return canvas


def render_anchor_sheet(
    image: np.ndarray,
    *,
    existing_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> np.ndarray:
    rendered = _bgr_image(image, label="anchor image")
    height, width = rendered.shape[:2]
    if set(existing_masks) != set(candidate_masks):
        raise ValueError("existing and candidate anchor object sets differ")
    for object_id in existing_masks:
        existing = _binary_mask(
            existing_masks[object_id],
            (height, width),
            label=f"{object_id} existing mask",
        )
        candidate = _binary_mask(
            candidate_masks[object_id],
            (height, width),
            label=f"{object_id} candidate mask",
        )
        _draw_boundary(rendered, existing, _EXISTING_COLOR)
        _draw_boundary(rendered, candidate, _CANDIDATE_COLOR)
        ys, xs = np.where(existing > 0)
        if len(xs):
            cv2.putText(
                rendered,
                object_id,
                (int(xs.min()), max(12, int(ys.min()) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                _EXISTING_COLOR,
                1,
                cv2.LINE_8,
            )
    lines = (
        f"case={metadata.get('case_id', '')} scene={metadata.get('scene_id', '')}",
        "existing=green candidate=magenta",
    )
    return _header(rendered, lines)


def _render_observation_tile(
    frame: np.ndarray,
    *,
    masks_by_object: Mapping[str, np.ndarray],
    states_by_object: Mapping[str, np.ndarray],
    observation_index: int,
    source_index: int,
) -> np.ndarray:
    rendered = _bgr_image(frame, label="observation frame")
    height, width = rendered.shape[:2]
    if set(masks_by_object) != set(states_by_object):
        raise ValueError("mask and state object sets differ")
    state_parts: list[str] = []
    for object_index, object_id in enumerate(masks_by_object):
        tube = np.asarray(masks_by_object[object_id])
        states = np.asarray(states_by_object[object_id])
        if tube.ndim != 3 or tube.shape[1:] != (height, width):
            raise ValueError(f"{object_id} mask tube shape differs from frames")
        if states.shape != (tube.shape[0],):
            raise ValueError(f"{object_id} states differ from mask tube")
        if not 0 <= observation_index < tube.shape[0]:
            raise ValueError("observation index is outside mask tube")
        mask = _binary_mask(
            tube[observation_index],
            (height, width),
            label=f"{object_id} observation mask",
        )
        color = _OBJECT_COLORS[object_index % len(_OBJECT_COLORS)]
        _draw_boundary(rendered, mask, color, thickness=2)
        state_parts.append(f"{object_id}:{int(states[observation_index])}")
    return _header(
        rendered,
        (
            f"obs={observation_index} source={source_index}",
            " ".join(state_parts),
        ),
    )


def render_dense_event_sheet(
    frames: Sequence[np.ndarray],
    *,
    masks_by_object: Mapping[str, np.ndarray],
    states_by_object: Mapping[str, np.ndarray],
    observation_indices: Sequence[int],
    source_indices: Sequence[int],
    columns: int = 4,
) -> np.ndarray:
    if columns <= 0:
        raise ValueError("columns must be positive")
    if not observation_indices or len(observation_indices) != len(source_indices):
        raise ValueError("observation/source indices must be non-empty and aligned")
    if not frames:
        raise ValueError("frames must not be empty")
    tiles = [
        _render_observation_tile(
            frames[int(observation_index)],
            masks_by_object=masks_by_object,
            states_by_object=states_by_object,
            observation_index=int(observation_index),
            source_index=int(source_index),
        )
        for observation_index, source_index in zip(
            observation_indices,
            source_indices,
            strict=True,
        )
    ]
    tile_height, tile_width = tiles[0].shape[:2]
    row_count = int(math.ceil(len(tiles) / columns))
    canvas = np.zeros((row_count * tile_height, columns * tile_width, 3), np.uint8)
    for tile_index, tile in enumerate(tiles):
        if tile.shape != tiles[0].shape:
            raise ValueError("all review frames must have identical dimensions")
        row, column = divmod(tile_index, columns)
        canvas[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    return canvas


def render_contact_sheet(
    frames: Sequence[np.ndarray],
    *,
    masks_by_object: Mapping[str, np.ndarray],
    states_by_object: Mapping[str, np.ndarray],
    source_indices: Sequence[int],
    columns: int = 5,
) -> np.ndarray:
    if len(frames) != len(source_indices):
        raise ValueError("contact-sheet frames and source indices differ")
    return render_dense_event_sheet(
        frames,
        masks_by_object=masks_by_object,
        states_by_object=states_by_object,
        observation_indices=tuple(range(len(frames))),
        source_indices=source_indices,
        columns=columns,
    )


__all__ = [
    "render_anchor_sheet",
    "render_contact_sheet",
    "render_dense_event_sheet",
]
