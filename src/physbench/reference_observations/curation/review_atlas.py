"""Compact, deterministic pages for human review of case evidence."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


def _render_tile(
    label: str,
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{label} must use uint8 HWC BGR layout")
    header_height = min(28, max(18, height // 5))
    content_height = height - header_height
    if content_height <= 0:
        raise ValueError("tile height is too small for its label")
    scale = min(width / value.shape[1], content_height / value.shape[0])
    resized = cv2.resize(
        value,
        (
            max(1, int(round(value.shape[1] * scale))),
            max(1, int(round(value.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.zeros((height, width, 3), np.uint8)
    y0 = header_height + (content_height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.putText(
        tile,
        str(label),
        (5, min(header_height - 5, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def render_review_atlas_pages(
    entries: Sequence[tuple[str, np.ndarray]],
    *,
    columns: int = 2,
    rows: int = 3,
    tile_width: int = 720,
    tile_height: int = 480,
) -> tuple[np.ndarray, ...]:
    """Render labeled images in caller order into fixed-size review pages."""
    if not entries:
        raise ValueError("entries must not be empty")
    if min(columns, rows, tile_width, tile_height) <= 0:
        raise ValueError("atlas dimensions must be positive")
    page_size = columns * rows
    pages = []
    for offset in range(0, len(entries), page_size):
        canvas = np.zeros(
            (rows * tile_height, columns * tile_width, 3),
            np.uint8,
        )
        for index, (label, image) in enumerate(entries[offset : offset + page_size]):
            row, column = divmod(index, columns)
            tile = _render_tile(
                label,
                image,
                width=tile_width,
                height=tile_height,
            )
            canvas[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = tile
        pages.append(canvas)
    expected_pages = int(math.ceil(len(entries) / page_size))
    if len(pages) != expected_pages:
        raise AssertionError("review atlas pagination is inconsistent")
    return tuple(pages)


__all__ = ["render_review_atlas_pages"]
