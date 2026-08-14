"""Typed contracts for frozen Dataset reference observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


class ObservationState(IntEnum):
    """Visibility state aligned to one reference-observation sample."""

    VISIBLE = 0
    OCCLUDED = 1
    OUT_OF_FRAME = 2
    UNRESOLVED = 3


@dataclass(frozen=True)
class TimelineSample:
    observation_index: int
    encoded_time_seconds: float
    physical_time_seconds: float
    source_frame_index: int
    source_time_seconds: float


@dataclass(frozen=True)
class ReferenceTimeline:
    sampling_rate_hz: float
    source_frame_count: int
    source_fps: float
    encoded_to_physical_speed: float
    samples: tuple[TimelineSample, ...]
    source_width: int | None = None
    source_height: int | None = None

    @property
    def source_duration_seconds(self) -> float:
        return (self.source_frame_count - 1) / self.source_fps


@dataclass(frozen=True)
class EntityObservation:
    object_id: str
    mask_id: str
    masks: np.ndarray
    centroid_xy: np.ndarray
    bbox_xyxy: np.ndarray
    area_pixels: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class ReferenceObservationBundle:
    case_id: str
    scene_id: str
    timeline: ReferenceTimeline
    entities: dict[str, EntityObservation]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


__all__ = [
    "EntityObservation",
    "ObservationState",
    "ReferenceObservationBundle",
    "ReferenceTimeline",
    "TimelineSample",
]
