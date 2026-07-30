from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class VisibilityState(str, Enum):
    """Evaluator-side state of a physical entity at one video frame."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    NOT_YET_PRESENT = "not_yet_present"
    UNKNOWN = "unknown"


class ReferenceCapability(str, Enum):
    """What a case reference is allowed to supervise."""

    SAME_CASE_GT = "same_case_gt"
    PHYSICS_PARENT = "physics_parent"
    ANNOTATION_ONLY = "annotation_only"


class LifecyclePolicy(str, Enum):
    """Allowed scene lifecycle for an expected physical identity."""

    PERSISTENT = "persistent"
    MAY_ENTER = "may_enter"
    MAY_EXIT = "may_exit"
    MAY_ENTER_AND_EXIT = "may_enter_and_exit"


@dataclass(frozen=True)
class EntitySpec:
    """Persistent physical identity declared by a scene adapter."""

    entity_id: str
    role_id: str
    entity_class: str
    parts: tuple[str, ...] = ()
    exchangeability_group: str | None = None
    lifecycle: LifecyclePolicy = LifecyclePolicy.PERSISTENT
    anchor: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must be non-empty")
        if not self.role_id.strip():
            raise ValueError("role_id must be non-empty")
        if not self.entity_class.strip():
            raise ValueError("entity_class must be non-empty")
        lifecycle = (
            self.lifecycle
            if isinstance(self.lifecycle, LifecyclePolicy)
            else LifecyclePolicy(self.lifecycle)
        )
        object.__setattr__(self, "lifecycle", lifecycle)


@dataclass(frozen=True)
class ObjectTrack:
    """Scene-independent observations for one prediction or reference track."""

    track_id: str
    xy: np.ndarray
    observed: np.ndarray
    visibility: tuple[VisibilityState, ...]
    matched_entity_id: str | None = None
    areas_px2: np.ndarray | None = None
    confidence: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track_id.strip():
            raise ValueError("track_id must be non-empty")
        xy = np.asarray(self.xy, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        if xy.ndim != 2 or xy.shape[1] < 1:
            raise ValueError("xy must have shape [frames, dimensions]")
        if observed.shape != (len(xy),):
            raise ValueError("observed must have one value per frame")
        if len(self.visibility) != len(xy):
            raise ValueError("visibility must have one state per frame")
        normalized_visibility = tuple(
            state
            if isinstance(state, VisibilityState)
            else VisibilityState(state)
            for state in self.visibility
        )
        if np.any(observed) and not np.isfinite(xy[observed]).all():
            raise ValueError("observed positions must be finite")
        areas = self._optional_series(
            self.areas_px2,
            expected_length=len(xy),
            name="areas_px2",
            minimum=0.0,
        )
        confidence = self._optional_series(
            self.confidence,
            expected_length=len(xy),
            name="confidence",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "visibility", normalized_visibility)
        object.__setattr__(self, "areas_px2", areas)
        object.__setattr__(self, "confidence", confidence)

    @staticmethod
    def _optional_series(
        value: np.ndarray | None,
        *,
        expected_length: int,
        name: str,
        minimum: float,
        maximum: float | None = None,
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (expected_length,):
            raise ValueError(f"{name} must have one value per frame")
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.min()) < minimum:
            raise ValueError(f"{name} values must be >= {minimum:g}")
        if (
            maximum is not None
            and finite.size
            and float(finite.max()) > maximum
        ):
            raise ValueError(f"{name} values must be <= {maximum:g}")
        return array

    def exposure(self) -> float:
        """Visible, actually observed entity-time exposure."""
        visible = np.asarray(
            [state is VisibilityState.VISIBLE for state in self.visibility],
            dtype=bool,
        )
        return float(np.count_nonzero(self.observed & visible))


@dataclass(frozen=True)
class EntityMatch:
    """One one-to-one entity/track association at a single frame."""

    frame_index: int
    entity_id: str
    track_id: str
    localization_quality: float
    normalized_distance: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not self.entity_id.strip() or not self.track_id.strip():
            raise ValueError("entity_id and track_id must be non-empty")
        if not math.isfinite(float(self.localization_quality)):
            raise ValueError("localization_quality must be finite")
        if not 0.0 <= float(self.localization_quality) <= 1.0:
            raise ValueError("localization_quality must be in [0, 1]")
        if not math.isfinite(float(self.normalized_distance)):
            raise ValueError("normalized_distance must be finite")
        if float(self.normalized_distance) < 0.0:
            raise ValueError("normalized_distance must be non-negative")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("weight must be finite and positive")


def exposure_by_id(
    tracks: Sequence[ObjectTrack],
    *,
    use_entity_ids: bool,
) -> dict[str, float]:
    """Return observed exposure keyed by physical ID or track ID."""
    output: dict[str, float] = {}
    for track in tracks:
        key = track.matched_entity_id if use_entity_ids else track.track_id
        if key is None:
            continue
        if key in output:
            raise ValueError(f"duplicate exposure key: {key}")
        output[key] = track.exposure()
    return output
