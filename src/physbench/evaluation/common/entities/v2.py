"""Capability-aware, open-world entity evaluation primitives.

This module is deliberately additive.  The collision evaluator's
``open_world_v1.2`` observer and scoring contracts remain unchanged; scene
adapters can opt into this module one at a time.

The central distinction from the v1 adapter is that an expected entity has
three independent evidence channels:

* existence says when the entity is expected and when cardinality is
  observable;
* localization says when future pixel position is a valid reference;
* association says when a persistent identity can be audited.

In particular, a physics-parent reference can supervise a persistent object
count without borrowing the parent's future pixel trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (
    Callable,
    Mapping,
    Sequence,
)

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import (
    EntityMatch,
    EntitySpec,
    LifecyclePolicy,
    ObjectTrack,
    ReferenceCapability,
    VisibilityState,
)
from .matching import FrozenAssignment, freeze_initial_assignment
from .observer import (
    OpenWorldObservation,
    ObjectDetection,
)
from .scoring import (
    EntityIntegrityScore,
    GatedCaseComposition,
    PositionComparison,
    compare_positions,
    compose_gated_case_score,
    score_entity_integrity,
)
from .timeline import CommonTimeGrid


OPEN_WORLD_V2_ID = "open_world_v2"
OPEN_WORLD_V2_VERSION = "2.0"
OPEN_WORLD_V2_DELETION_RESISTANCE_POWER = 3.0


def _readonly_bool(
    value: Sequence[bool] | np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have one value per time sample")
    if array.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values")
    output = np.array(array, dtype=bool, copy=True)
    output.setflags(write=False)
    return output


def _readonly_float(
    value: Sequence[float] | np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
    allow_nan: bool,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        qualifier = "finite or NaN" if allow_nan else "finite"
        raise ValueError(f"{name} must contain only {qualifier} values")
    output = np.array(array, dtype=np.float64, copy=True)
    output.setflags(write=False)
    return output


def _normalize_optional_masks(
    masks: Sequence[np.ndarray | None] | None,
    *,
    length: int,
) -> tuple[np.ndarray | None, ...]:
    if masks is None:
        return (None,) * length
    if len(masks) != length:
        raise ValueError("reference_masks must have one value per time sample")
    shape: tuple[int, int] | None = None
    output: list[np.ndarray | None] = []
    for mask in masks:
        if mask is None:
            output.append(None)
            continue
        binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        if binary.ndim != 2:
            raise ValueError("reference masks must be two-dimensional")
        if shape is None:
            shape = binary.shape
        elif binary.shape != shape:
            raise ValueError("reference masks must have a common shape")
        binary.setflags(write=False)
        output.append(binary)
    return tuple(output)


@dataclass(frozen=True)
class LifecycleResolution:
    """Expected in-frame presence, resolved without prediction evidence."""

    expected_exists: np.ndarray
    existence_supervised: np.ndarray
    source: str

    def __post_init__(self) -> None:
        expected = np.asarray(self.expected_exists)
        supervised = np.asarray(self.existence_supervised)
        if expected.ndim != 1:
            raise ValueError("expected_exists must be one-dimensional")
        object.__setattr__(
            self,
            "expected_exists",
            _readonly_bool(
                expected,
                length=len(expected),
                name="expected_exists",
            ),
        )
        object.__setattr__(
            self,
            "existence_supervised",
            _readonly_bool(
                supervised,
                length=len(expected),
                name="existence_supervised",
            ),
        )
        if not str(self.source).strip():
            raise ValueError("lifecycle source must be non-empty")


@dataclass(frozen=True)
class ExpectedPositionSample:
    """One scene-provided reference position on the common time grid."""

    xy: np.ndarray
    area_px2: float = 0.0
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        if xy.ndim != 1 or xy.size < 1 or not np.isfinite(xy).all():
            raise ValueError("expected position must be a finite vector")
        area = float(self.area_px2)
        if not math.isfinite(area) or area < 0.0:
            raise ValueError("area_px2 must be finite and non-negative")
        normalized_xy = np.array(xy, copy=True)
        normalized_xy.setflags(write=False)
        object.__setattr__(self, "xy", normalized_xy)
        object.__setattr__(self, "area_px2", area)
        if self.mask is not None:
            mask = np.where(np.asarray(self.mask) > 0, 255, 0).astype(
                np.uint8
            )
            if mask.ndim != 2:
                raise ValueError("expected-position mask must be 2-D")
            mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)


ExpectedPositionCallback = Callable[
    [EntitySpec, int, float],
    ExpectedPositionSample | Sequence[float] | np.ndarray | None,
]


@dataclass(frozen=True)
class ExpectedEntityTimeline:
    """Capability-aware supervision contract for one physical identity."""

    entity: EntitySpec
    capability: ReferenceCapability
    expected_exists: np.ndarray
    existence_supervised: np.ndarray
    localization_supervised: np.ndarray
    association_supervised: np.ndarray
    reference_xy: np.ndarray
    reference_area_px2: np.ndarray
    reference_masks: tuple[np.ndarray | None, ...] = ()
    lifecycle_source: str = "declaration"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        capability = (
            self.capability
            if isinstance(self.capability, ReferenceCapability)
            else ReferenceCapability(self.capability)
        )
        expected = np.asarray(self.expected_exists)
        if expected.ndim != 1:
            raise ValueError("expected_exists must be one-dimensional")
        frame_count = len(expected)
        expected = _readonly_bool(
            expected,
            length=frame_count,
            name="expected_exists",
        )
        existence = _readonly_bool(
            self.existence_supervised,
            length=frame_count,
            name="existence_supervised",
        )
        localization = _readonly_bool(
            self.localization_supervised,
            length=frame_count,
            name="localization_supervised",
        )
        association = _readonly_bool(
            self.association_supervised,
            length=frame_count,
            name="association_supervised",
        )
        if np.any(localization & ~(expected & existence)):
            raise ValueError(
                "localization supervision requires supervised existence"
            )
        if np.any(association & ~(expected & existence)):
            raise ValueError(
                "association supervision requires supervised existence"
            )
        xy = np.asarray(self.reference_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] != frame_count or xy.shape[1] < 1:
            raise ValueError(
                "reference_xy must have shape [time samples, dimensions]"
            )
        xy = _readonly_float(
            xy,
            shape=xy.shape,
            name="reference_xy",
            allow_nan=True,
        )
        if np.any(localization) and not np.isfinite(xy[localization]).all():
            raise ValueError(
                "localization-supervised positions must be finite"
            )
        area = _readonly_float(
            self.reference_area_px2,
            shape=(frame_count,),
            name="reference_area_px2",
            allow_nan=True,
        )
        finite_area = area[np.isfinite(area)]
        if finite_area.size and float(finite_area.min()) < 0.0:
            raise ValueError("reference areas must be non-negative")
        masks = _normalize_optional_masks(
            self.reference_masks or None,
            length=frame_count,
        )
        if capability is ReferenceCapability.PHYSICS_PARENT:
            future = np.arange(frame_count) > 0
            if np.any(localization & future):
                raise ValueError(
                    "physics-parent timelines cannot supervise future "
                    "pixel localization"
                )
        if not str(self.lifecycle_source).strip():
            raise ValueError("lifecycle_source must be non-empty")
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "expected_exists", expected)
        object.__setattr__(self, "existence_supervised", existence)
        object.__setattr__(self, "localization_supervised", localization)
        object.__setattr__(self, "association_supervised", association)
        object.__setattr__(self, "reference_xy", xy)
        object.__setattr__(self, "reference_area_px2", area)
        object.__setattr__(self, "reference_masks", masks)

    @property
    def entity_id(self) -> str:
        return self.entity.entity_id

    @property
    def frame_count(self) -> int:
        return len(self.expected_exists)

    def expected_exposure(self, time_grid: CommonTimeGrid) -> float:
        if len(time_grid.times_s) != self.frame_count:
            raise ValueError("timeline length differs from the time grid")
        mask = self.expected_exists & self.existence_supervised
        return float(np.dot(mask, time_grid.cell_weights_s))

    def association_exposure(self, time_grid: CommonTimeGrid) -> float:
        if len(time_grid.times_s) != self.frame_count:
            raise ValueError("timeline length differs from the time grid")
        return float(
            np.dot(self.association_supervised, time_grid.cell_weights_s)
        )

    def to_dict(self) -> dict[str, object]:
        xy = [
            [
                float(value) if math.isfinite(float(value)) else None
                for value in row
            ]
            for row in self.reference_xy
        ]
        area = [
            float(value) if math.isfinite(float(value)) else None
            for value in self.reference_area_px2
        ]
        return {
            "entity_id": self.entity_id,
            "entity_class": self.entity.entity_class,
            "lifecycle": self.entity.lifecycle.value,
            "capability": self.capability.value,
            "expected_exists": self.expected_exists.tolist(),
            "existence_supervised": self.existence_supervised.tolist(),
            "localization_supervised": (
                self.localization_supervised.tolist()
            ),
            "association_supervised": (
                self.association_supervised.tolist()
            ),
            "reference_xy": xy,
            "reference_area_px2": area,
            "reference_mask_available": [
                mask is not None for mask in self.reference_masks
            ],
            "lifecycle_source": self.lifecycle_source,
            "metadata": dict(self.metadata),
        }


def _validate_lifecycle_topology(
    mask: np.ndarray,
    lifecycle: LifecyclePolicy,
) -> None:
    values = np.asarray(mask, dtype=bool)
    transitions = np.diff(values.astype(np.int8))
    rises = int(np.count_nonzero(transitions == 1))
    falls = int(np.count_nonzero(transitions == -1))
    if lifecycle is LifecyclePolicy.PERSISTENT:
        if values.size and not bool(np.all(values)):
            raise ValueError("PERSISTENT lifecycle must exist for all samples")
    elif lifecycle is LifecyclePolicy.MAY_EXIT:
        if rises or falls > 1:
            raise ValueError("MAY_EXIT lifecycle may contain one final exit")
        if values.size and not bool(values[0]):
            raise ValueError("MAY_EXIT lifecycle must start present")
    elif lifecycle is LifecyclePolicy.MAY_ENTER:
        if falls or rises > 1:
            raise ValueError("MAY_ENTER lifecycle may contain one entry")
        if values.size and bool(values[0]) and rises:
            raise ValueError("MAY_ENTER lifecycle cannot re-enter")
    elif lifecycle is LifecyclePolicy.MAY_ENTER_AND_EXIT:
        if rises > 1 or falls > 1:
            raise ValueError(
                "MAY_ENTER_AND_EXIT must have one contiguous presence window"
            )
        present = np.flatnonzero(values)
        if present.size and not bool(np.all(values[present[0] : present[-1] + 1])):
            raise ValueError(
                "MAY_ENTER_AND_EXIT presence must be contiguous"
            )


def _reference_lifecycle_mask(
    entity: EntitySpec,
    reference_track: ObjectTrack,
    *,
    condition_present: bool,
) -> np.ndarray:
    lifecycle = entity.lifecycle
    frame_count = len(reference_track.visibility)
    if lifecycle is LifecyclePolicy.PERSISTENT:
        return np.ones(frame_count, dtype=bool)

    visible = {
        VisibilityState.VISIBLE,
        VisibilityState.OCCLUDED,
    }
    absent = {
        VisibilityState.OUT_OF_FRAME,
        VisibilityState.NOT_YET_PRESENT,
    }
    output = np.zeros(frame_count, dtype=bool)
    state = bool(condition_present)
    entered = state
    exited = False
    for index, visibility in enumerate(reference_track.visibility):
        if visibility in visible:
            if lifecycle is LifecyclePolicy.MAY_EXIT and exited:
                raise ValueError("reference re-enters after a MAY_EXIT event")
            state = True
            entered = True
        elif visibility in absent:
            if lifecycle is LifecyclePolicy.MAY_ENTER:
                state = entered
            elif lifecycle is LifecyclePolicy.MAY_EXIT:
                if entered:
                    state = False
                    exited = True
            else:
                state = False
                if entered:
                    exited = True
        # UNKNOWN never turns a detector miss into a lifecycle transition.
        output[index] = state
    _validate_lifecycle_topology(output, lifecycle)
    return output


def resolve_lifecycle(
    entity: EntitySpec,
    *,
    time_grid: CommonTimeGrid,
    capability: ReferenceCapability,
    reference_track: ObjectTrack | None = None,
    declared_expected_mask: Sequence[bool] | np.ndarray | None = None,
    condition_present: bool = True,
) -> LifecycleResolution:
    """Resolve expected presence from declaration/reference only.

    For a non-persistent lifecycle without reference or annotation evidence,
    only the condition sample is supervised.  This avoids both inventing an
    exit time and letting prediction disappearance choose its own denominator.
    """

    capability = (
        capability
        if isinstance(capability, ReferenceCapability)
        else ReferenceCapability(capability)
    )
    frame_count = len(time_grid.times_s)
    if reference_track is not None and len(reference_track.xy) != frame_count:
        raise ValueError("reference track length differs from the time grid")
    if declared_expected_mask is not None:
        expected = _readonly_bool(
            declared_expected_mask,
            length=frame_count,
            name="declared_expected_mask",
        )
        _validate_lifecycle_topology(expected, entity.lifecycle)
        supervised = np.ones(frame_count, dtype=bool)
        source = "declared_lifecycle_mask"
    elif entity.lifecycle is LifecyclePolicy.PERSISTENT:
        expected = np.ones(frame_count, dtype=bool)
        supervised = np.ones(frame_count, dtype=bool)
        source = "persistent_declaration"
    elif (
        reference_track is not None
        and capability is not ReferenceCapability.PHYSICS_PARENT
    ):
        expected = _reference_lifecycle_mask(
            entity,
            reference_track,
            condition_present=condition_present,
        )
        supervised = np.ones(frame_count, dtype=bool)
        source = f"{capability.value}_reference_visibility"
    else:
        expected = np.full(frame_count, bool(condition_present), dtype=bool)
        supervised = np.zeros(frame_count, dtype=bool)
        if frame_count:
            supervised[0] = True
        source = "condition_anchor_only"
    return LifecycleResolution(
        expected_exists=expected,
        existence_supervised=supervised,
        source=source,
    )


def _coerce_position_sample(
    value: ExpectedPositionSample | Sequence[float] | np.ndarray | None,
) -> ExpectedPositionSample | None:
    if value is None:
        return None
    if isinstance(value, ExpectedPositionSample):
        return value
    return ExpectedPositionSample(xy=np.asarray(value, dtype=np.float64))


def resolve_expected_entity_timeline(
    entity: EntitySpec,
    *,
    time_grid: CommonTimeGrid,
    capability: ReferenceCapability,
    reference_track: ObjectTrack | None = None,
    declared_expected_mask: Sequence[bool] | np.ndarray | None = None,
    existence_supervised_mask: Sequence[bool] | np.ndarray | None = None,
    association_supervised_mask: Sequence[bool] | np.ndarray | None = None,
    condition_position: ExpectedPositionSample | Sequence[float] | None = None,
    condition_present: bool = True,
    expected_position_callback: ExpectedPositionCallback | None = None,
    condition_identity_supervised: bool = False,
    reference_masks: Sequence[np.ndarray | None] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ExpectedEntityTimeline:
    """Build a three-channel timeline without consulting prediction pixels."""

    capability = (
        capability
        if isinstance(capability, ReferenceCapability)
        else ReferenceCapability(capability)
    )
    frame_count = len(time_grid.times_s)
    lifecycle = resolve_lifecycle(
        entity,
        time_grid=time_grid,
        capability=capability,
        reference_track=reference_track,
        declared_expected_mask=declared_expected_mask,
        condition_present=condition_present,
    )
    existence = (
        lifecycle.existence_supervised
        if existence_supervised_mask is None
        else _readonly_bool(
            existence_supervised_mask,
            length=frame_count,
            name="existence_supervised_mask",
        )
    )
    condition = _coerce_position_sample(condition_position)
    dimensions = (
        int(reference_track.xy.shape[1])
        if reference_track is not None
        else (len(condition.xy) if condition is not None else 2)
    )
    xy = np.full((frame_count, dimensions), np.nan, dtype=np.float64)
    area = np.full(frame_count, np.nan, dtype=np.float64)
    masks: list[np.ndarray | None] = list(
        _normalize_optional_masks(reference_masks, length=frame_count)
        if reference_masks is not None
        else (None,) * frame_count
    )
    localization = np.zeros(frame_count, dtype=bool)

    if (
        capability is ReferenceCapability.SAME_CASE_GT
        and reference_track is not None
    ):
        eligible = (
            reference_track.localization_eligible
            & lifecycle.expected_exists
            & existence
        )
        finite = np.isfinite(reference_track.xy).all(axis=1)
        localization = eligible & finite
        xy[:] = reference_track.xy
        if reference_track.areas_px2 is not None:
            area[:] = reference_track.areas_px2

    if expected_position_callback is not None:
        for index, time_s in enumerate(time_grid.times_s.tolist()):
            # A physics parent may contribute the condition anchor, never its
            # future pixel trajectory.
            if (
                capability is ReferenceCapability.PHYSICS_PARENT
                and index > 0
            ):
                continue
            sample = _coerce_position_sample(
                expected_position_callback(entity, index, float(time_s))
            )
            if sample is None:
                continue
            if sample.xy.shape != (dimensions,):
                if (
                    reference_track is None
                    and not np.isfinite(xy).any()
                ):
                    dimensions = len(sample.xy)
                    xy = np.full(
                        (frame_count, dimensions),
                        np.nan,
                        dtype=np.float64,
                    )
                else:
                    raise ValueError(
                        "position callback changed coordinate dimensions"
                    )
            xy[index] = sample.xy
            area[index] = sample.area_px2
            if sample.mask is not None:
                masks[index] = sample.mask
            localization[index] = bool(
                lifecycle.expected_exists[index] and existence[index]
            )

    if condition is not None and frame_count:
        if condition.xy.shape != (dimensions,):
            raise ValueError(
                "condition position dimensions differ from reference"
            )
        xy[0] = condition.xy
        area[0] = condition.area_px2
        if condition.mask is not None:
            masks[0] = condition.mask
        localization[0] = bool(
            lifecycle.expected_exists[0] and existence[0]
        )

    if capability is ReferenceCapability.PHYSICS_PARENT and frame_count > 1:
        localization[1:] = False
        xy[1:] = np.nan
        area[1:] = np.nan
        for index in range(1, frame_count):
            masks[index] = None

    if association_supervised_mask is not None:
        association = _readonly_bool(
            association_supervised_mask,
            length=frame_count,
            name="association_supervised_mask",
        )
    elif capability is ReferenceCapability.SAME_CASE_GT:
        association = np.array(localization, copy=True)
        if reference_track is not None:
            association &= reference_track.association_eligible
    elif condition_identity_supervised:
        association = lifecycle.expected_exists & existence
    else:
        association = np.zeros(frame_count, dtype=bool)
    if (
        capability is ReferenceCapability.PHYSICS_PARENT
        and np.any(association[1:])
        and not condition_identity_supervised
    ):
        raise ValueError(
            "physics-parent future identity supervision requires a "
            "condition-frozen identity hook"
        )
    return ExpectedEntityTimeline(
        entity=entity,
        capability=capability,
        expected_exists=lifecycle.expected_exists,
        existence_supervised=existence,
        localization_supervised=localization,
        association_supervised=association,
        reference_xy=xy,
        reference_area_px2=area,
        reference_masks=tuple(masks),
        lifecycle_source=lifecycle.source,
        metadata={} if metadata is None else metadata,
    )


ConditionIdentityCostHook = Callable[[EntitySpec, str], float]


def freeze_condition_identity(
    entities: Sequence[EntitySpec],
    condition_track_ids: Sequence[str],
    *,
    identity_cost_hook: ConditionIdentityCostHook,
    maximum_match_cost: float,
    unmatched_entity_cost: float = 1.0,
    unmatched_track_cost: float = 1.0,
    maximum_entities: int = 16,
) -> FrozenAssignment:
    """Freeze IDs from condition/initial anchors via a scene cost hook.

    The hook receives only an entity declaration and an initial-window track
    ID.  Scene adapters close over condition appearance/geometry; this API
    intentionally supplies no future prediction samples.
    """

    entity_values = list(entities)
    track_values = [str(value) for value in condition_track_ids]
    costs = np.empty(
        (len(entity_values), len(track_values)),
        dtype=np.float64,
    )
    for entity_index, entity in enumerate(entity_values):
        for track_index, track_id in enumerate(track_values):
            raw = float(identity_cost_hook(entity, track_id))
            if math.isnan(raw) or raw < 0.0:
                raise ValueError(
                    "identity_cost_hook must return non-negative costs "
                    "or positive infinity"
                )
            costs[entity_index, track_index] = raw
    return freeze_initial_assignment(
        entity_values,
        track_values,
        costs,
        maximum_match_cost=maximum_match_cost,
        unmatched_entity_cost=unmatched_entity_cost,
        unmatched_track_cost=unmatched_track_cost,
        maximum_entities=maximum_entities,
    )


@dataclass(frozen=True)
class PositionEvidence:
    """Scene-specific localization output used by null assignment."""

    score: float
    normalized_distance: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        distance = float(self.normalized_distance)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("position score must be finite and in [0, 1]")
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(
                "normalized position distance must be finite and non-negative"
            )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "normalized_distance", distance)


PositionSimilarityCallback = Callable[
    [ExpectedEntityTimeline, ObjectDetection, int, float],
    PositionEvidence | PositionComparison,
]


@dataclass(frozen=True)
class ObjectCentricComparisonV2:
    """Open-world comparison plus JSON and pixel-level audit artifacts."""

    integrity: EntityIntegrityScore
    matches: tuple[EntityMatch, ...]
    per_frame: tuple[Mapping[str, object], ...]
    reference_exposure: Mapping[str, float]
    prediction_exposure: Mapping[str, float]
    matched_masks: tuple[np.ndarray | None, ...]
    matched_mask_iou: tuple[float | None, ...]
    raw_integrity_gate: float | None = None
    deletion_resistant_gate_ceiling: float | None = None
    failed: bool = False
    failure_reason: str | None = None
    protocol_id: str = OPEN_WORLD_V2_ID
    protocol_version: str = OPEN_WORLD_V2_VERSION

    def __post_init__(self) -> None:
        if len(self.matched_masks) != len(self.per_frame):
            raise ValueError(
                "matched_masks must have one value per audit frame"
            )
        if len(self.matched_mask_iou) != len(self.per_frame):
            raise ValueError(
                "matched_mask_iou must have one value per audit frame"
            )
        masks = _normalize_optional_masks(
            self.matched_masks,
            length=len(self.per_frame),
        )
        iou: list[float | None] = []
        for value in self.matched_mask_iou:
            if value is None:
                iou.append(None)
                continue
            score = float(value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("matched-mask IoU must be in [0, 1]")
            iou.append(score)
        if self.failed and not str(self.failure_reason or "").strip():
            raise ValueError("failed comparisons require a failure reason")
        for name in (
            "raw_integrity_gate",
            "deletion_resistant_gate_ceiling",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        object.__setattr__(self, "matched_masks", masks)
        object.__setattr__(self, "matched_mask_iou", tuple(iou))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready artifact; full masks stay in ``matched_masks``."""

        mask_area = [
            None if mask is None else int(np.count_nonzero(mask))
            for mask in self.matched_masks
        ]
        return {
            "protocol": {
                "id": self.protocol_id,
                "version": self.protocol_version,
            },
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "integrity": self.integrity.to_dict(),
            "integrity_gate_policy": {
                "formula": (
                    "min(raw_presence_times_association,"
                    "exposure_recall_cubed)"
                ),
                "deletion_resistance_power": (
                    OPEN_WORLD_V2_DELETION_RESISTANCE_POWER
                ),
                "raw_integrity_gate": self.raw_integrity_gate,
                "deletion_resistant_gate_ceiling": (
                    self.deletion_resistant_gate_ceiling
                ),
                "effective_integrity_gate": (
                    self.integrity.integrity_gate
                ),
            },
            "reference_exposure": dict(self.reference_exposure),
            "prediction_exposure": dict(self.prediction_exposure),
            "matches": [
                {
                    "frame": match.frame_index,
                    "entity_id": match.entity_id,
                    "prediction_track_id": match.track_id,
                    "position_score": match.localization_quality,
                    "normalized_distance": match.normalized_distance,
                    "weight_s": match.weight,
                    "association_eligible": (
                        match.association_eligible
                    ),
                }
                for match in self.matches
            ],
            "per_frame": [dict(value) for value in self.per_frame],
            "matched_mask_area_px2": mask_area,
            "matched_mask_iou": list(self.matched_mask_iou),
        }


def _observation_index(
    observation: OpenWorldObservation,
    *,
    frame_count: int,
) -> tuple[
    dict[tuple[int, str], ObjectDetection],
    dict[str, float],
    dict[str, str],
]:
    index: dict[tuple[int, str], ObjectDetection] = {}
    weights: dict[str, float] = {}
    classes: dict[str, str] = {}
    for track in observation.tracks:
        weight = float(track.formal_exposure_weight)
        if not math.isfinite(weight):
            raise ValueError("prediction formal exposure must be finite")
        weights[track.track_id] = float(np.clip(weight, 0.0, 1.0))
        classes[track.track_id] = track.entity_class
        for detection in track.detections:
            if detection.frame_index >= frame_count:
                raise ValueError(
                    "prediction detection lies outside the time grid"
                )
            key = detection.frame_index, track.track_id
            if key in index:
                raise ValueError(
                    "prediction track has duplicate frame observations"
                )
            index[key] = detection
    return index, weights, classes


def build_matched_masks(
    matches: Sequence[EntityMatch],
    prediction_observation: OpenWorldObservation,
    *,
    frame_count: int,
) -> tuple[np.ndarray | None, ...]:
    """Union matched prediction instance masks at every time sample."""

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    detections, _, _ = _observation_index(
        prediction_observation,
        frame_count=frame_count,
    )
    by_frame: list[list[np.ndarray]] = [[] for _ in range(frame_count)]
    for match in matches:
        if match.frame_index >= frame_count:
            raise ValueError("match lies outside the requested frame count")
        detection = detections.get((match.frame_index, match.track_id))
        if detection is None:
            raise ValueError("match refers to an absent prediction detection")
        if detection.mask is not None:
            by_frame[match.frame_index].append(detection.mask)
    output: list[np.ndarray | None] = []
    for values in by_frame:
        if not values:
            output.append(None)
            continue
        shapes = {value.shape for value in values}
        if len(shapes) != 1:
            raise ValueError("matched masks in one frame have mixed shapes")
        shape = next(iter(shapes))
        union = np.zeros(shape, dtype=np.uint8)
        for mask in values:
            union = np.maximum(union, np.asarray(mask, dtype=np.uint8))
        union.setflags(write=False)
        output.append(union)
    return tuple(output)


def build_reference_union_masks(
    expected_timelines: Sequence[ExpectedEntityTimeline],
) -> tuple[np.ndarray | None, ...]:
    """Union reference subject masks without inventing unavailable pixels."""

    timelines = list(expected_timelines)
    if not timelines:
        return ()
    frame_count = timelines[0].frame_count
    if any(value.frame_count != frame_count for value in timelines):
        raise ValueError("expected timelines must have equal lengths")
    output: list[np.ndarray | None] = []
    for frame_index in range(frame_count):
        masks = [
            value.reference_masks[frame_index]
            for value in timelines
            if value.expected_exists[frame_index]
            and value.existence_supervised[frame_index]
            and value.reference_masks[frame_index] is not None
        ]
        if not masks:
            output.append(None)
            continue
        shapes = {mask.shape for mask in masks if mask is not None}
        if len(shapes) != 1:
            raise ValueError("reference masks in one frame have mixed shapes")
        shape = next(iter(shapes))
        union = np.zeros(shape, dtype=np.uint8)
        for mask in masks:
            union = np.maximum(union, np.asarray(mask, dtype=np.uint8))
        union.setflags(write=False)
        output.append(union)
    return tuple(output)


def matched_mask_iou_curve(
    reference_masks: Sequence[np.ndarray | None],
    prediction_masks: Sequence[np.ndarray | None],
) -> tuple[float | None, ...]:
    """Compute the Jensen-style physical-subject IoU curve when available."""

    if len(reference_masks) != len(prediction_masks):
        raise ValueError("mask timelines must have equal lengths")
    output: list[float | None] = []
    for reference, prediction in zip(reference_masks, prediction_masks):
        if reference is None or prediction is None:
            output.append(None)
            continue
        if reference.shape != prediction.shape:
            # Resolution alignment is a scene-adapter responsibility.  A
            # visualization-only mismatch must not invalidate entity scoring.
            output.append(None)
            continue
        ref = np.asarray(reference) > 0
        pred = np.asarray(prediction) > 0
        intersection = int(np.count_nonzero(ref & pred))
        union = int(np.count_nonzero(ref | pred))
        output.append(1.0 if union == 0 else intersection / union)
    return tuple(output)


def _position_evidence(
    timeline: ExpectedEntityTimeline,
    detection: ObjectDetection,
    *,
    frame_index: int,
    frame_diagonal_px: float,
    callback: PositionSimilarityCallback | None,
) -> PositionEvidence:
    if callback is not None:
        raw = callback(
            timeline,
            detection,
            frame_index,
            frame_diagonal_px,
        )
        if isinstance(raw, PositionEvidence):
            return raw
        if isinstance(raw, PositionComparison):
            return PositionEvidence(
                score=raw.score,
                normalized_distance=raw.normalized_distance,
                diagnostics={
                    "centroid_distance_px": raw.centroid_distance_px,
                    "scale_px": raw.scale_px,
                    "kernel": raw.kernel,
                },
            )
        raise TypeError(
            "position_similarity_callback must return PositionEvidence "
            "or PositionComparison"
        )
    area = timeline.reference_area_px2[frame_index]
    comparison = compare_positions(
        timeline.reference_xy[frame_index],
        detection.xy,
        reference_area_px2=(
            float(area) if np.isfinite(area) else 0.0
        ),
        frame_diagonal_px=frame_diagonal_px,
    )
    return PositionEvidence(
        score=comparison.score,
        normalized_distance=comparison.normalized_distance,
        diagnostics={
            "centroid_distance_px": comparison.centroid_distance_px,
            "scale_px": comparison.scale_px,
            "kernel": comparison.kernel,
        },
    )


def _matching_row(
    *,
    timeline: ExpectedEntityTimeline,
    detection: ObjectDetection,
    track_id: str,
    evidence: PositionEvidence,
    formal_weight: float,
    time_weight_s: float,
    association_eligible: bool,
    frozen: bool,
) -> tuple[EntityMatch | None, dict[str, object]]:
    weight = time_weight_s * formal_weight
    match = (
        EntityMatch(
            frame_index=detection.frame_index,
            entity_id=timeline.entity_id,
            track_id=track_id,
            localization_quality=evidence.score,
            normalized_distance=evidence.normalized_distance,
            weight=weight,
            association_eligible=association_eligible,
        )
        if weight > 0.0
        else None
    )
    row = {
        "entity_id": timeline.entity_id,
        "prediction_track_id": track_id,
        "position_score": evidence.score,
        "normalized_distance": evidence.normalized_distance,
        "formal_exposure_weight": formal_weight,
        "weight_s": weight,
        "localization_supervised": bool(
            timeline.localization_supervised[detection.frame_index]
        ),
        "association_eligible": association_eligible,
        "frozen_identity": frozen,
        "position_diagnostics": dict(evidence.diagnostics),
    }
    return match, row


def _compatible(
    timeline: ExpectedEntityTimeline,
    prediction_class: str,
) -> bool:
    return (
        not timeline.entity.entity_class
        or not prediction_class
        or timeline.entity.entity_class == prediction_class
    )


def _frame_matches(
    timelines: Sequence[ExpectedEntityTimeline],
    *,
    frame_index: int,
    time_weight_s: float,
    detections: Mapping[tuple[int, str], ObjectDetection],
    track_weights: Mapping[str, float],
    track_classes: Mapping[str, str],
    frozen_identity: FrozenAssignment | None,
    minimum_match_position_similarity: float,
    frame_diagonal_px: float,
    position_similarity_callback: PositionSimilarityCallback | None,
) -> tuple[
    list[EntityMatch],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    active = [
        timeline
        for timeline in timelines
        if timeline.expected_exists[frame_index]
        and timeline.existence_supervised[frame_index]
    ]
    available = {
        track_id: detection
        for (index, track_id), detection in detections.items()
        if index == frame_index and track_weights[track_id] > 0.0
    }
    matched: list[EntityMatch] = []
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    used_entities: set[str] = set()
    used_tracks: set[str] = set()
    locked_entities: set[str] = set()
    reserved_tracks: set[str] = set()

    if frozen_identity is not None:
        reserved_tracks = {
            track_id
            for track_id in frozen_identity.entity_to_track.values()
            if track_id is not None
        }
        for timeline in active:
            if timeline.entity_id not in frozen_identity.entity_to_track:
                continue
            locked_entities.add(timeline.entity_id)
            track_id = frozen_identity.entity_to_track[timeline.entity_id]
            if track_id is None:
                continue
            detection = available.get(track_id)
            if detection is None:
                continue
            if not _compatible(timeline, track_classes[track_id]):
                rejected.append(
                    {
                        "entity_id": timeline.entity_id,
                        "prediction_track_id": track_id,
                        "rejection_reason": "entity_class_mismatch",
                        "frozen_identity": True,
                    }
                )
                continue
            if timeline.localization_supervised[frame_index]:
                evidence = _position_evidence(
                    timeline,
                    detection,
                    frame_index=frame_index,
                    frame_diagonal_px=frame_diagonal_px,
                    callback=position_similarity_callback,
                )
            else:
                evidence = PositionEvidence(
                    score=1.0,
                    normalized_distance=0.0,
                    diagnostics={"channel": "existence_only"},
                )
            if evidence.score < minimum_match_position_similarity:
                rejected.append(
                    {
                        "entity_id": timeline.entity_id,
                        "prediction_track_id": track_id,
                        "position_score": evidence.score,
                        "normalized_distance": (
                            evidence.normalized_distance
                        ),
                        "rejection_reason": (
                            "position_similarity_below_minimum"
                        ),
                        "minimum_match_position_similarity": (
                            minimum_match_position_similarity
                        ),
                        "frozen_identity": True,
                    }
                )
                continue
            match, row = _matching_row(
                timeline=timeline,
                detection=detection,
                track_id=track_id,
                evidence=evidence,
                formal_weight=track_weights[track_id],
                time_weight_s=time_weight_s,
                association_eligible=bool(
                    timeline.association_supervised[frame_index]
                ),
                frozen=True,
            )
            if match is not None:
                matched.append(match)
            rows.append(row)
            used_entities.add(timeline.entity_id)
            used_tracks.add(track_id)

    remaining_entities = [
        timeline
        for timeline in active
        if timeline.entity_id not in used_entities
        and timeline.entity_id not in locked_entities
    ]
    remaining_tracks = [
        track_id
        for track_id in sorted(available)
        if track_id not in used_tracks and track_id not in reserved_tracks
    ]
    for tier_weight in sorted(
        {track_weights[value] for value in remaining_tracks},
        reverse=True,
    ):
        tier_tracks = [
            track_id
            for track_id in remaining_tracks
            if math.isclose(
                track_weights[track_id],
                tier_weight,
                abs_tol=1e-12,
            )
        ]
        if not remaining_entities or not tier_tracks:
            continue
        costs = np.full(
            (len(remaining_entities), len(tier_tracks)),
            1e9,
            dtype=np.float64,
        )
        evidence_by_edge: dict[tuple[int, int], PositionEvidence] = {}
        rejected_edges: list[dict[str, object]] = []
        for entity_index, timeline in enumerate(remaining_entities):
            for track_index, track_id in enumerate(tier_tracks):
                if not _compatible(timeline, track_classes[track_id]):
                    continue
                detection = available[track_id]
                if timeline.localization_supervised[frame_index]:
                    evidence = _position_evidence(
                        timeline,
                        detection,
                        frame_index=frame_index,
                        frame_diagonal_px=frame_diagonal_px,
                        callback=position_similarity_callback,
                    )
                else:
                    evidence = PositionEvidence(
                        score=1.0,
                        normalized_distance=0.0,
                        diagnostics={"channel": "existence_only"},
                    )
                if evidence.score < minimum_match_position_similarity:
                    rejected_edges.append(
                        {
                            "entity_id": timeline.entity_id,
                            "prediction_track_id": track_id,
                            "position_score": evidence.score,
                            "normalized_distance": (
                                evidence.normalized_distance
                            ),
                            "formal_exposure_weight": tier_weight,
                            "rejection_reason": (
                                "position_similarity_below_minimum"
                            ),
                            "minimum_match_position_similarity": (
                                minimum_match_position_similarity
                            ),
                            "frozen_identity": False,
                        }
                    )
                    continue
                evidence_by_edge[entity_index, track_index] = evidence
                # Stable sub-pixel tie-breaking for existence-only slots.
                costs[entity_index, track_index] = (
                    evidence.normalized_distance
                    + entity_index * 1e-12
                    + track_index * 1e-14
                )
        row_indices, column_indices = linear_sum_assignment(costs)
        matched_entity_indices: set[int] = set()
        matched_tier_tracks: set[str] = set()
        for entity_index, track_index in zip(
            row_indices.tolist(),
            column_indices.tolist(),
        ):
            if costs[entity_index, track_index] >= 1e8:
                continue
            timeline = remaining_entities[entity_index]
            track_id = tier_tracks[track_index]
            detection = available[track_id]
            evidence = evidence_by_edge[entity_index, track_index]
            match, row = _matching_row(
                timeline=timeline,
                detection=detection,
                track_id=track_id,
                evidence=evidence,
                formal_weight=tier_weight,
                time_weight_s=time_weight_s,
                association_eligible=bool(
                    timeline.association_supervised[frame_index]
                ),
                frozen=False,
            )
            if match is not None:
                matched.append(match)
            rows.append(row)
            used_entities.add(timeline.entity_id)
            used_tracks.add(track_id)
            matched_entity_indices.add(entity_index)
            matched_tier_tracks.add(track_id)
        remaining_entities = [
            value
            for index, value in enumerate(remaining_entities)
            if index not in matched_entity_indices
        ]
        remaining_tracks = [
            value
            for value in remaining_tracks
            if value not in matched_tier_tracks
        ]
        rejected.extend(rejected_edges)

    matched_entity_ids = {value.entity_id for value in matched}
    matched_track_ids = {value.track_id for value in matched}
    rejected = [
        row
        for row in rejected
        if row["entity_id"] not in matched_entity_ids
        and row["prediction_track_id"] not in matched_track_ids
    ]
    rejected.sort(
        key=lambda value: (
            str(value["entity_id"]),
            float(value.get("normalized_distance", math.inf)),
            str(value["prediction_track_id"]),
        )
    )
    return matched, rows, rejected


def compare_open_world_v2(
    *,
    expected_timelines: Sequence[ExpectedEntityTimeline],
    prediction_observation: OpenWorldObservation,
    time_grid: CommonTimeGrid,
    frame_diagonal_px: float,
    frozen_identity: FrozenAssignment | None = None,
    minimum_match_position_similarity: float = 0.1,
    position_similarity_callback: PositionSimilarityCallback | None = None,
) -> ObjectCentricComparisonV2:
    """Compare expected identities against every prediction-side object.

    Expected exposure comes exclusively from ``expected_timelines``.
    Prediction tracks, residuals, and overflow are all integrated on the same
    fixed time cells; a disappearing prediction therefore cannot shorten the
    expected denominator.
    """

    timelines = list(expected_timelines)
    frame_count = len(time_grid.times_s)
    if frame_count < 1:
        raise ValueError("open-world v2 requires a non-empty time grid")
    entity_ids = [value.entity_id for value in timelines]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("expected entity IDs must be unique")
    if any(value.frame_count != frame_count for value in timelines):
        raise ValueError("expected timeline length differs from time grid")
    diagonal = float(frame_diagonal_px)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("frame_diagonal_px must be finite and positive")
    minimum_similarity = float(minimum_match_position_similarity)
    if (
        not math.isfinite(minimum_similarity)
        or not 0.0 <= minimum_similarity <= 1.0
    ):
        raise ValueError(
            "minimum_match_position_similarity must be in [0, 1]"
        )
    if prediction_observation.overflow_counts.shape != (frame_count,):
        raise ValueError(
            "overflow_counts must have one value per time sample"
        )
    if frozen_identity is not None:
        unknown = set(frozen_identity.entity_to_track) - set(entity_ids)
        if unknown:
            raise ValueError(
                "frozen identity refers to unknown entities: "
                + ", ".join(sorted(unknown))
            )
    parent_identity_entities = {
        value.entity_id
        for value in timelines
        if value.capability is ReferenceCapability.PHYSICS_PARENT
        and np.any(value.association_supervised[1:])
    }
    if parent_identity_entities and (
        frozen_identity is None
        or not parent_identity_entities.issubset(
            frozen_identity.entity_to_track
        )
    ):
        raise ValueError(
            "physics-parent future association requires condition-frozen "
            "identity assignments for: "
            + ", ".join(sorted(parent_identity_entities))
        )

    detections, track_weights, track_classes = _observation_index(
        prediction_observation,
        frame_count=frame_count,
    )
    existence_evaluation_mask = (
        np.logical_or.reduce(
            [value.existence_supervised for value in timelines]
        )
        if timelines
        else np.ones(frame_count, dtype=bool)
    )
    association_evaluation_mask = (
        np.logical_or.reduce(
            [value.association_supervised for value in timelines]
        )
        if timelines
        else np.zeros(frame_count, dtype=bool)
    )
    reference_exposure = {
        value.entity_id: value.expected_exposure(time_grid)
        for value in timelines
    }
    prediction_exposure: dict[str, float] = {}
    prediction_association_exposure: dict[str, float] = {}
    for track in prediction_observation.tracks:
        observed = np.zeros(frame_count, dtype=bool)
        for detection in track.detections:
            observed[detection.frame_index] = True
        formal_weight = track_weights[track.track_id]
        prediction_exposure[track.track_id] = float(
            np.dot(
                observed & existence_evaluation_mask,
                time_grid.cell_weights_s,
            )
            * formal_weight
        )
        prediction_association_exposure[track.track_id] = float(
            np.dot(
                observed & association_evaluation_mask,
                time_grid.cell_weights_s,
            )
            * formal_weight
        )
    overflow_exposure = float(
        np.dot(
            prediction_observation.overflow_counts
            * existence_evaluation_mask.astype(np.float64),
            time_grid.cell_weights_s,
        )
    )
    if overflow_exposure > 0.0:
        if "__overflow__" in prediction_exposure:
            raise ValueError("prediction track ID '__overflow__' is reserved")
        prediction_exposure["__overflow__"] = overflow_exposure
        prediction_association_exposure["__overflow__"] = 0.0
    reference_association_exposure = {
        value.entity_id: value.association_exposure(time_grid)
        for value in timelines
    }

    all_matches: list[EntityMatch] = []
    per_frame: list[Mapping[str, object]] = []
    previous_prediction_tracks: set[str] = set()
    last_identity_track: dict[str, str] = {}
    for frame_index, time_s in enumerate(time_grid.times_s.tolist()):
        matches, match_rows, rejected = _frame_matches(
            timelines,
            frame_index=frame_index,
            time_weight_s=float(time_grid.cell_weights_s[frame_index]),
            detections=detections,
            track_weights=track_weights,
            track_classes=track_classes,
            frozen_identity=frozen_identity,
            minimum_match_position_similarity=minimum_similarity,
            frame_diagonal_px=diagonal,
            position_similarity_callback=position_similarity_callback,
        )
        all_matches.extend(matches)
        matched_entity_ids = {
            str(row["entity_id"]) for row in match_rows
        }
        matched_track_ids = {
            str(row["prediction_track_id"]) for row in match_rows
        }
        expected_active = [
            value
            for value in timelines
            if value.expected_exists[frame_index]
            and value.existence_supervised[frame_index]
        ]
        supervised_absent = [
            value.entity_id
            for value in timelines
            if value.existence_supervised[frame_index]
            and not value.expected_exists[frame_index]
        ]
        current_prediction_tracks = {
            track_id
            for (index, track_id), _ in detections.items()
            if index == frame_index and track_weights[track_id] > 0.0
            and existence_evaluation_mask[frame_index]
        }
        ambiguous_tracks = sorted(
            track_id
            for (index, track_id), _ in detections.items()
            if index == frame_index and track_weights[track_id] <= 0.0
        )
        extra_tracks = sorted(
            current_prediction_tracks - matched_track_ids
        )
        missing_entities = sorted(
            value.entity_id
            for value in expected_active
            if value.entity_id not in matched_entity_ids
        )
        id_switches: list[dict[str, str]] = []
        for row in match_rows:
            if not bool(row["association_eligible"]):
                continue
            entity_id = str(row["entity_id"])
            track_id = str(row["prediction_track_id"])
            previous = last_identity_track.get(entity_id)
            if previous is not None and previous != track_id:
                id_switches.append(
                    {
                        "entity_id": entity_id,
                        "from_track_id": previous,
                        "to_track_id": track_id,
                    }
                )
            last_identity_track[entity_id] = track_id

        position_terms: list[float] = []
        for timeline in expected_active:
            if not timeline.localization_supervised[frame_index]:
                continue
            row = next(
                (
                    value
                    for value in match_rows
                    if value["entity_id"] == timeline.entity_id
                ),
                None,
            )
            position_terms.append(
                0.0 if row is None else float(row["position_score"])
            )
        expected_cardinality = len(expected_active)
        formal_prediction_cardinality = float(
            sum(
                track_weights[track_id]
                for track_id in current_prediction_tracks
            )
            + (
                prediction_observation.overflow_counts[frame_index]
                if existence_evaluation_mask[frame_index]
                else 0.0
            )
        )
        matched_formal_cardinality = float(
            sum(
                float(row["formal_exposure_weight"])
                for row in match_rows
            )
        )
        per_frame.append(
            {
                "frame": frame_index,
                "time_s": float(time_s),
                "existence_supervised": bool(
                    existence_evaluation_mask[frame_index]
                ),
                "association_supervised": bool(
                    association_evaluation_mask[frame_index]
                ),
                "expected_entity_ids": sorted(
                    value.entity_id for value in expected_active
                ),
                "legally_absent_entity_ids": sorted(supervised_absent),
                "matches": match_rows,
                "missing_entity_ids": missing_entities,
                "extra_track_ids": extra_tracks,
                "ambiguous_candidate_track_ids": ambiguous_tracks,
                "rejected_candidate_matches": rejected,
                "id_switches": id_switches,
                "birth_track_ids": sorted(
                    current_prediction_tracks
                    - previous_prediction_tracks
                ),
                "death_track_ids": sorted(
                    previous_prediction_tracks
                    - current_prediction_tracks
                ),
                "expected_cardinality": expected_cardinality,
                "formal_prediction_cardinality": (
                    formal_prediction_cardinality
                ),
                "matched_formal_cardinality": (
                    matched_formal_cardinality
                ),
                "missing_exposure_s": float(
                    time_grid.cell_weights_s[frame_index]
                    * max(
                        expected_cardinality
                        - matched_formal_cardinality,
                        0.0,
                    )
                ),
                "extra_exposure_s": float(
                    time_grid.cell_weights_s[frame_index]
                    * max(
                        formal_prediction_cardinality
                        - matched_formal_cardinality,
                        0.0,
                    )
                ),
                "position_score": (
                    float(np.mean(position_terms))
                    if position_terms
                    else None
                ),
                "overflow_count": float(
                    prediction_observation.overflow_counts[frame_index]
                    if existence_evaluation_mask[frame_index]
                    else 0.0
                ),
            }
        )
        previous_prediction_tracks = current_prediction_tracks

    raw_integrity = score_entity_integrity(
        all_matches,
        reference_exposure=reference_exposure,
        prediction_exposure=prediction_exposure,
        reference_association_exposure=(
            reference_association_exposure
        ),
        prediction_association_exposure=(
            prediction_association_exposure
        ),
    )
    deletion_ceiling = float(
        np.clip(
            raw_integrity.exposure_recall
            ** OPEN_WORLD_V2_DELETION_RESISTANCE_POWER,
            0.0,
            1.0,
        )
    )
    integrity = replace(
        raw_integrity,
        integrity_gate=min(
            raw_integrity.integrity_gate,
            deletion_ceiling,
        ),
    )
    matched_masks = build_matched_masks(
        all_matches,
        prediction_observation,
        frame_count=frame_count,
    )
    reference_masks = build_reference_union_masks(timelines)
    if not timelines:
        reference_masks = (None,) * frame_count
    iou_curve = matched_mask_iou_curve(reference_masks, matched_masks)
    return ObjectCentricComparisonV2(
        integrity=integrity,
        matches=tuple(all_matches),
        per_frame=tuple(per_frame),
        reference_exposure=reference_exposure,
        prediction_exposure=prediction_exposure,
        matched_masks=matched_masks,
        matched_mask_iou=iou_curve,
        raw_integrity_gate=raw_integrity.integrity_gate,
        deletion_resistant_gate_ceiling=deletion_ceiling,
    )


def compose_object_centric_v2(
    comparison: ObjectCentricComparisonV2,
    *,
    content_components: Mapping[str, float | None],
    content_weights: Mapping[str, float],
) -> GatedCaseComposition:
    """Apply open-world integrity as the non-dilutable case-level gate."""

    composition = compose_gated_case_score(
        comparison.integrity.integrity_gate,
        content_components,
        content_weights,
    )
    if comparison.failed and composition.score is None:
        return GatedCaseComposition(
            score=0.0,
            integrity_gate=0.0,
            content=composition.content,
        )
    return composition


def fail_closed_open_world_v2(
    *,
    expected_timelines: Sequence[ExpectedEntityTimeline],
    time_grid: CommonTimeGrid,
    reason: str,
) -> ObjectCentricComparisonV2:
    """Return a finite conservative result after prediction-side failure."""

    text = str(reason).strip()
    if not text:
        raise ValueError("fail-closed reason must be non-empty")
    timelines = list(expected_timelines)
    frame_count = len(time_grid.times_s)
    if any(value.frame_count != frame_count for value in timelines):
        raise ValueError("expected timeline length differs from time grid")
    reference_exposure = {
        value.entity_id: value.expected_exposure(time_grid)
        for value in timelines
    }
    reference_association = {
        value.entity_id: value.association_exposure(time_grid)
        for value in timelines
    }
    failure_exposure = max(
        float(sum(reference_exposure.values())),
        float(time_grid.cell_weights_s.sum()),
        1.0,
    )
    prediction_exposure = {"__observer_failure__": failure_exposure}
    prediction_association = {"__observer_failure__": 0.0}
    integrity = score_entity_integrity(
        (),
        reference_exposure=reference_exposure,
        prediction_exposure=prediction_exposure,
        reference_association_exposure=reference_association,
        prediction_association_exposure=prediction_association,
    )
    per_frame: list[Mapping[str, object]] = []
    for frame_index, time_s in enumerate(time_grid.times_s.tolist()):
        missing = sorted(
            value.entity_id
            for value in timelines
            if value.expected_exists[frame_index]
            and value.existence_supervised[frame_index]
        )
        per_frame.append(
            {
                "frame": frame_index,
                "time_s": float(time_s),
                "existence_supervised": bool(
                    any(
                        value.existence_supervised[frame_index]
                        for value in timelines
                    )
                ),
                "expected_entity_ids": missing,
                "legally_absent_entity_ids": sorted(
                    value.entity_id
                    for value in timelines
                    if value.existence_supervised[frame_index]
                    and not value.expected_exists[frame_index]
                ),
                "matches": [],
                "missing_entity_ids": missing,
                "extra_track_ids": ["__observer_failure__"],
                "ambiguous_candidate_track_ids": [],
                "rejected_candidate_matches": [],
                "id_switches": [],
                "birth_track_ids": (
                    ["__observer_failure__"] if frame_index == 0 else []
                ),
                "death_track_ids": [],
                "expected_cardinality": len(missing),
                "formal_prediction_cardinality": 1.0,
                "matched_formal_cardinality": 0.0,
                "missing_exposure_s": float(
                    len(missing)
                    * time_grid.cell_weights_s[frame_index]
                ),
                "extra_exposure_s": float(
                    time_grid.cell_weights_s[frame_index]
                ),
                "position_score": 0.0 if missing else None,
                "overflow_count": 0.0,
                "failure_reason": text,
            }
        )
    reference_masks = build_reference_union_masks(timelines)
    if not timelines:
        reference_masks = (None,) * frame_count
    iou = tuple(
        0.0 if mask is not None else None for mask in reference_masks
    )
    return ObjectCentricComparisonV2(
        integrity=integrity,
        matches=(),
        per_frame=tuple(per_frame),
        reference_exposure=reference_exposure,
        prediction_exposure=prediction_exposure,
        matched_masks=(None,) * frame_count,
        matched_mask_iou=iou,
        raw_integrity_gate=integrity.integrity_gate,
        deletion_resistant_gate_ceiling=0.0,
        failed=True,
        failure_reason=text,
    )


def safe_compare_open_world_v2(
    *,
    expected_timelines: Sequence[ExpectedEntityTimeline],
    prediction_factory: Callable[[], OpenWorldObservation],
    time_grid: CommonTimeGrid,
    frame_diagonal_px: float,
    frozen_identity: FrozenAssignment | None = None,
    minimum_match_position_similarity: float = 0.1,
    position_similarity_callback: PositionSimilarityCallback | None = None,
) -> ObjectCentricComparisonV2:
    """Run prediction observation/comparison with a conservative fallback."""

    try:
        observation = prediction_factory()
        if not isinstance(observation, OpenWorldObservation):
            raise TypeError(
                "prediction_factory must return OpenWorldObservation"
            )
        return compare_open_world_v2(
            expected_timelines=expected_timelines,
            prediction_observation=observation,
            time_grid=time_grid,
            frame_diagonal_px=frame_diagonal_px,
            frozen_identity=frozen_identity,
            minimum_match_position_similarity=(
                minimum_match_position_similarity
            ),
            position_similarity_callback=position_similarity_callback,
        )
    except Exception as exc:  # prediction-side robustness boundary
        return fail_closed_open_world_v2(
            expected_timelines=expected_timelines,
            time_grid=time_grid,
            reason=f"{type(exc).__name__}: {exc}",
        )


def object_centric_artifact_payload(
    comparison: ObjectCentricComparisonV2,
) -> dict[str, object]:
    """Stable JSON payload helper for scene visualization/report writers."""

    return comparison.to_dict()


def build_per_frame_audit(
    comparison: ObjectCentricComparisonV2,
) -> tuple[dict[str, object], ...]:
    """Copy JSON-ready per-frame rows for CSV/overlay artifact writers."""

    return tuple(dict(value) for value in comparison.per_frame)


__all__ = [
    "OPEN_WORLD_V2_ID",
    "OPEN_WORLD_V2_DELETION_RESISTANCE_POWER",
    "OPEN_WORLD_V2_VERSION",
    "ConditionIdentityCostHook",
    "ExpectedEntityTimeline",
    "ExpectedPositionCallback",
    "ExpectedPositionSample",
    "LifecycleResolution",
    "ObjectCentricComparisonV2",
    "PositionEvidence",
    "PositionSimilarityCallback",
    "build_matched_masks",
    "build_per_frame_audit",
    "build_reference_union_masks",
    "compare_open_world_v2",
    "compose_object_centric_v2",
    "fail_closed_open_world_v2",
    "freeze_condition_identity",
    "matched_mask_iou_curve",
    "object_centric_artifact_payload",
    "resolve_expected_entity_timeline",
    "resolve_lifecycle",
    "safe_compare_open_world_v2",
]
