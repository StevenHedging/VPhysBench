"""Scene-aware identity and numbering checks for frozen Case subjects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class EntityIdentity:
    object_id: str
    mask_id: str
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def height(self) -> float:
        return max(1.0, float(self.bbox_xyxy[3]) - float(self.bbox_xyxy[1]))


@dataclass(frozen=True)
class IdentityIssue:
    code: str
    severity: str
    message: str
    object_id: str | None = None
    expected_order: tuple[str, ...] = ()
    observed_order: tuple[str, ...] = ()
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


def _finite_identity_values(entities: Iterable[EntityIdentity]) -> None:
    for entity in entities:
        values = (*entity.centroid_xy, *entity.bbox_xyxy)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(
                f"entity {entity.object_id} identity geometry must be finite"
            )


def _row_major_order(entities: tuple[EntityIdentity, ...]) -> tuple[str, ...]:
    if not entities:
        return ()
    tolerance = max(1.0, 0.5 * float(median(entity.height for entity in entities)))
    rows: list[list[EntityIdentity]] = []
    row_centers: list[float] = []
    for entity in sorted(entities, key=lambda value: value.centroid_xy[1]):
        y_center = float(entity.centroid_xy[1])
        if not rows or abs(y_center - row_centers[-1]) > tolerance:
            rows.append([entity])
            row_centers.append(y_center)
            continue
        rows[-1].append(entity)
        row_centers[-1] = sum(
            float(value.centroid_xy[1]) for value in rows[-1]
        ) / len(rows[-1])
    return tuple(
        entity.object_id
        for row in rows
        for entity in sorted(row, key=lambda value: value.centroid_xy[0])
    )


def audit_entity_order(
    scene_id: str,
    entities: Iterable[EntityIdentity],
) -> tuple[IdentityIssue, ...]:
    values = tuple(entities)
    _finite_identity_values(values)
    expected = tuple(entity.object_id for entity in values)
    if scene_id == "collision_1d":
        observed = tuple(
            entity.object_id
            for entity in sorted(values, key=lambda value: value.centroid_xy[0])
        )
        code = "collision_left_to_right_order"
    elif scene_id == "uniform_circular_motion":
        observed = _row_major_order(values)
        code = "row_major_object_order"
    else:
        return ()
    if observed == expected:
        return ()
    return (
        IdentityIssue(
            code=code,
            severity="error",
            message=(
                f"{scene_id} visual object order {observed} differs from "
                f"physics order {expected}"
            ),
            expected_order=expected,
            observed_order=observed,
        ),
    )


def audit_physics_key_binding(
    *,
    object_id: str,
    quantity_names: Iterable[str],
    physics_keys: Iterable[str],
) -> tuple[IdentityIssue, ...]:
    expected = {
        f"objects.{object_id}.{quantity_name}"
        for quantity_name in quantity_names
    }
    observed = set(physics_keys)
    missing = tuple(sorted(expected - observed))
    unexpected = tuple(sorted(observed - expected))
    if not missing and not unexpected:
        return ()
    return (
        IdentityIssue(
            code="mask_physics_keys_mismatch",
            severity="error",
            message=(
                f"{object_id} first-frame mask physics keys differ from "
                "the formal object quantities"
            ),
            object_id=object_id,
            missing_keys=missing,
            unexpected_keys=unexpected,
        ),
    )


__all__ = [
    "EntityIdentity",
    "IdentityIssue",
    "audit_entity_order",
    "audit_physics_key_binding",
]
