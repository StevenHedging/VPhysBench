#!/usr/bin/env python3
"""Build repair candidates by splicing reviewed SAM samples into canonical tubes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from physbench.reference_observations import (
    EntityObservation,
    load_entity_observation,
    load_reference_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import load_curation_cases


def splice_entity_samples(
    canonical_masks: np.ndarray,
    canonical_states: np.ndarray,
    candidate_masks: np.ndarray,
    candidate_states: np.ndarray,
    *,
    candidate_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(canonical_masks, dtype=np.uint8).copy()
    states = np.asarray(canonical_states, dtype=np.uint8).copy()
    replacement_masks = np.asarray(candidate_masks, dtype=np.uint8)
    replacement_states = np.asarray(candidate_states, dtype=np.uint8)
    if masks.shape != replacement_masks.shape or states.shape != replacement_states.shape:
        raise ValueError("canonical and candidate entity arrays must align")
    if states.shape != (masks.shape[0],):
        raise ValueError("entity states must align with the mask timeline")
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(states)):
        raise ValueError("candidate splice index is outside the timeline")
    masks[indices] = replacement_masks[indices]
    states[indices] = replacement_states[indices]
    return masks, states


def _touches_horizontal_boundary(mask: np.ndarray) -> bool:
    values = np.asarray(mask, dtype=bool)
    return bool(np.any(values[:, 0]) or np.any(values[:, -1]))


def should_replace_observation_zero(
    *,
    canonical_area: int,
    candidate_area: int,
    candidate_visible_areas: np.ndarray,
) -> bool:
    """Use a stable candidate to repair an obviously mis-scaled old anchor."""
    visible = np.asarray(candidate_visible_areas, dtype=np.int64)
    visible = visible[visible > 0]
    if not len(visible):
        return False
    typical_area = float(np.median(visible))
    candidate_is_plausible = 0.20 * typical_area <= candidate_area <= 3.0 * typical_area
    canonical_is_implausible = (
        canonical_area < 0.35 * typical_area
        or canonical_area > 2.5 * typical_area
    )
    return bool(candidate_is_plausible and canonical_is_implausible)


def maybe_replace_observation_zero(
    masks: np.ndarray,
    states: np.ndarray,
    candidate_masks: np.ndarray,
    candidate_states: np.ndarray,
    *,
    authorized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a bad anchor only when the reviewed case plan authorizes it."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    candidates = np.asarray(candidate_masks, dtype=np.uint8)
    candidate_lifecycle = np.asarray(candidate_states, dtype=np.uint8)
    if result_masks.shape != candidates.shape or result_states.shape != candidate_lifecycle.shape:
        raise ValueError("observation-zero replacement arrays must align")
    if not authorized:
        return result_masks, result_states
    candidate_areas = candidates.reshape(len(candidates), -1).sum(
        axis=1, dtype=np.int64
    )
    candidate_visible_areas = candidate_areas[
        (candidate_lifecycle == 0) & (candidate_areas > 0)
    ]
    if should_replace_observation_zero(
        canonical_area=int(np.count_nonzero(result_masks[0])),
        candidate_area=int(candidate_areas[0]),
        candidate_visible_areas=candidate_visible_areas,
    ):
        result_masks[0] = candidates[0]
        result_states[0] = 0
    return result_masks, result_states


def apply_lifecycle_overrides(
    masks: np.ndarray,
    states: np.ndarray,
    overrides: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply reviewed lifecycle ranges and clear every non-visible mask."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    if result_masks.ndim != 3 or result_states.shape != (len(result_masks),):
        raise ValueError("lifecycle overrides require aligned THW/T arrays")
    for override in overrides:
        start = int(override["start_index"])
        end = int(override["end_index"])
        state = int(override["state"])
        if start < 0 or end < start or end >= len(result_states) or state not in (0, 1, 2, 3):
            raise ValueError("lifecycle override has an invalid range or state")
        selection = slice(start, end + 1)
        if state == 0 and not np.all(
            result_masks[selection].reshape(end - start + 1, -1).any(axis=1)
        ):
            raise ValueError("visible lifecycle override requires nonempty masks")
        result_states[selection] = state
        if state != 0:
            result_masks[selection] = 0
    return result_masks, result_states


def apply_frame_overrides(
    masks: np.ndarray,
    states: np.ndarray,
    overrides: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply reviewed single-frame copies or clipped circular masks."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    if result_masks.ndim != 3 or result_states.shape != (len(result_masks),):
        raise ValueError("frame overrides require aligned THW/T arrays")
    height, width = result_masks.shape[1:]
    yy, xx = np.ogrid[:height, :width]
    for override in overrides:
        target = int(override["target_index"])
        if target < 0 or target >= len(result_masks):
            raise ValueError("frame override target is outside the timeline")
        has_source = "source_index" in override
        has_circle = "circle_xy_radius" in override
        if has_source == has_circle:
            raise ValueError("frame override requires exactly one source or circle")
        if has_source:
            source = int(override["source_index"])
            if source < 0 or source >= len(result_masks):
                raise ValueError("frame override source is outside the timeline")
            if not np.any(result_masks[source]):
                raise ValueError("frame override source mask is empty")
            result_masks[target] = result_masks[source]
        else:
            x, y, radius = (float(value) for value in override["circle_xy_radius"])
            if radius <= 0:
                raise ValueError("frame override circle radius must be positive")
            result_masks[target] = (
                (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            ).astype(np.uint8)
        result_states[target] = 0
    return result_masks, result_states


def apply_circle_source_ranges(
    masks: np.ndarray,
    states: np.ndarray,
    source_centroid_xy: np.ndarray,
    overrides: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild collapsed masks at reviewed candidate centers with fixed radius."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    centers = np.asarray(source_centroid_xy, dtype=np.float64)
    if result_masks.ndim != 3 or result_states.shape != (len(result_masks),):
        raise ValueError("circle-source overrides require aligned THW/T arrays")
    if centers.shape != (len(result_masks), 2):
        raise ValueError("circle-source centers must align with the mask timeline")
    height, width = result_masks.shape[1:]
    yy, xx = np.ogrid[:height, :width]
    for override in overrides:
        start = int(override["start_index"])
        end = int(override["end_index"])
        radius = float(override["radius"])
        if start < 0 or end < start or end >= len(result_masks) or radius <= 0:
            raise ValueError("circle-source override has an invalid range or radius")
        for index in range(start, end + 1):
            x, y = centers[index]
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError("circle-source override requires finite centers")
            result_masks[index] = (
                (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= radius**2
            ).astype(np.uint8)
            result_states[index] = 0
    return result_masks, result_states


def apply_circle_motion_ranges(
    masks: np.ndarray,
    states: np.ndarray,
    overrides: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a reviewed range from an explicit constant-velocity circle."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    if result_masks.ndim != 3 or result_states.shape != (len(result_masks),):
        raise ValueError("circle-motion overrides require aligned THW/T arrays")
    height, width = result_masks.shape[1:]
    yy, xx = np.ogrid[:height, :width]
    for override in overrides:
        start = int(override["start_index"])
        end = int(override["end_index"])
        center = np.asarray(override["center_start_xy"], dtype=np.float64)
        velocity = np.asarray(override["velocity_xy"], dtype=np.float64)
        radius = float(override["radius"])
        if (
            start < 0
            or end < start
            or end >= len(result_masks)
            or center.shape != (2,)
            or velocity.shape != (2,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(velocity))
            or radius <= 0
        ):
            raise ValueError("circle-motion override is invalid")
        for index in range(start, end + 1):
            x, y = center + (index - start) * velocity
            result_masks[index] = (
                (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= radius**2
            ).astype(np.uint8)
            result_states[index] = 0
    return result_masks, result_states


def apply_source_ranges(
    masks: np.ndarray,
    states: np.ndarray,
    source_masks: np.ndarray,
    source_states: np.ndarray,
    overrides: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Copy reviewed ranges from another candidate entity after an ID switch."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    replacements = np.asarray(source_masks, dtype=np.uint8)
    replacement_states = np.asarray(source_states, dtype=np.uint8)
    if result_masks.shape != replacements.shape or result_states.shape != replacement_states.shape:
        raise ValueError("source range entities must align")
    for override in overrides:
        start = int(override["start_index"])
        end = int(override["end_index"])
        if start < 0 or end < start or end >= len(result_states):
            raise ValueError("source range is outside the timeline")
        selection = slice(start, end + 1)
        if np.any(replacement_states[selection] != 0):
            raise ValueError("reviewed source range must be fully visible")
        result_masks[selection] = replacements[selection]
        result_states[selection] = replacement_states[selection]
    return result_masks, result_states


def _is_round_subject_mask(mask: np.ndarray) -> bool:
    """Reject rail/ruler fragments while allowing clipped boundary disks."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return False
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect = max(width, height) / max(1, min(width, height))
    fill = len(xs) / float(width * height)
    touches_boundary = bool(xs.min() == 0 or xs.max() == mask.shape[1] - 1)
    maximum_aspect = 3.5 if touches_boundary else 2.0
    return bool(aspect <= maximum_aspect and fill >= 0.35)


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.asarray([np.nan, np.nan], np.float64)
    return np.asarray([xs.mean(), ys.mean()], np.float64)


def split_ordered_mask_overlap(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign shared pixels at contact using collision left-to-right identity."""
    left = (np.asarray(left_mask) > 0).astype(np.uint8)
    right = (np.asarray(right_mask) > 0).astype(np.uint8)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("ordered overlap split requires aligned HW masks")
    left_center = _mask_centroid(left)
    right_center = _mask_centroid(right)
    if not np.all(np.isfinite(left_center)) or not np.all(np.isfinite(right_center)):
        raise ValueError("ordered overlap split requires nonempty masks")
    if left_center[0] >= right_center[0]:
        raise ValueError("ordered overlap split requires increasing x centers")
    overlap = np.logical_and(left > 0, right > 0)
    if not np.any(overlap):
        return left, right
    midpoint = 0.5 * (float(left_center[0]) + float(right_center[0]))
    columns = np.arange(left.shape[1], dtype=np.float64)[None, :]
    right[np.logical_and(overlap, columns <= midpoint)] = 0
    left[np.logical_and(overlap, columns > midpoint)] = 0
    return left, right


def _translated_mask(mask: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
    shift_x, shift_y = (int(round(float(value))) for value in shift_xy)
    height, width = mask.shape
    result = np.zeros_like(mask, dtype=np.uint8)
    source_x0, source_x1 = max(0, -shift_x), min(width, width - shift_x)
    source_y0, source_y1 = max(0, -shift_y), min(height, height - shift_y)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return result
    result[
        source_y0 + shift_y : source_y1 + shift_y,
        source_x0 + shift_x : source_x1 + shift_x,
    ] = mask[source_y0:source_y1, source_x0:source_x1]
    return result


def interpolate_short_visible_gaps(
    masks: np.ndarray,
    states: np.ndarray,
    *,
    occupied_masks: np.ndarray,
    maximum_gap: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate neighboring reviewed masks across short internal failures."""
    result_masks = np.asarray(masks, dtype=np.uint8).copy()
    result_states = np.asarray(states, dtype=np.uint8).copy()
    occupied = np.asarray(occupied_masks, dtype=np.uint8)
    missing = np.flatnonzero(np.isin(result_states, (1, 3)))
    if not len(missing):
        return result_masks, result_states
    starts = missing[np.r_[True, np.diff(missing) > 1]]
    ends = missing[np.r_[np.diff(missing) > 1, True]]
    for start_value, end_value in zip(starts, ends):
        start, end = int(start_value), int(end_value)
        if end - start + 1 > maximum_gap or start == 0 or end + 1 >= len(result_states):
            continue
        left, right = start - 1, end + 1
        if result_states[left] != 0 or result_states[right] != 0:
            continue
        left_center = _mask_centroid(result_masks[left])
        right_center = _mask_centroid(result_masks[right])
        if not np.all(np.isfinite(left_center)) or not np.all(np.isfinite(right_center)):
            continue
        span = right - left
        for index in range(start, end + 1):
            alpha = (index - left) / span
            target_center = (1.0 - alpha) * left_center + alpha * right_center
            template_index = left if alpha <= 0.5 else right
            template_center = left_center if template_index == left else right_center
            replacement = _translated_mask(
                result_masks[template_index], target_center - template_center
            )
            replacement = np.logical_and(replacement > 0, occupied[index] == 0).astype(
                np.uint8
            )
            if _is_round_subject_mask(replacement):
                result_masks[index] = replacement
                result_states[index] = 0
    return result_masks, result_states


def repair_unresolved_samples(
    canonical_masks: np.ndarray,
    canonical_states: np.ndarray,
    candidate_masks: np.ndarray,
    candidate_states: np.ndarray,
    *,
    occupied_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair failed/unknown collision samples from plausible round candidates."""
    masks = np.asarray(canonical_masks, dtype=np.uint8).copy()
    states = np.asarray(canonical_states, dtype=np.uint8).copy()
    candidates = np.asarray(candidate_masks, dtype=np.uint8)
    candidate_lifecycle = np.asarray(candidate_states, dtype=np.uint8)
    occupied = np.asarray(occupied_masks, dtype=np.uint8)
    if masks.shape != candidates.shape or masks.shape != occupied.shape:
        raise ValueError("canonical, candidate, and occupied masks must align")
    if states.shape != candidate_lifecycle.shape or states.shape != (len(masks),):
        raise ValueError("canonical and candidate lifecycle arrays must align")

    canonical_area = masks.reshape(len(masks), -1).sum(axis=1, dtype=np.int64)
    candidate_area = candidates.reshape(len(candidates), -1).sum(
        axis=1, dtype=np.int64
    )
    visible_area = candidate_area[
        (candidate_lifecycle == 0) & (candidate_area > 0)
    ]
    if not len(visible_area):
        visible_area = canonical_area[(states == 0) & (canonical_area > 0)]
    if not len(visible_area):
        raise ValueError("unresolved repair requires a visible canonical area reference")
    typical_area = float(np.median(visible_area))
    minimum_area = max(4.0, 0.20 * typical_area)
    maximum_area = 3.0 * typical_area

    unresolved = np.flatnonzero(np.isin(states, (1, 3)))
    if not len(unresolved):
        return masks, states
    starts = unresolved[np.r_[True, np.diff(unresolved) > 1]]
    ends = unresolved[np.r_[np.diff(unresolved) > 1, True]]
    for start, end in zip(starts, ends):
        next_state = int(states[end + 1]) if end + 1 < len(states) else None
        previous_nonempty = next(
            (index for index in range(int(start) - 1, -1, -1) if canonical_area[index]),
            None,
        )
        terminal_boundary_exit = (
            end + 1 == len(states)
            and previous_nonempty is not None
            and _touches_horizontal_boundary(masks[previous_nonempty])
        )
        boundary_exit = next_state == 2 or terminal_boundary_exit
        for index in range(int(start), int(end) + 1):
            replacement = np.logical_and(candidates[index] > 0, occupied[index] == 0)
            replacement_area = int(np.count_nonzero(replacement))
            plausible = (
                candidate_lifecycle[index] == 0
                and minimum_area <= replacement_area <= maximum_area
                and _is_round_subject_mask(replacement)
            )
            if boundary_exit:
                plausible = plausible and _touches_horizontal_boundary(replacement)
            if plausible:
                masks[index] = replacement.astype(np.uint8)
                states[index] = 0
            else:
                masks[index] = 0
                states[index] = 2 if boundary_exit else 1
    return masks, states


def select_entity_base(
    canonical_masks: np.ndarray,
    canonical_states: np.ndarray,
    candidate_masks: np.ndarray,
    candidate_states: np.ndarray,
    *,
    full_candidate: bool,
    repair_unresolved: bool,
    occupied_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the reviewed full tube before considering generic repair."""
    if full_candidate:
        return splice_entity_samples(
            canonical_masks,
            canonical_states,
            candidate_masks,
            candidate_states,
            candidate_indices=tuple(range(len(canonical_states))),
        )
    if repair_unresolved:
        return repair_unresolved_samples(
            canonical_masks,
            canonical_states,
            candidate_masks,
            candidate_states,
            occupied_masks=occupied_masks,
        )
    return splice_entity_samples(
        canonical_masks,
        canonical_states,
        candidate_masks,
        candidate_states,
        candidate_indices=(0,),
    )


def _trajectory(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(masks)
    centroid = np.full((count, 2), np.nan, np.float32)
    bbox = np.full((count, 4), np.nan, np.float32)
    area = masks.reshape(count, -1).sum(axis=1, dtype=np.int64)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            centroid[index] = (float(xs.mean()), float(ys.mean()))
            bbox[index] = (xs.min(), ys.min(), xs.max(), ys.max())
    return centroid, bbox, area


def _load_candidate_entity(root: Path, object_id: str, samples: int) -> EntityObservation:
    entity_root = root / "entities" / object_id
    return load_entity_observation(
        entity_root / "mask_tube.npz",
        entity_root / "trajectory.npz",
        object_id=object_id,
        mask_id=f"{int(object_id.rsplit('_', 1)[-1]):02d}",
        expected_samples=samples,
    )


def build_spliced_case(
    case,
    *,
    candidate_case_root: Path,
    output_root: Path,
    full_candidate_entities: set[str],
    repair_unresolved: bool = False,
    lifecycle_overrides: Sequence[Mapping[str, object]] = (),
    frame_overrides: Sequence[Mapping[str, object]] = (),
    source_overrides: Sequence[Mapping[str, object]] = (),
    circle_source_overrides: Sequence[Mapping[str, object]] = (),
    circle_motion_overrides: Sequence[Mapping[str, object]] = (),
    observation_zero_entities: set[str] | None = None,
) -> Path:
    canonical = load_reference_observation(
        case.asset_root,
        case.observation_manifest_path,
        bundle_root=case.asset_root,
    )
    output = output_root / case.scene_id / case.case_id
    sample_count = len(case.timeline["samples"])
    candidates = {
        catalog_entity.identity.object_id: _load_candidate_entity(
            candidate_case_root,
            catalog_entity.identity.object_id,
            sample_count,
        )
        for catalog_entity in case.entities
    }
    entities: dict[str, EntityObservation] = {}
    authorized_observation_zero = observation_zero_entities or set()
    for index, catalog_entity in enumerate(case.entities, 1):
        object_id = catalog_entity.identity.object_id
        original = canonical.entities[object_id]
        candidate = candidates[object_id]
        occupied = np.zeros_like(original.masks, dtype=np.uint8)
        if repair_unresolved and object_id not in full_candidate_entities:
            for other_id, other in canonical.entities.items():
                if other_id != object_id:
                    occupied |= other.masks.astype(np.uint8)
                    occupied |= candidates[other_id].masks.astype(np.uint8)
        masks, states = select_entity_base(
            original.masks,
            original.state,
            candidate.masks,
            candidate.state,
            full_candidate=object_id in full_candidate_entities,
            repair_unresolved=repair_unresolved,
            occupied_masks=occupied,
        )
        if repair_unresolved and object_id not in full_candidate_entities:
            masks, states = interpolate_short_visible_gaps(
                masks,
                states,
                occupied_masks=occupied,
            )
            masks, states = maybe_replace_observation_zero(
                masks,
                states,
                candidate.masks,
                candidate.state,
                authorized=object_id in authorized_observation_zero,
            )
        entity_lifecycle = tuple(
            item for item in lifecycle_overrides if item.get("object_id") == object_id
        )
        entity_frames = tuple(
            item for item in frame_overrides if item.get("object_id") == object_id
        )
        entity_sources = tuple(
            item for item in source_overrides if item.get("object_id") == object_id
        )
        entity_circle_sources = tuple(
            item
            for item in circle_source_overrides
            if item.get("object_id") == object_id
        )
        entity_circle_motions = tuple(
            item
            for item in circle_motion_overrides
            if item.get("object_id") == object_id
        )
        for source_override in entity_sources:
            source_id = str(source_override["source_object_id"])
            if source_id not in candidates:
                raise ValueError(f"unknown source entity: {source_id}")
            masks, states = apply_source_ranges(
                masks,
                states,
                candidates[source_id].masks,
                candidates[source_id].state,
                (source_override,),
            )
        for circle_override in entity_circle_sources:
            source_id = str(circle_override.get("source_object_id", object_id))
            if source_id not in candidates:
                raise ValueError(f"unknown circle-source entity: {source_id}")
            masks, states = apply_circle_source_ranges(
                masks,
                states,
                candidates[source_id].centroid_xy,
                (circle_override,),
            )
        masks, states = apply_circle_motion_ranges(
            masks, states, entity_circle_motions
        )
        masks, states = apply_frame_overrides(masks, states, entity_frames)
        masks, states = apply_lifecycle_overrides(masks, states, entity_lifecycle)
        centroid, bbox, area = _trajectory(masks)
        entity = EntityObservation(
            object_id=object_id,
            mask_id=f"{index:02d}",
            masks=masks,
            centroid_xy=centroid,
            bbox_xyxy=bbox,
            area_pixels=area,
            state=states,
        )
        entities[object_id] = entity
        write_entity_observation(output / "entities" / object_id, entity)

    overlap_summary: dict[str, int] = {}
    object_ids = tuple(entities)
    for left_index, left_id in enumerate(object_ids):
        for right_id in object_ids[left_index + 1 :]:
            left = entities[left_id]
            right = entities[right_id]
            overlap = np.logical_and(left.masks > 0, right.masks > 0).reshape(
                sample_count, -1
            ).sum(axis=1)
            for frame_index in np.flatnonzero(overlap > 32):
                left_changed = not np.array_equal(
                    left.masks[frame_index],
                    canonical.entities[left_id].masks[frame_index],
                )
                right_changed = not np.array_equal(
                    right.masks[frame_index],
                    canonical.entities[right_id].masks[frame_index],
                )
                if not (left_changed or right_changed):
                    raise ValueError(
                        f"canonical tubes overlap excessively for {left_id}/{right_id} "
                        f"at observation {frame_index}"
                    )
                try:
                    split_left, split_right = split_ordered_mask_overlap(
                        left.masks[frame_index], right.masks[frame_index]
                    )
                except ValueError:
                    for object_id in (left_id, right_id):
                        original = canonical.entities[object_id]
                        entity = entities[object_id]
                        if not np.array_equal(
                            entity.masks[frame_index], original.masks[frame_index]
                        ):
                            entity.masks[frame_index] = original.masks[frame_index]
                            entity.state[frame_index] = original.state[frame_index]
                else:
                    left.masks[frame_index] = split_left
                    right.masks[frame_index] = split_right
                    left.state[frame_index] = 0
                    right.state[frame_index] = 0
            left.centroid_xy[:], left.bbox_xyxy[:], left.area_pixels[:] = _trajectory(
                left.masks
            )
            right.centroid_xy[:], right.bbox_xyxy[:], right.area_pixels[:] = _trajectory(
                right.masks
            )
            overlap = np.logical_and(left.masks > 0, right.masks > 0).reshape(
                sample_count, -1
            ).sum(axis=1)
            maximum = int(overlap.max())
            overlap_summary[f"{left_id}:{right_id}"] = maximum
            if maximum > 32:
                raise ValueError(
                    f"spliced tubes overlap excessively for {left_id}/{right_id}: {maximum}"
                )

    for object_id, entity in entities.items():
        write_entity_observation(output / "entities" / object_id, entity)

    metadata = output / "candidate_tracking.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "model_id": "reviewed_canonical_plus_sam2_splice",
                "strategy": (
                    "candidate_for_canonical_unresolved_samples"
                    if repair_unresolved
                    else "candidate_full_for_selected_entities_else_candidate_observation_zero"
                ),
                "full_candidate_entities": sorted(full_candidate_entities),
                "lifecycle_overrides": list(lifecycle_overrides),
                "frame_overrides": list(frame_overrides),
                "source_overrides": list(source_overrides),
                "circle_source_overrides": list(circle_source_overrides),
                "circle_motion_overrides": list(circle_motion_overrides),
                "observation_zero_entities": sorted(authorized_observation_zero),
                "overlap_max_pixels": overlap_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--repair-unresolved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.plan is None and not arguments.repair_unresolved:
        raise ValueError("a splice plan or --repair-unresolved is required")
    case_ids = tuple(arguments.case_list.read_text(encoding="utf-8").split())
    plan = (
        json.loads(arguments.plan.read_text(encoding="utf-8"))
        if arguments.plan is not None
        else {}
    )
    unknown = set(plan) - set(case_ids)
    if unknown:
        raise ValueError(f"splice plan references unrequested Cases: {sorted(unknown)}")
    cases = load_curation_cases(arguments.dataset, case_ids=case_ids)
    for case in cases:
        case_plan = plan.get(case.case_id, {})
        candidate = next(
            arguments.candidate_root.rglob(f"{case.case_id}/candidate_tracking.json")
        ).parent
        build_spliced_case(
            case,
            candidate_case_root=candidate,
            output_root=arguments.output_root,
            full_candidate_entities=set(case_plan.get("full_candidate_entities", [])),
            repair_unresolved=(
                arguments.repair_unresolved or bool(case_plan.get("repair_unresolved", False))
            ),
            lifecycle_overrides=tuple(case_plan.get("lifecycle", ())),
            frame_overrides=tuple(case_plan.get("frame_overrides", ())),
            source_overrides=tuple(case_plan.get("source_overrides", ())),
            circle_source_overrides=tuple(
                case_plan.get("circle_source_overrides", ())
            ),
            circle_motion_overrides=tuple(
                case_plan.get("circle_motion_overrides", ())
            ),
            observation_zero_entities=set(
                case_plan.get("observation_zero_entities", ())
            ),
        )
    print(json.dumps({"cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
