"""Adapters from existing entity matches and masks to CSTI Tubes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from physbench.evaluation.common.entities import (
    EntityMatch,
    OpenWorldObservation,
    ReferenceCapability,
)

from .contracts import CSTIContractError, CSTIEntityTube, CSTIInput


def normalize_observer_mask(mask: np.ndarray, *, frame_shape: tuple[int, int]) -> np.ndarray:
    """Losslessly convert an observer binary integer mask to read-only bool."""

    shape = _validate_frame_shape(frame_shape)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != shape:
        actual = getattr(mask, "shape", None)
        raise CSTIContractError(
            "csti_observer_mask_shape_invalid",
            f"CSTI observer mask must be a two-dimensional array of shape {shape}, got {actual}",
        )
    if mask.dtype != np.bool_ and not np.issubdtype(mask.dtype, np.integer):
        raise CSTIContractError(
            "csti_observer_mask_type_invalid",
            f"CSTI observer mask must be bool or integer, got {mask.dtype}",
        )
    normalized = np.array(mask > 0, dtype=bool, copy=True)
    normalized.setflags(write=False)
    return normalized


def build_csti_input_from_frame_matches(
    *,
    reference_capability: ReferenceCapability,
    times_s: Sequence[float],
    frame_shape: tuple[int, int],
    expected_entities: Sequence[tuple[str, str]],
    reference_masks_by_entity: Mapping[str, Sequence[np.ndarray]],
    prediction_observation: OpenWorldObservation,
    matches: Sequence[EntityMatch],
) -> CSTIInput:
    """Materialize already-computed frame associations as per-entity Tubes."""

    times = _normalize_times(times_s)
    shape = _validate_frame_shape(frame_shape)
    expected = _validate_expected_entities(expected_entities)
    expected_ids = {entity_id for entity_id, _ in expected}
    _require_exact_keys(
        reference_masks_by_entity,
        expected_ids=expected_ids,
        label="reference_masks_by_entity",
    )

    references = {
        entity_id: _normalize_sequence(
            reference_masks_by_entity[entity_id],
            frame_count=len(times),
            frame_shape=shape,
            label=f"reference entity {entity_id}",
        )
        for entity_id, _ in expected
    }

    if not isinstance(prediction_observation, OpenWorldObservation):
        raise CSTIContractError(
            "csti_prediction_observation_invalid",
            "CSTI frame-match adapter requires an OpenWorldObservation",
        )
    if prediction_observation.overflow_counts.shape != (len(times),):
        raise CSTIContractError(
            "csti_prediction_frame_count_mismatch",
            "Prediction observation must use the CSTI timeline frame count",
        )

    detections: dict[tuple[str, int], object] = {}
    known_tracks: set[str] = set()
    for track in prediction_observation.tracks:
        known_tracks.add(track.track_id)
        for detection in track.detections:
            if detection.frame_index >= len(times):
                raise CSTIContractError(
                    "csti_detection_frame_out_of_range",
                    f"Detection frame {detection.frame_index} lies outside the CSTI timeline",
                )
            slot = (track.track_id, detection.frame_index)
            if slot in detections:
                raise CSTIContractError(
                    "csti_detection_slot_duplicate",
                    f"Prediction detection slot {slot!r} is duplicated",
                )
            detections[slot] = detection

    empty_template = tuple(_read_only_empty(shape) for _ in times)
    predictions: dict[str, list[np.ndarray] | None] = {entity_id: None for entity_id, _ in expected}
    matched_track_ids: dict[str, set[str]] = {entity_id: set() for entity_id, _ in expected}
    entity_frame_slots: set[tuple[str, int]] = set()
    track_frame_slots: set[tuple[str, int]] = set()
    for match in matches:
        if match.entity_id not in expected_ids:
            raise CSTIContractError(
                "csti_match_entity_unknown",
                f"CSTI match references unknown entity {match.entity_id!r}",
            )
        if match.track_id not in known_tracks:
            raise CSTIContractError(
                "csti_match_track_unknown",
                f"CSTI match references unknown track {match.track_id!r}",
            )
        if match.frame_index >= len(times):
            raise CSTIContractError(
                "csti_match_frame_out_of_range",
                f"CSTI match frame {match.frame_index} lies outside the timeline",
            )
        entity_slot = (match.entity_id, match.frame_index)
        track_slot = (match.track_id, match.frame_index)
        if entity_slot in entity_frame_slots:
            raise CSTIContractError(
                "csti_match_entity_frame_duplicate",
                f"CSTI entity-frame slot {entity_slot!r} is duplicated",
            )
        if track_slot in track_frame_slots:
            raise CSTIContractError(
                "csti_match_track_frame_duplicate",
                f"CSTI track-frame slot {track_slot!r} is duplicated",
            )
        detection = detections.get(track_slot)
        if detection is None:
            raise CSTIContractError(
                "csti_match_detection_missing",
                f"CSTI match has no detection at track-frame slot {track_slot!r}",
            )
        entity_frame_slots.add(entity_slot)
        track_frame_slots.add(track_slot)
        if predictions[match.entity_id] is None:
            predictions[match.entity_id] = list(empty_template)
        if detection.mask is not None:
            predictions[match.entity_id][match.frame_index] = normalize_observer_mask(
                detection.mask,
                frame_shape=shape,
            )
        matched_track_ids[match.entity_id].add(match.track_id)

    entities = tuple(
        CSTIEntityTube(
            entity_id=entity_id,
            role_id=role_id,
            reference_masks=references[entity_id],
            prediction_masks=(
                tuple(predictions[entity_id])
                if predictions[entity_id] is not None
                else None
            ),
            matched_prediction_track_ids=tuple(sorted(matched_track_ids[entity_id])),
        )
        for entity_id, role_id in expected
    )
    return CSTIInput(
        reference_capability=reference_capability,
        times_s=times,
        frame_shape=shape,
        entities=entities,
    )


def build_csti_input_from_aligned_masks(
    *,
    reference_capability: ReferenceCapability,
    times_s: Sequence[float],
    frame_shape: tuple[int, int],
    expected_entities: Sequence[tuple[str, str]],
    reference_masks_by_entity: Mapping[str, Sequence[np.ndarray]],
    prediction_masks_by_entity: Mapping[str, Sequence[np.ndarray] | None],
    matched_track_ids_by_entity: Mapping[str, Sequence[str]],
) -> CSTIInput:
    """Build CSTI input from Scene results that already align entity masks."""

    times = _normalize_times(times_s)
    shape = _validate_frame_shape(frame_shape)
    expected = _validate_expected_entities(expected_entities)
    expected_ids = {entity_id for entity_id, _ in expected}
    for label, mapping in (
        ("reference_masks_by_entity", reference_masks_by_entity),
        ("prediction_masks_by_entity", prediction_masks_by_entity),
        ("matched_track_ids_by_entity", matched_track_ids_by_entity),
    ):
        _require_exact_keys(mapping, expected_ids=expected_ids, label=label)

    entities: list[CSTIEntityTube] = []
    for entity_id, role_id in expected:
        reference = _normalize_sequence(
            reference_masks_by_entity[entity_id],
            frame_count=len(times),
            frame_shape=shape,
            label=f"reference entity {entity_id}",
        )
        raw_prediction = prediction_masks_by_entity[entity_id]
        prediction = (
            None
            if raw_prediction is None
            else _normalize_sequence(
                raw_prediction,
                frame_count=len(times),
                frame_shape=shape,
                label=f"prediction entity {entity_id}",
            )
        )
        track_ids = _normalize_track_ids(
            matched_track_ids_by_entity[entity_id],
            entity_id=entity_id,
        )
        entities.append(
            CSTIEntityTube(
                entity_id=entity_id,
                role_id=role_id,
                reference_masks=reference,
                prediction_masks=prediction,
                matched_prediction_track_ids=track_ids,
            )
        )
    return CSTIInput(
        reference_capability=reference_capability,
        times_s=times,
        frame_shape=shape,
        entities=tuple(entities),
    )


def _normalize_sequence(
    masks: Sequence[np.ndarray],
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
    label: str,
) -> tuple[np.ndarray, ...]:
    if not isinstance(masks, (tuple, list)) or len(masks) != frame_count:
        raise CSTIContractError(
            "csti_adapter_tube_length_mismatch",
            f"CSTI {label} must contain {frame_count} masks",
        )
    return tuple(normalize_observer_mask(mask, frame_shape=frame_shape) for mask in masks)


def _read_only_empty(frame_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=bool)
    mask.setflags(write=False)
    return mask


def _validate_frame_shape(frame_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(frame_shape, (tuple, list))
        or len(frame_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in frame_shape)
    ):
        raise CSTIContractError(
            "csti_frame_shape_invalid",
            "CSTI frame shape must contain two positive integers",
        )
    return (int(frame_shape[0]), int(frame_shape[1]))


def _validate_expected_entities(
    expected_entities: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(expected_entities, (tuple, list)) or not expected_entities:
        raise CSTIContractError(
            "csti_expected_entities_invalid",
            "CSTI expected entities must contain non-empty entity and role IDs",
        )
    normalized: list[tuple[str, str]] = []
    for item in expected_entities:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise CSTIContractError(
                "csti_expected_entities_invalid",
                "CSTI expected entities must contain entity/role pairs",
            )
        entity_id, role_id = item
        if (
            not isinstance(entity_id, str)
            or not entity_id.strip()
            or not isinstance(role_id, str)
            or not role_id.strip()
        ):
            raise CSTIContractError(
                "csti_expected_entities_invalid",
                "CSTI expected entity and role IDs must be non-empty strings",
            )
        normalized.append((entity_id, role_id))
    ids = [entity_id for entity_id, _ in normalized]
    if len(ids) != len(set(ids)):
        raise CSTIContractError(
            "csti_expected_entity_duplicate",
            "CSTI expected entity IDs must be unique",
        )
    return tuple(normalized)


def _normalize_times(times_s: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(times_s, (tuple, list)) or not times_s:
        raise CSTIContractError(
            "csti_timeline_invalid",
            "CSTI timeline must contain at least one timestamp",
        )
    normalized: list[float] = []
    for value in times_s:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
        ):
            raise CSTIContractError(
                "csti_timeline_invalid",
                "CSTI timestamps must be finite numbers",
            )
        normalized.append(float(value))
    return tuple(normalized)


def _normalize_track_ids(
    track_ids: Sequence[str],
    *,
    entity_id: str,
) -> tuple[str, ...]:
    if not isinstance(track_ids, (tuple, list)) or any(
        not isinstance(track_id, str) or not track_id.strip()
        for track_id in track_ids
    ):
        raise CSTIContractError(
            "csti_track_ids_invalid",
            f"CSTI matched track IDs for {entity_id!r} must be non-empty strings",
        )
    return tuple(sorted(set(track_ids)))


def _require_exact_keys(
    mapping: Mapping[str, object],
    *,
    expected_ids: set[str],
    label: str,
) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != expected_ids:
        actual = sorted(mapping) if isinstance(mapping, Mapping) else None
        raise CSTIContractError(
            "csti_adapter_entity_coverage_mismatch",
            f"CSTI {label} keys differ: expected={sorted(expected_ids)}, got={actual}",
        )
