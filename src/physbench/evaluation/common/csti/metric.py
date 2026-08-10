"""Exact postcondition spatio-temporal soft Tube IoU."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from physbench.evaluation.common.entities import ReferenceCapability

from .contracts import CSTIConfig, CSTIContractError, CSTIInput


def score_postcondition_tube(
    reference_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    *,
    times_s: Sequence[float],
    config: CSTIConfig,
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    """Score one matched Tube after excluding configured initial samples."""

    return _score_postcondition_tube(
        reference_masks,
        prediction_masks,
        times_s=times_s,
        config=config,
        crop_finite_support=True,
    )


def score_postcondition_tube_reference(
    reference_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    *,
    times_s: Sequence[float],
    config: CSTIConfig,
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    """Full-domain correctness oracle for calibrated CSTI."""

    return _score_postcondition_tube(
        reference_masks,
        prediction_masks,
        times_s=times_s,
        config=config,
        crop_finite_support=False,
    )


def _score_postcondition_tube(
    reference_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    *,
    times_s: Sequence[float],
    config: CSTIConfig,
    crop_finite_support: bool,
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    if not isinstance(config, CSTIConfig):
        raise CSTIContractError(
            "csti_config_invalid",
            "CSTI config must be a CSTIConfig instance",
        )
    reference = _stack_strict_binary_masks(reference_masks, label="reference")
    prediction = _stack_strict_binary_masks(prediction_masks, label="prediction")
    if reference.shape != prediction.shape:
        raise CSTIContractError(
            "csti_tube_shape_mismatch",
            f"CSTI Tube shapes differ: reference={reference.shape}, prediction={prediction.shape}",
        )
    normalized_times, delta_t_s, terminal_frames_excluded = _normalize_times_s(
        times_s,
        frame_count=reference.shape[0],
    )
    if terminal_frames_excluded:
        reference = reference[:-terminal_frames_excluded]
        prediction = prediction[:-terminal_frames_excluded]
    return _score_postcondition_tube_arrays(
        reference,
        prediction,
        times_s=normalized_times,
        delta_t_s=delta_t_s,
        config=config,
        crop_finite_support=crop_finite_support,
    )


def _score_postcondition_tube_arrays(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    times_s: tuple[float, ...],
    delta_t_s: float,
    config: CSTIConfig,
    crop_finite_support: bool,
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    start = config.initial_frames_excluded
    if start >= reference.shape[0]:
        raise CSTIContractError(
            "csti_initial_exclusion_invalid",
            "CSTI initial_frames_excluded must leave at least one scored frame: "
            f"excluded={start}, frames={reference.shape[0]}",
        )
    reference_score = reference[start:]
    prediction_score = prediction[start:]
    _, height, width = reference_score.shape
    spacing = (
        delta_t_s / config.temporal_tolerance_s,
        1.0
        / max(height - 1, 1)
        / config.spatial_tolerance_fraction,
        1.0
        / max(width - 1, 1)
        / config.spatial_tolerance_fraction,
    )
    formal_score = _soft_tube_iou_arrays(
        reference_score,
        prediction_score,
        spacing=spacing,
        crop_finite_support=crop_finite_support,
    )

    scored_frame_count = reference_score.shape[0]
    checkpoint_fractions: dict[int, float] = {}
    for fraction in config.diagnostic_prefix_fractions:
        end = int(math.ceil(fraction * scored_frame_count))
        checkpoint_fractions[end] = fraction

    diagnostics: list[dict[str, float | int]] = []
    for end, fraction in sorted(checkpoint_fractions.items()):
        prefix_score = (
            formal_score
            if end == scored_frame_count
            else _soft_tube_iou_arrays(
                reference_score[:end],
                prediction_score[:end],
                spacing=spacing,
                crop_finite_support=crop_finite_support,
            )
        )
        diagnostics.append(
            {
                "fraction": float(fraction),
                "end_frame_index": int(start + end - 1),
                "end_time_s": float(times_s[start + end - 1]),
                "score": float(prefix_score),
            }
        )
    return float(formal_score), tuple(diagnostics)


def _soft_tube_iou_arrays(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    crop_finite_support: bool,
) -> float:
    reference_any = bool(np.any(reference))
    prediction_any = bool(np.any(prediction))
    if not reference_any and not prediction_any:
        return 1.0
    if not reference_any or not prediction_any:
        return 0.0

    if crop_finite_support:
        spatial_crop = _finite_support_spatial_crop(
            reference,
            prediction,
            spacing=spacing,
            radius=1.0,
        )
        reference = reference[spatial_crop]
        prediction = prediction[spatial_crop]
    reference_soft = _soft_membership(reference, spacing=spacing)
    prediction_soft = _soft_membership(prediction, spacing=spacing)
    intersection = float(
        np.minimum(reference_soft, prediction_soft).sum(dtype=np.float64)
    )
    union = float(
        np.maximum(reference_soft, prediction_soft).sum(dtype=np.float64)
    )
    score = intersection / union
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CSTIContractError(
            "csti_score_invalid",
            f"CSTI Tube score is invalid: {score!r}",
        )
    return float(score)


def evaluate_csti(
    value: CSTIInput,
    *,
    expected_entities: Sequence[tuple[str, str]],
    config: CSTIConfig,
) -> dict[str, Any]:
    """Validate and score a manifest-complete CSTI input."""

    if not isinstance(value, CSTIInput):
        raise CSTIContractError(
            "csti_input_invalid",
            "CSTI input must be a CSTIInput instance",
        )
    if not isinstance(config, CSTIConfig):
        raise CSTIContractError(
            "csti_config_invalid",
            "CSTI config must be a CSTIConfig instance",
        )
    if value.reference_capability is not ReferenceCapability.SAME_CASE_GT:
        return not_applicable_csti_metric(
            config=config,
            reason_code="csti_requires_same_case_gt",
        )

    expected = _validate_expected_entities(expected_entities)
    (
        times_s,
        frame_shape,
        delta_t_s,
        terminal_frames_excluded,
    ) = _validate_timeline(value)
    frame_count = len(value.times_s)
    regular_frame_count = len(times_s)
    if config.initial_frames_excluded >= regular_frame_count:
        raise CSTIContractError(
            "csti_initial_exclusion_invalid",
            "CSTI initial_frames_excluded must leave at least one scored frame: "
            f"excluded={config.initial_frames_excluded}, "
            f"regular_frames={regular_frame_count}",
        )
    by_id: dict[str, Any] = {}
    for entity in value.entities:
        if entity.entity_id in by_id:
            raise CSTIContractError(
                "csti_entity_duplicate",
                f"CSTI input repeats entity {entity.entity_id!r}",
            )
        by_id[entity.entity_id] = entity
    expected_ids = {entity_id for entity_id, _ in expected}
    if set(by_id) != expected_ids:
        raise CSTIContractError(
            "csti_entity_coverage_mismatch",
            f"CSTI entity coverage differs: expected={sorted(expected_ids)}, got={sorted(by_id)}",
        )

    objects: list[dict[str, Any]] = []
    object_scores: list[float] = []
    for entity_id, role_id in expected:
        entity = by_id[entity_id]
        if entity.role_id != role_id:
            raise CSTIContractError(
                "csti_entity_role_mismatch",
                f"CSTI role differs for {entity_id!r}: expected={role_id!r}, got={entity.role_id!r}",
            )
        reference = _stack_strict_binary_masks(
            entity.reference_masks,
            label=f"reference entity {entity_id}",
        )
        _validate_tube_shape(
            reference,
            frame_count=frame_count,
            frame_shape=frame_shape,
            entity_id=entity_id,
        )
        track_ids = _validate_track_ids(
            entity.matched_prediction_track_ids,
            entity_id=entity_id,
        )

        if entity.prediction_masks is None:
            score = 0.0
            diagnostic_prefix_curve: list[dict[str, float | int]] | None = None
            matched = False
        else:
            if not track_ids:
                raise CSTIContractError(
                    "csti_track_ids_missing",
                    f"Matched CSTI entity {entity_id!r} must identify at least one prediction track",
                )
            prediction = _stack_strict_binary_masks(
                entity.prediction_masks,
                label=f"prediction entity {entity_id}",
            )
            _validate_tube_shape(
                prediction,
                frame_count=frame_count,
                frame_shape=frame_shape,
                entity_id=entity_id,
            )
            score, raw_diagnostics = _score_postcondition_tube_arrays(
                reference[:regular_frame_count],
                prediction[:regular_frame_count],
                times_s=times_s,
                delta_t_s=delta_t_s,
                config=config,
                crop_finite_support=True,
            )
            diagnostic_prefix_curve = [
                dict(point) for point in raw_diagnostics
            ]
            matched = True

        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise CSTIContractError(
                "csti_score_invalid",
                f"CSTI object score is invalid: {score!r}",
            )
        object_scores.append(score)
        objects.append(
            {
                "entity_id": entity_id,
                "role_id": role_id,
                "matched": matched,
                "matched_prediction_track_ids": list(track_ids),
                "score": score,
                "diagnostic_prefix_curve": diagnostic_prefix_curve,
            }
        )

    score = float(np.mean(object_scores, dtype=np.float64))
    return {
        "status": "evaluated",
        "score": score,
        **_metric_metadata(config),
        "frame_count": frame_count,
        "terminal_frames_excluded": terminal_frames_excluded,
        "initial_frames_excluded": config.initial_frames_excluded,
        "scored_frame_count": (
            regular_frame_count - config.initial_frames_excluded
        ),
        "frame_shape": list(frame_shape),
        "objects": objects,
    }


def zero_csti_metric(
    *,
    expected_entities: Sequence[tuple[str, str]],
    config: CSTIConfig,
    degradation_code: str,
    degradation_reason: str,
) -> dict[str, Any]:
    """Build a manifest-complete evaluated zero for prediction degradation."""

    expected = _validate_expected_entities(expected_entities)
    return {
        "status": "evaluated",
        "score": 0.0,
        **_metric_metadata(config),
        "degradation": {
            "code": str(degradation_code),
            "reason": str(degradation_reason),
        },
        "objects": [
            {
                "entity_id": entity_id,
                "role_id": role_id,
                "matched": False,
                "matched_prediction_track_ids": [],
                "score": 0.0,
                "diagnostic_prefix_curve": None,
            }
            for entity_id, role_id in expected
        ],
    }


def not_applicable_csti_metric(
    *,
    config: CSTIConfig,
    reason_code: str,
) -> dict[str, Any]:
    """Build the explicit result used when no same-case GT Tube exists."""

    if not isinstance(config, CSTIConfig):
        raise CSTIContractError(
            "csti_config_invalid",
            "CSTI config must be a CSTIConfig instance",
        )
    return {
        "status": "not_applicable",
        "score": None,
        "reason_code": str(reason_code),
        **_metric_metadata(config),
    }


def _metric_metadata(config: CSTIConfig) -> dict[str, Any]:
    return {
        "algorithm": config.algorithm,
        "parameters": {
            "spatial_tolerance_fraction": config.spatial_tolerance_fraction,
            "temporal_tolerance_s": config.temporal_tolerance_s,
            "condition_frame_policy": config.condition_frame_policy,
            "initial_frames_excluded": config.initial_frames_excluded,
            "terminal_frame_policy": "exclude_off_grid_endpoint",
        },
        "aggregation": {
            "time": config.score_aggregation,
            "entities": "mean_all_gt_entities",
        },
    }


def _validate_expected_entities(
    expected_entities: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(expected_entities, (tuple, list)) or not expected_entities:
        raise CSTIContractError(
            "csti_expected_entities_invalid",
            "CSTI expected entities must be a non-empty sequence",
        )
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in expected_entities:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise CSTIContractError(
                "csti_expected_entities_invalid",
                "Each CSTI expected entity must contain entity_id and role_id",
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
        if entity_id in seen:
            raise CSTIContractError(
                "csti_expected_entity_duplicate",
                f"CSTI expected entity {entity_id!r} is duplicated",
            )
        seen.add(entity_id)
        normalized.append((entity_id, role_id))
    return tuple(normalized)


def _validate_timeline(
    value: CSTIInput,
) -> tuple[tuple[float, ...], tuple[int, int], float, int]:
    normalized_times, delta_t_s, terminal_frames_excluded = (
        _normalize_times_s(value.times_s)
    )
    shape = value.frame_shape
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise CSTIContractError(
            "csti_frame_shape_invalid",
            "CSTI frame shape must contain two positive integers",
        )
    return (
        normalized_times,
        (int(shape[0]), int(shape[1])),
        delta_t_s,
        terminal_frames_excluded,
    )


def _normalize_times_s(
    times_s: Sequence[float],
    *,
    frame_count: int | None = None,
) -> tuple[tuple[float, ...], float, int]:
    if not isinstance(times_s, (tuple, list)) or len(times_s) < 2:
        raise CSTIContractError(
            "csti_timeline_too_short",
            "CSTI timeline must contain a condition frame and at least one scored frame",
        )
    if frame_count is not None and len(times_s) != frame_count:
        raise CSTIContractError(
            "csti_timeline_length_mismatch",
            f"CSTI timeline has {len(times_s)} timestamps for {frame_count} masks",
        )
    normalized: list[float] = []
    for timestamp in times_s:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise CSTIContractError(
                "csti_timeline_invalid",
                "CSTI timestamps must be finite numbers",
            )
        value = float(timestamp)
        if not np.isfinite(value):
            raise CSTIContractError(
                "csti_timeline_invalid",
                "CSTI timestamps must be finite numbers",
            )
        normalized.append(value)
    deltas = np.diff(np.asarray(normalized, dtype=np.float64))
    if np.any(deltas <= 0.0):
        raise CSTIContractError(
            "csti_timeline_invalid",
            "CSTI timestamps must be strictly increasing",
        )
    if np.allclose(deltas, deltas[0], rtol=1e-6, atol=1e-9):
        return tuple(normalized), float(deltas[0]), 0
    preceding = deltas[:-1]
    has_short_terminal_endpoint = (
        preceding.size > 0
        and np.allclose(
            preceding,
            preceding[0],
            rtol=1e-6,
            atol=1e-9,
        )
        and deltas[-1] < preceding[0]
        and not np.isclose(
            deltas[-1],
            preceding[0],
            rtol=1e-6,
            atol=1e-9,
        )
    )
    if not has_short_terminal_endpoint:
        raise CSTIContractError(
            "csti_timeline_nonuniform",
            "CSTI timestamps must use a uniform physical time grid with at "
            "most one shorter terminal endpoint interval",
        )
    return tuple(normalized[:-1]), float(preceding[0]), 1


def _validate_tube_shape(
    tube: np.ndarray,
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
    entity_id: str,
) -> None:
    expected = (frame_count, *frame_shape)
    if tube.shape != expected:
        raise CSTIContractError(
            "csti_tube_shape_mismatch",
            f"CSTI Tube for {entity_id!r} has shape {tube.shape}, expected {expected}",
        )


def _validate_track_ids(
    track_ids: tuple[str, ...],
    *,
    entity_id: str,
) -> tuple[str, ...]:
    if not isinstance(track_ids, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in track_ids
    ):
        raise CSTIContractError(
            "csti_track_ids_invalid",
            f"CSTI matched track IDs for {entity_id!r} must be non-empty strings in a tuple",
        )
    if tuple(sorted(set(track_ids))) != track_ids:
        raise CSTIContractError(
            "csti_track_ids_invalid",
            f"CSTI matched track IDs for {entity_id!r} must be sorted and unique",
        )
    return track_ids


def _stack_strict_binary_masks(
    masks: Sequence[np.ndarray],
    *,
    label: str,
) -> np.ndarray:
    if not isinstance(masks, (tuple, list)) or not masks:
        raise CSTIContractError(
            "csti_tube_empty",
            f"CSTI {label} Tube must contain at least one mask",
        )

    normalized: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for index, value in enumerate(masks):
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.size == 0:
            raise CSTIContractError(
                "csti_mask_invalid",
                f"CSTI {label} mask {index} must be a non-empty two-dimensional NumPy array",
            )
        if value.dtype == np.bool_:
            mask = value
        elif np.issubdtype(value.dtype, np.integer) and bool(
            np.all((value == 0) | (value == 1))
        ):
            mask = value.astype(bool, copy=False)
        else:
            raise CSTIContractError(
                "csti_mask_not_binary",
                f"CSTI {label} mask {index} must be bool or integer with values in {{0, 1}}",
            )
        shape = (int(mask.shape[0]), int(mask.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise CSTIContractError(
                "csti_mask_shape_mismatch",
                f"CSTI {label} masks must share one shape: expected={expected_shape}, got={shape}",
            )
        normalized.append(mask)
    return np.stack(normalized, axis=0)


def _soft_membership(
    mask: np.ndarray,
    *,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    distance = distance_transform_edt(~mask, sampling=spacing)
    return np.maximum(0.0, 1.0 - distance)


def _finite_support_spatial_crop(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    radius: float,
) -> tuple[slice, slice, slice]:
    """Crop spatial voxels whose soft occupancy is zero for both Tubes."""

    spatial_foreground = np.any(reference | prediction, axis=0)
    rows, columns = np.nonzero(spatial_foreground)
    if not len(rows):
        raise CSTIContractError(
            "csti_support_crop_empty",
            "CSTI finite-support crop requires non-empty joint foreground",
        )
    row_padding = int(math.ceil(radius / spacing[1]))
    column_padding = int(math.ceil(radius / spacing[2]))
    row_start = max(int(rows.min()) - row_padding, 0)
    row_stop = min(int(rows.max()) + row_padding + 1, reference.shape[1])
    column_start = max(int(columns.min()) - column_padding, 0)
    column_stop = min(
        int(columns.max()) + column_padding + 1,
        reference.shape[2],
    )
    return (
        slice(None),
        slice(row_start, row_stop),
        slice(column_start, column_stop),
    )
