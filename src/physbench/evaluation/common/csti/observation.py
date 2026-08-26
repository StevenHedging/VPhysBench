"""Text-prompted candidate binding and locked CSTI Tube construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from physbench.evaluation.common.entities import EntitySpec

from .contracts import (
    CSTIConfig,
    CSTIContractError,
    CSTIEntityTube,
    CSTIInput,
)
from .metric import zero_csti_metric


@dataclass(frozen=True)
class PromptGroupConfig:
    group_id: str
    text: str
    entity_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, value in (("id", self.group_id), ("text", self.text)):
            if not isinstance(value, str) or not value.strip():
                raise CSTIContractError(
                    "csti_observer_prompt_invalid",
                    f"CSTI observer prompt-group {label} must be non-empty",
                )
        if (
            not isinstance(self.entity_classes, tuple)
            or not self.entity_classes
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.entity_classes
            )
            or len(set(self.entity_classes)) != len(self.entity_classes)
        ):
            raise CSTIContractError(
                "csti_observer_prompt_invalid",
                "CSTI observer prompt-group entity_classes must be unique "
                "non-empty strings",
            )


@dataclass(frozen=True)
class CSTIObserverConfig:
    prompt_groups: tuple[PromptGroupConfig, ...]
    initial_match_iou_threshold: float
    initial_match_ambiguity_margin: float
    termination_patience: int
    minimum_mask_pixels: int
    minimum_observation_confidence: float
    segmenter: Mapping[str, Any] = field(default_factory=dict)
    debug_outputs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_groups, tuple) or not self.prompt_groups:
            raise CSTIContractError(
                "csti_observer_prompt_invalid",
                "CSTI observer requires at least one prompt group",
            )
        group_ids = [group.group_id for group in self.prompt_groups]
        if len(group_ids) != len(set(group_ids)):
            raise CSTIContractError(
                "csti_observer_prompt_invalid",
                "CSTI observer prompt-group IDs must be unique",
            )
        classes = [
            entity_class
            for group in self.prompt_groups
            for entity_class in group.entity_classes
        ]
        if len(classes) != len(set(classes)):
            raise CSTIContractError(
                "csti_observer_prompt_invalid",
                "Each entity class must belong to exactly one prompt group",
            )
        for label, value in (
            ("initial_match_iou_threshold", self.initial_match_iou_threshold),
            (
                "initial_match_ambiguity_margin",
                self.initial_match_ambiguity_margin,
            ),
            (
                "minimum_observation_confidence",
                self.minimum_observation_confidence,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise CSTIContractError(
                    "csti_observer_threshold_invalid",
                    f"CSTI observer {label} must lie in [0, 1]",
                )
        if (
            isinstance(self.termination_patience, bool)
            or not isinstance(self.termination_patience, int)
            or self.termination_patience < 1
        ):
            raise CSTIContractError(
                "csti_observer_patience_invalid",
                "CSTI observer termination_patience must be positive",
            )
        if (
            isinstance(self.minimum_mask_pixels, bool)
            or not isinstance(self.minimum_mask_pixels, int)
            or self.minimum_mask_pixels < 1
        ):
            raise CSTIContractError(
                "csti_observer_mask_threshold_invalid",
                "CSTI observer minimum_mask_pixels must be positive",
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CSTIObserverConfig":
        if not isinstance(value, Mapping):
            raise CSTIContractError(
                "csti_observer_config_invalid",
                "CSTI observer configuration must be a mapping",
            )
        required = {
            "prompt_groups",
            "initial_match_iou_threshold",
            "initial_match_ambiguity_margin",
            "termination_patience",
            "minimum_mask_pixels",
            "minimum_observation_confidence",
        }
        optional = {"segmenter", "debug_outputs"}
        missing = required - set(value)
        extra = set(value) - required - optional
        if missing or extra:
            raise CSTIContractError(
                "csti_observer_config_invalid",
                "CSTI observer configuration keys are invalid: "
                f"missing={sorted(missing)}, extra={sorted(extra)}",
            )
        raw_groups = value["prompt_groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise CSTIContractError(
                "csti_observer_prompt_invalid",
                "CSTI observer prompt_groups must be a non-empty array",
            )
        groups: list[PromptGroupConfig] = []
        for raw in raw_groups:
            if not isinstance(raw, Mapping) or set(raw) != {
                "id",
                "text",
                "entity_classes",
            }:
                raise CSTIContractError(
                    "csti_observer_prompt_invalid",
                    "Each CSTI observer prompt group must contain only id, "
                    "text, and entity_classes",
                )
            raw_classes = raw["entity_classes"]
            if not isinstance(raw_classes, list):
                raise CSTIContractError(
                    "csti_observer_prompt_invalid",
                    "CSTI observer entity_classes must be an array",
                )
            groups.append(
                PromptGroupConfig(
                    group_id=raw["id"],
                    text=raw["text"],
                    entity_classes=tuple(raw_classes),
                )
            )
        segmenter = value.get("segmenter", {})
        if not isinstance(segmenter, Mapping):
            raise CSTIContractError(
                "csti_observer_config_invalid",
                "CSTI observer segmenter must be a mapping",
            )
        debug_outputs = value.get("debug_outputs", False)
        if not isinstance(debug_outputs, bool):
            raise CSTIContractError(
                "csti_observer_config_invalid",
                "CSTI observer debug_outputs must be boolean",
            )
        return cls(
            prompt_groups=tuple(groups),
            initial_match_iou_threshold=_unit_float(
                value["initial_match_iou_threshold"],
                "initial_match_iou_threshold",
            ),
            initial_match_ambiguity_margin=_unit_float(
                value["initial_match_ambiguity_margin"],
                "initial_match_ambiguity_margin",
            ),
            termination_patience=_positive_int(
                value["termination_patience"], "termination_patience"
            ),
            minimum_mask_pixels=_positive_int(
                value["minimum_mask_pixels"], "minimum_mask_pixels"
            ),
            minimum_observation_confidence=_unit_float(
                value["minimum_observation_confidence"],
                "minimum_observation_confidence",
            ),
            segmenter=dict(segmenter),
            debug_outputs=debug_outputs,
        )

    def group_for_class(self, entity_class: str) -> PromptGroupConfig | None:
        return next(
            (
                group
                for group in self.prompt_groups
                if entity_class in group.entity_classes
            ),
            None,
        )


def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CSTIContractError(
            "csti_observer_threshold_invalid",
            f"CSTI observer {label} must be numeric",
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CSTIContractError(
            "csti_observer_threshold_invalid",
            f"CSTI observer {label} must lie in [0, 1]",
        )
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CSTIContractError(
            "csti_observer_integer_invalid",
            f"CSTI observer {label} must be a positive integer",
        )
    return value


@dataclass(frozen=True)
class SemanticCandidateTube:
    candidate_id: str
    prompt_group_id: str
    backend_object_id: int
    masks: np.ndarray
    boxes_xywh: np.ndarray
    confidences: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise CSTIContractError(
                "csti_candidate_invalid", "candidate_id must be non-empty"
            )
        if not isinstance(self.prompt_group_id, str) or not self.prompt_group_id:
            raise CSTIContractError(
                "csti_candidate_invalid", "prompt_group_id must be non-empty"
            )
        if isinstance(self.backend_object_id, bool) or not isinstance(
            self.backend_object_id, int
        ):
            raise CSTIContractError(
                "csti_candidate_invalid", "backend_object_id must be an integer"
            )
        masks = np.asarray(self.masks)
        if masks.ndim != 3 or not all(dimension > 0 for dimension in masks.shape):
            raise CSTIContractError(
                "csti_candidate_mask_invalid", "candidate masks must use THW layout"
            )
        if masks.dtype == np.bool_:
            normalized_masks = np.array(masks, dtype=bool, copy=True)
        elif np.issubdtype(masks.dtype, np.integer) and bool(
            np.all((masks == 0) | (masks == 1))
        ):
            normalized_masks = np.array(masks, dtype=bool, copy=True)
        else:
            raise CSTIContractError(
                "csti_candidate_mask_invalid",
                "candidate masks must be bool or binary integer arrays",
            )
        frame_count = masks.shape[0]
        boxes = np.asarray(self.boxes_xywh, dtype=np.float32)
        confidences = np.asarray(self.confidences, dtype=np.float32)
        if boxes.shape != (frame_count, 4) or not np.isfinite(boxes).all():
            raise CSTIContractError(
                "csti_candidate_boxes_invalid",
                "candidate boxes must be finite with shape [T,4]",
            )
        if (
            confidences.shape != (frame_count,)
            or not np.isfinite(confidences).all()
            or np.any((confidences < 0.0) | (confidences > 1.0))
        ):
            raise CSTIContractError(
                "csti_candidate_confidence_invalid",
                "candidate confidences must be finite [0,1] values with shape [T]",
            )
        normalized_masks.setflags(write=False)
        boxes = np.array(boxes, copy=True)
        confidences = np.array(confidences, copy=True)
        boxes.setflags(write=False)
        confidences.setflags(write=False)
        object.__setattr__(self, "masks", normalized_masks)
        object.__setattr__(self, "boxes_xywh", boxes)
        object.__setattr__(self, "confidences", confidences)


@dataclass(frozen=True)
class EvaluatorInitFailure:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitialIdentityMatch:
    success: bool
    initial_matching: Mapping[str, str]
    matching_iou: Mapping[str, float]
    ignored_candidate_ids: tuple[str, ...]
    failure: EvaluatorInitFailure | None = None


@dataclass(frozen=True)
class TrackObservationValidity:
    valid: bool
    reason: str | None
    area_pixels: int


@dataclass(frozen=True)
class LockedPredictionTubes:
    prediction_masks_by_entity: Mapping[str, tuple[np.ndarray, ...]]
    termination_frame_per_subject: Mapping[str, int | None]
    final_state_per_subject: Mapping[str, str]
    pending_invalid_frames_per_subject: Mapping[str, int]
    invalid_reason_per_frame: Mapping[str, tuple[str | None, ...]]


@dataclass(frozen=True)
class CSTIObservationResult:
    evaluator_init_success: bool
    csti_input: CSTIInput | None
    initial_match: InitialIdentityMatch
    locked_tubes: LockedPredictionTubes | None
    candidate_count: int


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / union) if union else 1.0


def _failure(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> InitialIdentityMatch:
    return InitialIdentityMatch(
        success=False,
        initial_matching={},
        matching_iou={},
        ignored_candidate_ids=tuple(sorted(candidate_ids)),
        failure=EvaluatorInitFailure(code=code, message=message, details=details),
    )


def _best_alternative_score(
    ious: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    threshold: float,
) -> float | None:
    best: float | None = None
    for forbidden_row, forbidden_column in zip(rows, columns, strict=True):
        alternative_cost = 1.0 - ious
        alternative_cost = np.array(alternative_cost, copy=True)
        alternative_cost[forbidden_row, forbidden_column] = 1e6
        alt_rows, alt_columns = linear_sum_assignment(alternative_cost)
        if any(
            row == forbidden_row and column == forbidden_column
            for row, column in zip(alt_rows, alt_columns, strict=True)
        ):
            continue
        if len(alt_rows) != ious.shape[0]:
            continue
        selected = ious[alt_rows, alt_columns]
        if np.any(selected < threshold):
            continue
        score = float(np.mean(selected, dtype=np.float64))
        if best is None or score > best:
            best = score
    return best


def match_initial_identities(
    *,
    entities: Sequence[EntitySpec],
    reference_masks_by_entity: Mapping[str, np.ndarray],
    candidates: Sequence[SemanticCandidateTube],
    config: CSTIObserverConfig,
) -> InitialIdentityMatch:
    """Bind GT entities to compatible SAM tracks once on frame zero."""

    entity_tuple = tuple(entities)
    candidate_tuple = tuple(candidates)
    entity_ids = tuple(entity.entity_id for entity in entity_tuple)
    if not entity_tuple or len(entity_ids) != len(set(entity_ids)):
        raise CSTIContractError(
            "csti_observer_entities_invalid",
            "CSTI observer entities must be non-empty and unique",
        )
    if set(reference_masks_by_entity) != set(entity_ids):
        raise CSTIContractError(
            "csti_observer_reference_coverage_mismatch",
            "CSTI observer first-frame references must cover every entity",
        )
    candidate_ids = tuple(candidate.candidate_id for candidate in candidate_tuple)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CSTIContractError(
            "csti_candidate_duplicate", "CSTI observer candidate IDs must be unique"
        )

    normalized_reference: dict[str, np.ndarray] = {}
    frame_shape: tuple[int, int] | None = None
    for entity_id in entity_ids:
        mask = np.asarray(reference_masks_by_entity[entity_id])
        if mask.ndim != 2 or mask.dtype != np.bool_:
            if not (
                mask.ndim == 2
                and np.issubdtype(mask.dtype, np.integer)
                and bool(np.all((mask == 0) | (mask == 1)))
            ):
                raise CSTIContractError(
                    "csti_observer_reference_mask_invalid",
                    "CSTI observer references must be binary two-dimensional masks",
                )
        normalized = np.asarray(mask, dtype=bool)
        if frame_shape is None:
            frame_shape = normalized.shape
        elif normalized.shape != frame_shape:
            raise CSTIContractError(
                "csti_observer_reference_shape_mismatch",
                "CSTI observer reference masks must share one shape",
            )
        normalized_reference[entity_id] = normalized

    mappings: dict[str, str] = {}
    matching_ious: dict[str, float] = {}
    used_candidates: set[str] = set()
    group_for_entity: dict[str, PromptGroupConfig] = {}
    for entity in entity_tuple:
        group = config.group_for_class(entity.entity_class)
        if group is None:
            raise CSTIContractError(
                "csti_observer_entity_class_unmapped",
                f"No CSTI text prompt maps entity class {entity.entity_class!r}",
            )
        group_for_entity[entity.entity_id] = group

    for group in config.prompt_groups:
        group_entities = tuple(
            entity
            for entity in entity_tuple
            if group_for_entity[entity.entity_id].group_id == group.group_id
        )
        if not group_entities:
            continue
        group_candidates = tuple(
            candidate
            for candidate in candidate_tuple
            if candidate.prompt_group_id == group.group_id
        )
        if len(group_candidates) < len(group_entities):
            return _failure(
                "insufficient_semantic_candidates",
                "SAM detected fewer compatible instances than the manifest expects",
                details={
                    "prompt_group_id": group.group_id,
                    "expected_count": len(group_entities),
                    "detected_count": len(group_candidates),
                },
                candidate_ids=candidate_ids,
            )
        assert frame_shape is not None
        for candidate in group_candidates:
            if candidate.masks.shape[1:] != frame_shape:
                raise CSTIContractError(
                    "csti_candidate_shape_mismatch",
                    "SAM candidate mask canvas differs from aligned GT masks",
                )
        ious = np.asarray(
            [
                [
                    _mask_iou(
                        normalized_reference[entity.entity_id],
                        candidate.masks[0],
                    )
                    for candidate in group_candidates
                ]
                for entity in group_entities
            ],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(1.0 - ious)
        if len(rows) != len(group_entities):
            return _failure(
                "incomplete_initial_assignment",
                "SAM candidates cannot form a complete one-to-one assignment",
                details={"prompt_group_id": group.group_id},
                candidate_ids=candidate_ids,
            )
        selected = ious[rows, columns]
        group_matching_iou = {
            group_entities[row].entity_id: float(ious[row, column])
            for row, column in zip(rows, columns, strict=True)
        }
        if np.any(selected < config.initial_match_iou_threshold):
            return _failure(
                "initial_match_iou_below_threshold",
                "At least one initial SAM-to-GT match is below the configured IoU",
                details={
                    "prompt_group_id": group.group_id,
                    "threshold": config.initial_match_iou_threshold,
                    "matching_iou": group_matching_iou,
                },
                candidate_ids=candidate_ids,
            )
        best_score = float(np.mean(selected, dtype=np.float64))
        alternative_score = _best_alternative_score(
            ious,
            rows,
            columns,
            threshold=config.initial_match_iou_threshold,
        )
        margin = (
            math.inf
            if alternative_score is None
            else best_score - alternative_score
        )
        if (
            alternative_score is not None
            and margin < config.initial_match_ambiguity_margin
        ):
            return _failure(
                "initial_assignment_ambiguous",
                "Multiple complete initial identity assignments are too similar",
                details={
                    "prompt_group_id": group.group_id,
                    "best_assignment_score": best_score,
                    "alternative_assignment_score": alternative_score,
                    "assignment_margin": margin,
                    "required_margin": config.initial_match_ambiguity_margin,
                },
                candidate_ids=candidate_ids,
            )
        for row, column in zip(rows, columns, strict=True):
            entity_id = group_entities[row].entity_id
            candidate_id = group_candidates[column].candidate_id
            mappings[entity_id] = candidate_id
            matching_ious[entity_id] = float(ious[row, column])
            used_candidates.add(candidate_id)

    if set(mappings) != set(entity_ids):
        return _failure(
            "incomplete_initial_assignment",
            "Initial identity mapping does not cover every legal entity",
            details={
                "expected_entities": list(entity_ids),
                "matched_entities": sorted(mappings),
            },
            candidate_ids=candidate_ids,
        )
    return InitialIdentityMatch(
        success=True,
        initial_matching=dict(mappings),
        matching_iou=dict(matching_ious),
        ignored_candidate_ids=tuple(sorted(set(candidate_ids) - used_candidates)),
        failure=None,
    )


def is_valid_track_observation(
    mask: object,
    *,
    confidence: object,
    frame_shape: tuple[int, int],
    config: CSTIObserverConfig,
) -> TrackObservationValidity:
    """Validate only documented backend mask and confidence evidence."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.shape != frame_shape:
        return TrackObservationValidity(False, "mask_shape_invalid", 0)
    if value.dtype == np.bool_:
        binary = value
    elif np.issubdtype(value.dtype, np.integer) and bool(
        np.all((value == 0) | (value == 1))
    ):
        binary = value > 0
    else:
        return TrackObservationValidity(False, "mask_format_invalid", 0)
    area = int(np.count_nonzero(binary))
    if area == 0:
        return TrackObservationValidity(False, "mask_empty", 0)
    if area < config.minimum_mask_pixels:
        return TrackObservationValidity(False, "mask_area_below_minimum", area)
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float, np.integer, np.floating)
    ):
        return TrackObservationValidity(False, "confidence_invalid", area)
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        return TrackObservationValidity(False, "confidence_invalid", area)
    if confidence_value < config.minimum_observation_confidence:
        return TrackObservationValidity(False, "confidence_below_minimum", area)
    return TrackObservationValidity(True, None, area)


def _read_only_mask(mask: np.ndarray, *, frame_shape: tuple[int, int]) -> np.ndarray:
    value = np.array(mask, dtype=bool, copy=True)
    if value.shape != frame_shape:
        raise CSTIContractError(
            "csti_candidate_shape_mismatch",
            "CSTI prediction mask does not match the evaluator frame shape",
        )
    value.setflags(write=False)
    return value


def _empty_mask(frame_shape: tuple[int, int]) -> np.ndarray:
    return _read_only_mask(np.zeros(frame_shape, dtype=bool), frame_shape=frame_shape)


def build_locked_prediction_tubes(
    *,
    entities: Sequence[EntitySpec],
    candidates: Sequence[SemanticCandidateTube],
    initial_match: InitialIdentityMatch,
    frame_count: int,
    frame_shape: tuple[int, int],
    config: CSTIObserverConfig,
) -> LockedPredictionTubes:
    """Build full-length Tubes without changing the frame-zero identity map."""

    entity_tuple = tuple(entities)
    entity_ids = tuple(entity.entity_id for entity in entity_tuple)
    if not initial_match.success:
        raise CSTIContractError(
            "csti_initial_match_unsuccessful",
            "Cannot build locked Tubes from an unsuccessful initial match",
        )
    if set(initial_match.initial_matching) != set(entity_ids):
        raise CSTIContractError(
            "csti_locked_mapping_incomplete",
            "Locked CSTI identity mapping must cover every entity",
        )
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise CSTIContractError(
            "csti_observer_frame_count_invalid",
            "CSTI observer frame_count must be positive",
        )
    if (
        not isinstance(frame_shape, (tuple, list))
        or len(frame_shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in frame_shape
        )
    ):
        raise CSTIContractError(
            "csti_observer_frame_shape_invalid",
            "CSTI observer frame_shape must contain two positive integers",
        )
    normalized_shape = (int(frame_shape[0]), int(frame_shape[1]))
    by_candidate = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_candidate) != len(tuple(candidates)):
        raise CSTIContractError(
            "csti_candidate_duplicate", "CSTI observer candidate IDs must be unique"
        )

    prediction_masks: dict[str, tuple[np.ndarray, ...]] = {}
    termination_frames: dict[str, int | None] = {}
    final_states: dict[str, str] = {}
    pending_counts: dict[str, int] = {}
    invalid_reasons: dict[str, tuple[str | None, ...]] = {}
    for entity in entity_tuple:
        candidate_id = initial_match.initial_matching[entity.entity_id]
        candidate = by_candidate.get(candidate_id)
        if candidate is None:
            raise CSTIContractError(
                "csti_locked_candidate_missing",
                f"Locked SAM candidate {candidate_id!r} is absent",
            )
        if candidate.masks.shape != (frame_count, *normalized_shape):
            raise CSTIContractError(
                "csti_candidate_shape_mismatch",
                "Locked SAM candidate Tube differs from the evaluation timeline/canvas",
            )
        if candidate.confidences.shape != (frame_count,):
            raise CSTIContractError(
                "csti_candidate_confidence_invalid",
                "Locked SAM confidence series differs from the evaluation timeline",
            )

        output = [_empty_mask(normalized_shape) for _ in range(frame_count)]
        reasons: list[str | None] = [None] * frame_count
        consecutive_invalid = 0
        candidate_termination_frame: int | None = None
        termination_frame: int | None = None
        terminated = False
        for frame_index in range(frame_count):
            if terminated:
                reasons[frame_index] = "track_terminated"
                continue
            validity = is_valid_track_observation(
                candidate.masks[frame_index],
                confidence=float(candidate.confidences[frame_index]),
                frame_shape=normalized_shape,
                config=config,
            )
            if validity.valid:
                output[frame_index] = _read_only_mask(
                    candidate.masks[frame_index], frame_shape=normalized_shape
                )
                consecutive_invalid = 0
                candidate_termination_frame = None
                continue
            reasons[frame_index] = validity.reason
            if consecutive_invalid == 0:
                candidate_termination_frame = frame_index
            consecutive_invalid += 1
            if consecutive_invalid >= config.termination_patience:
                assert candidate_termination_frame is not None
                termination_frame = candidate_termination_frame
                for clear_index in range(termination_frame, frame_count):
                    output[clear_index] = _empty_mask(normalized_shape)
                terminated = True

        prediction_masks[entity.entity_id] = tuple(output)
        termination_frames[entity.entity_id] = termination_frame
        final_states[entity.entity_id] = (
            "TERMINATED" if terminated else "VIDEO_END"
        )
        pending_counts[entity.entity_id] = 0 if terminated else consecutive_invalid
        invalid_reasons[entity.entity_id] = tuple(reasons)

    return LockedPredictionTubes(
        prediction_masks_by_entity=prediction_masks,
        termination_frame_per_subject=termination_frames,
        final_state_per_subject=final_states,
        pending_invalid_frames_per_subject=pending_counts,
        invalid_reason_per_frame=invalid_reasons,
    )


def observe_csti_tubes(
    *,
    aligned_input: CSTIInput,
    entities: Sequence[EntitySpec],
    candidates: Sequence[SemanticCandidateTube],
    config: CSTIObserverConfig,
) -> CSTIObservationResult:
    """Replace only aligned CSTI prediction Tubes after one initial match."""

    if not isinstance(aligned_input, CSTIInput):
        raise CSTIContractError(
            "csti_observer_input_invalid",
            "CSTI observer requires an aligned CSTIInput",
        )
    entity_tuple = tuple(entities)
    specs_by_id = {entity.entity_id: entity for entity in entity_tuple}
    aligned_by_id = {
        entity.entity_id: entity for entity in aligned_input.entities
    }
    if (
        len(specs_by_id) != len(entity_tuple)
        or len(aligned_by_id) != len(aligned_input.entities)
        or set(specs_by_id) != set(aligned_by_id)
    ):
        raise CSTIContractError(
            "csti_observer_entity_coverage_mismatch",
            "CSTI observer entities must exactly cover the aligned input",
        )
    for entity_id, spec in specs_by_id.items():
        if aligned_by_id[entity_id].role_id != spec.role_id:
            raise CSTIContractError(
                "csti_observer_role_mismatch",
                f"CSTI observer role differs for {entity_id!r}",
            )
    candidate_tuple = tuple(candidates)
    initial = match_initial_identities(
        entities=entity_tuple,
        reference_masks_by_entity={
            entity.entity_id: entity.reference_masks[0]
            for entity in aligned_input.entities
        },
        candidates=candidate_tuple,
        config=config,
    )
    if not initial.success:
        return CSTIObservationResult(
            evaluator_init_success=False,
            csti_input=None,
            initial_match=initial,
            locked_tubes=None,
            candidate_count=len(candidate_tuple),
        )
    locked = build_locked_prediction_tubes(
        entities=entity_tuple,
        candidates=candidate_tuple,
        initial_match=initial,
        frame_count=len(aligned_input.times_s),
        frame_shape=aligned_input.frame_shape,
        config=config,
    )
    observed_entities = tuple(
        CSTIEntityTube(
            entity_id=entity.entity_id,
            role_id=entity.role_id,
            reference_masks=entity.reference_masks,
            prediction_masks=locked.prediction_masks_by_entity[
                entity.entity_id
            ],
            matched_prediction_track_ids=(
                initial.initial_matching[entity.entity_id],
            ),
        )
        for entity in aligned_input.entities
    )
    return CSTIObservationResult(
        evaluator_init_success=True,
        csti_input=CSTIInput(
            reference_capability=aligned_input.reference_capability,
            times_s=aligned_input.times_s,
            frame_shape=aligned_input.frame_shape,
            entities=observed_entities,
        ),
        initial_match=initial,
        locked_tubes=locked,
        candidate_count=len(candidate_tuple),
    )


def _failure_reason_mapping(
    failure: EvaluatorInitFailure | None,
) -> dict[str, Any] | None:
    if failure is None:
        return None
    return {
        "code": failure.code,
        "message": failure.message,
        "details": dict(failure.details),
    }


def decorate_csti_metric(
    metric: Mapping[str, Any],
    *,
    observation: CSTIObservationResult,
) -> dict[str, Any]:
    """Add observer audit fields around an already-computed CSTI metric."""

    if not observation.evaluator_init_success or observation.locked_tubes is None:
        raise CSTIContractError(
            "csti_observer_not_initialized",
            "Cannot decorate CSTI from a failed observer initialization",
        )
    value = dict(metric)
    if value.get("status") != "evaluated" or not isinstance(
        value.get("objects"), list
    ):
        raise CSTIContractError(
            "csti_metric_decoration_invalid",
            "Only an evaluated CSTI metric can receive observer diagnostics",
        )
    per_subject = {
        str(item["entity_id"]): float(item["score"])
        for item in value["objects"]
    }
    value.update(
        {
            "csti_video": float(value["score"]),
            "csti_per_subject": per_subject,
            "subject_count": len(per_subject),
            "evaluator_init_success": True,
            "evaluator_init_failure_reason": None,
            "initial_matching": dict(
                observation.initial_match.initial_matching
            ),
            "initial_matching_iou": dict(
                observation.initial_match.matching_iou
            ),
            "termination_frame_per_subject": dict(
                observation.locked_tubes.termination_frame_per_subject
            ),
            "observer": {
                "candidate_count": observation.candidate_count,
                "ignored_candidate_ids": list(
                    observation.initial_match.ignored_candidate_ids
                ),
                "final_state_per_subject": dict(
                    observation.locked_tubes.final_state_per_subject
                ),
                "pending_invalid_frames_per_subject": dict(
                    observation.locked_tubes.pending_invalid_frames_per_subject
                ),
                "invalid_reason_per_frame": {
                    entity_id: list(reasons)
                    for entity_id, reasons in (
                        observation.locked_tubes.invalid_reason_per_frame.items()
                    )
                },
            },
        }
    )
    return value


def evaluator_init_failure_metric(
    *,
    observation: CSTIObservationResult,
    expected_entities: Sequence[tuple[str, str]],
    config: CSTIConfig,
) -> dict[str, Any]:
    """Build a visible null-score record for evaluator initialization loss."""

    if observation.evaluator_init_success:
        raise CSTIContractError(
            "csti_observer_initialized",
            "Successful CSTI observation is not an initialization failure",
        )
    value = zero_csti_metric(
        expected_entities=expected_entities,
        config=config,
        degradation_code="evaluator_init_failure",
        degradation_reason=(
            observation.initial_match.failure.message
            if observation.initial_match.failure is not None
            else "CSTI evaluator initialization failed"
        ),
    )
    value["status"] = "evaluator_init_failure"
    value["score"] = None
    value.pop("degradation", None)
    for item in value["objects"]:
        item["score"] = None
    expected = tuple(expected_entities)
    value.update(
        {
            "csti_video": None,
            "csti_per_subject": {
                entity_id: None for entity_id, _ in expected
            },
            "subject_count": len(expected),
            "evaluator_init_success": False,
            "evaluator_init_failure_reason": _failure_reason_mapping(
                observation.initial_match.failure
            ),
            "initial_matching": dict(
                observation.initial_match.initial_matching
            ),
            "initial_matching_iou": dict(
                observation.initial_match.matching_iou
            ),
            "termination_frame_per_subject": {
                entity_id: None for entity_id, _ in expected
            },
            "observer": {
                "candidate_count": observation.candidate_count,
                "ignored_candidate_ids": list(
                    observation.initial_match.ignored_candidate_ids
                ),
            },
        }
    )
    return value


__all__ = [
    "CSTIObserverConfig",
    "CSTIObservationResult",
    "EvaluatorInitFailure",
    "InitialIdentityMatch",
    "LockedPredictionTubes",
    "PromptGroupConfig",
    "SemanticCandidateTube",
    "TrackObservationValidity",
    "build_locked_prediction_tubes",
    "decorate_csti_metric",
    "evaluator_init_failure_metric",
    "is_valid_track_observation",
    "match_initial_identities",
    "observe_csti_tubes",
]
