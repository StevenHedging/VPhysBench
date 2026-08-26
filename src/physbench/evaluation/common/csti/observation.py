"""Text-prompted candidate binding and locked CSTI Tube construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from physbench.evaluation.common.entities import EntitySpec

from .contracts import CSTIContractError


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


__all__ = [
    "CSTIObserverConfig",
    "EvaluatorInitFailure",
    "InitialIdentityMatch",
    "PromptGroupConfig",
    "SemanticCandidateTube",
    "match_initial_identities",
]
