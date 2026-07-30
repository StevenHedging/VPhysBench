from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import EntityMatch


@dataclass(frozen=True)
class PositionComparison:
    centroid_distance_px: float
    tolerance_px: float
    scale_px: float
    normalized_distance: float
    score: float
    kernel: str


@dataclass(frozen=True)
class GOSPADecomposition:
    distance: float
    normalized_distance: float
    localization_power: float
    missed_power: float
    false_power: float
    missed_exposure: float
    false_exposure: float
    cutoff: float
    order: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        return {
            "distance": self.distance,
            "normalized_distance": self.normalized_distance,
            "localization_power": self.localization_power,
            "missed_power": self.missed_power,
            "false_power": self.false_power,
            "missed_exposure": self.missed_exposure,
            "false_exposure": self.false_exposure,
            "cutoff": self.cutoff,
            "order": self.order,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class EntityIntegrityScore:
    score: float
    integrity_gate: float
    soft_detection_accuracy: float
    association_accuracy: float
    matched_localization_quality: float
    matched_exposure: float
    reference_exposure: float
    prediction_exposure: float
    detection_recall: float
    detection_precision: float
    pair_association: dict[str, float]
    gospa: GOSPADecomposition

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "integrity_gate": self.integrity_gate,
            "soft_detection_accuracy": self.soft_detection_accuracy,
            "association_accuracy": self.association_accuracy,
            "matched_localization_quality": (
                self.matched_localization_quality
            ),
            "matched_exposure": self.matched_exposure,
            "reference_exposure": self.reference_exposure,
            "prediction_exposure": self.prediction_exposure,
            "detection_recall": self.detection_recall,
            "detection_precision": self.detection_precision,
            "pair_association": self.pair_association,
            "gospa": self.gospa.to_dict(),
        }


@dataclass(frozen=True)
class GeometricComposition:
    score: float | None
    weights_used: dict[str, float]
    omitted_components: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "weights_used": self.weights_used,
            "omitted_components": list(self.omitted_components),
        }


@dataclass(frozen=True)
class GatedCaseComposition:
    score: float | None
    integrity_gate: float
    content: GeometricComposition

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "integrity_gate": self.integrity_gate,
            "content": self.content.to_dict(),
            "formula": "integrity_gate_times_content",
        }


def distance_similarity(
    normalized_distance: float,
    *,
    kernel: str = "cauchy",
) -> float:
    """Continuous position score with non-zero support beyond mask overlap."""
    distance = max(float(normalized_distance), 0.0)
    if not math.isfinite(distance):
        return 0.0
    if kernel == "cauchy":
        return float(1.0 / (1.0 + distance * distance))
    if kernel == "gaussian":
        exponent = min(0.5 * distance * distance, 745.0)
        return float(math.exp(-exponent))
    raise ValueError(f"unsupported distance kernel: {kernel}")


def compare_positions(
    reference_xy: Sequence[float],
    prediction_xy: Sequence[float],
    *,
    reference_area_px2: float,
    frame_diagonal_px: float,
    radius_multiplier: float = 2.0,
    diagonal_floor_fraction: float = 0.02,
    jitter_tolerance_radius_fraction: float = 0.05,
    kernel: str = "cauchy",
) -> PositionComparison:
    """Compare centroids using only reference/condition-side scale."""
    parameters = {
        "radius_multiplier": radius_multiplier,
        "diagonal_floor_fraction": diagonal_floor_fraction,
        "jitter_tolerance_radius_fraction": (
            jitter_tolerance_radius_fraction
        ),
    }
    for name, raw in parameters.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if radius_multiplier == 0.0 and diagonal_floor_fraction == 0.0:
        raise ValueError("at least one position scale must be positive")
    reference = np.asarray(reference_xy, dtype=np.float64)
    prediction = np.asarray(prediction_xy, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 1:
        raise ValueError("positions must have the same one-dimensional shape")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        return PositionComparison(
            centroid_distance_px=float("inf"),
            tolerance_px=0.0,
            scale_px=1.0,
            normalized_distance=float("inf"),
            score=0.0,
            kernel=kernel,
        )
    area = max(float(reference_area_px2), 0.0)
    diagonal = max(float(frame_diagonal_px), 0.0)
    if not math.isfinite(area):
        area = 0.0
    if not math.isfinite(diagonal):
        diagonal = 0.0
    radius = math.sqrt(area / math.pi)
    scale = max(
        radius_multiplier * radius,
        diagonal_floor_fraction * diagonal,
        1e-9,
    )
    tolerance = max(jitter_tolerance_radius_fraction * radius, 0.0)
    distance = float(np.linalg.norm(reference - prediction))
    normalized = max(distance - tolerance, 0.0) / scale
    return PositionComparison(
        centroid_distance_px=distance,
        tolerance_px=tolerance,
        scale_px=scale,
        normalized_distance=normalized,
        score=distance_similarity(normalized, kernel=kernel),
        kernel=kernel,
    )


def compose_weighted_geometric(
    components: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> GeometricComposition:
    """Combine comparable [0,1] components without letting a high term hide zero."""
    usable: dict[str, tuple[float, float]] = {}
    omitted: list[str] = []
    for name, value in components.items():
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            omitted.append(name)
            continue
        if value is None or not math.isfinite(float(value)):
            omitted.append(name)
            continue
        usable[name] = (float(np.clip(value, 0.0, 1.0)), weight)
    if not usable:
        return GeometricComposition(
            score=None,
            weights_used={},
            omitted_components=tuple(sorted(omitted)),
        )
    total_weight = sum(weight for _, weight in usable.values())
    normalized_weights = {
        name: weight / total_weight
        for name, (_, weight) in usable.items()
    }
    if any(value == 0.0 for value, _ in usable.values()):
        score = 0.0
    else:
        score = math.exp(
            sum(
                normalized_weights[name] * math.log(value)
                for name, (value, _) in usable.items()
            )
        )
    return GeometricComposition(
        score=float(np.clip(score, 0.0, 1.0)),
        weights_used=normalized_weights,
        omitted_components=tuple(sorted(omitted)),
    )


def compose_gated_case_score(
    integrity_gate: float,
    content_components: Mapping[str, float | None],
    content_weights: Mapping[str, float],
) -> GatedCaseComposition:
    """Apply cardinality/identity integrity as a non-dilutable case gate."""
    gate = float(integrity_gate)
    if not math.isfinite(gate):
        gate = 0.0
    gate = float(np.clip(gate, 0.0, 1.0))
    content = compose_weighted_geometric(
        content_components,
        content_weights,
    )
    score = (
        None
        if content.score is None
        else float(np.clip(gate * content.score, 0.0, 1.0))
    )
    return GatedCaseComposition(
        score=score,
        integrity_gate=gate,
        content=content,
    )


def score_entity_integrity(
    matches: Sequence[EntityMatch],
    *,
    reference_exposure: Mapping[str, float],
    prediction_exposure: Mapping[str, float],
    gospa_order: float = 2.0,
    gospa_cutoff: float = 1.0,
    gospa_alpha: float = 2.0,
) -> EntityIntegrityScore:
    """Continuous HOTA-style entity-set and identity score.

    `prediction_exposure` must include unmatched residual tracks. This is what
    makes persistent extra or duplicated physical objects lower the score.
    """
    reference = _sanitize_exposure(reference_exposure, "reference")
    prediction = _sanitize_exposure(prediction_exposure, "prediction")
    _validate_one_to_one(matches)
    for match in matches:
        if match.entity_id not in reference:
            raise ValueError(
                f"match references unknown entity {match.entity_id!r}"
            )
        if match.track_id not in prediction:
            raise ValueError(
                f"match references unknown track {match.track_id!r}"
            )

    reference_total = float(sum(reference.values()))
    prediction_total = float(sum(prediction.values()))
    matched_exposure = float(sum(match.weight for match in matches))
    maximum_matched = min(reference_total, prediction_total)
    if matched_exposure > maximum_matched + 1e-9:
        raise ValueError(
            "matched exposure exceeds reference or prediction exposure"
        )
    matched_localization_quality = float(
        sum(
            match.weight * match.localization_quality
            for match in matches
        )
    )
    matched_localization_quality = float(
        np.clip(matched_localization_quality, 0.0, maximum_matched)
    )
    denominator = (
        reference_total + prediction_total - matched_localization_quality
    )
    if denominator <= 1e-12:
        detection_accuracy = 1.0
    else:
        detection_accuracy = matched_localization_quality / denominator

    pair_exposure: dict[tuple[str, str], float] = {}
    for match in matches:
        key = match.entity_id, match.track_id
        pair_exposure[key] = pair_exposure.get(key, 0.0) + match.weight
    pair_association: dict[str, float] = {}
    association_numerator = 0.0
    for (entity_id, track_id), exposure in pair_exposure.items():
        pair_maximum = min(reference[entity_id], prediction[track_id])
        if exposure > pair_maximum + 1e-9:
            raise ValueError(
                f"pair exposure exceeds lifecycle for {entity_id}/{track_id}"
            )
        pair_union = (
            reference[entity_id] + prediction[track_id] - exposure
        )
        association = exposure / pair_union if pair_union > 1e-12 else 1.0
        pair_association[f"{entity_id}::{track_id}"] = float(association)
        association_numerator += exposure * association
    association_accuracy = (
        association_numerator / matched_exposure
        if matched_exposure > 1e-12
        else (1.0 if reference_total <= 1e-12 and prediction_total <= 1e-12 else 0.0)
    )
    integrity_gate = (
        max(detection_accuracy, 0.0) * max(association_accuracy, 0.0)
    )
    score = math.sqrt(integrity_gate)
    recall = (
        matched_exposure / reference_total
        if reference_total > 1e-12
        else (1.0 if prediction_total <= 1e-12 else 0.0)
    )
    precision = (
        matched_exposure / prediction_total
        if prediction_total > 1e-12
        else (1.0 if reference_total <= 1e-12 else 0.0)
    )
    gospa = _gospa(
        matches,
        reference_total=reference_total,
        prediction_total=prediction_total,
        matched_exposure=matched_exposure,
        order=gospa_order,
        cutoff=gospa_cutoff,
        alpha=gospa_alpha,
    )
    return EntityIntegrityScore(
        score=float(np.clip(score, 0.0, 1.0)),
        integrity_gate=float(np.clip(integrity_gate, 0.0, 1.0)),
        soft_detection_accuracy=float(
            np.clip(detection_accuracy, 0.0, 1.0)
        ),
        association_accuracy=float(
            np.clip(association_accuracy, 0.0, 1.0)
        ),
        matched_localization_quality=matched_localization_quality,
        matched_exposure=matched_exposure,
        reference_exposure=reference_total,
        prediction_exposure=prediction_total,
        detection_recall=float(np.clip(recall, 0.0, 1.0)),
        detection_precision=float(np.clip(precision, 0.0, 1.0)),
        pair_association=pair_association,
        gospa=gospa,
    )


def _sanitize_exposure(
    values: Mapping[str, float],
    label: str,
) -> dict[str, float]:
    output = {}
    for item_id, raw in values.items():
        value = float(raw)
        if not item_id:
            raise ValueError(f"{label} exposure IDs must be non-empty")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{label} exposure for {item_id!r} must be finite and >= 0"
            )
        output[str(item_id)] = value
    return output


def _validate_one_to_one(matches: Sequence[EntityMatch]) -> None:
    entity_slots: set[tuple[int, str]] = set()
    track_slots: set[tuple[int, str]] = set()
    for match in matches:
        entity_slot = match.frame_index, match.entity_id
        track_slot = match.frame_index, match.track_id
        if entity_slot in entity_slots:
            raise ValueError(
                f"entity {match.entity_id!r} matched twice in frame "
                f"{match.frame_index}"
            )
        if track_slot in track_slots:
            raise ValueError(
                f"track {match.track_id!r} matched twice in frame "
                f"{match.frame_index}"
            )
        entity_slots.add(entity_slot)
        track_slots.add(track_slot)


def _gospa(
    matches: Sequence[EntityMatch],
    *,
    reference_total: float,
    prediction_total: float,
    matched_exposure: float,
    order: float,
    cutoff: float,
    alpha: float,
) -> GOSPADecomposition:
    if order < 1.0:
        raise ValueError("GOSPA order must be >= 1")
    if cutoff <= 0.0:
        raise ValueError("GOSPA cutoff must be positive")
    if not 0.0 < alpha <= 2.0:
        raise ValueError("GOSPA alpha must be in (0, 2]")
    localization_power = float(
        sum(
            match.weight
            * min(cutoff, match.normalized_distance) ** order
            for match in matches
        )
    )
    missed_exposure = max(reference_total - matched_exposure, 0.0)
    false_exposure = max(prediction_total - matched_exposure, 0.0)
    unit_cardinality_power = cutoff**order / alpha
    missed_power = unit_cardinality_power * missed_exposure
    false_power = unit_cardinality_power * false_exposure
    total_power = localization_power + missed_power + false_power
    distance = total_power ** (1.0 / order)
    union_exposure = max(
        reference_total + prediction_total - matched_exposure,
        1.0,
    )
    normalized_distance = (
        total_power / union_exposure
    ) ** (1.0 / order) / cutoff
    return GOSPADecomposition(
        distance=float(distance),
        normalized_distance=float(normalized_distance),
        localization_power=localization_power,
        missed_power=float(missed_power),
        false_power=float(false_power),
        missed_exposure=float(missed_exposure),
        false_exposure=float(false_exposure),
        cutoff=float(cutoff),
        order=float(order),
        alpha=float(alpha),
    )
