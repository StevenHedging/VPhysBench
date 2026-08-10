from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import (
    EntityMatch,
    ObjectTrack,
    VisibilityState,
)
from .scoring import (
    EntityIntegrityScore,
    compare_positions,
    score_entity_integrity,
)
from .timeline import CommonTimeGrid


class EvidenceTier(str, Enum):
    """Protocol-frozen strength assigned by the evaluator, not the model."""

    PARTICIPANT = "participant"
    INDEPENDENT_SALIENT = "independent_salient"
    TENTATIVE = "tentative"
    AMBIGUOUS = "ambiguous"


FORMAL_EXPOSURE_WEIGHTS: Mapping[EvidenceTier, float] = {
    EvidenceTier.PARTICIPANT: 1.0,
    EvidenceTier.INDEPENDENT_SALIENT: 0.5,
    EvidenceTier.TENTATIVE: 0.25,
    EvidenceTier.AMBIGUOUS: 0.0,
}


@dataclass(frozen=True)
class ObjectDetection:
    """One evaluator-produced object observation at a shared time sample."""

    frame_index: int
    detection_id: str
    xy: np.ndarray
    area_px2: float
    entity_class: str
    mask: np.ndarray | None = None
    confidence: float = 1.0
    evidence_tier: EvidenceTier = EvidenceTier.TENTATIVE
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not self.detection_id:
            raise ValueError("detection_id must be non-empty")
        if not self.entity_class:
            raise ValueError("entity_class must be non-empty")
        xy = np.asarray(self.xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("xy must contain two finite coordinates")
        area = float(self.area_px2)
        confidence = float(self.confidence)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("area_px2 must be finite and positive")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        tier = (
            self.evidence_tier
            if isinstance(self.evidence_tier, EvidenceTier)
            else EvidenceTier(self.evidence_tier)
        )
        mask = self.mask
        if mask is not None:
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.ndim != 2:
                raise ValueError("mask must be a two-dimensional array")
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            mask.setflags(write=False)
        normalized_xy = np.array(xy, copy=True)
        normalized_xy.setflags(write=False)
        object.__setattr__(self, "xy", normalized_xy)
        object.__setattr__(self, "area_px2", area)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_tier", tier)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(
            self,
            "sources",
            tuple(sorted({str(value) for value in self.sources if str(value)})),
        )

    @property
    def formal_exposure_weight(self) -> float:
        return float(FORMAL_EXPOSURE_WEIGHTS[self.evidence_tier])


@dataclass(frozen=True)
class OpenWorldTrack:
    track_id: str
    detections: tuple[ObjectDetection, ...]
    confirmed: bool
    evidence_tier: EvidenceTier

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be non-empty")
        if not self.detections:
            raise ValueError("an open-world track requires observations")
        indices = [item.frame_index for item in self.detections]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError(
                "track detections must have unique increasing frame indices"
            )
        classes = {item.entity_class for item in self.detections}
        if len(classes) != 1:
            raise ValueError("one track cannot change entity class")
        tier = (
            self.evidence_tier
            if isinstance(self.evidence_tier, EvidenceTier)
            else EvidenceTier(self.evidence_tier)
        )
        object.__setattr__(self, "evidence_tier", tier)

    @property
    def entity_class(self) -> str:
        return self.detections[0].entity_class

    @property
    def formal_exposure_weight(self) -> float:
        return float(FORMAL_EXPOSURE_WEIGHTS[self.evidence_tier])

    def to_object_track(
        self,
        *,
        frame_count: int,
        time_weights_s: Sequence[float],
        matched_entity_id: str | None = None,
    ) -> ObjectTrack:
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        if len(time_weights_s) != frame_count:
            raise ValueError("time weights must have one value per frame")
        xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
        area = np.full(frame_count, np.nan, dtype=np.float64)
        confidence = np.zeros(frame_count, dtype=np.float64)
        observed = np.zeros(frame_count, dtype=bool)
        for detection in self.detections:
            if detection.frame_index >= frame_count:
                raise ValueError("track detection lies outside the time grid")
            index = detection.frame_index
            xy[index] = detection.xy
            area[index] = detection.area_px2
            confidence[index] = detection.confidence
            observed[index] = True
        visibility = tuple(
            VisibilityState.VISIBLE if value else VisibilityState.UNKNOWN
            for value in observed
        )
        return ObjectTrack(
            track_id=self.track_id,
            matched_entity_id=matched_entity_id,
            xy=xy,
            observed=observed,
            visibility=visibility,
            areas_px2=area,
            confidence=confidence,
            existence_observed=observed,
            localization_eligible=observed,
            association_eligible=observed,
            time_weights_s=np.asarray(time_weights_s, dtype=np.float64),
            metadata={
                "entity_class": self.entity_class,
                "evidence_tier": self.evidence_tier.value,
                "formal_exposure_weight": self.formal_exposure_weight,
                "confirmed": self.confirmed,
            },
        )


@dataclass(frozen=True)
class OpenWorldObservation:
    tracks: tuple[OpenWorldTrack, ...]
    overflow_counts: np.ndarray
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError(
                "open-world observation track IDs must be unique"
            )
        overflow = np.asarray(self.overflow_counts, dtype=np.float64)
        if overflow.ndim != 1:
            raise ValueError("overflow_counts must be one-dimensional")
        if not np.isfinite(overflow).all() or np.any(overflow < 0.0):
            raise ValueError(
                "overflow_counts must be finite and non-negative"
            )
        normalized = np.array(overflow, copy=True)
        normalized.setflags(write=False)
        object.__setattr__(self, "overflow_counts", normalized)


@dataclass(frozen=True)
class OpenWorldComparison:
    integrity: EntityIntegrityScore
    matches: tuple[EntityMatch, ...]
    per_frame: tuple[Mapping[str, object], ...]
    reference_exposure: Mapping[str, float]
    prediction_exposure: Mapping[str, float]


@dataclass
class _MutableTrack:
    track_id: str
    detections: list[ObjectDetection]


def _mask_overlap(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    first_binary = first > 0
    second_binary = second > 0
    intersection = int(np.logical_and(first_binary, second_binary).sum())
    first_area = int(first_binary.sum())
    second_area = int(second_binary.sum())
    union = first_area + second_area - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / max(min(first_area, second_area), 1)
    return float(iou), float(containment)


def deduplicate_frame_detections(
    detections: Iterable[ObjectDetection],
    *,
    mask_iou_threshold: float = 0.7,
    containment_threshold: float = 0.9,
) -> list[ObjectDetection]:
    """Deduplicate proposal sources only when they share pixel support.

    Nearby or synchronously moving objects are deliberately not merged.
    """

    values = list(detections)
    if not 0.0 <= mask_iou_threshold <= 1.0:
        raise ValueError("mask_iou_threshold must be in [0, 1]")
    if not 0.0 <= containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be in [0, 1]")
    if len({item.frame_index for item in values}) > 1:
        raise ValueError("deduplication accepts detections from one frame")
    retained: list[ObjectDetection] = []
    for candidate in sorted(
        values,
        key=lambda value: (
            value.formal_exposure_weight,
            value.confidence,
            value.area_px2,
        ),
        reverse=True,
    ):
        duplicate = False
        for existing in retained:
            if (
                candidate.entity_class != existing.entity_class
                or candidate.mask is None
                or existing.mask is None
            ):
                continue
            iou, containment = _mask_overlap(
                candidate.mask, existing.mask
            )
            if (
                iou >= mask_iou_threshold
                or containment >= containment_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return sorted(retained, key=lambda value: value.detection_id)


def _predicted_xy(track: _MutableTrack, time_s: float, times: np.ndarray) -> np.ndarray:
    latest = track.detections[-1]
    if len(track.detections) < 2:
        return latest.xy
    previous = track.detections[-2]
    delta_time = (
        times[latest.frame_index] - times[previous.frame_index]
    )
    if delta_time <= 1e-12:
        return latest.xy
    velocity = (latest.xy - previous.xy) / delta_time
    horizon = max(time_s - times[latest.frame_index], 0.0)
    return latest.xy + velocity * horizon


def _track_cost(
    track: _MutableTrack,
    detection: ObjectDetection,
    *,
    time_s: float,
    times: np.ndarray,
    minimum_scale_px: float,
    area_weight: float,
) -> float:
    latest = track.detections[-1]
    if latest.entity_class != detection.entity_class:
        return float("inf")
    latest_partition = latest.metadata.get(
        "exclusive_tracking_partition"
    )
    detection_partition = detection.metadata.get(
        "exclusive_tracking_partition"
    )
    if (
        latest_partition is not None
        or detection_partition is not None
    ) and latest_partition != detection_partition:
        # Scene adapters may reserve a condition-anchored directed stream so
        # a nearby residual/shadow cannot take over its ID.  This is opt-in;
        # historical detections without a partition retain frozen behavior.
        return float("inf")
    radius = max(
        math.sqrt(latest.area_px2 / math.pi),
        math.sqrt(detection.area_px2 / math.pi),
        minimum_scale_px,
    )
    distance = float(
        np.linalg.norm(
            _predicted_xy(track, time_s, times) - detection.xy
        )
        / radius
    )
    area = abs(math.log(detection.area_px2 / latest.area_px2))
    return distance + area_weight * area


def _confirmed_track(
    detections: Sequence[ObjectDetection],
    *,
    immediate_confidence: float,
) -> bool:
    if any(
        len(item.sources) >= 2
        or (
            item.evidence_tier is EvidenceTier.PARTICIPANT
            and item.confidence >= immediate_confidence
        )
        for item in detections
    ):
        return True
    if len(detections) < 2:
        return False
    frames = [item.frame_index for item in detections]
    return any(
        frames[index + 1] - frames[index] <= 2
        for index in range(len(frames) - 1)
    )


def _formal_track_tier(
    detections: Sequence[ObjectDetection],
    *,
    confirmed: bool,
) -> EvidenceTier:
    if detections and all(
        bool(
            item.metadata.get(
                "suppress_persistence_only_promotion", False
            )
        )
        for item in detections
    ):
        # Some scene adapters retain low-specificity proposals purely for
        # audit. Repetition alone must not turn a stationary apparatus mark
        # into a physical participant; independent evidence can still emit a
        # separate, stronger detection and track.
        return EvidenceTier.AMBIGUOUS
    if confirmed and any(
        item.evidence_tier is EvidenceTier.PARTICIPANT
        for item in detections
    ):
        return EvidenceTier.PARTICIPANT
    if confirmed and any(
        item.evidence_tier is EvidenceTier.INDEPENDENT_SALIENT
        for item in detections
    ):
        return EvidenceTier.INDEPENDENT_SALIENT
    if confirmed and any(
        item.evidence_tier is EvidenceTier.TENTATIVE
        for item in detections
    ):
        # Tentative participant proposals pay a small amount immediately.
        # Once temporally confirmed, the whole track is retrospectively a
        # full participant exposure; there is no free grace window.
        return EvidenceTier.PARTICIPANT
    if any(
        item.evidence_tier is not EvidenceTier.AMBIGUOUS
        for item in detections
    ):
        return EvidenceTier.TENTATIVE
    if confirmed:
        # Repeated low-specificity evidence is not free, but persistence
        # alone cannot turn a ruler mark or printed circle into a full
        # physical participant.
        return EvidenceTier.TENTATIVE
    return EvidenceTier.AMBIGUOUS


def track_open_world_detections(
    detections_by_frame: Sequence[Sequence[ObjectDetection]],
    *,
    time_grid: CommonTimeGrid,
    maximum_gap_s: float = 0.25,
    maximum_assignment_cost: float = 4.0,
    minimum_scale_px: float = 4.0,
    area_weight: float = 0.4,
    immediate_confidence: float = 0.9,
    maximum_tracks: int = 64,
) -> OpenWorldObservation:
    """Build causal tracks while preserving every unmatched residual."""

    frame_count = len(time_grid.times_s)
    if len(detections_by_frame) != frame_count:
        raise ValueError("detections must have one sequence per time sample")
    numeric = {
        "maximum_gap_s": maximum_gap_s,
        "maximum_assignment_cost": maximum_assignment_cost,
        "minimum_scale_px": minimum_scale_px,
        "area_weight": area_weight,
        "immediate_confidence": immediate_confidence,
    }
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numeric.values()):
        raise ValueError("tracking thresholds must be finite and non-negative")
    if not 0.0 <= immediate_confidence <= 1.0:
        raise ValueError("immediate_confidence must be in [0, 1]")
    if maximum_tracks < 1:
        raise ValueError("maximum_tracks must be positive")

    tracks: list[_MutableTrack] = []
    overflow = np.zeros(frame_count, dtype=np.float64)
    created = 0
    for frame_index, raw in enumerate(detections_by_frame):
        values = deduplicate_frame_detections(raw)
        if any(item.frame_index != frame_index for item in values):
            raise ValueError("detection frame_index differs from its bucket")
        current_time = float(time_grid.times_s[frame_index])
        active = [
            track
            for track in tracks
            if current_time
            - float(time_grid.times_s[track.detections[-1].frame_index])
            <= maximum_gap_s + 1e-12
        ]
        matched_detection_indices: set[int] = set()
        if active and values:
            costs = np.asarray(
                [
                    [
                        _track_cost(
                            track,
                            detection,
                            time_s=current_time,
                            times=time_grid.times_s,
                            minimum_scale_px=minimum_scale_px,
                            area_weight=area_weight,
                        )
                        for detection in values
                    ]
                    for track in active
                ],
                dtype=np.float64,
            )
            finite = np.isfinite(costs)
            if finite.any():
                safe = np.where(
                    finite,
                    costs,
                    maximum_assignment_cost + 1e6,
                )
                rows, columns = linear_sum_assignment(safe)
                for row, column in zip(rows.tolist(), columns.tolist()):
                    if safe[row, column] > maximum_assignment_cost:
                        continue
                    active[row].detections.append(values[column])
                    matched_detection_indices.add(column)
        unmatched = [
            value
            for index, value in enumerate(values)
            if index not in matched_detection_indices
        ]
        available = max(maximum_tracks - len(tracks), 0)
        accepted = unmatched[:available]
        for detection in accepted:
            tracks.append(
                _MutableTrack(
                    track_id=f"track_{created:04d}",
                    detections=[detection],
                )
            )
            created += 1
        for detection in unmatched[available:]:
            overflow[frame_index] += max(
                detection.formal_exposure_weight,
                FORMAL_EXPOSURE_WEIGHTS[EvidenceTier.TENTATIVE],
            )

    finalized: list[OpenWorldTrack] = []
    for track in tracks:
        confirmed = _confirmed_track(
            track.detections,
            immediate_confidence=immediate_confidence,
        )
        finalized.append(
            OpenWorldTrack(
                track_id=track.track_id,
                detections=tuple(track.detections),
                confirmed=confirmed,
                evidence_tier=_formal_track_tier(
                    track.detections,
                    confirmed=confirmed,
                ),
            )
        )
    return OpenWorldObservation(
        tracks=tuple(finalized),
        overflow_counts=overflow,
        diagnostics={
            "input_detections": int(
                sum(len(value) for value in detections_by_frame)
            ),
            "tracks": len(finalized),
            "overflow_observations": float(overflow.sum()),
            "maximum_tracks": maximum_tracks,
        },
    )


def _weighted_exposure(track: ObjectTrack) -> float:
    weight = float(track.metadata.get("formal_exposure_weight", 1.0))
    return weight * track.exposure()


def _frame_assignment(
    reference: Sequence[tuple[str, ObjectTrack]],
    prediction: Sequence[ObjectTrack],
    *,
    frame_index: int,
    frame_diagonal_px: float,
    minimum_match_position_similarity: float,
    fixed_entity_track_ids: Mapping[str, str] | None = None,
) -> tuple[
    list[EntityMatch],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if not reference or not prediction:
        return [], [], []
    matches: list[EntityMatch] = []
    rows_out: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    weighted_prediction = [
        (
            track,
            min(
                max(
                    float(
                        track.metadata.get(
                            "formal_exposure_weight",
                            1.0,
                        )
                    ),
                    0.0,
                ),
                1.0,
            ),
        )
        for track in prediction
    ]
    weighted_prediction = [
        value for value in weighted_prediction if value[1] > 0.0
    ]
    remaining_reference = list(reference)
    for tier_weight in sorted(
        {weight for _, weight in weighted_prediction},
        reverse=True,
    ):
        tier_prediction = [
            track
            for track, weight in weighted_prediction
            if math.isclose(weight, tier_weight, abs_tol=1e-12)
        ]
        if not remaining_reference or not tier_prediction:
            continue
        costs = np.full(
            (len(remaining_reference), len(tier_prediction)),
            1e9,
            dtype=np.float64,
        )
        comparisons: dict[tuple[int, int], object] = {}
        for ref_index, (entity_id, ref_track) in enumerate(
            remaining_reference
        ):
            ref_class = str(
                ref_track.metadata.get("entity_class", "")
            )
            for pred_index, pred_track in enumerate(tier_prediction):
                if (
                    fixed_entity_track_ids is not None
                    and entity_id in fixed_entity_track_ids
                    and pred_track.track_id
                    != fixed_entity_track_ids[entity_id]
                ):
                    continue
                pred_class = str(
                    pred_track.metadata.get("entity_class", "")
                )
                if ref_class and pred_class and ref_class != pred_class:
                    continue
                reference_area = (
                    float(ref_track.areas_px2[frame_index])
                    if ref_track.areas_px2 is not None
                    and np.isfinite(ref_track.areas_px2[frame_index])
                    else 0.0
                )
                comparison = compare_positions(
                    ref_track.xy[frame_index],
                    pred_track.xy[frame_index],
                    reference_area_px2=reference_area,
                    frame_diagonal_px=frame_diagonal_px,
                )
                if (
                    comparison.score
                    < minimum_match_position_similarity
                ):
                    rejected_rows.append(
                        {
                            "entity_id": remaining_reference[ref_index][0],
                            "prediction_track_id": pred_track.track_id,
                            "normalized_distance": (
                                comparison.normalized_distance
                            ),
                            "position_score": comparison.score,
                            "formal_exposure_weight": tier_weight,
                            "evidence_tier": pred_track.metadata.get(
                                "evidence_tier"
                            ),
                            "rejection_reason": (
                                "position_similarity_below_minimum"
                            ),
                            "minimum_match_position_similarity": (
                                minimum_match_position_similarity
                            ),
                        }
                    )
                    continue
                comparisons[ref_index, pred_index] = comparison
                costs[ref_index, pred_index] = (
                    comparison.normalized_distance
                )
        rows, columns = linear_sum_assignment(costs)
        matched_reference_indices: set[int] = set()
        for ref_index, pred_index in zip(
            rows.tolist(),
            columns.tolist(),
        ):
            if costs[ref_index, pred_index] >= 1e8:
                continue
            entity_id, ref_track = remaining_reference[ref_index]
            pred_track = tier_prediction[pred_index]
            comparison = comparisons[ref_index, pred_index]
            time_weight = (
                float(ref_track.time_weights_s[frame_index])
                if ref_track.time_weights_s is not None
                else 1.0
            )
            weight = time_weight * tier_weight
            association = bool(
                ref_track.association_eligible[frame_index]
                and pred_track.association_eligible[frame_index]
            )
            matches.append(
                EntityMatch(
                    frame_index=frame_index,
                    entity_id=entity_id,
                    track_id=pred_track.track_id,
                    localization_quality=comparison.score,
                    normalized_distance=comparison.normalized_distance,
                    weight=weight,
                    association_eligible=association,
                )
            )
            rows_out.append(
                {
                    "entity_id": entity_id,
                    "prediction_track_id": pred_track.track_id,
                    "normalized_distance": (
                        comparison.normalized_distance
                    ),
                    "position_score": comparison.score,
                    "weight_s": weight,
                    "formal_exposure_weight": tier_weight,
                    "evidence_tier": pred_track.metadata.get(
                        "evidence_tier"
                    ),
                }
            )
            matched_reference_indices.add(ref_index)
        remaining_reference = [
            value
            for index, value in enumerate(remaining_reference)
            if index not in matched_reference_indices
        ]
    matched_entity_ids = {match.entity_id for match in matches}
    matched_track_ids = {match.track_id for match in matches}
    # A rejected edge is a formal null-assignment diagnostic only when both
    # endpoints remain unmatched after all evidence tiers. Edges rejected
    # from a high tier must not hide a valid lower-tier correspondence.
    rejected_rows = [
        row
        for row in rejected_rows
        if row["entity_id"] not in matched_entity_ids
        and row["prediction_track_id"] not in matched_track_ids
    ]
    rejected_rows.sort(
        key=lambda row: (
            str(row["entity_id"]),
            -float(row["formal_exposure_weight"]),
            float(row["normalized_distance"]),
            str(row["prediction_track_id"]),
        )
    )
    return matches, rows_out, rejected_rows


def compare_open_world_tracks(
    *,
    reference_tracks: Sequence[ObjectTrack],
    prediction_observation: OpenWorldObservation,
    time_grid: CommonTimeGrid,
    frame_diagonal_px: float,
    minimum_match_position_similarity: float = 0.0,
    fixed_entity_track_ids: Mapping[str, str] | None = None,
) -> OpenWorldComparison:
    """Compare fixed GT identities with every prediction track and residual.

    A positive ``minimum_match_position_similarity`` adds a null hypothesis
    to each evidence-tiered Hungarian assignment. A candidate below the
    reference-scaled continuous position threshold is reported as a rejected
    edge, leaving the reference entity missing and the prediction track extra.
    ``fixed_entity_track_ids`` makes declared identities terminal: an entity
    may match only its frame-zero-bound directed track, so residual discovery
    cannot rename or reactivate it after uncertainty.
    """

    frame_count = len(time_grid.times_s)
    if frame_count < 1:
        raise ValueError("open-world comparison requires a non-empty grid")
    diagonal = float(frame_diagonal_px)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("frame_diagonal_px must be finite and positive")
    minimum_similarity = float(minimum_match_position_similarity)
    if (
        not math.isfinite(minimum_similarity)
        or not 0.0 <= minimum_similarity <= 1.0
    ):
        raise ValueError(
            "minimum_match_position_similarity must be finite and in [0, 1]"
        )
    if prediction_observation.overflow_counts.shape != (frame_count,):
        raise ValueError(
            "overflow_counts must have one value per common time sample"
        )
    reference_by_entity: dict[str, ObjectTrack] = {}
    for track in reference_tracks:
        if len(track.xy) != frame_count:
            raise ValueError("reference track length differs from time grid")
        entity_id = track.matched_entity_id
        if not entity_id:
            raise ValueError("reference tracks require matched_entity_id")
        if entity_id in reference_by_entity:
            raise ValueError(f"duplicate reference entity ID: {entity_id}")
        reference_by_entity[entity_id] = track

    fixed_tracks = None
    if fixed_entity_track_ids is not None:
        fixed_tracks = {
            str(entity_id): str(track_id)
            for entity_id, track_id in fixed_entity_track_ids.items()
        }
        if (
            set(fixed_tracks) - set(reference_by_entity)
            or any(not value for value in fixed_tracks.values())
            or len(set(fixed_tracks.values())) != len(fixed_tracks)
        ):
            raise ValueError(
                "fixed entity-track identities must be unique, non-empty, "
                "and reference-backed"
            )

    prediction_tracks = [
        value.to_object_track(
            frame_count=frame_count,
            time_weights_s=time_grid.cell_weights_s,
        )
        for value in prediction_observation.tracks
    ]
    reference_exposure = {
        entity_id: track.exposure()
        for entity_id, track in reference_by_entity.items()
    }
    prediction_exposure = {
        track.track_id: _weighted_exposure(track)
        for track in prediction_tracks
    }
    overflow_exposure = float(
        np.dot(
            prediction_observation.overflow_counts,
            time_grid.cell_weights_s,
        )
    )
    if overflow_exposure > 0.0:
        prediction_exposure["__overflow__"] = overflow_exposure

    matches: list[EntityMatch] = []
    per_frame: list[Mapping[str, object]] = []
    for frame_index, time_s in enumerate(time_grid.times_s.tolist()):
        visible_reference = [
            (entity_id, track)
            for entity_id, track in reference_by_entity.items()
            if track.localization_eligible[frame_index]
        ]
        visible_prediction = [
            track
            for track in prediction_tracks
            if track.localization_eligible[frame_index]
        ]
        current, match_rows, rejected_rows = _frame_assignment(
            visible_reference,
            visible_prediction,
            frame_index=frame_index,
            frame_diagonal_px=diagonal,
            minimum_match_position_similarity=minimum_similarity,
            fixed_entity_track_ids=fixed_tracks,
        )
        matches.extend(current)
        matched_entities = {item.entity_id for item in current}
        matched_tracks = {item.track_id for item in current}
        missing_entity_ids = sorted(
            entity_id
            for entity_id, _ in visible_reference
            if entity_id not in matched_entities
        )
        unmatched_prediction = [
            track
            for track in visible_prediction
            if track.track_id not in matched_tracks
        ]
        formal_residual = [
            track
            for track in unmatched_prediction
            if float(
                track.metadata.get("formal_exposure_weight", 1.0)
            )
            > 0.0
        ]
        ambiguous_candidates = [
            track
            for track in unmatched_prediction
            if float(
                track.metadata.get("formal_exposure_weight", 1.0)
            )
            <= 0.0
        ]
        formal_cardinality = float(
            sum(
                min(
                    max(
                        float(
                            track.metadata.get(
                                "formal_exposure_weight",
                                1.0,
                            )
                        ),
                        0.0,
                    ),
                    1.0,
                )
                for track in visible_prediction
            )
            + prediction_observation.overflow_counts[frame_index]
        )
        nearest_rejected_score: dict[str, float] = {}
        for row in rejected_rows:
            entity_id = str(row["entity_id"])
            nearest_rejected_score[entity_id] = max(
                nearest_rejected_score.get(entity_id, 0.0),
                float(row["position_score"]),
            )
        matched_position_score = (
            float(
                np.mean(
                    [
                        float(row["position_score"])
                        for row in match_rows
                    ]
                )
            )
            if match_rows
            else 0.0
        )
        position_terms = [
            float(row["position_score"]) for row in match_rows
        ] + [
            nearest_rejected_score.get(entity_id, 0.0)
            for entity_id in missing_entity_ids
        ]
        position_diagnostic_score = (
            float(np.mean(position_terms)) if position_terms else 0.0
        )
        per_frame.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "matches": match_rows,
                "missing_entity_ids": missing_entity_ids,
                "residual_track_ids": sorted(
                    track.track_id
                    for track in formal_residual
                ),
                "ambiguous_candidate_track_ids": sorted(
                    track.track_id
                    for track in ambiguous_candidates
                ),
                "rejected_candidate_matches": rejected_rows,
                "matched_position_score": matched_position_score,
                "position_diagnostic_score": position_diagnostic_score,
                "formal_prediction_cardinality": formal_cardinality,
                "participant_prediction_count": sum(
                    float(
                        track.metadata.get(
                            "formal_exposure_weight",
                            1.0,
                        )
                    )
                    >= 1.0 - 1e-12
                    for track in visible_prediction
                ),
                "raw_prediction_candidate_count": len(
                    visible_prediction
                ),
                "overflow_count": float(
                    prediction_observation.overflow_counts[frame_index]
                ),
            }
        )

    reference_association_exposure = {
        entity_id: float(
            np.dot(
                track.association_eligible.astype(np.float64),
                time_grid.cell_weights_s,
            )
        )
        for entity_id, track in reference_by_entity.items()
    }
    prediction_association_exposure = {
        track.track_id: float(
            np.dot(
                track.association_eligible.astype(np.float64),
                time_grid.cell_weights_s,
            )
            * float(
                track.metadata.get("formal_exposure_weight", 1.0)
            )
        )
        for track in prediction_tracks
    }
    if overflow_exposure > 0.0:
        prediction_association_exposure["__overflow__"] = 0.0
    integrity = score_entity_integrity(
        matches,
        reference_exposure=reference_exposure,
        prediction_exposure=prediction_exposure,
        reference_association_exposure=reference_association_exposure,
        prediction_association_exposure=prediction_association_exposure,
    )
    return OpenWorldComparison(
        integrity=integrity,
        matches=tuple(matches),
        per_frame=tuple(per_frame),
        reference_exposure=reference_exposure,
        prediction_exposure=prediction_exposure,
    )


def detections_from_instance_masks(
    instance_masks: Sequence[Sequence[np.ndarray]],
    *,
    entity_class: str,
    source: str,
    evidence_tier: EvidenceTier = EvidenceTier.PARTICIPANT,
    confidence: float = 1.0,
    minimum_area_px2: int = 1,
) -> list[list[ObjectDetection]]:
    """Convert per-instance masks into per-frame open-world detections."""

    if not instance_masks:
        return []
    frame_count = len(instance_masks[0])
    if any(len(instance) != frame_count for instance in instance_masks):
        raise ValueError("instance mask timelines must have equal length")
    output: list[list[ObjectDetection]] = [
        [] for _ in range(frame_count)
    ]
    for instance_index, instance in enumerate(instance_masks):
        for frame_index, mask in enumerate(instance):
            binary = np.where(np.asarray(mask) > 0, 255, 0).astype(
                np.uint8
            )
            area = int(np.count_nonzero(binary))
            if area < minimum_area_px2:
                continue
            moments = cv2.moments(binary, binaryImage=True)
            if abs(float(moments["m00"])) <= 1e-12:
                continue
            xy = np.asarray(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ],
                dtype=np.float64,
            )
            output[frame_index].append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"{source}_{instance_index:03d}_{frame_index:05d}"
                    ),
                    xy=xy,
                    area_px2=float(area),
                    entity_class=entity_class,
                    mask=binary,
                    confidence=confidence,
                    evidence_tier=evidence_tier,
                    sources=(source,),
                    metadata={"instance_index": instance_index},
                )
            )
    return output


__all__ = [
    "FORMAL_EXPOSURE_WEIGHTS",
    "EvidenceTier",
    "ObjectDetection",
    "OpenWorldComparison",
    "OpenWorldObservation",
    "OpenWorldTrack",
    "compare_open_world_tracks",
    "deduplicate_frame_detections",
    "detections_from_instance_masks",
    "track_open_world_detections",
]
