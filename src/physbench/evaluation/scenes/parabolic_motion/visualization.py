from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ...common.artifacts.open_world_v2 import write_open_world_v2_artifacts
from ...common.entities.contracts import ReferenceCapability
from ...common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
)
from ...common.entities.v2 import ExpectedEntityTimeline


@dataclass(frozen=True)
class _VisualizationComparison:
    per_frame: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": "parabolic_motion_visual_audit_v1",
            "failed": False,
            "failure_reason": None,
            "per_frame": [dict(row) for row in self.per_frame],
        }


def _prediction_observation(
    observation: Any,
    *,
    frame_count: int,
) -> OpenWorldObservation | None:
    detections: list[ObjectDetection] = []
    for index in range(frame_count):
        if not bool(observation.observed[index]):
            continue
        mask = observation.masks[index]
        detections.append(
            ObjectDetection(
                frame_index=index,
                detection_id=f"projectile_{index:04d}",
                xy=observation.xy[index],
                area_px2=float(max(np.count_nonzero(mask), 1)),
                entity_class="ball",
                mask=mask,
                confidence=1.0,
                evidence_tier=EvidenceTier.PARTICIPANT,
                sources=("parabolic_projectile_observer",),
                metadata={
                    "interpolated": bool(observation.interpolated[index]),
                    "candidate_cardinality": int(
                        observation.cardinality[index]
                    ),
                },
            )
        )
    if not detections:
        return None
    selected = np.asarray(observation.observed, dtype=np.int64)
    residual = np.maximum(
        np.asarray(observation.cardinality, dtype=np.int64) - selected,
        0,
    )
    return OpenWorldObservation(
        tracks=(
            OpenWorldTrack(
                track_id="projectile_track",
                detections=tuple(detections),
                confirmed=True,
                evidence_tier=EvidenceTier.PARTICIPANT,
            ),
        ),
        overflow_counts=residual.astype(np.float64),
        diagnostics={
            "adapter": "parabolic_motion_visualization_v1",
            **dict(observation.diagnostics),
        },
    )


def _expected_timeline(
    entity: Any,
    *,
    capability: ReferenceCapability,
    reference: Any,
    expected: np.ndarray,
) -> ExpectedEntityTimeline:
    expected_exists = np.asarray(expected, dtype=bool)
    localization = expected_exists & np.asarray(reference.observed, dtype=bool)
    areas = np.asarray(
        [float(np.count_nonzero(mask)) for mask in reference.masks],
        dtype=np.float64,
    )
    return ExpectedEntityTimeline(
        entity=entity,
        capability=capability,
        expected_exists=expected_exists,
        existence_supervised=expected_exists,
        localization_supervised=localization,
        association_supervised=localization,
        reference_xy=reference.xy,
        reference_area_px2=areas,
        reference_masks=tuple(reference.masks),
        lifecycle_source="parabolic_reference_frozen_may_exit",
        metadata={"visualization_adapter": "parabolic_motion_v1"},
    )


def _comparison_rows(
    entity_id: str,
    *,
    reference: Any,
    prediction: Any,
    scored: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    binding_accepted = bool(scored["binding"]["accepted"])
    observed = np.asarray(prediction.observed, dtype=bool)
    for index in range(len(observed)):
        expected = bool(scored["expected"][index])
        matched = bool(scored["matched"][index])
        prediction_present = bool(observed[index])
        overflow = max(
            int(scored["prediction_cardinality"][index])
            - int(prediction_present),
            0,
        )
        matches = (
            [
                {
                    "entity_id": entity_id,
                    "prediction_track_id": "projectile_track",
                    "position_score": float(
                        scored["position_scores"][index]
                    ),
                }
            ]
            if matched
            else []
        )
        rows.append(
            {
                "frame_index": index,
                "expected_cardinality": int(expected),
                "formal_prediction_cardinality": int(
                    scored["prediction_cardinality"][index]
                ),
                "expected_entity_ids": [entity_id] if expected else [],
                "legally_absent_entity_ids": [] if expected else [entity_id],
                "matches": matches,
                "missing_entity_ids": [entity_id]
                if expected and not matched
                else [],
                "extra_track_ids": ["projectile_track"]
                if prediction_present and not matched
                else [],
                "ambiguous_candidate_track_ids": [],
                "rejected_candidate_matches": (
                    [
                        {
                            "entity_id": entity_id,
                            "prediction_track_id": "projectile_track",
                            "position_score": float(
                                scored["position_scores"][index]
                            ),
                            "reason": "frame_zero_binding_rejected",
                        }
                    ]
                    if prediction_present and expected and not binding_accepted
                    else []
                ),
                "overflow_count": float(overflow),
                "position_score": float(scored["position_scores"][index]),
                "birth_track_ids": ["projectile_track"]
                if prediction_present and (index == 0 or not observed[index - 1])
                else [],
                "death_track_ids": ["projectile_track"]
                if index > 0 and observed[index - 1] and not prediction_present
                else [],
                "id_switches": [],
            }
        )
    return tuple(rows)


def _derivative(
    values: np.ndarray,
    observed: np.ndarray,
    times_s: Sequence[float],
    index: int,
    axis: int,
) -> float | None:
    if index < 1 or not (observed[index - 1] and observed[index]):
        return None
    step = float(times_s[index] - times_s[index - 1])
    if step <= 0.0:
        return None
    return float((values[index, axis] - values[index - 1, axis]) / step)


def _acceleration(
    values: np.ndarray,
    observed: np.ndarray,
    times_s: Sequence[float],
    index: int,
) -> float | None:
    if index < 2 or not np.all(observed[index - 2 : index + 1]):
        return None
    first = _derivative(values, observed, times_s, index - 1, 1)
    second = _derivative(values, observed, times_s, index, 1)
    step = float(times_s[index] - times_s[index - 1])
    if first is None or second is None or step <= 0.0:
        return None
    return float((second - first) / step)


def write_parabolic_visualization(
    request: Any,
    *,
    config: Mapping[str, Any],
    times_s: Sequence[float],
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    entity: Any,
    capability: ReferenceCapability,
    reference: Any,
    prediction: Any,
    scored: Mapping[str, Any],
    prediction_available: Sequence[bool],
    reference_role: str,
    score_summary: Mapping[str, Any],
    has_issues: bool,
) -> dict[str, Any]:
    """Adapt the projectile observer to the unified Open-World v2 renderer."""

    rows = _comparison_rows(
        entity.entity_id,
        reference=reference,
        prediction=prediction,
        scored=scored,
    )
    expected_timeline = _expected_timeline(
        entity,
        capability=capability,
        reference=reference,
        expected=np.asarray(scored["expected"], dtype=bool),
    )
    diagnostics = []
    for index in range(len(times_s)):
        diagnostics.append(
            {
                "position similarity": float(
                    scored["position_scores"][index]
                ),
                "prediction vx(px/s)": _derivative(
                    prediction.xy,
                    prediction.observed,
                    times_s,
                    index,
                    0,
                ),
                "prediction ay(px/s2)": _acceleration(
                    prediction.xy,
                    prediction.observed,
                    times_s,
                    index,
                ),
            }
        )
    return write_open_world_v2_artifacts(
        request,
        scene_name="Parabolic motion",
        times_s=times_s,
        reference_frames=reference_frames,
        prediction_frames=prediction_frames,
        expected_timelines=[expected_timeline],
        prediction_observation=_prediction_observation(
            prediction,
            frame_count=len(times_s),
        ),
        comparison=_VisualizationComparison(rows),
        config=config,
        reference_union_masks=reference.masks,
        prediction_union_masks=prediction.masks,
        full_subject_ious=scored["ious"],
        prediction_available=prediction_available,
        reference_role=reference_role,
        score_summary=score_summary,
        per_frame_diagnostics=diagnostics,
        has_issues=has_issues,
    )


__all__ = ["write_parabolic_visualization"]
