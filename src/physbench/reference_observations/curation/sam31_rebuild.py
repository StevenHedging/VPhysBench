"""Per-Case orchestration and quality gates for SAM 3.1 collision GT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import ObservationState
from .sam31_gt import (
    GtCandidate,
    OrderedGtIdentity,
    bind_row_major_identities,
    select_collision_candidates,
    validate_physics_caption_binding,
)
from .sam31_predictor import Sam31GtPredictor, Sam31GtTrack


@dataclass(frozen=True)
class Sam31GtConfig:
    primary_prompt: str = "small round object"
    fallback_prompts: tuple[str, ...] = ("ball", "steel ball", "glass ball")
    ambiguity_margin: float = 0.03
    border_crop_fraction: float = 0.6
    near_duplicate_tube_iou: float = 0.85

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Sam31GtConfig":
        return cls(
            primary_prompt=str(value.get("primary_prompt", cls.primary_prompt)),
            fallback_prompts=tuple(
                str(item) for item in value.get("fallback_prompts", cls.fallback_prompts)
            ),
            ambiguity_margin=float(
                value.get("ambiguity_margin", cls.ambiguity_margin)
            ),
            border_crop_fraction=float(
                value.get("border_crop_fraction", cls.border_crop_fraction)
            ),
            near_duplicate_tube_iou=float(
                value.get("near_duplicate_tube_iou", cls.near_duplicate_tube_iou)
            ),
        )

    def __post_init__(self) -> None:
        if not self.primary_prompt.strip():
            raise ValueError("SAM3.1 GT primary prompt must be non-empty")
        if any(not value.strip() for value in self.fallback_prompts):
            raise ValueError("SAM3.1 GT fallback prompts must be non-empty")
        if not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("SAM3.1 GT ambiguity margin must lie in [0,1]")
        if not 0.5 <= self.border_crop_fraction < 1:
            raise ValueError("SAM3.1 GT border crop fraction must lie in [0.5,1)")
        if not 0 < self.near_duplicate_tube_iou <= 1:
            raise ValueError("SAM3.1 GT duplicate Tube IoU must lie in (0,1]")


@dataclass(frozen=True)
class GtQualityFinding:
    code: str
    severity: str
    message: str
    object_ids: tuple[str, ...] = ()
    observation_indices: tuple[int, ...] = ()
    statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sam31GtCaseCandidate:
    case_id: str
    identities: tuple[OrderedGtIdentity, ...]
    masks_by_object: dict[str, np.ndarray]
    states_by_object: dict[str, np.ndarray]
    source_frame_indices: tuple[int, ...]
    physical_times: tuple[float, ...]
    findings: tuple[GtQualityFinding, ...]
    discovery_attempts: tuple[dict[str, Any], ...]
    predictor_provenance: Mapping[str, Any]

    @property
    def accepted(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)


def _validate_track(
    track: Sam31GtTrack,
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
) -> None:
    if track.masks.shape != (frame_count, *frame_shape):
        raise ValueError(
            f"SAM3.1 GT track {track.semantic_id} has shape {track.masks.shape}; "
            f"expected {(frame_count, *frame_shape)}"
        )
    if track.boxes_xywh.shape != (frame_count, 4):
        raise ValueError(f"SAM3.1 GT track {track.semantic_id} boxes are misaligned")
    if track.confidences.shape != (frame_count,):
        raise ValueError(
            f"SAM3.1 GT track {track.semantic_id} confidences are misaligned"
        )
    if not np.isfinite(track.boxes_xywh).all() or not np.isfinite(
        track.confidences
    ).all():
        raise ValueError(f"SAM3.1 GT track {track.semantic_id} is non-finite")


def _pad_crop_track(
    track: Sam31GtTrack,
    *,
    full_shape: tuple[int, int],
    left: int,
    source: str,
) -> Sam31GtTrack:
    height, width = full_shape
    crop_height, crop_width = track.masks.shape[1:]
    if crop_height != height or left < 0 or left + crop_width > width:
        raise ValueError("SAM3.1 GT crop track does not fit the full frame")
    masks = np.zeros((len(track.masks), height, width), dtype=bool)
    masks[:, :, left : left + crop_width] = track.masks
    boxes = track.boxes_xywh.astype(np.float32, copy=True)
    boxes[:, 0] = (boxes[:, 0] * crop_width + left) / width
    boxes[:, 2] = boxes[:, 2] * crop_width / width
    return Sam31GtTrack(
        semantic_id=track.semantic_id,
        backend_object_id=track.backend_object_id,
        masks=masks,
        boxes_xywh=boxes,
        confidences=track.confidences.copy(),
        prompt=track.prompt,
        source=source,
    )


def _candidate_from_track(
    track: Sam31GtTrack,
    *,
    candidate_id: str,
) -> GtCandidate:
    return GtCandidate.from_mask(
        candidate_id=candidate_id,
        prompt=track.prompt,
        backend_object_id=track.backend_object_id,
        mask=track.masks[0],
        confidence=float(track.confidences[0]),
        source=track.source,
    )


def _try_select(
    tracks_by_candidate: Mapping[str, Sam31GtTrack],
    *,
    expected_radii: tuple[float, ...],
    ambiguity_margin: float,
) -> tuple[GtCandidate, ...]:
    candidates = tuple(
        _candidate_from_track(track, candidate_id=candidate_id)
        for candidate_id, track in tracks_by_candidate.items()
        if track.masks[0].any()
    )
    return select_collision_candidates(
        candidates,
        expected_radii=expected_radii,
        ambiguity_margin=ambiguity_margin,
    )


def _discover_tracks(
    frames: tuple[np.ndarray, ...],
    *,
    predictor: Sam31GtPredictor,
    config: Sam31GtConfig,
    expected_radii: tuple[float, ...],
) -> tuple[tuple[GtCandidate, ...], dict[str, Sam31GtTrack], tuple[dict[str, Any], ...]]:
    frame_shape = frames[0].shape[:2]
    frame_count = len(frames)
    tracks: dict[str, Sam31GtTrack] = {}
    attempts: list[dict[str, Any]] = []
    last_error: ValueError | None = None

    def add_discovery(
        discovery_frames: tuple[np.ndarray, ...],
        *,
        prompts: tuple[str, ...],
        prefix: str,
        left: int = 0,
    ) -> None:
        raw_tracks = predictor.discover(discovery_frames, prompts=prompts)
        attempts.append(
            {
                "source": prefix,
                "prompts": list(prompts),
                "candidate_count": len(raw_tracks),
                "frame_shape": list(discovery_frames[0].shape[:2]),
            }
        )
        for index, raw_track in enumerate(raw_tracks):
            track = raw_track
            if discovery_frames[0].shape[:2] != frame_shape:
                track = _pad_crop_track(
                    raw_track,
                    full_shape=frame_shape,
                    left=left,
                    source=prefix,
                )
            _validate_track(track, frame_count=frame_count, frame_shape=frame_shape)
            candidate_id = f"{prefix}:{index:02d}:{track.semantic_id}"
            tracks[candidate_id] = track

    rounds = (
        ((config.primary_prompt,), "primary"),
        (config.fallback_prompts, "fallback"),
    )
    for prompts, prefix in rounds:
        if not prompts:
            continue
        add_discovery(frames, prompts=prompts, prefix=prefix)
        try:
            selected = _try_select(
                tracks,
                expected_radii=expected_radii,
                ambiguity_margin=config.ambiguity_margin,
            )
            return selected, tracks, tuple(attempts)
        except ValueError as exc:
            last_error = exc

    height, width = frame_shape
    crop_width = max(1, int(round(width * config.border_crop_fraction)))
    all_prompts = tuple(dict.fromkeys((config.primary_prompt, *config.fallback_prompts)))
    for side, left in (("left", 0), ("right", width - crop_width)):
        crop_frames = tuple(frame[:, left : left + crop_width] for frame in frames)
        add_discovery(
            crop_frames,
            prompts=all_prompts,
            prefix=f"border_{side}",
            left=left,
        )
        try:
            selected = _try_select(
                tracks,
                expected_radii=expected_radii,
                ambiguity_margin=config.ambiguity_margin,
            )
            return selected, tracks, tuple(attempts)
        except ValueError as exc:
            last_error = exc
    assert last_error is not None
    raise ValueError(f"SAM3.1 GT discovery failed after all fallbacks: {last_error}")


def _centroid_x(mask: np.ndarray) -> float:
    _, xs = np.nonzero(mask)
    return float(xs.mean())


def _states_and_findings(
    object_id: str,
    masks: np.ndarray,
) -> tuple[np.ndarray, tuple[GtQualityFinding, ...]]:
    present = masks.reshape(len(masks), -1).any(axis=1)
    states = np.full(len(masks), int(ObservationState.VISIBLE), dtype=np.uint8)
    findings: list[GtQualityFinding] = []
    if not present[0]:
        states[0] = int(ObservationState.UNRESOLVED)
        findings.append(
            GtQualityFinding(
                "empty_frame_zero",
                "error",
                f"{object_id} is empty on frame zero",
                (object_id,),
                (0,),
            )
        )
    missing = np.flatnonzero(~present)
    if not len(missing):
        return states, tuple(findings)
    last_present = int(np.flatnonzero(present)[-1]) if present.any() else -1
    internal = tuple(int(index) for index in missing if index < last_present)
    if internal:
        states[list(internal)] = int(ObservationState.UNRESOLVED)
        findings.append(
            GtQualityFinding(
                "internal_track_gap",
                "error",
                f"{object_id} disappears and later reappears",
                (object_id,),
                internal,
            )
        )
    trailing = tuple(int(index) for index in missing if index > last_present)
    if trailing and last_present >= 0:
        mask = masks[last_present]
        touches_border = bool(
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        )
        if touches_border:
            states[list(trailing)] = int(ObservationState.OUT_OF_FRAME)
        else:
            states[list(trailing)] = int(ObservationState.UNRESOLVED)
            findings.append(
                GtQualityFinding(
                    "unexplained_trailing_disappearance",
                    "error",
                    f"{object_id} disappears without reaching an image boundary",
                    (object_id,),
                    trailing,
                )
            )
    return states, tuple(findings)


def _tube_iou_series(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(left, right).reshape(len(left), -1).sum(axis=1)
    union = np.logical_or(left, right).reshape(len(left), -1).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(left), dtype=np.float64),
        where=union > 0,
    )


def _joint_findings(
    masks_by_object: Mapping[str, np.ndarray],
    *,
    duplicate_iou_threshold: float,
) -> tuple[GtQualityFinding, ...]:
    object_ids = tuple(masks_by_object)
    findings: list[GtQualityFinding] = []
    inversions: list[int] = []
    for frame_index in range(len(next(iter(masks_by_object.values())))):
        masks = [masks_by_object[object_id][frame_index] for object_id in object_ids]
        if not all(mask.any() for mask in masks):
            continue
        centers = [_centroid_x(mask) for mask in masks]
        if any(right <= left for left, right in zip(centers, centers[1:])):
            inversions.append(frame_index)
    if inversions:
        findings.append(
            GtQualityFinding(
                "identity_order_inversion",
                "error",
                "locked collision identities invert their visual x order",
                object_ids,
                tuple(inversions),
            )
        )
    for left_index, left_id in enumerate(object_ids):
        for right_id in object_ids[left_index + 1 :]:
            values = _tube_iou_series(
                masks_by_object[left_id], masks_by_object[right_id]
            )
            both_present = np.logical_and(
                masks_by_object[left_id].reshape(len(values), -1).any(axis=1),
                masks_by_object[right_id].reshape(len(values), -1).any(axis=1),
            )
            median_iou = float(np.median(values[both_present])) if both_present.any() else 0.0
            if median_iou >= duplicate_iou_threshold:
                findings.append(
                    GtQualityFinding(
                        "near_duplicate_tubes",
                        "error",
                        f"{left_id} and {right_id} track nearly the same subject",
                        (left_id, right_id),
                        statistics={"median_visible_iou": median_iou},
                    )
                )
    return tuple(findings)


def rebuild_collision_case(
    case: Any,
    *,
    frames: Sequence[np.ndarray],
    predictor: Sam31GtPredictor,
    config: Sam31GtConfig,
) -> Sam31GtCaseCandidate:
    if case.scene_id != "collision_1d":
        raise ValueError("SAM3.1 collision GT rebuild accepts collision_1d only")
    frame_values = tuple(np.asarray(frame) for frame in frames)
    if not frame_values:
        raise ValueError("SAM3.1 collision GT rebuild requires aligned frames")
    first_shape = frame_values[0].shape
    if (
        len(first_shape) != 3
        or first_shape[2] != 3
        or any(frame.dtype != np.uint8 or frame.shape != first_shape for frame in frame_values)
    ):
        raise ValueError("SAM3.1 collision GT frames must be equal uint8 HWC images")
    samples = tuple(case.timeline["samples"])
    if len(samples) != len(frame_values):
        raise ValueError(
            "SAM3.1 collision GT frame count differs from the frozen timeline"
        )
    if tuple(int(item["observation_index"]) for item in samples) != tuple(
        range(len(samples))
    ):
        raise ValueError("SAM3.1 collision GT timeline indices must be contiguous")
    binding = validate_physics_caption_binding(case.physics, case.caption)
    selected, tracks, attempts = _discover_tracks(
        frame_values,
        predictor=predictor,
        config=config,
        expected_radii=binding.radii_m,
    )
    identities = bind_row_major_identities(selected, binding)
    masks_by_object: dict[str, np.ndarray] = {}
    states_by_object: dict[str, np.ndarray] = {}
    findings: list[GtQualityFinding] = []
    for identity in identities:
        masks = tracks[identity.candidate.candidate_id].masks.astype(bool, copy=True)
        masks_by_object[identity.object_id] = masks
        states, entity_findings = _states_and_findings(identity.object_id, masks)
        states_by_object[identity.object_id] = states
        findings.extend(entity_findings)
    findings.extend(
        _joint_findings(
            masks_by_object,
            duplicate_iou_threshold=config.near_duplicate_tube_iou,
        )
    )
    return Sam31GtCaseCandidate(
        case_id=str(case.case_id),
        identities=identities,
        masks_by_object=masks_by_object,
        states_by_object=states_by_object,
        source_frame_indices=tuple(int(item["source_frame_index"]) for item in samples),
        physical_times=tuple(float(item["physical_time_seconds"]) for item in samples),
        findings=tuple(findings),
        discovery_attempts=attempts,
        predictor_provenance=dict(predictor.describe()),
    )


__all__ = [
    "GtQualityFinding",
    "Sam31GtCaseCandidate",
    "Sam31GtConfig",
    "rebuild_collision_case",
]
