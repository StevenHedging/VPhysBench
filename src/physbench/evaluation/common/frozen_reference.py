"""V14-only loading and evaluator-canvas alignment of frozen GT tubes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import BadZipFile

import cv2
import numpy as np

from ...reference_observations import (
    ObservationState,
    load_reference_observation,
)
from ..contracts import CaseEvaluationRequest
from .errors import ReferenceAnalysisError
from .frozen_subject import transform_frozen_subject_mask


POLICY = "frozen_dataset_reference_observation_v1"
_DATA_ERRORS = (
    BadZipFile,
    EOFError,
    KeyError,
    IndexError,
    OSError,
    TypeError,
    UnicodeError,
    ValueError,
    cv2.error,
)


@dataclass(frozen=True)
class FrozenReferenceEntity:
    entity_id: str
    mask_id: str
    masks: np.ndarray
    centroid_xy: np.ndarray
    bbox_xyxy: np.ndarray
    area_pixels: np.ndarray
    state: np.ndarray
    visible: np.ndarray


@dataclass(frozen=True)
class FrozenReferenceObservation:
    case_id: str
    scene_id: str
    policy: str
    times_s: np.ndarray
    source_observation_indices: np.ndarray
    entities: Mapping[str, FrozenReferenceEntity]
    manifest_path: Path
    manifest_sha256: str
    provenance: Mapping[str, object]


def _manifest_path(request: CaseEvaluationRequest) -> tuple[str, Path]:
    value = request.case.get("assets", {}).get(
        "reference_observation_manifest"
    )
    if not isinstance(value, str) or not value:
        raise ReferenceAnalysisError(
            "reference_observation_manifest_missing",
            "V14 Case does not declare "
            "assets.reference_observation_manifest",
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReferenceAnalysisError(
            "reference_observation_manifest_path_escape",
            "reference observation manifest must stay within asset_root",
        )
    root = request.asset_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            "reference_observation_manifest_path_escape",
            "reference observation manifest escapes asset_root",
        ) from exc
    return value, path


def _source_indices(
    source_times: np.ndarray,
    requested_times: np.ndarray,
    *,
    sampling_rate_hz: float,
) -> np.ndarray:
    if source_times.ndim != 1 or source_times.size == 0:
        raise ValueError("frozen reference physical-time grid is empty")
    if np.any(np.diff(source_times) <= 0.0):
        raise ValueError(
            "frozen reference physical-time grid is not strictly increasing"
        )
    if requested_times.ndim != 1 or requested_times.size == 0:
        raise ValueError("evaluator physical-time grid is empty")
    if not np.isfinite(requested_times).all() or np.any(
        np.diff(requested_times) <= 0.0
    ):
        raise ValueError(
            "evaluator physical-time grid must be finite and increasing"
        )
    right = np.searchsorted(source_times, requested_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.clip(right - 1, 0, source_times.size - 1)
    choose_left = (
        np.abs(requested_times - source_times[left])
        <= np.abs(requested_times - source_times[right])
    )
    indices = np.where(choose_left, left, right).astype(np.int64)
    tolerance = max(1e-6, 1e-4 / float(sampling_rate_hz))
    errors = np.abs(source_times[indices] - requested_times)
    if np.any(errors > tolerance):
        raise ValueError(
            "evaluator physical-time grid is not represented exactly by the "
            "frozen reference observation"
        )
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(
            "evaluator physical-time grid maps duplicate frozen observations"
        )
    return indices


def _geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (
            np.full(2, np.nan, dtype=np.float64),
            np.full(4, np.nan, dtype=np.float64),
            0,
        )
    return (
        np.asarray([xs.mean(), ys.mean()], dtype=np.float64),
        np.asarray(
            [xs.min(), ys.min(), xs.max(), ys.max()],
            dtype=np.float64,
        ),
        int(xs.size),
    )


def _align_entity(
    entity,
    *,
    indices: np.ndarray,
    spatial_transform: Mapping[str, object],
) -> FrozenReferenceEntity:
    masks = []
    centroids = []
    boxes = []
    areas = []
    states = np.asarray(entity.state[indices], dtype=np.uint8)
    for output_index, source_index in enumerate(indices.tolist()):
        mask = transform_frozen_subject_mask(
            entity.masks[source_index],
            spatial_transform,
        )
        centroid, box, area = _geometry(mask)
        state = ObservationState(int(states[output_index]))
        if state is ObservationState.VISIBLE and area == 0:
            raise ValueError(
                f"{entity.object_id} is VISIBLE but empty after spatial transform"
            )
        if state is not ObservationState.VISIBLE and area != 0:
            raise ValueError(
                f"{entity.object_id} has a non-visible state with a non-empty mask"
            )
        masks.append(mask)
        centroids.append(centroid)
        boxes.append(box)
        areas.append(area)
    mask_array = np.stack(masks).astype(np.uint8, copy=False)
    centroid_array = np.stack(centroids).astype(np.float64, copy=False)
    box_array = np.stack(boxes).astype(np.float64, copy=False)
    area_array = np.asarray(areas, dtype=np.int64)
    visible = states == int(ObservationState.VISIBLE)
    for value in (
        mask_array,
        centroid_array,
        box_array,
        area_array,
        states,
        visible,
    ):
        value.setflags(write=False)
    return FrozenReferenceEntity(
        entity_id=entity.object_id,
        mask_id=entity.mask_id,
        masks=mask_array,
        centroid_xy=centroid_array,
        bbox_xyxy=box_array,
        area_pixels=area_array,
        state=states,
        visible=visible,
    )


def load_frozen_reference_observation(
    request: CaseEvaluationRequest,
    *,
    times_s: Sequence[float],
    spatial_transform: Mapping[str, object],
    expected_entity_ids: Sequence[str],
) -> FrozenReferenceObservation:
    """Load authoritative V14 GT and align it to one evaluator time/canvas."""

    declared_path, manifest_path = _manifest_path(request)
    try:
        bundle = load_reference_observation(
            request.asset_root,
            manifest_path,
        )
        if bundle.case_id != request.case.get("case_id"):
            raise ValueError(
                "frozen reference Case identity differs from evaluated Case"
            )
        if bundle.scene_id != request.case.get("scene_id"):
            raise ValueError(
                "frozen reference Scene identity differs from evaluated Scene"
            )
        expected = tuple(str(value) for value in expected_entity_ids)
        actual = tuple(bundle.entities)
        if not expected or actual != expected:
            raise ValueError(
                "frozen reference entity identities differ from evaluator "
                f"manifest: expected={expected}, actual={actual}"
            )
        quality = bundle.quality
        if not isinstance(quality, dict) or quality.get("status") != "pass":
            raise ValueError("frozen reference quality status is not pass")
        requested_times = np.asarray(times_s, dtype=np.float64)
        source_times = np.asarray(
            [
                sample.physical_time_seconds
                for sample in bundle.timeline.samples
            ],
            dtype=np.float64,
        )
        indices = _source_indices(
            source_times,
            requested_times,
            sampling_rate_hz=bundle.timeline.sampling_rate_hz,
        )
        entities = {
            entity_id: _align_entity(
                bundle.entities[entity_id],
                indices=indices,
                spatial_transform=spatial_transform,
            )
            for entity_id in expected
        }
    except ReferenceAnalysisError:
        raise
    except _DATA_ERRORS as exc:
        raise ReferenceAnalysisError(
            "reference_observation_invalid",
            "V14 frozen reference observation is invalid: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    requested_times.setflags(write=False)
    indices.setflags(write=False)
    return FrozenReferenceObservation(
        case_id=bundle.case_id,
        scene_id=bundle.scene_id,
        policy=POLICY,
        times_s=requested_times,
        source_observation_indices=indices,
        entities=entities,
        manifest_path=bundle.manifest_path,
        manifest_sha256=bundle.manifest_sha256,
        provenance={
            "policy": POLICY,
            "manifest": declared_path,
            "manifest_sha256": bundle.manifest_sha256,
            "source_observation_indices": indices.tolist(),
            "spatial_transform": dict(spatial_transform),
        },
    )


__all__ = [
    "FrozenReferenceEntity",
    "FrozenReferenceObservation",
    "POLICY",
    "load_frozen_reference_observation",
]
