"""Lossless storage and loading for frozen reference observations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..identifiers import require_safe_id
from ..io import load_json, sha256_file, write_json
from .contracts import (
    EntityObservation,
    ObservationState,
    ReferenceObservationBundle,
    ReferenceTimeline,
)
from .paths import resolve_dataset_file
from .timeline import timeline_from_dict, timeline_to_dict


_MASK_KEYS = {"packed_masks", "height", "width", "observation_index", "state"}
_TRAJECTORY_KEYS = {
    "centroid_xy",
    "bbox_xyxy",
    "area_pixels",
    "observation_index",
    "state",
}
_FILE_RECORD_KEYS = {"path", "size_bytes", "sha256"}


def _binary_uint8_thw(masks: np.ndarray) -> np.ndarray:
    array = np.asarray(masks)
    if array.dtype != np.uint8 or array.ndim != 3:
        raise ValueError("mask tube must use uint8 THW layout")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError("mask tube values must be binary")
    return array


def pack_mask_tube(masks: np.ndarray) -> np.ndarray:
    """Bit-pack the native-width dimension of one binary THW mask tube."""

    return np.packbits(_binary_uint8_thw(masks), axis=2, bitorder="little")


def unpack_mask_tube(packed: np.ndarray, *, width: int) -> np.ndarray:
    array = np.asarray(packed)
    if array.dtype != np.uint8 or array.ndim != 3:
        raise ValueError("packed mask tube must use uint8 THB layout")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("mask width must be a positive integer")
    if array.shape[2] != (width + 7) // 8:
        raise ValueError("packed mask width metadata does not match storage")
    return np.unpackbits(
        array,
        axis=2,
        count=width,
        bitorder="little",
    ).astype(np.uint8)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_entity(entity: EntityObservation) -> tuple[int, int, int]:
    require_safe_id(entity.object_id, label="reference observation object_id")
    if not entity.object_id.startswith("object_"):
        raise ValueError("reference observation object_id must use object_N")
    if not isinstance(entity.mask_id, str) or not entity.mask_id.isdigit():
        raise ValueError("reference observation mask_id must be numeric")
    masks = _binary_uint8_thw(entity.masks)
    count, height, width = masks.shape
    expected = {
        "centroid_xy": ((count, 2), np.float32),
        "bbox_xyxy": ((count, 4), np.float32),
        "area_pixels": ((count,), np.int64),
        "state": ((count,), np.uint8),
    }
    for name, (shape, dtype) in expected.items():
        value = np.asarray(getattr(entity, name))
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"entity {name} must use {dtype} shape {shape}")
    states = set(np.asarray(entity.state).tolist())
    allowed_states = {int(value) for value in ObservationState}
    if not states <= allowed_states:
        raise ValueError("entity state contains an unknown value")
    return count, height, width


def write_entity_observation(directory: str | Path, entity: EntityObservation) -> None:
    output = Path(directory)
    count, height, width = _validate_entity(entity)
    indices = np.arange(count, dtype=np.int64)
    _atomic_savez(
        output / "mask_tube.npz",
        packed_masks=pack_mask_tube(entity.masks),
        height=np.asarray(height, dtype=np.int64),
        width=np.asarray(width, dtype=np.int64),
        observation_index=indices,
        state=np.asarray(entity.state, dtype=np.uint8),
    )
    _atomic_savez(
        output / "trajectory.npz",
        centroid_xy=np.asarray(entity.centroid_xy, dtype=np.float32),
        bbox_xyxy=np.asarray(entity.bbox_xyxy, dtype=np.float32),
        area_pixels=np.asarray(entity.area_pixels, dtype=np.int64),
        observation_index=indices,
        state=np.asarray(entity.state, dtype=np.uint8),
    )


def _scalar_int(value: np.ndarray, *, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.int64:
        raise ValueError(f"{label} must be an int64 scalar")
    return int(array)


def load_entity_observation(
    mask_path: str | Path,
    trajectory_path: str | Path,
    *,
    object_id: str,
    mask_id: str,
    expected_samples: int,
) -> EntityObservation:
    with np.load(mask_path, allow_pickle=False) as mask_payload:
        if set(mask_payload.files) != _MASK_KEYS:
            raise ValueError("mask tube NPZ fields are invalid")
        packed = np.asarray(mask_payload["packed_masks"])
        height = _scalar_int(mask_payload["height"], label="mask height")
        width = _scalar_int(mask_payload["width"], label="mask width")
        mask_indices = np.asarray(mask_payload["observation_index"])
        mask_state = np.asarray(mask_payload["state"])
    masks = unpack_mask_tube(packed, width=width)
    if masks.shape[1] != height:
        raise ValueError("mask height metadata does not match storage")
    with np.load(trajectory_path, allow_pickle=False) as trajectory_payload:
        if set(trajectory_payload.files) != _TRAJECTORY_KEYS:
            raise ValueError("trajectory NPZ fields are invalid")
        centroid = np.asarray(trajectory_payload["centroid_xy"])
        bbox = np.asarray(trajectory_payload["bbox_xyxy"])
        area = np.asarray(trajectory_payload["area_pixels"])
        trajectory_indices = np.asarray(trajectory_payload["observation_index"])
        trajectory_state = np.asarray(trajectory_payload["state"])
    expected_indices = np.arange(expected_samples, dtype=np.int64)
    for indices, label in (
        (mask_indices, "mask observation indices"),
        (trajectory_indices, "trajectory observation indices"),
    ):
        if indices.dtype != np.int64 or indices.shape != (expected_samples,):
            raise ValueError(f"{label} must use contiguous int64 values")
    if not (
        np.array_equal(mask_indices, expected_indices)
        and np.array_equal(trajectory_indices, expected_indices)
    ):
        raise ValueError("entity observation indices do not match timeline")
    if not np.array_equal(mask_state, trajectory_state):
        raise ValueError("entity state differs between mask and trajectory")
    entity = EntityObservation(
        object_id=object_id,
        mask_id=mask_id,
        masks=masks,
        centroid_xy=centroid,
        bbox_xyxy=bbox,
        area_pixels=area,
        state=mask_state,
    )
    _validate_entity(entity)
    if masks.shape[0] != expected_samples:
        raise ValueError("entity sample count does not match timeline")
    return entity


def write_timeline(path: str | Path, case_id: str, timeline: ReferenceTimeline) -> None:
    require_safe_id(case_id, label="reference observation case_id")
    write_json(path, timeline_to_dict(case_id, timeline))


def _verified_record(
    asset_root: Path,
    record: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(record, dict) or set(record) != _FILE_RECORD_KEYS:
        raise ValueError(f"{label} file record is invalid")
    size = record["size_bytes"]
    digest = record["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} size is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    path = resolve_dataset_file(asset_root, record["path"], label=label)
    if path.stat().st_size != size:
        raise ValueError(f"{label} size mismatch")
    if sha256_file(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def load_reference_observation(
    asset_root: str | Path,
    manifest_path: str | Path,
    *,
    bundle_root: str | Path | None = None,
) -> ReferenceObservationBundle:
    root = Path(asset_root).resolve(strict=True)
    allowed_bundle_root = (
        root
        if bundle_root is None
        else Path(bundle_root).resolve(strict=True)
    )
    manifest_file = resolve_dataset_file(
        allowed_bundle_root,
        manifest_path,
        label="reference observation manifest",
        allow_absolute=True,
    )
    child_root = manifest_file.parent
    manifest = load_json(manifest_file)
    required = {
        "schema_version",
        "case_id",
        "scene_id",
        "source",
        "generator",
        "timeline",
        "quality",
        "entities",
    }
    if set(manifest) != required or manifest.get("schema_version") != "1.0":
        raise ValueError("reference observation manifest has invalid fields")
    case_id = require_safe_id(
        manifest.get("case_id"),
        label="reference observation case_id",
    )
    scene_id = require_safe_id(
        manifest.get("scene_id"),
        label="reference observation scene_id",
    )
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {
        "reference_video",
        "first_frame_mask_manifest",
    }:
        raise ValueError("reference observation source has invalid fields")
    _verified_record(root, source["reference_video"], label="reference video")
    anchor_path = _verified_record(
        root,
        source["first_frame_mask_manifest"],
        label="first-frame mask manifest",
    )
    anchor_manifest = load_json(anchor_path)
    if anchor_manifest.get("case_id") != case_id:
        raise ValueError("first-frame mask manifest Case mismatch")
    timeline_path = _verified_record(
        child_root,
        manifest["timeline"],
        label="timeline",
    )
    quality_path = _verified_record(
        child_root,
        manifest["quality"],
        label="quality",
    )
    timeline_value = load_json(timeline_path)
    if timeline_value.get("case_id") != case_id:
        raise ValueError("reference observation timeline Case mismatch")
    timeline = timeline_from_dict(timeline_value)
    quality = load_json(quality_path)
    if quality.get("case_id") != case_id:
        raise ValueError("reference observation quality Case mismatch")
    raw_entities = manifest["entities"]
    if not isinstance(raw_entities, list) or not raw_entities:
        raise ValueError("reference observation entities must be non-empty")
    entities: dict[str, EntityObservation] = {}
    for index, record in enumerate(raw_entities, 1):
        if not isinstance(record, dict) or set(record) != {
            "object_id",
            "mask_id",
            "mask_tube",
            "trajectory",
        }:
            raise ValueError("reference observation entity fields are invalid")
        object_id = record["object_id"]
        if object_id != f"object_{index}" or object_id in entities:
            raise ValueError("reference observation object IDs must be contiguous")
        mask_id = record["mask_id"]
        if mask_id != f"{index:02d}":
            raise ValueError("reference observation mask IDs must be contiguous")
        mask_path = _verified_record(
            child_root,
            record["mask_tube"],
            label=f"{object_id} mask tube",
        )
        trajectory_path = _verified_record(
            child_root,
            record["trajectory"],
            label=f"{object_id} trajectory",
        )
        entities[object_id] = load_entity_observation(
            mask_path,
            trajectory_path,
            object_id=object_id,
            mask_id=mask_id,
            expected_samples=len(timeline.samples),
        )
    return ReferenceObservationBundle(
        case_id=case_id,
        scene_id=scene_id,
        timeline=timeline,
        entities=entities,
        manifest=manifest,
        manifest_path=manifest_file,
        manifest_sha256=sha256_file(manifest_file),
    )


__all__ = [
    "load_entity_observation",
    "load_reference_observation",
    "pack_mask_tube",
    "unpack_mask_tube",
    "write_entity_observation",
    "write_timeline",
]
