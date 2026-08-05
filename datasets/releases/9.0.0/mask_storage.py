#!/usr/bin/env python3
"""Storage contract for first-frame physical-subject masks."""

from __future__ import annotations

import copy
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import cv2
import numpy as np


NPZ_KEYS = {"masks", "mask_ids", "object_ids", "frame_index"}


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a non-empty 2-D uint8 mask encoded with values 0 and 1."""

    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a numpy array")
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError("mask must be a non-empty 2-D array")
    if mask.dtype != np.uint8:
        raise ValueError("mask dtype must be uint8")
    values = set(int(value) for value in np.unique(mask))
    if not values <= {0, 1} and not values <= {0, 255}:
        raise ValueError("mask values must use either {0,1} or {0,255}")
    if not np.any(mask > 0):
        raise ValueError("mask must contain foreground pixels")
    return np.ascontiguousarray(mask > 0, dtype=np.uint8)


def read_binary_png(
    path: Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode a PNG and normalize either approved binary representation."""

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"cannot decode mask PNG: {path}")
    binary = normalize_binary_mask(mask)
    if expected_shape is not None and binary.shape != expected_shape:
        raise ValueError(
            f"mask PNG shape {binary.shape} does not match {expected_shape}: {path}"
        )
    return binary


def _storage_metadata(instances: list[dict[str, Any]]) -> dict[str, Any]:
    if not instances:
        raise ValueError("storage metadata requires at least one instance")
    parents: set[PurePosixPath] = set()
    for instance in instances:
        asset = instance.get("asset")
        if not isinstance(asset, str):
            raise ValueError("instance PNG asset must be a string")
        relative = PurePosixPath(asset)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("instance PNG asset must be a safe relative path")
        parents.add(relative.parent)
    if len(parents) != 1:
        raise ValueError("all instance assets must share one mask directory")
    parent = next(iter(parents))
    return {
        "model": {
            "asset_pattern": (parent / "{mask_id}.npz").as_posix(),
            "array_key": "masks",
            "layout": "1HW",
            "dtype": "uint8",
            "values": [0, 1],
        },
        "visualization": {
            "asset_pattern": (relative.parent / "{mask_id}.png").as_posix(),
            "dtype": "uint8",
            "values": [0, 255],
        },
    }


def _validated_bundle(
    masks: list[np.ndarray],
    instances: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not masks or len(masks) != len(instances):
        raise ValueError("masks and instances must have equal non-zero length")
    normalized: list[np.ndarray] = []
    mask_ids: list[str] = []
    object_ids: list[str] = []
    expected_shape: tuple[int, int] | None = None
    for index, (mask, instance) in enumerate(zip(masks, instances), start=1):
        expected_id = f"{index:02d}"
        if not isinstance(instance, dict) or instance.get("mask_id") != expected_id:
            raise ValueError("instance mask IDs must be contiguous from 01")
        object_id = instance.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("every instance must have a non-empty object_id")
        asset = instance.get("asset")
        if not isinstance(asset, str) or PurePosixPath(asset).name != f"{expected_id}.png":
            raise ValueError("instance PNG asset must match its mask_id")
        binary = normalize_binary_mask(mask)
        if expected_shape is None:
            expected_shape = binary.shape
        elif binary.shape != expected_shape:
            raise ValueError("all masks in a bundle must have the same shape")
        if "area_pixels" in instance:
            area = instance["area_pixels"]
            if not isinstance(area, int) or area != int(np.count_nonzero(binary)):
                raise ValueError("instance area_pixels does not match its mask")
        normalized.append(binary)
        mask_ids.append(expected_id)
        object_ids.append(object_id)
    stack = np.stack(normalized).astype(np.uint8, copy=False)
    if np.any(np.sum(stack, axis=0) > 1):
        raise ValueError("instance masks must not overlap")
    mask_width = max(len(value) for value in mask_ids)
    object_width = max(len(value) for value in object_ids)
    return (
        stack,
        np.asarray(mask_ids, dtype=f"<U{mask_width}"),
        np.asarray(object_ids, dtype=f"<U{object_width}"),
    )


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".masks-",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_save_png(path: Path, mask: np.ndarray) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        if not cv2.imwrite(str(temporary), mask * 255):
            raise ValueError(f"failed to write PNG {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_mask_bundle(
    mask_directory: Path,
    masks: list[np.ndarray],
    instances: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically write one model archive and one visible PNG per object."""

    stack, mask_ids, object_ids = _validated_bundle(masks, instances)
    metadata = _storage_metadata(instances)
    mask_directory.mkdir(parents=True, exist_ok=True)
    for index, mask_id in enumerate(mask_ids.tolist()):
        arrays = {
            "masks": stack[index : index + 1],
            "mask_ids": mask_ids[index : index + 1],
            "object_ids": object_ids[index : index + 1],
            "frame_index": np.asarray(0, dtype=np.int64),
        }
        _atomic_save_npz(mask_directory / f"{mask_id}.npz", arrays)
        _atomic_save_png(mask_directory / f"{mask_id}.png", stack[index])
    return metadata


def load_mask_npz(path: Path) -> dict[str, np.ndarray]:
    """Load an approved Mask archive without enabling pickle."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != NPZ_KEYS:
            raise ValueError("mask NPZ has an unexpected key set")
        arrays = {key: archive[key].copy() for key in archive.files}
    masks = arrays["masks"]
    mask_ids = arrays["mask_ids"]
    object_ids = arrays["object_ids"]
    frame_index = arrays["frame_index"]
    if masks.dtype != np.uint8 or masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("masks must be non-empty uint8 [O,H,W]")
    values = set(int(value) for value in np.unique(masks))
    if not values <= {0, 1} or 1 not in values:
        raise ValueError("masks must use non-empty {0,1} values")
    if mask_ids.dtype.kind != "U" or mask_ids.shape != (masks.shape[0],):
        raise ValueError("mask_ids must be a Unicode [O] array")
    if object_ids.dtype.kind != "U" or object_ids.shape != (masks.shape[0],):
        raise ValueError("object_ids must be a Unicode [O] array")
    if frame_index.dtype != np.int64 or frame_index.shape != () or int(frame_index) != 0:
        raise ValueError("frame_index must be the int64 scalar zero")
    return arrays


def upgrade_manifest_storage(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return a schema 1.2 per-object manifest without mutating the input."""

    output = copy.deepcopy(manifest)
    instances = output.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("manifest instances must be a non-empty list")
    for index, instance in enumerate(instances):
        if instance.get("mask_id") != f"{index + 1:02d}":
            raise ValueError("manifest mask IDs must be contiguous from 01")
        if not isinstance(instance.get("object_id"), str):
            raise ValueError("manifest instance object_id must be a string")
        asset = instance.get("asset")
        if not isinstance(asset, str):
            raise ValueError("manifest instance PNG asset must be a string")
        instance["npz_asset"] = PurePosixPath(asset).with_suffix(".npz").as_posix()
        instance.pop("npz_index", None)
    output["schema_version"] = "1.2"
    output.pop("dtype", None)
    output.pop("values", None)
    output["storage"] = _storage_metadata(instances)
    return output
