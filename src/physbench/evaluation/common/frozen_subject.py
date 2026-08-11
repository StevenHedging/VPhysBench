"""Strict evaluator-only loading of frozen first-frame subject masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ...io import load_json, sha256_file
from ..contracts import CaseEvaluationRequest
from .errors import ReferenceAnalysisError


@dataclass(frozen=True)
class FrozenSubjectAnchor:
    case_id: str
    logical_entity_id: str
    dataset_object_id: str
    entity_class: str
    manifest_path: Path
    npz_path: Path
    source_mask: np.ndarray
    mask: np.ndarray
    centroid_xy: np.ndarray
    area_px2: float
    equivalent_radius_px: float
    provenance: Mapping[str, object]


def transform_frozen_subject_mask(
    mask: np.ndarray,
    spatial_transform: Mapping[str, object],
) -> np.ndarray:
    """Apply an evaluator media transform to a binary Dataset mask."""

    source = np.asarray(mask)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("frozen subject mask must be a non-empty 2D array")
    source_binary = np.where(source > 0, 255, 0).astype(np.uint8)
    source_height, source_width = source_binary.shape
    declared_source = _integer_pair(
        spatial_transform.get("source_size"), name="source_size"
    )
    if declared_source != (source_width, source_height):
        raise ValueError(
            "mask shape differs from spatial transform source_size"
        )
    target_width, target_height = _integer_pair(
        spatial_transform.get("target_size"), name="target_size"
    )
    policy = spatial_transform.get("policy")
    if policy == "reference_content_crop_resize_no_pad":
        x, y, crop_width, crop_height = _integer_quad(
            spatial_transform.get("crop_xywh"), name="crop_xywh"
        )
        if (
            x < 0
            or y < 0
            or crop_width <= 0
            or crop_height <= 0
            or x + crop_width > source_width
            or y + crop_height > source_height
            or crop_width * target_height != crop_height * target_width
        ):
            raise ValueError("invalid no-pad mask crop")
        scale = _finite_positive(spatial_transform.get("scale"), name="scale")
        if not math.isclose(
            scale,
            target_width / crop_width,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("no-pad mask scale differs from crop geometry")
        output = cv2.resize(
            source_binary[y : y + crop_height, x : x + crop_width],
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    elif policy == "preserve_aspect_ratio_letterbox":
        scale = _finite_positive(spatial_transform.get("scale"), name="scale")
        offset_x, offset_y = _integer_pair(
            spatial_transform.get("offset_xy"),
            name="offset_xy",
            allow_zero=True,
        )
        expected_scale = min(
            target_width / source_width,
            target_height / source_height,
        )
        if not math.isclose(
            scale, expected_scale, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "letterbox mask scale differs from source/target geometry"
            )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        expected_offset = (
            (target_width - resized_width) // 2,
            (target_height - resized_height) // 2,
        )
        if (offset_x, offset_y) != expected_offset:
            raise ValueError(
                "letterbox mask offset differs from centered geometry"
            )
        if (
            offset_x + resized_width > target_width
            or offset_y + resized_height > target_height
        ):
            raise ValueError("letterbox mask placement exceeds target canvas")
        resized = cv2.resize(
            source_binary,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        output[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
    else:
        raise ValueError(f"unsupported mask spatial policy: {policy!r}")
    return np.where(output > 0, 255, 0).astype(np.uint8)


def load_frozen_subject_anchor(
    request: CaseEvaluationRequest,
    *,
    logical_entity_id: str,
    entity_class: str,
    spatial_transform: Mapping[str, object],
    dataset_object_id: str | None = None,
    error_namespace: str = "reference_subject",
) -> FrozenSubjectAnchor:
    """Load exactly one annotated subject or fail closed as reference data."""

    if not logical_entity_id:
        raise ValueError("logical_entity_id must be non-empty")
    if not entity_class:
        raise ValueError("entity_class must be non-empty")
    if not error_namespace:
        raise ValueError("error_namespace must be non-empty")
    root = request.asset_root.resolve()
    manifest_value = request.case.get("assets", {}).get(
        "first_frame_mask_manifest"
    )
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ReferenceAnalysisError(
            f"{error_namespace}_manifest_missing",
            "Case does not declare assets.first_frame_mask_manifest",
        )
    manifest_path = _resolve_asset(
        root,
        manifest_value,
        code=f"{error_namespace}_manifest_path_escape",
        label="frozen subject mask manifest",
    )
    if not manifest_path.is_file():
        raise ReferenceAnalysisError(
            f"{error_namespace}_manifest_missing",
            f"frozen subject mask manifest does not exist: {manifest_path}",
        )
    try:
        manifest = load_json(manifest_path)
        instance, image_shape = _validate_manifest(
            manifest,
            request=request,
            entity_class=entity_class,
            dataset_object_id=dataset_object_id,
        )
    except ReferenceAnalysisError:
        raise
    except Exception as exc:
        raise ReferenceAnalysisError(
            f"{error_namespace}_manifest_invalid",
            "frozen subject mask manifest violates schema 1.2: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    npz_value = instance.get("npz_asset")
    npz_path = _resolve_asset(
        root,
        npz_value,
        code=f"{error_namespace}_mask_path_escape",
        label="frozen subject mask NPZ",
    )
    if not npz_path.is_file():
        raise ReferenceAnalysisError(
            f"{error_namespace}_mask_missing",
            f"frozen subject mask NPZ does not exist: {npz_path}",
        )
    try:
        source_mask, npz_object_id = _load_npz_mask(
            npz_path,
            instance=instance,
            image_shape=image_shape,
            logical_entity_id=logical_entity_id,
        )
        bbox_policy = _validate_geometry(source_mask, instance=instance)
    except Exception as exc:
        raise ReferenceAnalysisError(
            f"{error_namespace}_mask_invalid",
            "frozen subject mask NPZ violates the per-object contract: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        transformed = transform_frozen_subject_mask(
            source_mask, spatial_transform
        )
    except Exception as exc:
        raise ReferenceAnalysisError(
            f"{error_namespace}_mask_transform_invalid",
            "frozen subject mask cannot use the evaluator transform: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    ys, xs = np.where(transformed > 0)
    if xs.size == 0:
        raise ReferenceAnalysisError(
            f"{error_namespace}_mask_transform_invalid",
            "frozen subject mask is empty after the evaluator transform",
        )
    source_mask.setflags(write=False)
    transformed.setflags(write=False)
    area = float(xs.size)
    selected_object_id = str(instance["object_id"])
    return FrozenSubjectAnchor(
        case_id=str(request.case["case_id"]),
        logical_entity_id=logical_entity_id,
        dataset_object_id=selected_object_id,
        entity_class=entity_class,
        manifest_path=manifest_path,
        npz_path=npz_path,
        source_mask=source_mask,
        mask=transformed,
        centroid_xy=np.asarray([xs.mean(), ys.mean()], dtype=np.float64),
        area_px2=area,
        equivalent_radius_px=float(math.sqrt(area / math.pi)),
        provenance={
            "policy": "frozen_dataset_subject_annotation_v1",
            "path_policy": "dataset_relative_asset_reference_v1",
            "manifest": manifest_value,
            "manifest_sha256": sha256_file(manifest_path),
            "npz": str(npz_value),
            "npz_sha256": sha256_file(npz_path),
            "mask_id": str(instance["mask_id"]),
            "dataset_object_id": selected_object_id,
            "logical_entity_id": logical_entity_id,
            "npz_object_id": npz_object_id,
            "bbox_policy": bbox_policy,
            "entity_class": entity_class,
            "source_shape_hw": list(source_mask.shape),
            "target_shape_hw": list(transformed.shape),
            "spatial_transform": dict(spatial_transform),
        },
    )


def _resolve_asset(root: Path, value: object, *, code: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReferenceAnalysisError(code, f"{label} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReferenceAnalysisError(
            code, f"{label} escapes the Dataset root: {value}"
        )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            code, f"{label} escapes the Dataset root: {value}"
        ) from exc
    return path


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    request: CaseEvaluationRequest,
    entity_class: str,
    dataset_object_id: str | None,
) -> tuple[Mapping[str, Any], tuple[int, int]]:
    if manifest.get("schema_version") != "1.2":
        raise ValueError("schema_version must be '1.2'")
    if manifest.get("case_id") != request.case.get("case_id"):
        raise ValueError("case_id differs from the evaluated Case")
    if manifest.get("scene_id") != request.case.get("scene_id"):
        raise ValueError("scene_id differs from the evaluated Case")
    if manifest.get("frame_index") != 0:
        raise ValueError("frame_index must be zero")
    if manifest.get("frame_scope") != "first_frame_only":
        raise ValueError("frame_scope must be 'first_frame_only'")
    if manifest.get("source_first_frame") != request.case.get("assets", {}).get(
        "first_frame"
    ):
        raise ValueError("source_first_frame differs from the Case asset")
    image_shape = _integer_pair(
        manifest.get("image_shape_hw"), name="image_shape_hw"
    )
    storage = manifest.get("storage")
    if not isinstance(storage, Mapping):
        raise TypeError("storage must be an object")
    model = storage.get("model")
    if not isinstance(model, Mapping) or (
        model.get("array_key") != "masks"
        or model.get("layout") != "1HW"
        or model.get("dtype") != "uint8"
        or model.get("values") != [0, 1]
    ):
        raise ValueError("storage.model differs from the per-object contract")
    instances = manifest.get("instances")
    if not isinstance(instances, list):
        raise TypeError("instances must be a list")
    matches = [
        item
        for item in instances
        if isinstance(item, Mapping)
        and item.get("entity_class") == entity_class
        and (
            dataset_object_id is None
            or item.get("object_id") == dataset_object_id
        )
    ]
    if len(matches) != 1:
        qualifier = (
            entity_class
            if dataset_object_id is None
            else f"{entity_class}/{dataset_object_id}"
        )
        raise ValueError(
            f"manifest must declare exactly one subject matching {qualifier}"
        )
    instance = matches[0]
    for key in (
        "mask_id",
        "object_id",
        "npz_asset",
        "area_pixels",
        "bbox_xyxy",
        "centroid_xy",
    ):
        if key not in instance:
            raise KeyError(key)
    if not isinstance(instance["mask_id"], str) or not instance["mask_id"]:
        raise TypeError("mask_id must be a non-empty string")
    if not isinstance(instance["object_id"], str) or not instance["object_id"]:
        raise TypeError("object_id must be a non-empty string")
    return instance, image_shape


def _load_npz_mask(
    path: Path,
    *,
    instance: Mapping[str, Any],
    image_shape: tuple[int, int],
    logical_entity_id: str,
) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "masks",
            "mask_ids",
            "object_ids",
            "frame_index",
        }:
            raise ValueError("NPZ keys differ from the per-object contract")
        masks = payload["masks"]
        mask_ids = payload["mask_ids"]
        object_ids = payload["object_ids"]
        frame_index = payload["frame_index"]
        if masks.dtype != np.uint8 or masks.shape != (1, *image_shape):
            raise ValueError("masks must be uint8 [1,H,W]")
        if set(int(value) for value in np.unique(masks)) - {0, 1}:
            raise ValueError("masks values must be binary {0,1}")
        if (
            mask_ids.shape != (1,)
            or mask_ids.dtype.kind != "U"
            or str(mask_ids[0]) != instance["mask_id"]
        ):
            raise ValueError("mask_ids does not identify this mask")
        npz_object_id = str(object_ids[0]) if object_ids.shape == (1,) else ""
        if (
            object_ids.shape != (1,)
            or object_ids.dtype.kind != "U"
            or npz_object_id
            not in {logical_entity_id, str(instance["object_id"])}
        ):
            raise ValueError(
                "object_ids identifies neither the logical nor selected "
                "Dataset entity"
            )
        if (
            frame_index.shape != ()
            or frame_index.dtype != np.int64
            or int(frame_index) != 0
        ):
            raise ValueError("frame_index must be int64 scalar zero")
        output = np.array(masks[0], copy=True)
    if not np.any(output):
        raise ValueError("subject mask must not be empty")
    return output, npz_object_id


def _validate_geometry(
    mask: np.ndarray, *, instance: Mapping[str, Any]
) -> str:
    ys, xs = np.where(mask > 0)
    area = int(xs.size)
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    centroid = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
    if instance["area_pixels"] != area:
        raise ValueError("area_pixels differs from the NPZ mask")
    inclusive_bbox = [bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1]
    if instance["bbox_xyxy"] == bbox:
        bbox_policy = "xyxy_half_open"
    elif instance["bbox_xyxy"] == inclusive_bbox:
        bbox_policy = "xyxy_inclusive_max_legacy"
    else:
        raise ValueError("bbox_xyxy differs from the NPZ mask")
    declared_centroid = np.asarray(instance["centroid_xy"], dtype=np.float64)
    if declared_centroid.shape != (2,) or not np.allclose(
        centroid, declared_centroid, rtol=0.0, atol=1e-6
    ):
        raise ValueError("centroid_xy differs from the NPZ mask")
    return bbox_policy


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or output <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return output


def _integer_pair(
    value: object, *, name: str, allow_zero: bool = False
) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
    ):
        raise TypeError(f"{name} must contain two integers")
    output = (int(value[0]), int(value[1]))
    minimum = 0 if allow_zero else 1
    if min(output) < minimum:
        raise ValueError(f"{name} entries must be >= {minimum}")
    return output


def _integer_quad(value: object, *, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
    ):
        raise TypeError(f"{name} must contain four integers")
    return tuple(int(item) for item in value)


__all__ = [
    "FrozenSubjectAnchor",
    "load_frozen_subject_anchor",
    "transform_frozen_subject_mask",
]
