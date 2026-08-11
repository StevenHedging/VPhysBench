from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..evaluation.common.masks.sam2 import MaskPrompt, Sam2VideoSegmenter
from ..io import load_json, sha256_file, write_json


DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"


def _contained(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the Dataset root") from exc
    return resolved


def resolve_training_mask_manifest(
    reference_video: str | Path,
    *,
    dataset_root: str | Path,
    case_id: str,
) -> Path:
    """Find mask supervision beside an authorized canonical training video."""

    root = Path(dataset_root).resolve()
    reference = _contained(Path(reference_video), root, label="reference video")
    if reference.name != "reference.mp4" or reference.parent.name != "canonical":
        raise ValueError(
            "training mask discovery requires canonical/reference.mp4"
        )
    manifest = _contained(
        reference.parent / "masks" / "manifest.json",
        root,
        label="mask manifest",
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"training mask manifest not found: {manifest}")
    value = load_json(manifest)
    if value.get("case_id") != case_id:
        raise ValueError(
            f"mask manifest case_id {value.get('case_id')!r} does not match "
            f"training case_id {case_id!r}"
        )
    return manifest


def contain_mask(
    mask: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Apply WAN contain geometry to a binary mask with zero-valued padding."""

    if mask.ndim != 2:
        raise ValueError("contain mask expects one HW plane")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"subject masks must be binary, got {sorted(values)}")
    source_height, source_width = mask.shape
    scale = min(output_width / source_width, output_height / source_height)
    resized_width = max(1, min(output_width, int(round(source_width * scale))))
    resized_height = max(1, min(output_height, int(round(source_height * scale))))
    resized = cv2.resize(
        mask.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((output_height, output_width), dtype=np.uint8)
    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open normalized training video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"normalized training video has no frames: {path}")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("normalized training video changes spatial shape")
    return frames


def _asset_path(value: str, *, dataset_root: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"mask asset must be Dataset-relative: {value!r}")
    path = _contained(dataset_root / relative, dataset_root, label="mask asset")
    if not path.is_file():
        raise FileNotFoundError(f"mask asset not found: {path}")
    return path


def _load_seed_masks(
    manifest: dict[str, Any],
    *,
    dataset_root: Path,
    output_width: int,
    output_height: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    instances = manifest.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("training mask manifest requires at least one instance")
    seeds: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for instance in instances:
        value = instance.get("npz_asset")
        if not isinstance(value, str) or not value:
            raise ValueError("training mask instance requires npz_asset")
        path = _asset_path(value, dataset_root=dataset_root)
        with np.load(path, allow_pickle=False) as payload:
            if "masks" not in payload:
                raise ValueError(f"mask asset has no masks array: {path}")
            masks = payload["masks"]
        if masks.ndim != 3 or masks.shape[0] != 1:
            raise ValueError(
                "current VPhysBench training supervision requires one first-frame "
                f"mask per instance, got {masks.shape}"
            )
        values = set(np.unique(masks).tolist())
        if not values.issubset({0, 1}):
            raise ValueError(
                f"subject masks must be binary, got {sorted(values)} in {path}"
            )
        transformed = contain_mask(
            masks[0],
            output_width=output_width,
            output_height=output_height,
        )
        if not bool(transformed.any()):
            raise ValueError(f"subject mask becomes empty after WAN geometry: {path}")
        seeds.append(transformed)
        sources.append({
            "mask_id": instance.get("mask_id"),
            "object_id": instance.get("object_id"),
            "path": str(path),
            "sha256": sha256_file(path),
            "foreground_pixels": int(transformed.sum()),
        })
    return seeds, sources


def _prompt_from_mask(
    mask: np.ndarray,
    *,
    instance: dict[str, Any],
) -> MaskPrompt:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("cannot build a SAM2 prompt from an empty mask")
    box = np.asarray(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )
    points = np.asarray(
        [[float(np.median(xs)), float(np.median(ys))]],
        dtype=np.float32,
    )
    return MaskPrompt(
        frame_index=0,
        box_xyxy=box,
        points_xy=points,
        point_labels=np.ones((1,), dtype=np.int32),
        metadata={
            "mask_id": instance.get("mask_id"),
            "object_id": instance.get("object_id"),
            "source": "vphysbench_first_frame_subject_mask",
        },
    )


def _cached_audit(
    output: Path,
    audit_path: Path,
    *,
    source_fingerprint: dict[str, Any],
) -> dict[str, Any] | None:
    if not output.is_file() or not audit_path.is_file():
        return None
    audit = load_json(audit_path)
    if audit.get("source_fingerprint") != source_fingerprint:
        return None
    if audit.get("tube_sha256") != sha256_file(output):
        return None
    with np.load(output, allow_pickle=False) as payload:
        tube = payload.get("masks")
        if tube is None or tuple(tube.shape) != tuple(audit.get("shape_thw", [])):
            return None
        if tube.dtype != np.uint8 or not set(np.unique(tube).tolist()).issubset({0, 1}):
            return None
    return {**audit, "status": "cached"}


def materialize_subject_mask_tube(
    *,
    normalized_video: str | Path,
    mask_manifest: str | Path,
    dataset_root: str | Path,
    output: str | Path,
    case_id: str,
    segmenter: Any | None = None,
) -> dict[str, Any]:
    """Propagate first-frame instances into an auditable normalized THW tube."""

    root = Path(dataset_root).resolve()
    video = Path(normalized_video).resolve()
    manifest_path = _contained(
        Path(mask_manifest), root, label="mask manifest"
    )
    output_path = Path(output).resolve()
    audit_path = output_path.with_suffix(".audit.json")
    manifest = load_json(manifest_path)
    if manifest.get("case_id") != case_id:
        raise ValueError("mask manifest case_id does not match training case_id")
    if not video.is_file():
        raise FileNotFoundError(f"normalized training video not found: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot probe normalized training video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width < 1 or height < 1 or frame_count < 1:
        raise ValueError("normalized training video has invalid dimensions")

    seeds, mask_sources = _load_seed_masks(
        manifest,
        dataset_root=root,
        output_width=width,
        output_height=height,
    )
    model_id = str(
        getattr(segmenter, "model_id", DEFAULT_SAM2_MODEL_ID)
    )
    source_fingerprint = {
        "case_id": case_id,
        "normalized_video_sha256": sha256_file(video),
        "mask_manifest_sha256": sha256_file(manifest_path),
        "mask_asset_sha256": [item["sha256"] for item in mask_sources],
        "segmenter_model_id": model_id,
        "target_shape_thw": [frame_count, height, width],
    }
    cached = _cached_audit(
        output_path,
        audit_path,
        source_fingerprint=source_fingerprint,
    )
    if cached is not None:
        return cached

    frames = _read_video(video)
    if len(frames) != frame_count:
        frame_count = len(frames)
        source_fingerprint["target_shape_thw"] = [frame_count, height, width]
    prompts = [
        _prompt_from_mask(seed, instance=instance)
        for seed, instance in zip(seeds, manifest["instances"], strict=True)
    ]
    if segmenter is None:
        segmenter = Sam2VideoSegmenter({
            "model_id": DEFAULT_SAM2_MODEL_ID,
            "device": "cuda",
        })
    instance_tubes, propagation = segmenter.segment_instances(
        frames,
        prompts=prompts,
        temporary_prefix="physbench_wan_st_tube_",
        exclusive_masks=False,
    )
    if len(instance_tubes) != len(seeds):
        raise ValueError("SAM2 instance count differs from seed-mask count")
    union = np.zeros((frame_count, height, width), dtype=np.uint8)
    for masks in instance_tubes:
        if len(masks) != frame_count:
            raise ValueError("SAM2 tube frame count differs from normalized video")
        for index, mask in enumerate(masks):
            plane = np.asarray(mask)
            if plane.shape != (height, width):
                raise ValueError("SAM2 mask shape differs from normalized video")
            union[index] |= (plane > 0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        masks=union,
        layout=np.asarray("THW"),
        case_id=np.asarray(case_id),
    )
    audit = {
        "schema_version": "1.0",
        "status": "materialized",
        "case_id": case_id,
        "source": "sam2_propagated_first_frame_subject_masks",
        "normalized_video": str(video),
        "mask_manifest": str(manifest_path),
        "output": str(output_path),
        "shape_thw": list(union.shape),
        "dtype": str(union.dtype),
        "values": sorted(np.unique(union).tolist()),
        "instance_count": len(seeds),
        "frame_count": frame_count,
        "foreground_fraction": float(union.mean()),
        "mask_sources": mask_sources,
        "propagation": propagation,
        "spatial_mapping": "wan_normalized_video_canvas_nearest_mask_seed",
        "temporal_mapping": "propagate_on_exact_normalized_training_frames",
        "source_fingerprint": source_fingerprint,
        "tube_sha256": sha256_file(output_path),
    }
    write_json(audit_path, audit)
    return audit


__all__ = [
    "DEFAULT_SAM2_MODEL_ID",
    "contain_mask",
    "materialize_subject_mask_tube",
    "resolve_training_mask_manifest",
]
