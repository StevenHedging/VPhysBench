from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..storage import unpack_mask_tube


@dataclass(frozen=True)
class CandidateFile:
    source: Path
    target_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CandidateBundle:
    case_id: str
    manifest_path: Path
    files: tuple[CandidateFile, ...]

    @property
    def digest(self) -> str:
        return _sha256(self.manifest_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} escapes the candidate bundle")
    return value


def _scalar_int(value: np.ndarray, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.int64:
        raise ValueError(f"{label} must be an int64 scalar")
    return int(array)


def _validate_reductions(mask_path: Path, trajectory_path: Path) -> None:
    with np.load(mask_path, allow_pickle=False) as payload:
        required = {"packed_masks", "height", "width", "observation_index", "state"}
        if set(payload.files) != required:
            raise ValueError("mask tube fields are invalid")
        height = _scalar_int(payload["height"], "height")
        width = _scalar_int(payload["width"], "width")
        masks = unpack_mask_tube(np.asarray(payload["packed_masks"]), width=width)
        mask_indices = np.asarray(payload["observation_index"])
        mask_state = np.asarray(payload["state"])
    if masks.shape[1] != height:
        raise ValueError("mask height is inconsistent")
    with np.load(trajectory_path, allow_pickle=False) as payload:
        required = {
            "centroid_xy", "bbox_xyxy", "area_pixels", "observation_index", "state"
        }
        if set(payload.files) != required:
            raise ValueError("trajectory fields are invalid")
        centroid = np.asarray(payload["centroid_xy"])
        bbox = np.asarray(payload["bbox_xyxy"])
        area = np.asarray(payload["area_pixels"])
        trajectory_indices = np.asarray(payload["observation_index"])
        trajectory_state = np.asarray(payload["state"])
    count = masks.shape[0]
    expected_indices = np.arange(count, dtype=np.int64)
    if not np.array_equal(mask_indices, expected_indices) or not np.array_equal(
        trajectory_indices, expected_indices
    ):
        raise ValueError("observation_index is not contiguous")
    if not np.array_equal(mask_state, trajectory_state):
        raise ValueError("state differs between mask tube and trajectory")
    expected_area = masks.reshape(count, -1).sum(axis=1, dtype=np.int64)
    if area.dtype != np.int64 or not np.array_equal(area, expected_area):
        raise ValueError("area_pixels is not the exact reduction of mask pixels")
    expected_centroid = np.full((count, 2), np.nan, np.float32)
    expected_bbox = np.full((count, 4), np.nan, np.float32)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            expected_centroid[index] = (float(xs.mean()), float(ys.mean()))
            expected_bbox[index] = (xs.min(), ys.min(), xs.max(), ys.max())
    if centroid.dtype != np.float32 or not np.allclose(
        centroid, expected_centroid, rtol=0.0, atol=1e-4, equal_nan=True
    ):
        raise ValueError("centroid_xy is not derived from mask pixels")
    if bbox.dtype != np.float32 or not np.allclose(
        bbox, expected_bbox, rtol=0.0, atol=0.0, equal_nan=True
    ):
        raise ValueError("bbox_xyxy is not derived from mask pixels")


def validate_candidate_bundle(manifest_path: str | Path) -> CandidateBundle:
    manifest_file = Path(manifest_path).resolve(strict=True)
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    required_fields = {"schema_version", "case_id", "required_targets", "files"}
    if set(value) != required_fields or value["schema_version"] != "1.0":
        raise ValueError("candidate bundle manifest fields are invalid")
    if not isinstance(value["case_id"], str) or not value["case_id"]:
        raise ValueError("candidate bundle case_id is invalid")
    required_targets = tuple(
        _safe_relative(item, label="required target") for item in value["required_targets"]
    )
    if len(set(required_targets)) != len(required_targets):
        raise ValueError("candidate bundle has duplicate required targets")
    files: list[CandidateFile] = []
    target_to_file: dict[str, CandidateFile] = {}
    for raw in value["files"]:
        if not isinstance(raw, dict) or set(raw) != {
            "candidate_path", "target_path", "size_bytes", "sha256"
        }:
            raise ValueError("candidate file record is invalid")
        candidate_path = _safe_relative(raw["candidate_path"], label="candidate path")
        target_path = _safe_relative(raw["target_path"], label="target path")
        source = (manifest_file.parent / candidate_path).resolve(strict=True)
        try:
            source.relative_to(manifest_file.parent)
        except ValueError as exc:
            raise ValueError("candidate path escapes the bundle") from exc
        size = raw["size_bytes"]
        digest = raw["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("candidate file size is invalid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("candidate file SHA-256 is invalid")
        if source.stat().st_size != size or _sha256(source) != digest:
            raise ValueError(f"candidate file digest mismatch: {candidate_path}")
        record = CandidateFile(source, target_path, size, digest)
        if target_path in target_to_file:
            raise ValueError(f"duplicate candidate target: {target_path}")
        target_to_file[target_path] = record
        files.append(record)
    if set(target_to_file) != set(required_targets):
        raise ValueError("candidate files do not exactly cover required targets")
    for target, mask_record in target_to_file.items():
        suffix = "/mask_tube.npz"
        if not target.endswith(suffix):
            continue
        trajectory_target = target[: -len(suffix)] + "/trajectory.npz"
        trajectory_record = target_to_file.get(trajectory_target)
        if trajectory_record is None:
            raise ValueError(f"mask tube lacks trajectory target: {target}")
        _validate_reductions(mask_record.source, trajectory_record.source)
    return CandidateBundle(
        case_id=value["case_id"],
        manifest_path=manifest_file,
        files=tuple(files),
    )
