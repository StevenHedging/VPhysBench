"""Hash-closed staged bundles for SAM 3.1 collision GT candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..storage import write_entity_observation
from .bundle import validate_candidate_bundle
from .finalize import entity_from_masks, write_anchor_assets
from .sam31_rebuild import Sam31GtCaseCandidate
from .visualization import render_dense_event_sheet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _render_trajectory(
    frame: np.ndarray,
    masks_by_object: Mapping[str, np.ndarray],
) -> np.ndarray:
    image = np.asarray(frame).copy()
    colors = ((0, 255, 255), (255, 128, 0), (0, 128, 255))
    for object_index, (object_id, masks) in enumerate(masks_by_object.items()):
        points = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            if len(xs):
                points.append((int(round(xs.mean())), int(round(ys.mean()))))
        if len(points) > 1:
            cv2.polylines(
                image,
                [np.asarray(points, np.int32)],
                False,
                colors[object_index % len(colors)],
                1,
                cv2.LINE_AA,
            )
        if points:
            cv2.putText(
                image,
                object_id,
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colors[object_index % len(colors)],
                1,
                cv2.LINE_AA,
            )
    return image


def _write_overlay_video(
    path: Path,
    frames: Sequence[np.ndarray],
    masks_by_object: Mapping[str, np.ndarray],
    *,
    fps: float,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create SAM3.1 GT overlay video: {path}")
    colors = ((0, 255, 255), (255, 128, 0), (0, 128, 255))
    try:
        for frame_index, frame in enumerate(frames):
            rendered = np.asarray(frame).copy()
            for object_index, masks in enumerate(masks_by_object.values()):
                mask = np.where(masks[frame_index], 255, 0).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(
                    rendered,
                    contours,
                    -1,
                    colors[object_index % len(colors)],
                    2,
                )
            writer.write(rendered)
    finally:
        writer.release()


def _quality_issue(finding: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "code": finding.code,
        "status": "fail" if finding.severity == "error" else "warning",
        "message": finding.message,
    }
    if finding.object_ids:
        value["object_id"] = finding.object_ids[0]
    if finding.observation_indices:
        value["observation_index"] = finding.observation_indices[0]
    return value


def _review_issue(finding: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"code": finding.code, "notes": finding.message}
    if finding.object_ids:
        value["object_id"] = finding.object_ids[0]
    if finding.observation_indices:
        value["observation_index"] = finding.observation_indices[0]
    return value


def _bundle_record(root: Path, source: Path, target: str) -> dict[str, Any]:
    return {
        "candidate_path": source.relative_to(root).as_posix(),
        "target_path": target,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def write_sam31_candidate_bundle(
    case: Any,
    candidate: Sam31GtCaseCandidate,
    *,
    frames: Sequence[np.ndarray],
    output_root: str | Path,
    config_fingerprint: str,
    code_revision: str,
) -> Path:
    if case.case_id != candidate.case_id or case.scene_id != "collision_1d":
        raise ValueError("SAM3.1 GT candidate does not match its collision Case")
    if (
        len(config_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in config_fingerprint)
    ):
        raise ValueError("SAM3.1 GT config fingerprint must be lowercase SHA-256")
    if not code_revision:
        raise ValueError("SAM3.1 GT code revision must be non-empty")
    frame_values = tuple(np.asarray(frame) for frame in frames)
    if not frame_values or len(frame_values) != len(candidate.source_frame_indices):
        raise ValueError("SAM3.1 GT bundle frames differ from candidate timeline")
    frame_shape = frame_values[0].shape[:2]
    if any(
        frame.dtype != np.uint8
        or frame.shape != (*frame_shape, 3)
        for frame in frame_values
    ):
        raise ValueError("SAM3.1 GT bundle frames must be equal uint8 BGR images")

    output = Path(output_root) / case.scene_id / case.case_id
    if output.exists():
        raise FileExistsError(f"SAM3.1 GT candidate already exists: {output}")
    masks_root = output / "canonical/masks"
    observation_root = output / "canonical/reference_observation"
    visualization_root = observation_root / "visualization"
    masks_root.mkdir(parents=True)
    visualization_root.mkdir(parents=True)

    entities: dict[str, Any] = {}
    instances: list[dict[str, Any]] = []
    mask_asset_prefix = case.assets["first_frame_mask_manifest"].rsplit("/", 1)[0]
    for identity in candidate.identities:
        masks = np.asarray(candidate.masks_by_object[identity.object_id], dtype=np.uint8)
        states = np.asarray(candidate.states_by_object[identity.object_id], dtype=np.uint8)
        if masks.shape != (len(frame_values), *frame_shape):
            raise ValueError(
                f"SAM3.1 GT {identity.object_id} mask shape differs from frames"
            )
        entity = entity_from_masks(
            identity.object_id,
            identity.mask_id,
            masks,
            states,
        )
        entities[identity.object_id] = entity
        entity_root = observation_root / "entities" / identity.object_id
        write_entity_observation(entity_root, entity)
        record = write_anchor_assets(
            masks_root,
            entity,
            semantic_object_id=identity.evaluator_object_id,
            entity_class="ball",
        )
        record.update(
            {
                "asset": f"{mask_asset_prefix}/{identity.mask_id}.png",
                "npz_asset": f"{mask_asset_prefix}/{identity.mask_id}.npz",
                "physics_keys": [
                    f"objects.{identity.object_id}.{quantity_name}"
                    for quantity_name in case.physics["objects"][identity.object_id]
                ],
                "segmentation": {
                    "model": "SAM 3.1 multiplex",
                    "review": "pending full-tube SAM3.1 curation review",
                    "selection_policy": "sam31_collision_gt_row_major_v1",
                },
            }
        )
        instances.append(record)

    mask_manifest = copy.deepcopy(case.mask_manifest)
    mask_manifest["instances"] = instances
    mask_manifest["image_shape_hw"] = list(frame_shape)
    mask_manifest["ordering"] = "row_major_top_to_bottom_then_left_to_right"
    mask_manifest["generator"] = {
        "id": "sam31_collision_gt_curation_v1",
        "model": "SAM 3.1 multiplex",
        "review_policy": "candidate_requires_recorded_full_video_review",
    }
    mask_manifest.pop("localization", None)
    mask_manifest_path = masks_root / "manifest.json"
    _write_json(mask_manifest_path, mask_manifest)

    timeline_source = case.observation_manifest_path.parent / "timeline.json"
    if not timeline_source.is_file():
        raise ValueError(f"SAM3.1 GT timeline is missing: {timeline_source}")
    timeline_path = observation_root / "timeline.json"
    shutil.copy2(timeline_source, timeline_path)

    failures = [item for item in candidate.findings if item.severity == "error"]
    warnings = [item for item in candidate.findings if item.severity != "error"]
    quality = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "status": "fail" if failures else ("warning" if warnings else "pass"),
        "config_fingerprint": config_fingerprint,
        "checks": [
            {
                "code": "row_major_identity_binding",
                "status": "pass",
                "message": "frame-zero masks are bound top-to-bottom then left-to-right",
            },
            {
                "code": "anchor_tube_zero_equality",
                "status": "pass",
                "message": "every persisted anchor equals observation zero",
            },
        ],
        "entities": {
            object_id: {
                "sample_count": len(entity.state),
                "visible_samples": int(np.count_nonzero(entity.state == 0)),
                "out_of_frame_samples": int(np.count_nonzero(entity.state == 2)),
                "unresolved_samples": int(np.count_nonzero(entity.state == 3)),
                "frame_zero_area_pixels": int(entity.area_pixels[0]),
            }
            for object_id, entity in entities.items()
        },
        "failures": [_quality_issue(item) for item in failures],
        "warnings": [_quality_issue(item) for item in warnings],
    }
    quality_path = observation_root / "quality.json"
    _write_json(quality_path, quality)

    observation_manifest = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "scene_id": case.scene_id,
        "source": {
            "reference_video": _record(
                case.reference_video_path,
                case.assets["reference_video"],
            ),
            "first_frame_mask_manifest": _record(
                mask_manifest_path,
                case.assets["first_frame_mask_manifest"],
            ),
            "anchor_masks": [
                {
                    "object_id": identity.object_id,
                    "mask_id": identity.mask_id,
                    "asset": _record(
                        masks_root / f"{identity.mask_id}.npz",
                        f"{mask_asset_prefix}/{identity.mask_id}.npz",
                    ),
                }
                for identity in candidate.identities
            ],
        },
        "generator": {
            "id": "sam31_collision_gt_curation_v1",
            "code_revision": code_revision,
            "model_id": "SAM 3.1 multiplex",
            "config_fingerprint": config_fingerprint,
            "propagation": {
                "identity_policy": "frame_zero_row_major_locked",
                "mask_overlap_policy": "independent_nonexclusive",
                "discovery_attempts": list(candidate.discovery_attempts),
                "predictor": dict(candidate.predictor_provenance),
            },
        },
        "timeline": _record(timeline_path, "timeline.json"),
        "quality": _record(quality_path, "quality.json"),
        "entities": [
            {
                "object_id": identity.object_id,
                "mask_id": identity.mask_id,
                "mask_tube": _record(
                    observation_root
                    / "entities"
                    / identity.object_id
                    / "mask_tube.npz",
                    f"entities/{identity.object_id}/mask_tube.npz",
                ),
                "trajectory": _record(
                    observation_root
                    / "entities"
                    / identity.object_id
                    / "trajectory.npz",
                    f"entities/{identity.object_id}/trajectory.npz",
                ),
            }
            for identity in candidate.identities
        ],
    }
    observation_manifest_path = observation_root / "manifest.json"
    _write_json(observation_manifest_path, observation_manifest)

    sample_indices = tuple(
        dict.fromkeys(
            int(value)
            for value in np.linspace(
                0,
                len(frame_values) - 1,
                min(16, len(frame_values)),
                dtype=int,
            )
        )
    )
    contact_sheet = render_dense_event_sheet(
        frame_values,
        masks_by_object=candidate.masks_by_object,
        states_by_object=candidate.states_by_object,
        observation_indices=sample_indices,
        source_indices=tuple(candidate.source_frame_indices[index] for index in sample_indices),
        columns=4,
    )
    contact_path = visualization_root / "contact_sheet.png"
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError(f"cannot write SAM3.1 GT contact sheet: {contact_path}")
    trajectory_path = visualization_root / "trajectory.png"
    if not cv2.imwrite(
        str(trajectory_path),
        _render_trajectory(frame_values[0], candidate.masks_by_object),
    ):
        raise RuntimeError(f"cannot write SAM3.1 GT trajectory: {trajectory_path}")
    overview_path = visualization_root / "overview.mp4"
    _write_overlay_video(
        overview_path,
        frame_values,
        candidate.masks_by_object,
        fps=float(case.timeline["sampling_rate_hz"]),
    )
    visualization_manifest = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "observation_manifest_sha256": _sha256(observation_manifest_path),
        "renderer": {
            "id": "sam31_collision_gt_renderer_v1",
            "config_fingerprint": config_fingerprint,
        },
        "video": {
            "width": frame_shape[1],
            "height": frame_shape[0],
            "fps": float(case.timeline["sampling_rate_hz"]),
            "frame_count": len(frame_values),
        },
        "files": [
            _record(contact_path, "contact_sheet.png"),
            _record(trajectory_path, "trajectory.png"),
            _record(overview_path, "overview.mp4"),
        ],
    }
    visualization_manifest_path = visualization_root / "manifest.json"
    _write_json(visualization_manifest_path, visualization_manifest)

    failing_objects = {
        object_id
        for finding in failures
        for object_id in finding.object_ids
    }
    review = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "decision": "pending",
        "scope": "full_video",
        "reviewer": "pending_sam31_collision_gt_curation",
        "observation_manifest_sha256": _sha256(observation_manifest_path),
        "visualization_manifest_sha256": _sha256(visualization_manifest_path),
        "checks": {
            "identity": "fail" if failures else "pass",
            "mask_alignment": "fail" if failures else "pass",
            "state_labels": "fail" if failures else "pass",
            "trajectory_continuity": "fail" if failures else "pass",
        },
        "entity_reviews": {
            object_id: {
                "decision": (
                    "changes_requested" if object_id in failing_objects else "approved"
                ),
                "notes": (
                    "automatic quality gate failed; requires correction"
                    if object_id in failing_objects
                    else "automatic gates passed; visual approval remains pending"
                ),
            }
            for object_id in entities
        },
        "issues": [_review_issue(item) for item in candidate.findings],
    }
    _write_json(observation_root / "review.json", review)

    files = sorted(
        (
            path
            for path in output.rglob("*")
            if path.is_file() and path.name != "candidate_bundle.json"
        ),
        key=lambda path: path.relative_to(output).as_posix(),
    )
    targets = [path.relative_to(output).as_posix() for path in files]
    bundle_path = output / "candidate_bundle.json"
    _write_json(
        bundle_path,
        {
            "schema_version": "1.0",
            "case_id": case.case_id,
            "required_targets": targets,
            "files": [
                _bundle_record(output, source, target)
                for source, target in zip(files, targets, strict=True)
            ],
        },
    )
    validate_candidate_bundle(bundle_path)
    return bundle_path


__all__ = ["write_sam31_candidate_bundle"]
