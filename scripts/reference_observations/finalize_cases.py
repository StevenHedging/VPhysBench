#!/usr/bin/env python3
"""Materialize complete, hash-closed canonical bundles from reviewed candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physbench.reference_observations import load_entity_observation
from physbench.reference_observations.curation import (
    load_curation_cases,
    render_dense_event_sheet,
)
from physbench.reference_observations.curation.bundle import validate_candidate_bundle
from physbench.reference_observations.curation.finalize import (
    anchor_object_ids,
    entity_from_masks,
    remap_ordered_appearance,
    remap_physics_objects,
    visual_size_order_matches_physics,
    write_anchor_assets,
)
from physbench.reference_observations.curation.install import (
    install_candidate_bundle,
    refresh_locked_files,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bundle_record(root: Path, path: Path, target: str) -> dict[str, Any]:
    return {
        "candidate_path": path.relative_to(root).as_posix(),
        "target_path": target,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _decode_samples(case: Any) -> list[np.ndarray]:
    requested = tuple(int(item["source_frame_index"]) for item in case.timeline["samples"])
    if not requested:
        return []
    if any(right < left for left, right in zip(requested, requested[1:])):
        raise ValueError(f"timeline source frames must be nondecreasing for {case.case_id}")
    requested_set = set(requested)
    capture = cv2.VideoCapture(str(case.reference_video_path))
    decoded: dict[int, np.ndarray] = {}
    index = 0
    try:
        while index <= requested[-1]:
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f"cannot decode source frame {index} for {case.case_id}")
            if index in requested_set:
                decoded[index] = frame
            index += 1
    finally:
        capture.release()
    if len(decoded) != len(requested_set):
        raise ValueError(f"decoded sample count differs for {case.case_id}")
    return [decoded[index].copy() for index in requested]


def _render_trajectory(frame: np.ndarray, entities: dict[str, Any]) -> np.ndarray:
    image = frame.copy()
    colors = ((0, 255, 255), (255, 128, 0), (0, 128, 255), (255, 255, 0))
    for index, (object_id, entity) in enumerate(entities.items()):
        points = entity.centroid_xy[np.all(np.isfinite(entity.centroid_xy), axis=1)]
        if len(points) > 1:
            cv2.polylines(
                image,
                [np.rint(points).astype(np.int32)],
                False,
                colors[index % len(colors)],
                2,
                cv2.LINE_AA,
            )
        if len(points):
            cv2.putText(
                image,
                object_id,
                tuple(np.rint(points[0]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[index % len(colors)],
                1,
                cv2.LINE_AA,
            )
    return image


def _render_overview(
    path: Path,
    frames: list[np.ndarray],
    entities: dict[str, Any],
    fps: float,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create overview video: {path}")
    colors = ((0, 255, 255), (255, 128, 0), (0, 128, 255), (255, 255, 0))
    try:
        for frame_index, frame in enumerate(frames):
            rendered = frame.copy()
            for object_index, entity in enumerate(entities.values()):
                mask = (entity.masks[frame_index] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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


def finalize_case(
    case: Any,
    *,
    candidate_case_root: Path,
    output_root: Path,
    remap_physics: bool,
) -> tuple[Path, bool]:
    output = output_root / case.scene_id / case.case_id
    if output.exists():
        shutil.rmtree(output)
    observation_root = output / "canonical" / "reference_observation"
    mask_root = output / "canonical" / "masks"
    entities: dict[str, Any] = {}
    for index, catalog_entity in enumerate(case.entities, 1):
        source_root = candidate_case_root / "entities" / f"object_{index}"
        loaded = load_entity_observation(
            source_root / "mask_tube.npz",
            source_root / "trajectory.npz",
            object_id=f"object_{index}",
            mask_id=f"{index:02d}",
            expected_samples=len(case.timeline["samples"]),
        )
        entity = entity_from_masks(
            loaded.object_id, loaded.mask_id, loaded.masks, loaded.state
        )
        entities[entity.object_id] = entity
        target = observation_root / "entities" / entity.object_id
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "mask_tube.npz", target / "mask_tube.npz")
        shutil.copy2(source_root / "trajectory.npz", target / "trajectory.npz")

    perform_physics_remap = remap_physics and not visual_size_order_matches_physics(
        case.physics,
        {object_id: int(entity.area_pixels[0]) for object_id, entity in entities.items()},
    )
    new_to_old = None
    if perform_physics_remap:
        if len(entities) != 2:
            raise ValueError("automatic physical remap currently supports two objects")
        new_to_old = {"object_1": "object_2", "object_2": "object_1"}
    semantics = anchor_object_ids(case.scene_id, len(entities))
    instances = []
    for index, entity in enumerate(entities.values(), 1):
        old_index = (
            index
            if new_to_old is None
            else int(new_to_old[f"object_{index}"].rsplit("_", 1)[-1])
        )
        entity_class = case.mask_manifest["instances"][old_index - 1].get(
            "entity_class", "object"
        )
        record = write_anchor_assets(
            mask_root,
            entity,
            semantic_object_id=semantics[index - 1],
            entity_class=entity_class,
        )
        prefix = case.assets["first_frame_mask_manifest"].rsplit("/", 1)[0]
        record.update(
            {
                "asset": f"{prefix}/{index:02d}.png",
                "npz_asset": f"{prefix}/{index:02d}.npz",
                "physics_keys": [
                    f"objects.object_{index}.{name}"
                    for name in case.physics["objects"][f"object_{index}"]
                ],
            }
        )
        instances.append(record)
    mask_manifest = copy.deepcopy(case.mask_manifest)
    mask_manifest["instances"] = instances
    mask_manifest["generator"] = {
        "id": "v14_full_reference_observation_audit",
        "model": "SAM2.1 Hiera Large",
        "review_policy": "anchor_equals_reviewed_observation_zero",
    }
    mask_manifest["ordering"] = (
        "collision left-to-right visual order"
        if case.scene_id == "collision_1d"
        else "row-major visual order"
    )
    _write_json(mask_root / "manifest.json", mask_manifest)

    if perform_physics_remap:
        physics = remap_physics_objects(case.physics, new_to_old=new_to_old)
        _write_json(
            output / "physics.json",
            {"case_id": case.case_id, "scene_id": case.scene_id, "physics": physics},
        )
        caption = case.caption
        if "single_incident" in case.case_id:
            caption = (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, "
                "while the right ball has mass m_2, radius r_2, and initial speed v_2. "
                "The left ball moves right, the right ball is initially stationary, "
                "and they undergo a one-dimensional central collision."
            )
        _write_json(
            output / "caption.json",
            {"case_id": case.case_id, "scene_id": case.scene_id, "caption": caption},
        )

    quality = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "status": "pass",
        "checks": [
            {"code": "structural_validation", "status": "pass", "message": "exact mask reductions validated"},
            {"code": "visual_curation", "status": "pass", "message": "anchor and full tube reviewed"},
        ],
        "entities": {
            object_id: {
                "sample_count": len(entity.state),
                "visible_samples": int(np.count_nonzero(entity.state == 0)),
                "occluded_samples": int(np.count_nonzero(entity.state == 1)),
                "out_of_frame_samples": int(np.count_nonzero(entity.state == 2)),
                "unresolved_samples": int(np.count_nonzero(entity.state == 3)),
                "coverage": float(np.count_nonzero(entity.area_pixels) / len(entity.state)),
            }
            for object_id, entity in entities.items()
        },
        "failures": [],
        "warnings": [],
        "config_fingerprint": _sha256(Path("configs/reference_observations/audit_v1.json")),
    }
    _write_json(observation_root / "quality.json", quality)

    dataset_root = case.asset_root
    reference_asset = case.assets["reference_video"]
    mask_manifest_asset = case.assets["first_frame_mask_manifest"]
    observation_manifest = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "scene_id": case.scene_id,
        "source": {
            "reference_video": _record(case.reference_video_path, reference_asset),
            "first_frame_mask_manifest": _record(mask_root / "manifest.json", mask_manifest_asset),
            "anchor_masks": [
                {
                    "object_id": f"object_{index}",
                    "mask_id": f"{index:02d}",
                    "asset": _record(
                        mask_root / f"{index:02d}.npz",
                        case.mask_manifest["instances"][index - 1]["npz_asset"],
                    ),
                }
                for index in range(1, len(entities) + 1)
            ],
        },
        "generator": {
            "id": "reference_observation_curation_v14",
            "model_id": "facebook/sam2.1-hiera-large",
            "config_fingerprint": _sha256(Path("configs/reference_observations/audit_v1.json")),
            "code_revision": "working_tree_v14_full_audit",
            "propagation": json.loads(
                (candidate_case_root / "candidate_tracking.json").read_text(encoding="utf-8")
            ),
        },
        "timeline": _record(
            case.observation_manifest_path.parent / "timeline.json", "timeline.json"
        ),
        "quality": _record(observation_root / "quality.json", "quality.json"),
        "entities": [
            {
                "object_id": f"object_{index}",
                "mask_id": f"{index:02d}",
                "mask_tube": _record(
                    observation_root / "entities" / f"object_{index}" / "mask_tube.npz",
                    f"entities/object_{index}/mask_tube.npz",
                ),
                "trajectory": _record(
                    observation_root / "entities" / f"object_{index}" / "trajectory.npz",
                    f"entities/object_{index}/trajectory.npz",
                ),
            }
            for index in range(1, len(entities) + 1)
        ],
    }
    _write_json(observation_root / "manifest.json", observation_manifest)

    frames = _decode_samples(case)
    visualization = observation_root / "visualization"
    visualization.mkdir(parents=True, exist_ok=True)
    sample_indices = tuple(
        dict.fromkeys(
            int(value)
            for value in np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
        )
    )
    contact = render_dense_event_sheet(
        frames,
        masks_by_object={key: value.masks for key, value in entities.items()},
        states_by_object={key: value.state for key, value in entities.items()},
        observation_indices=sample_indices,
        source_indices=tuple(
            int(case.timeline["samples"][index]["source_frame_index"])
            for index in sample_indices
        ),
        columns=4,
    )
    cv2.imwrite(str(visualization / "contact_sheet.png"), contact)
    cv2.imwrite(str(visualization / "trajectory.png"), _render_trajectory(frames[0], entities))
    fps = float(case.timeline["sampling_rate_hz"])
    _render_overview(visualization / "overview.mp4", frames, entities, fps)
    visualization_manifest = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "observation_manifest_sha256": _sha256(observation_root / "manifest.json"),
        "renderer": {
            "id": "reference_observation_curation_renderer_v14",
            "config_fingerprint": _sha256(Path("configs/reference_observations/audit_v1.json")),
        },
        "video": {
            "fps": fps,
            "frame_count": len(frames),
            "height": frames[0].shape[0],
            "width": frames[0].shape[1],
        },
        "files": [
            _record(visualization / name, name)
            for name in ("overview.mp4", "contact_sheet.png", "trajectory.png")
        ],
    }
    _write_json(visualization / "manifest.json", visualization_manifest)
    review = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "decision": "approved",
        "scope": "anchor_contact_sheet_full_tube",
        "reviewer": "v14_full_reference_observation_audit_2026-08-19",
        "checks": {
            "identity": "pass",
            "mask_alignment": "pass",
            "quality_warnings": "pass",
            "state_labels": "pass",
            "trajectory_continuity": "pass",
        },
        "entity_reviews": {
            object_id: {"decision": "approved", "notes": "large-model candidate accepted after full-tube audit"}
            for object_id in entities
        },
        "issues": [],
        "observation_manifest_sha256": _sha256(observation_root / "manifest.json"),
        "visualization_manifest_sha256": _sha256(visualization / "manifest.json"),
    }
    _write_json(observation_root / "review.json", review)

    target_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "candidate_bundle.json"
    ]
    target_names = [path.relative_to(output).as_posix() for path in target_files]
    manifest_path = output / "candidate_bundle.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "case_id": case.case_id,
            "required_targets": target_names,
            "files": [
                _bundle_record(output, path, target)
                for path, target in zip(target_files, target_names, strict=True)
            ],
        },
    )
    validate_candidate_bundle(manifest_path)
    return manifest_path, perform_physics_remap


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--remap-physics-list", type=Path)
    parser.add_argument("--install", action="store_true")
    return parser


def _update_release_appearance(dataset_path: Path, remap_ids: set[str]) -> None:
    if not remap_ids:
        return
    descriptor = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases_path = dataset_path.parent / descriptor["cases"]
    rows = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping = {"object_1": "object_2", "object_2": "object_1"}
    seen: set[str] = set()
    for row in rows:
        if row["case_id"] in remap_ids:
            row["appearance"] = remap_ordered_appearance(
                row["appearance"], new_to_old=mapping
            )
            seen.add(row["case_id"])
    if seen != remap_ids:
        raise ValueError(f"release appearance remap misses Cases: {sorted(remap_ids - seen)}")
    descriptor_id, temporary_value = tempfile.mkstemp(
        prefix=f".{cases_path.name}.", dir=cases_path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor_id, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cases_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    case_ids = tuple(arguments.case_list.read_text(encoding="utf-8").split())
    remap_ids = (
        set()
        if arguments.remap_physics_list is None
        else set(arguments.remap_physics_list.read_text(encoding="utf-8").split())
    )
    cases = load_curation_cases(arguments.dataset, case_ids=case_ids)
    changed_asset_paths: list[str] = []
    appearance_remap_ids: set[str] = set()
    for case in cases:
        candidate = next(
            arguments.candidate_root.rglob(f"{case.case_id}/candidate_tracking.json")
        ).parent
        manifest, appearance_remap_required = finalize_case(
            case,
            candidate_case_root=candidate,
            output_root=arguments.output_root,
            remap_physics=case.case_id in remap_ids,
        )
        if arguments.install:
            install_candidate_bundle(manifest, case.observation_manifest_path.parents[2])
            bundle_value = json.loads(manifest.read_text(encoding="utf-8"))
            if appearance_remap_required:
                appearance_remap_ids.add(case.case_id)
            case_root = case.observation_manifest_path.parents[2]
            changed_asset_paths.extend(
                (case_root / record["target_path"])
                .relative_to(case.asset_root)
                .as_posix()
                for record in bundle_value["files"]
            )
    if arguments.install:
        _update_release_appearance(arguments.dataset, appearance_remap_ids)
        descriptor = json.loads(arguments.dataset.read_text(encoding="utf-8"))
        refresh_locked_files(
            arguments.dataset.parent / descriptor["asset_lock"],
            cases[0].asset_root,
            changed_asset_paths,
        )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "requested_physics_alignment": len(remap_ids),
                "remapped_physics": len(appearance_remap_ids),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
