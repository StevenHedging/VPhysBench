#!/usr/bin/env python3
"""Run a read-only, one-record-per-Case V14 observation audit."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from physbench.reference_observations import load_reference_observation
from physbench.reference_observations.curation import (
    audit_anchor_tube_zero,
    audit_case_bindings,
    audit_mask_trajectory_reductions,
    audit_visual_order,
    fast_entity_diagnostics,
    load_curation_cases,
    pixel_entity_diagnostics,
    render_anchor_sheet,
)
from physbench.reference_observations.curation.anchors import (
    build_independent_anchor_candidates,
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_diagnostic_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: int(item["release_index"]))
    descriptor, temporary_value = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in ordered:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_anchor(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        masks = np.asarray(payload["masks"])
    if masks.ndim == 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim != 2:
        raise ValueError(f"anchor mask must use HW layout: {path}")
    return (masks > 0).astype(np.uint8)


def _spaced_video_frames(path: Path, count: int = 5) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, frame_count - 1), count, dtype=int)
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f"cannot decode frame {index}: {path}")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def audit_case(case: Any, release_index: int, render_root: Path | None) -> dict[str, Any]:
    observation = load_reference_observation(
        case.asset_root,
        case.observation_manifest_path,
        bundle_root=case.asset_root,
    )
    binding = [*audit_case_bindings(case), *audit_visual_order(case)]
    physical_time = np.asarray(
        [sample.physical_time_seconds for sample in observation.timeline.samples],
        np.float64,
    )
    entity_records: dict[str, Any] = {}
    anchors: dict[str, np.ndarray] = {}
    for catalog_entity in case.entities:
        object_id = catalog_entity.identity.object_id
        entity = observation.entities[object_id]
        anchor = _load_anchor(catalog_entity.anchor_npz_path)
        anchors[object_id] = anchor
        fast = fast_entity_diagnostics(
            area=entity.area_pixels,
            centroid_xy=entity.centroid_xy,
            state=entity.state,
            physical_time=physical_time,
            scene_id=case.scene_id,
        )
        pixel = pixel_entity_diagnostics(
            masks=entity.masks,
            state=entity.state,
            scene_id=case.scene_id,
        )
        exact = audit_mask_trajectory_reductions(
            masks=entity.masks,
            area_pixels=entity.area_pixels,
            centroid_xy=entity.centroid_xy,
            bbox_xyxy=entity.bbox_xyxy,
        )
        anchor_issue = audit_anchor_tube_zero(anchor, entity.masks[0])
        entity_records[object_id] = {
            "fast": fast,
            "pixel": pixel,
            "exact_reduction_issues": exact,
            "anchor_tube_zero_issue": anchor_issue,
        }

    independent: dict[str, np.ndarray] = {}
    independent_status: dict[str, Any] = {"status": "not_rendered"}
    evidence: dict[str, str] = {}
    if render_root is not None:
        case_root = render_root / case.scene_id / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        frames = _spaced_video_frames(case.reference_video_path)
        try:
            candidates = build_independent_anchor_candidates(
                frames,
                scene_id=case.scene_id,
                expected_count=len(case.entities),
                comparison_masks=anchors,
            )
            independent = {candidate.object_id: candidate.mask for candidate in candidates}
            independent_status = {
                "status": "candidate",
                "comparison_iou": {
                    candidate.object_id: candidate.comparison_iou for candidate in candidates
                },
            }
        except ValueError as exc:
            independent = dict(anchors)
            independent_status = {"status": "localization_failed", "message": str(exc)}
        anchor_sheet = render_anchor_sheet(
            frames[0],
            existing_masks=anchors,
            candidate_masks=independent,
            metadata={"case_id": case.case_id, "scene_id": case.scene_id},
        )
        anchor_path = case_root / "anchor.png"
        if not cv2.imwrite(str(anchor_path), anchor_sheet):
            raise RuntimeError(f"cannot write anchor evidence: {anchor_path}")
        evidence["anchor_sheet"] = anchor_path.as_posix()
    visualization_root = case.visualization_manifest_path.parent
    for record in case.visualization_manifest.get("files", []):
        if isinstance(record, dict) and record.get("path") in {
            "contact_sheet.png", "overview.mp4", "trajectory.png"
        }:
            evidence[Path(record["path"]).stem] = (
                visualization_root / record["path"]
            ).as_posix()
    return {
        "release_index": release_index,
        "case_id": case.case_id,
        "scene_id": case.scene_id,
        "binding_issues": binding,
        "entities": entity_records,
        "independent_anchor": independent_status,
        "evidence": evidence,
        "existing_review_decision_ignored": case.review.get("decision"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--no-install", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    cases = load_curation_cases(
        arguments.dataset,
        case_ids=arguments.case_id or None,
    )
    records = [
        audit_case(case, index, arguments.render_root)
        for index, case in enumerate(cases)
    ]
    write_diagnostic_records(arguments.output, records)
    print(json.dumps({"cases": len(records), "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
