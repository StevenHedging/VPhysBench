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

from physbench.datasets import load_dataset
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


def merge_diagnostic_records(
    output_path: str | Path,
    input_paths: Iterable[str | Path],
) -> int:
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_cases: set[str] = set()
    for input_path in input_paths:
        for line in Path(input_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            release_index = int(record["release_index"])
            case_id = str(record["case_id"])
            if release_index in seen_indices:
                raise ValueError(f"duplicate release index {release_index}")
            if case_id in seen_cases:
                raise ValueError(f"duplicate Case {case_id}")
            seen_indices.add(release_index)
            seen_cases.add(case_id)
            records.append(record)
    if seen_indices != set(range(len(records))):
        raise ValueError("merged diagnostics do not have contiguous release indices")
    write_diagnostic_records(output_path, records)
    return len(records)


def shard_release_indices(
    case_count: int,
    shard_index: int,
    shard_count: int,
) -> tuple[int, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("audit shard index/count are invalid")
    return tuple(range(shard_index, case_count, shard_count))


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
    indices = tuple(
        int(value) for value in np.linspace(0, max(0, frame_count - 1), count, dtype=int)
    )
    requested = set(indices)
    frames: list[np.ndarray] = []
    try:
        index = 0
        while index <= indices[-1]:
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f"cannot decode frame {index}: {path}")
            if index in requested:
                frames.append(frame)
            index += 1
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
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--merge-input", action="append", type=Path, default=[])
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-install", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.merge_input:
        count = merge_diagnostic_records(arguments.output, arguments.merge_input)
        print(json.dumps({"cases": count, "output": str(arguments.output)}))
        return 0
    if arguments.dataset is None:
        raise ValueError("--dataset is required unless --merge-input is used")
    snapshot = load_dataset(arguments.dataset)
    indexed_ids = [(index, str(case["case_id"])) for index, case in enumerate(snapshot.cases)]
    requested = set(arguments.case_id)
    if requested:
        known = {case_id for _, case_id in indexed_ids}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown requested Case IDs: {sorted(unknown)}")
        indexed_ids = [item for item in indexed_ids if item[1] in requested]
    selected_release_indices = set(
        shard_release_indices(
            len(snapshot.cases), arguments.shard_index, arguments.shard_count
        )
    )
    indexed_ids = [item for item in indexed_ids if item[0] in selected_release_indices]
    cases = load_curation_cases(
        arguments.dataset,
        case_ids={case_id for _, case_id in indexed_ids},
    )
    release_index_by_id = {case_id: index for index, case_id in indexed_ids}
    records = [
        audit_case(case, release_index_by_id[case.case_id], arguments.render_root)
        for case in cases
    ]
    write_diagnostic_records(arguments.output, records)
    print(json.dumps({"cases": len(records), "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
