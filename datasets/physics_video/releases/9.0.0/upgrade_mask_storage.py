#!/usr/bin/env python3
"""Upgrade the 9.0.0 first-frame masks to model and visualization formats."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any

import cv2
import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parent
PHYSICS_VIDEO_ROOT = RELEASE_ROOT.parents[1]
sys.path.insert(0, str(RELEASE_ROOT))

from mask_storage import (  # noqa: E402
    load_mask_npz,
    read_binary_png,
    upgrade_manifest_storage,
    write_mask_bundle,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _asset_path(physics_video_root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe Dataset asset path: {value}")
    root = physics_video_root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Dataset asset escapes its root: {value}") from exc
    return path


def _without_npz_fields(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = json.loads(json.dumps(instances))
    for instance in values:
        instance.pop("npz_index", None)
        instance.pop("npz_asset", None)
    return values


def _validate_geometry(mask: np.ndarray, instance: dict[str, Any]) -> None:
    ys, xs = np.where(mask > 0)
    area = int(len(xs))
    if instance.get("area_pixels") != area:
        raise ValueError(f"area metadata mismatch: {instance.get('asset')}")
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    if instance.get("bbox_xyxy") != bbox:
        raise ValueError(f"bbox metadata mismatch: {instance.get('asset')}")
    centroid = [float(xs.mean()), float(ys.mean())]
    if not np.allclose(instance.get("centroid_xy"), centroid, atol=1e-9):
        raise ValueError(f"centroid metadata mismatch: {instance.get('asset')}")


def _preflight_complete_case(
    case: dict[str, Any],
    record: dict[str, Any],
    physics_video_root: Path,
) -> dict[str, Any]:
    instances = record.get("instances")
    expected = record.get("expected_subject_count")
    if not isinstance(instances, list) or not isinstance(expected, int):
        raise ValueError(f"invalid mask record: {case['case_id']}")
    if len(instances) != expected or expected < 1:
        raise ValueError(f"mask cardinality mismatch: {case['case_id']}")
    manifest_asset = record.get("manifest_asset")
    if not isinstance(manifest_asset, str):
        raise ValueError(f"complete Case lacks manifest: {case['case_id']}")
    if case.get("assets", {}).get("first_frame_mask_manifest") != manifest_asset:
        raise ValueError(f"Case/record manifest mismatch: {case['case_id']}")
    manifest_path = _asset_path(physics_video_root, manifest_asset)
    manifest = read_json(manifest_path)
    if manifest.get("case_id") != case["case_id"]:
        raise ValueError(f"manifest Case mismatch: {case['case_id']}")
    if manifest.get("frame_index") != 0 or manifest.get("frame_scope") != "first_frame_only":
        raise ValueError(f"manifest is not first-frame-only: {case['case_id']}")
    if _without_npz_fields(manifest.get("instances", [])) != _without_npz_fields(instances):
        raise ValueError(f"manifest/index instances mismatch: {case['case_id']}")
    first_frame_asset = case.get("assets", {}).get("first_frame")
    if not isinstance(first_frame_asset, str):
        raise ValueError(f"Case lacks first_frame: {case['case_id']}")
    first_frame = cv2.imread(
        str(_asset_path(physics_video_root, first_frame_asset)),
        cv2.IMREAD_UNCHANGED,
    )
    if first_frame is None:
        raise ValueError(f"cannot decode first frame: {case['case_id']}")
    expected_shape = first_frame.shape[:2]
    if manifest.get("image_shape_hw") != list(expected_shape):
        raise ValueError(f"manifest image shape mismatch: {case['case_id']}")
    masks: list[np.ndarray] = []
    for index, instance in enumerate(instances, start=1):
        mask_id = f"{index:02d}"
        if instance.get("mask_id") != mask_id:
            raise ValueError(f"non-contiguous mask IDs: {case['case_id']}")
        role = f"first_frame_subject_mask_{mask_id}"
        if case["assets"].get(role) != instance.get("asset"):
            raise ValueError(f"Case/instance PNG mismatch: {case['case_id']}")
        mask = read_binary_png(
            _asset_path(physics_video_root, instance["asset"]),
            expected_shape,
        )
        _validate_geometry(mask, instance)
        masks.append(mask)
    stack = np.stack(masks)
    if np.any(np.sum(stack, axis=0) > 1):
        raise ValueError(f"overlapping instance masks: {case['case_id']}")
    mask_asset_parent = PurePosixPath(manifest_asset).parent
    summary_npz_asset = (mask_asset_parent / "masks.npz").as_posix()
    exposed_npz = case["assets"].get("first_frame_masks_npz")
    if exposed_npz not in (None, summary_npz_asset):
        raise ValueError(f"Case exposes the wrong NPZ asset: {case['case_id']}")
    summary_npz_path = _asset_path(physics_video_root, summary_npz_asset)
    object_npz_assets = [
        (mask_asset_parent / f"{item['mask_id']}.npz").as_posix()
        for item in instances
    ]
    object_npz_paths = [
        _asset_path(physics_video_root, asset) for asset in object_npz_assets
    ]
    expected_mask_ids = [item["mask_id"] for item in instances]
    expected_object_ids = [item["object_id"] for item in instances]
    if summary_npz_path.exists():
        archive = load_mask_npz(summary_npz_path)
        np.testing.assert_array_equal(archive["masks"], stack)
        if archive["mask_ids"].tolist() != expected_mask_ids:
            raise ValueError(f"summary NPZ mask IDs mismatch: {case['case_id']}")
        if archive["object_ids"].tolist() != expected_object_ids:
            raise ValueError(f"summary NPZ object IDs mismatch: {case['case_id']}")
        source_kind = "summary"
    else:
        if not all(path.is_file() for path in object_npz_paths):
            raise ValueError(f"Case has neither complete summary nor object NPZ: {case['case_id']}")
        for index, path in enumerate(object_npz_paths):
            archive = load_mask_npz(path)
            if archive["masks"].shape != (1, *expected_shape):
                raise ValueError(f"object NPZ shape mismatch: {object_npz_assets[index]}")
            np.testing.assert_array_equal(archive["masks"][0], stack[index])
            if archive["mask_ids"].tolist() != [expected_mask_ids[index]]:
                raise ValueError(f"object NPZ mask ID mismatch: {object_npz_assets[index]}")
            if archive["object_ids"].tolist() != [expected_object_ids[index]]:
                raise ValueError(f"object NPZ object ID mismatch: {object_npz_assets[index]}")
        source_kind = "objects"
    for index, asset in enumerate(object_npz_assets, start=1):
        role = f"first_frame_subject_mask_npz_{index:02d}"
        exposed = case["assets"].get(role)
        if exposed not in (None, asset):
            raise ValueError(f"Case exposes the wrong object NPZ: {case['case_id']}")
    return {
        "case_id": case["case_id"],
        "mask_directory": manifest_path.parent,
        "manifest_path": manifest_path,
        "manifest_asset": manifest_asset,
        "summary_npz_path": summary_npz_path,
        "object_npz_assets": object_npz_assets,
        "source_kind": source_kind,
        "instances": instances,
        "mask_assets": [item["asset"] for item in instances],
        "expected_shape": expected_shape,
    }


def preflight_release(
    release_root: Path = RELEASE_ROOT,
    physics_video_root: Path = PHYSICS_VIDEO_ROOT,
) -> list[dict[str, Any]]:
    """Validate all records before any existing Mask asset is replaced."""

    cases = read_jsonl(release_root / "cases.jsonl")
    records = read_jsonl(release_root / "masks.jsonl")
    if len(cases) != len(records):
        raise ValueError("Case and mask record counts differ")
    by_case = {case["case_id"]: case for case in cases}
    if len(by_case) != len(cases):
        raise ValueError("duplicate Case IDs")
    if {record["case_id"] for record in records} != set(by_case):
        raise ValueError("mask record IDs differ from Case IDs")
    operations: list[dict[str, Any]] = []
    for record in records:
        case = by_case[record["case_id"]]
        status = record.get("status")
        if status == "skipped":
            if record.get("instances") or record.get("manifest_asset") is not None:
                raise ValueError(f"skipped Case exposes masks: {case['case_id']}")
            assets = case.get("assets", {})
            if assets.get("first_frame_mask_manifest") is not None:
                raise ValueError(f"skipped Case exposes manifest: {case['case_id']}")
            if assets.get("first_frame_masks_npz") is not None:
                raise ValueError(f"skipped Case exposes NPZ: {case['case_id']}")
            continue
        if status != "complete":
            raise ValueError(f"invalid mask status: {case['case_id']}")
        operations.append(_preflight_complete_case(case, record, physics_video_root))
    return operations


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".json",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def upgrade_release(
    *,
    release_root: Path = RELEASE_ROOT,
    physics_video_root: Path = PHYSICS_VIDEO_ROOT,
    materialize: bool,
    report_path: Path | None,
) -> dict[str, Any]:
    """Preflight the release and optionally publish every upgraded bundle."""

    records = read_jsonl(release_root / "masks.jsonl")
    operations = preflight_release(release_root, physics_video_root)
    removed_summary_npz_files = 0
    if materialize:
        for operation in operations:
            masks = [
                read_binary_png(
                    _asset_path(physics_video_root, asset),
                    operation["expected_shape"],
                )
                for asset in operation["mask_assets"]
            ]
            storage = write_mask_bundle(
                operation["mask_directory"],
                masks,
                operation["instances"],
            )
            manifest = upgrade_manifest_storage(read_json(operation["manifest_path"]))
            if manifest["storage"] != storage:
                raise ValueError(f"storage metadata mismatch: {operation['case_id']}")
            for index, instance in enumerate(manifest["instances"]):
                archive = load_mask_npz(
                    _asset_path(physics_video_root, instance["npz_asset"])
                )
                if archive["masks"].shape != (1, *operation["expected_shape"]):
                    raise ValueError(f"written object NPZ shape mismatch: {instance['npz_asset']}")
                np.testing.assert_array_equal(archive["masks"][0], masks[index])
                if archive["mask_ids"].tolist() != [instance["mask_id"]]:
                    raise ValueError(f"written object NPZ mask ID mismatch: {instance['npz_asset']}")
                if archive["object_ids"].tolist() != [instance["object_id"]]:
                    raise ValueError(f"written object NPZ object ID mismatch: {instance['npz_asset']}")
            _write_json_atomic(operation["manifest_path"], manifest)
            if operation["summary_npz_path"].exists():
                operation["summary_npz_path"].unlink()
                removed_summary_npz_files += 1
    skipped_ids = [
        record["case_id"] for record in records if record.get("status") == "skipped"
    ]
    report = {
        "schema_version": "1.2",
        "release": "9.0.0",
        "materialized": materialize,
        "selected_cases": len(records),
        "complete_cases": len(operations),
        "skipped_cases": len(skipped_ids),
        "mask_files": sum(len(item["instances"]) for item in operations),
        "npz_files": sum(len(item["instances"]) for item in operations),
        "removed_summary_npz_files": removed_summary_npz_files,
        "skipped_case_ids": skipped_ids,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(report_path, report)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    report = upgrade_release(
        materialize=bool(arguments.materialize),
        report_path=arguments.report,
    )
    print(
        f"selected={report['selected_cases']} complete={report['complete_cases']} "
        f"skipped={report['skipped_cases']} masks={report['mask_files']} "
        f"npz={report['npz_files']} materialized={report['materialized']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
