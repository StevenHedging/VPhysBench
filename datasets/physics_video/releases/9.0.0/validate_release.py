#!/usr/bin/env python3
"""Independently validate the 9.0.0 first-frame mask release."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parent
PHYSICS_VIDEO_ROOT = RELEASE_ROOT.parents[1]
REPOSITORY_ROOT = PHYSICS_VIDEO_ROOT.parents[1]
BASE_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "8.0.0"
sys.path.insert(0, str(RELEASE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mask_storage import load_mask_npz  # noqa: E402
from physbench.datasets.loader import load_dataset  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_mask_assets(case: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(case))
    value["assets"] = {
        key: item
        for key, item in value["assets"].items()
        if key != "first_frame_mask_manifest"
        and key != "first_frame_masks_npz"
        and not key.startswith("first_frame_subject_mask_")
    }
    return value


def validate() -> dict[str, Any]:
    base_cases = read_jsonl(BASE_RELEASE_ROOT / "cases.jsonl")
    cases = read_jsonl(RELEASE_ROOT / "cases.jsonl")
    records = read_jsonl(RELEASE_ROOT / "masks.jsonl")
    descriptor = read_json(RELEASE_ROOT / "dataset.json")
    if len(base_cases) != 799 or len(cases) != 799 or len(records) != 799:
        raise ValueError("base Cases, 9.0.0 Cases, and mask records must each be 799")
    if len({case["case_id"] for case in cases}) != 799:
        raise ValueError("9.0.0 contains duplicate Case IDs")
    if [case["case_id"] for case in cases] != [
        case["case_id"] for case in base_cases
    ]:
        raise ValueError("9.0.0 Case order or identity differs from 8.0.0")
    for base_case, case in zip(base_cases, cases):
        if without_mask_assets(case) != base_case:
            raise ValueError(f"non-mask Case content changed: {case['case_id']}")

    if descriptor.get("mask_annotations", {}).get("schema_version") != "1.1":
        raise ValueError("dataset mask annotation schema must be 1.1")

    if descriptor["mask_annotations"]["sha256"] != sha256(
        RELEASE_ROOT / descriptor["mask_annotations"]["path"]
    ):
        raise ValueError("masks.jsonl SHA-256 does not match dataset.json")
    by_case = {record["case_id"]: record for record in records}
    if set(by_case) != {case["case_id"] for case in cases}:
        raise ValueError("mask record IDs do not match Case IDs")

    complete_cases = 0
    skipped_cases = []
    mask_files = 0
    npz_files = 0
    scene_counts: dict[str, dict[str, int]] = {}
    for case in cases:
        record = by_case[case["case_id"]]
        scene = case["scene_id"]
        scene_counts.setdefault(scene, {"complete_cases": 0, "mask_files": 0})
        expected = int(record["expected_subject_count"])
        if record["status"] == "skipped":
            skipped_cases.append(case["case_id"])
            if (
                record["instances"]
                or record["manifest_asset"] is not None
                or record.get("npz_asset") is not None
            ):
                raise ValueError(f"skipped Case exposes masks: {case['case_id']}")
            if case["assets"]["first_frame_mask_manifest"] is not None:
                raise ValueError(f"skipped Case exposes a manifest: {case['case_id']}")
            if case["assets"].get("first_frame_masks_npz") is not None:
                raise ValueError(f"skipped Case exposes an NPZ: {case['case_id']}")
            continue
        if record["status"] != "complete":
            raise ValueError(f"invalid mask status for {case['case_id']}")
        if record.get("schema_version") != "1.1":
            raise ValueError(f"mask record schema is not 1.1: {case['case_id']}")
        instances = record["instances"]
        if len(instances) != expected:
            raise ValueError(f"mask cardinality mismatch: {case['case_id']}")
        manifest_path = PHYSICS_VIDEO_ROOT / record["manifest_asset"]
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_json(manifest_path)
        if manifest["case_id"] != case["case_id"]:
            raise ValueError(f"manifest Case mismatch: {case['case_id']}")
        if manifest["frame_index"] != 0 or manifest["frame_scope"] != "first_frame_only":
            raise ValueError(f"manifest is not first-frame-only: {case['case_id']}")
        if manifest.get("schema_version") != "1.1":
            raise ValueError(f"manifest schema is not 1.1: {case['case_id']}")
        if manifest["instances"] != instances:
            raise ValueError(f"manifest/index instance mismatch: {case['case_id']}")
        first_frame = cv2.imread(
            str(PHYSICS_VIDEO_ROOT / case["assets"]["first_frame"]),
            cv2.IMREAD_UNCHANGED,
        )
        if first_frame is None:
            raise ValueError(f"cannot decode first frame: {case['case_id']}")
        npz_asset = record.get("npz_asset")
        if not isinstance(npz_asset, str):
            raise ValueError(f"complete Case lacks NPZ asset: {case['case_id']}")
        if case["assets"].get("first_frame_masks_npz") != npz_asset:
            raise ValueError(f"Case/index NPZ asset mismatch: {case['case_id']}")
        expected_storage = {
            "model": {
                "asset": npz_asset,
                "array_key": "masks",
                "layout": "OHW",
                "dtype": "uint8",
                "values": [0, 1],
            },
            "visualization": {
                "asset_pattern": str(Path(npz_asset).parent / "{mask_id}.png"),
                "dtype": "uint8",
                "values": [0, 255],
            },
        }
        if manifest.get("storage") != expected_storage:
            raise ValueError(f"manifest storage contract mismatch: {case['case_id']}")
        archive = load_mask_npz(PHYSICS_VIDEO_ROOT / npz_asset)
        if archive["masks"].shape != (expected, *first_frame.shape[:2]):
            raise ValueError(f"NPZ mask shape mismatch: {case['case_id']}")
        expected_mask_ids = [f"{index:02d}" for index in range(1, expected + 1)]
        if archive["mask_ids"].tolist() != expected_mask_ids:
            raise ValueError(f"NPZ mask IDs mismatch: {case['case_id']}")
        if archive["object_ids"].tolist() != [
            instance["object_id"] for instance in instances
        ]:
            raise ValueError(f"NPZ object IDs mismatch: {case['case_id']}")
        masks = []
        centroids = []
        for index, instance in enumerate(instances, start=1):
            expected_id = f"{index:02d}"
            if instance["mask_id"] != expected_id:
                raise ValueError(f"non-contiguous mask IDs: {case['case_id']}")
            if instance.get("npz_index") != index - 1:
                raise ValueError(f"non-contiguous NPZ indices: {case['case_id']}")
            asset_role = f"first_frame_subject_mask_{expected_id}"
            if case["assets"].get(asset_role) != instance["asset"]:
                raise ValueError(f"Case/index asset mismatch: {case['case_id']}")
            mask = cv2.imread(
                str(PHYSICS_VIDEO_ROOT / instance["asset"]),
                cv2.IMREAD_UNCHANGED,
            )
            if mask is None:
                raise ValueError(f"cannot decode mask: {instance['asset']}")
            if mask.ndim != 2 or mask.shape != first_frame.shape[:2]:
                raise ValueError(f"mask shape mismatch: {instance['asset']}")
            if mask.dtype != np.uint8:
                raise ValueError(f"mask dtype is not uint8: {instance['asset']}")
            values = set(int(value) for value in np.unique(mask))
            if not values <= {0, 255} or 255 not in values:
                raise ValueError(
                    f"PNG values are not non-empty {{0,255}}: {instance['asset']}"
                )
            binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
            if not np.array_equal(binary, archive["masks"][index - 1]):
                raise ValueError(f"PNG/NPZ pixels differ: {instance['asset']}")
            area = int(np.count_nonzero(mask))
            if area != instance["area_pixels"]:
                raise ValueError(f"mask area metadata mismatch: {instance['asset']}")
            ys, xs = np.where(mask > 0)
            centroid = [float(xs.mean()), float(ys.mean())]
            if not np.allclose(centroid, instance["centroid_xy"], atol=1e-9):
                raise ValueError(f"mask centroid metadata mismatch: {instance['asset']}")
            masks.append(binary)
            centroids.append(centroid)
        for first_index in range(len(masks)):
            for second_index in range(first_index + 1, len(masks)):
                if np.array_equal(masks[first_index], masks[second_index]):
                    raise ValueError(f"duplicate instance masks: {case['case_id']}")
                if np.any((masks[first_index] > 0) & (masks[second_index] > 0)):
                    raise ValueError(f"overlapping instance masks: {case['case_id']}")
        if len(centroids) > 1:
            if scene == "collision_1d":
                if centroids != sorted(centroids, key=lambda xy: xy[0]):
                    raise ValueError(f"collision masks are not left-to-right: {case['case_id']}")
                if [item["object_id"] for item in instances] != [
                    f"ball_{index}" for index in range(1, expected + 1)
                ]:
                    raise ValueError(f"collision physics mapping mismatch: {case['case_id']}")
            elif centroids != sorted(centroids, key=lambda xy: (xy[1], xy[0])):
                raise ValueError(f"masks are not in row-major order: {case['case_id']}")
        complete_cases += 1
        mask_files += len(masks)
        npz_files += 1
        scene_counts[scene]["complete_cases"] += 1
        scene_counts[scene]["mask_files"] += len(masks)

    if (
        complete_cases != 797
        or len(skipped_cases) != 2
        or mask_files != 1205
        or npz_files != 797
    ):
        raise ValueError("release totals differ from the frozen contract")

    base_lock = read_json(BASE_RELEASE_ROOT / "assets.lock.json")
    lock = read_json(RELEASE_ROOT / "assets.lock.json")
    base_by_path = {item["path"]: item for item in base_lock["files"]}
    current_by_path = {item["path"]: item for item in lock["files"]}
    for path, item in base_by_path.items():
        if current_by_path.get(path) != item:
            raise ValueError(f"8.0.0 locked asset changed in 9.0.0: {path}")
    if len(lock["files"]) != 4831:
        raise ValueError("9.0.0 asset lock must contain 4,831 files")
    snapshot = load_dataset(
        RELEASE_ROOT / "dataset.json",
        check_assets=True,
        check_asset_hashes=True,
    )
    release = read_json(RELEASE_ROOT / "release.json")
    if snapshot.digest != release["dataset_digest"]:
        raise ValueError("loader/release digest mismatch")
    return {
        "schema_version": "1.0",
        "dataset_id": descriptor["dataset_id"],
        "dataset_digest": snapshot.digest,
        "case_count": len(cases),
        "complete_case_count": complete_cases,
        "skipped_case_count": len(skipped_cases),
        "mask_file_count": mask_files,
        "npz_file_count": npz_files,
        "asset_file_count": len(lock["files"]),
        "base_locked_assets_verified_unchanged": len(base_by_path),
        "scene_counts": scene_counts,
        "skipped_cases": skipped_cases,
        "checks": [
            "Case identity/order and all non-mask content equal 8.0.0",
            "first-frame-only manifest contract",
            "per-Case subject cardinality",
            "NPZ keys/layout/dtype/value/ID/non-empty contract",
            "PNG shape/dtype/{0,255}/non-empty contract",
            "pixel equality between every PNG and NPZ object plane",
            "mask metadata area and centroid",
            "multi-instance uniqueness, exclusivity, and ordering",
            "mask index SHA-256",
            "all 8.0.0 locked assets unchanged",
            "9.0.0 loader validation with every asset SHA-256",
        ],
    }


if __name__ == "__main__":
    result = validate()
    output = RELEASE_ROOT / "validation_report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
