#!/usr/bin/env python3
"""Build the 9.0.0 metadata snapshot from 8.0.0 plus mask results."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


RELEASE_ROOT = Path(__file__).resolve().parent
PHYSICS_VIDEO_ROOT = RELEASE_ROOT.parents[1]
BASE_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "8.0.0"
REPORT_PATH = RELEASE_ROOT / "mask_generation_report.json"
DATASET_ID = "physics_video_six_scene_v9"
CHECKPOINT_SHA256 = (
    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_subject_count(case: dict[str, Any]) -> int:
    scene = case["scene_id"]
    if scene == "collision_1d":
        indices = {
            int(key.split("_")[1])
            for key in case["physics"]
            if key.startswith("ball_") and key.endswith("_mass")
        }
        if not indices or indices != set(range(1, max(indices) + 1)):
            raise ValueError(f"{case['case_id']} has non-contiguous collision balls")
        return max(indices)
    if scene == "uniform_circular_motion":
        count = case.get("appearance", {}).get("object_count")
        if count not in (1, 2):
            raise ValueError(f"{case['case_id']} has invalid circular object_count")
        return int(count)
    if scene in {
        "inclined_plane_slide",
        "parabolic_motion",
        "pendulum",
        "push_bottle",
    }:
        return 1
    raise ValueError(f"unsupported scene {scene!r}")


def copy_frozen_metadata() -> None:
    for directory in ("scenes", "views"):
        shutil.copytree(
            BASE_RELEASE_ROOT / directory,
            RELEASE_ROOT / directory,
            dirs_exist_ok=True,
        )
    for filename in (
        "annotation_corrections.json",
        "asset_directory_mapping.json",
        "ball_spec_catalog.json",
        "legacy_split_audit_7.0.0.json",
        "migration_audit.json",
        "split_audit.json",
    ):
        shutil.copy2(BASE_RELEASE_ROOT / filename, RELEASE_ROOT / filename)


def main() -> int:
    base_cases = read_jsonl(BASE_RELEASE_ROOT / "cases.jsonl")
    report = read_json(REPORT_PATH)
    if report.get("selected_cases") != len(base_cases):
        raise ValueError("mask report does not cover every base Case")
    by_case = {item["case_id"]: item for item in report["results"]}
    if set(by_case) != {case["case_id"] for case in base_cases}:
        raise ValueError("mask report Case IDs do not match 8.0.0")

    copy_frozen_metadata()
    output_cases = []
    mask_records = []
    completed = 0
    skipped = []
    total_masks = 0
    for base_case in base_cases:
        case = json.loads(json.dumps(base_case))
        result = by_case[case["case_id"]]
        expected = expected_subject_count(case)
        complete = result.get("status") == "complete"
        instances = result.get("instances", []) if complete else []
        if complete and len(instances) != expected:
            raise ValueError(
                f"{case['case_id']} has {len(instances)} masks; expected {expected}"
            )
        manifest_asset = (
            f"{result['mask_directory']}/manifest.json" if complete else None
        )
        case["assets"]["first_frame_mask_manifest"] = manifest_asset
        for index in range(1, expected + 1):
            role = f"first_frame_subject_mask_{index:02d}"
            case["assets"][role] = (
                instances[index - 1]["asset"] if complete else None
            )
        output_cases.append(case)

        record: dict[str, Any] = {
            "schema_version": "1.0",
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "status": "complete" if complete else "skipped",
            "frame_index": 0,
            "frame_scope": "first_frame_only",
            "source_first_frame": case["assets"]["first_frame"],
            "ordering": "row_major_top_to_bottom_then_left_to_right",
            "expected_subject_count": expected,
            "manifest_asset": manifest_asset,
            "instances": instances,
        }
        if complete:
            completed += 1
            total_masks += len(instances)
        else:
            record["error"] = result.get("error", "unspecified uncertainty")
            skipped.append(
                {
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                    "reason": record["error"],
                }
            )
        mask_records.append(record)

    write_jsonl(RELEASE_ROOT / "cases.jsonl", output_cases)
    write_jsonl(RELEASE_ROOT / "masks.jsonl", mask_records)
    mask_index_sha256 = sha256(RELEASE_ROOT / "masks.jsonl")
    descriptor = {
        "schema_version": "4.0",
        "dataset_id": DATASET_ID,
        "release": "9.0.0",
        "cases": "cases.jsonl",
        "scene_catalog": "scenes",
        "views": {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        },
        "asset_root": "../..",
        "asset_lock": "assets.lock.json",
        "release_manifest": "release.json",
        "mask_annotations": {
            "path": "masks.jsonl",
            "sha256": mask_index_sha256,
            "schema_version": "1.0",
        },
    }
    write_json(RELEASE_ROOT / "dataset.json", descriptor)
    base_release = read_json(BASE_RELEASE_ROOT / "release.json")
    write_json(
        RELEASE_ROOT / "mask_release_audit.json",
        {
            "schema_version": "1.0",
            "base_release": "8.0.0",
            "base_dataset_id": "physics_video_six_scene_v8",
            "base_dataset_digest": base_release["dataset_digest"],
            "output_release": "9.0.0",
            "output_dataset_id": DATASET_ID,
            "case_count": len(output_cases),
            "complete_case_count": completed,
            "skipped_case_count": len(skipped),
            "mask_file_count": total_masks,
            "frame_scope": "first_frame_only",
            "subject_policy": {
                "collision_1d": "every visible collision ball",
                "inclined_plane_slide": "sliding block only; exclude incline",
                "parabolic_motion": "projectile ball only",
                "pendulum": "bob only; exclude cord",
                "push_bottle": "bottle only; exclude hand",
                "uniform_circular_motion": "every orbiting block",
            },
            "ordering": "row-major scan order: top-to-bottom, then left-to-right within a row",
            "generator": "sam2.1_hiera_tiny_first_frame_physical_subject_v1",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "mask_index_sha256": mask_index_sha256,
            "skipped_cases": skipped,
            "old_case_and_media_policy": "8.0.0 Case facts and pre-existing media are copied or referenced without modification",
        },
    )
    print(
        f"cases={len(output_cases)} complete={completed} "
        f"skipped={len(skipped)} masks={total_masks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
