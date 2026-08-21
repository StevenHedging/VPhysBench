#!/usr/bin/env python3
"""Repair evaluator-facing IDs in v14 curated first-frame anchor NPZs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from physbench.io import write_json
from physbench.reference_observations.curation.finalize import (
    anchor_geometry,
    anchor_object_ids,
    rewrite_anchor_object_id,
)
from physbench.reference_observations.curation.install import refresh_locked_files


EXPECTED_CURATED_CASES = {
    ("collision_1d", "human_reviewed_sam2_1_hiera_large_collision_anchor_v1"): 7,
    ("collision_1d", "v14_full_reference_observation_audit"): 142,
    ("inclined_plane_slide", "v14_full_reference_observation_audit"): 5,
    ("parabolic_motion", "v14_full_reference_observation_audit"): 69,
    ("pendulum", "v14_full_reference_observation_audit"): 20,
    ("push_bottle", "v14_full_reference_observation_audit"): 16,
    ("uniform_circular_motion", "v14_full_reference_observation_audit"): 1,
    ("vertical_spring_oscillator", "v14_full_reference_observation_audit"): 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _curated_cases(
    dataset_path: Path,
) -> tuple[
    list[tuple[dict[str, Any], tuple[str, ...]]],
    Counter[tuple[str, str]],
]:
    descriptor = _load_json(dataset_path)
    release_root = dataset_path.parent
    asset_root = (release_root / str(descriptor["asset_root"])).resolve()
    cases_path = release_root / str(descriptor["cases"])
    selected: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        manifest_path = asset_root / case["assets"]["first_frame_mask_manifest"]
        manifest = _load_json(manifest_path)
        generator_id = manifest.get("generator", {}).get("id")
        key = (case["scene_id"], generator_id)
        if key not in EXPECTED_CURATED_CASES:
            continue
        instances = manifest["instances"]
        if generator_id == (
            "human_reviewed_sam2_1_hiera_large_collision_anchor_v1"
        ):
            expected_ids = tuple(instance["object_id"] for instance in instances)
        else:
            expected_ids = anchor_object_ids(case["scene_id"], len(instances))
        counts[key] += 1
        selected.append((case, expected_ids))
    if dict(counts) != EXPECTED_CURATED_CASES:
        raise ValueError(
            "curated case closure differs from the sealed v14 repair set: "
            f"{dict(counts)}"
        )
    return selected, counts


def _repair_root(
    root: Path,
    cases: list[tuple[dict[str, Any], tuple[str, ...]]],
) -> tuple[int, set[str]]:
    rewritten = 0
    locked_paths: set[str] = set()
    anchor_paths: set[str] = set()
    for case, expected_ids in cases:
        mask_manifest_relative = case["assets"]["first_frame_mask_manifest"]
        mask_manifest_path = root / mask_manifest_relative
        mask_manifest = _load_json(mask_manifest_path)
        instances = mask_manifest["instances"]
        if len(instances) != len(expected_ids):
            raise ValueError(f"anchor count changed for {case['case_id']}")
        for instance, object_id in zip(instances, expected_ids, strict=True):
            anchor_relative = instance["npz_asset"]
            anchor_path = root / anchor_relative
            rewritten += int(rewrite_anchor_object_id(anchor_path, object_id))
            with np.load(anchor_path, allow_pickle=False) as payload:
                geometry = anchor_geometry(payload["masks"][0])
            instance.update(geometry)
            anchor_paths.add(anchor_relative)
            locked_paths.add(anchor_relative)
        write_json(mask_manifest_path, mask_manifest)
        locked_paths.add(mask_manifest_relative)

        observation_relative = case["assets"]["reference_observation_manifest"]
        observation_path = root / observation_relative
        observation = _load_json(observation_path)
        observation["source"]["first_frame_mask_manifest"] = _record(
            mask_manifest_path,
            mask_manifest_relative,
        )
        source_anchors = {
            item["mask_id"]: item for item in observation["source"]["anchor_masks"]
        }
        for instance in instances:
            source = source_anchors[instance["mask_id"]]
            source["asset"] = _record(
                root / instance["npz_asset"],
                instance["npz_asset"],
            )
        write_json(observation_path, observation)
        locked_paths.add(observation_relative)

        visualization_relative = case["assets"][
            "reference_observation_visualization_manifest"
        ]
        visualization_path = root / visualization_relative
        visualization = _load_json(visualization_path)
        visualization["observation_manifest_sha256"] = _sha256(observation_path)
        write_json(visualization_path, visualization)
        locked_paths.add(visualization_relative)
    expected_anchor_count = sum(len(ids) for _, ids in cases)
    if len(anchor_paths) != expected_anchor_count:
        raise ValueError("curated anchor paths are not unique")
    return rewritten, locked_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--asset-root",
        required=True,
        action="append",
        type=Path,
        help="Datasets root containing assets/; may be repeated for mirrored trees",
    )
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()

    cases, counts = _curated_cases(args.dataset.resolve(strict=True))
    roots = [path.resolve(strict=True) for path in args.asset_root]
    rewrite_counts: dict[str, int] = {}
    locked_paths: set[str] | None = None
    for root in roots:
        rewritten, root_locked_paths = _repair_root(root, cases)
        rewrite_counts[str(root)] = rewritten
        if locked_paths is None:
            locked_paths = root_locked_paths
        elif root_locked_paths != locked_paths:
            raise RuntimeError("repaired asset roots selected different closures")

    assert locked_paths is not None
    for relative in locked_paths:
        digests = {_sha256(root / relative) for root in roots}
        if len(digests) != 1:
            raise RuntimeError(f"repaired asset mirrors differ: {relative}")
    refresh_locked_files(args.lock, roots[0], locked_paths)
    print(
        json.dumps(
            {
                "curated_cases": {
                    f"{scene_id}:{generator_id}": count
                    for (scene_id, generator_id), count in counts.items()
                },
                "anchor_files": sum(len(ids) for _, ids in cases),
                "hash_closed_files": len(locked_paths),
                "rewritten_by_root": rewrite_counts,
                "lock": str(args.lock.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
