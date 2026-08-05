#!/usr/bin/env python3
"""Independently validate the immutable Dataset 10.0.0 release."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
V9_RELEASE_ROOT = DATASETS_ROOT / "releases" / "9.0.0"
V10_RELEASE_ROOT = DATASETS_ROOT / "releases" / "10.0.0"
V10_DATASET = V10_RELEASE_ROOT / "dataset.json"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "10.0.0"
EXPECTED_RELEASE_ENTRIES = {
    "README.md",
    "dataset.json",
    "release.json",
    "cases.jsonl",
    "assets.lock.json",
    "scenes",
    "views",
}
EXPECTED_PHYSICS_FINGERPRINT = (
    "ca8c593007ce4e3d6d7437656f9aa0c3f32b5ff245a4592c570297ba9f957d1a"
)


from physbench.datasets import load_dataset  # noqa: E402
from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
)


def _physics_fingerprint(cases: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        [(case["case_id"], case["physics"]) for case in cases]
    )


def validate_v10(
    dataset_path: Path = V10_DATASET,
    *,
    write_report: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    release_root = dataset_path.parent
    if {path.name for path in release_root.iterdir()} != EXPECTED_RELEASE_ENTRIES:
        raise ValueError("V10 release tree is not the minimal seven-entry snapshot")

    descriptor = load_json(dataset_path)
    if descriptor.get("dataset_id") != "physics_video_six_scene_v10":
        raise ValueError("unexpected V10 Dataset ID")
    if descriptor.get("release") != "10.0.0":
        raise ValueError("unexpected V10 release identity")
    if "mask_annotations" in descriptor:
        raise ValueError("minimal V10 descriptor must not contain mask_annotations")

    v9_cases = load_jsonl(V9_RELEASE_ROOT / "cases.jsonl")
    v10_cases = load_jsonl(release_root / "cases.jsonl")
    if len(v9_cases) != 799 or len(v10_cases) != 799:
        raise ValueError("V9 and V10 must each contain 799 Cases")
    physics_paths: set[str] = set()
    asset_root = (release_root / descriptor["asset_root"]).resolve()
    for old, new in zip(v9_cases, v10_cases, strict=True):
        comparable = copy.deepcopy(new)
        relative = comparable["assets"].pop("physics_annotation", None)
        if not isinstance(relative, str) or not relative.endswith("/physics.json"):
            raise ValueError(
                f"case {new['case_id']} lacks a Case-local physics document"
            )
        if old != comparable:
            raise ValueError(f"V10 changes V9 Case facts: {new['case_id']}")
        if relative in physics_paths:
            raise ValueError(f"duplicate V10 physics path: {relative}")
        physics_paths.add(relative)
        path = (asset_root / relative).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as exc:
            raise ValueError(f"physics path escapes asset_root: {relative}") from exc
        document = load_json(path)
        if document != {
            "schema_version": "1.0",
            "case_id": new["case_id"],
            "scene_id": new["scene_id"],
            "physics": new["physics"],
        }:
            raise ValueError(f"invalid physics document: {relative}")
    if len(physics_paths) != 799:
        raise ValueError("V10 must contain 799 unique physics documents")

    v9_fingerprint = _physics_fingerprint(v9_cases)
    v10_fingerprint = _physics_fingerprint(v10_cases)
    if v9_fingerprint != EXPECTED_PHYSICS_FINGERPRINT:
        raise ValueError(f"unexpected V9 physics fingerprint: {v9_fingerprint}")
    if v10_fingerprint != v9_fingerprint:
        raise ValueError("V10 physics fingerprint differs from V9")

    v9_lock = load_json(V9_RELEASE_ROOT / "assets.lock.json")
    v10_lock = load_json(release_root / "assets.lock.json")
    old_by_path = {item["path"]: item for item in v9_lock["files"]}
    new_by_path = {item["path"]: item for item in v10_lock["files"]}
    if len(old_by_path) != 5239 or len(new_by_path) != 6038:
        raise ValueError("unexpected V9 or V10 locked-asset count")
    for path, old in old_by_path.items():
        if new_by_path.get(path) != old:
            raise ValueError(f"V10 changed an existing locked asset: {path}")
    added = set(new_by_path) - set(old_by_path)
    if added != physics_paths:
        raise ValueError("V10 lock additions are not exactly the physics documents")
    for relative, locked in new_by_path.items():
        path = (asset_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != locked["size_bytes"]:
            raise ValueError(f"locked asset size mismatch: {relative}")

    migration = load_json(PROVENANCE_ROOT / "migration.json")
    removals = migration.get("removed_legacy_directories")
    if not isinstance(removals, list) or len(removals) != 32:
        raise ValueError("V10 migration evidence must record 32 removals")
    removal_directories = [item["directory"] for item in removals]
    if len(set(removal_directories)) != 32:
        raise ValueError("V10 migration removal paths must be unique")
    for relative in removal_directories:
        if (DATASETS_ROOT / relative).exists():
            raise ValueError(f"legacy directory still exists: {relative}")

    snapshot = load_dataset(dataset_path, check_asset_hashes=True)
    release = load_json(release_root / "release.json")
    report = {
        "schema_version": "1.0",
        "status": "valid",
        "dataset_id": snapshot.dataset_id,
        "release": descriptor["release"],
        "dataset_digest": snapshot.digest,
        "asset_files_digest": v10_lock["files_digest"],
        "physics_fingerprint": v10_fingerprint,
        "counts": {
            "cases": len(v10_cases),
            "physics_documents": len(physics_paths),
            "locked_assets": len(new_by_path),
            "preserved_v9_locked_assets": len(old_by_path),
            "removed_legacy_directories": len(removals),
        },
        "checks": {
            "minimal_release_tree": True,
            "v9_case_facts_preserved": True,
            "physics_documents_equal_inline": True,
            "asset_sha256_verified": True,
            "legacy_directories_absent": True,
        },
    }
    if release["dataset_digest"] != report["dataset_digest"]:
        raise ValueError("V10 release manifest digest mismatch")
    if write_report:
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(PROVENANCE_ROOT / "validation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=V10_DATASET)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = validate_v10(args.dataset, write_report=args.write_report)
    print(
        f"status={report['status']} cases={report['counts']['cases']} "
        f"physics={report['counts']['physics_documents']} "
        f"locked_assets={report['counts']['locked_assets']} "
        f"dataset_digest={report['dataset_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
