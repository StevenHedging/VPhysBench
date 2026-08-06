#!/usr/bin/env python3
"""Independently validate the single active Dataset 12.0.0."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

from physbench.datasets import load_dataset
from physbench.datasets.physics import iter_physics_quantities
from physbench.io import load_json, load_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
V12_RELEASE_ROOT = DATASETS_ROOT / "releases" / "12.0.0"
V12_DATASET = V12_RELEASE_ROOT / "dataset.json"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "12.0.0"
EXPECTED_ENTRIES = {
    "dataset.json",
    "cases.jsonl",
    "scenes",
    "views",
}


def _symbol_in_prompt(symbol: str, prompt: str) -> bool:
    boundary = r"[A-Za-z0-9_]"
    return re.search(
        rf"(?<!{boundary}){re.escape(symbol)}(?!{boundary})", prompt
    ) is not None


def validate_v12(
    dataset_path: Path = V12_DATASET,
    *,
    write_report: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    if dataset_path != V12_DATASET.resolve():
        raise ValueError(f"V12 validator requires {V12_DATASET}")
    if {path.name for path in V12_RELEASE_ROOT.iterdir()} != EXPECTED_ENTRIES:
        raise ValueError("V12 runtime Release tree is not minimal")
    snapshot = load_dataset(dataset_path)
    if snapshot.dataset_id != "physics_video_six_scene_v12":
        raise ValueError(f"unexpected Dataset ID: {snapshot.dataset_id}")
    if snapshot.descriptor["schema_version"] != "5.0":
        raise ValueError("V12 requires Dataset schema 5.0")
    if len(snapshot.cases) != 799 or len(snapshot.scene_configs) != 6:
        raise ValueError("V12 Case/Scene count mismatch")
    if snapshot.asset_lock is not None:
        raise ValueError("V12 must not use an asset lock")

    indexed_cases = load_jsonl(V12_RELEASE_ROOT / "cases.jsonl")
    provenance_cases = load_jsonl(PROVENANCE_ROOT / "cases.jsonl")
    loaded_by_id = {case["case_id"]: case for case in snapshot.cases}
    physics_paths: set[str] = set()
    caption_paths: set[str] = set()
    quantity_count = 0
    for indexed_case in indexed_cases:
        if set(indexed_case) != {
            "case_id",
            "scene_id",
            "assets",
            "appearance",
            "temporal",
        }:
            raise ValueError(
                f"non-runtime field remains in Case index: {indexed_case['case_id']}"
            )
        if set(indexed_case["temporal"]) != {"encoded_to_physical_speed"}:
            raise ValueError(
                f"non-runtime temporal field remains for {indexed_case['case_id']}"
            )
        allowed_assets = {
            "caption",
            "first_frame",
            "first_frame_mask_manifest",
            "physics_annotation",
            "reference_video",
        }
        if not {
            "caption",
            "first_frame",
            "physics_annotation",
            "reference_video",
        } <= set(indexed_case["assets"]) or not set(
            indexed_case["assets"]
        ) <= allowed_assets:
            raise ValueError(f"invalid runtime assets for {indexed_case['case_id']}")
        case = loaded_by_id[indexed_case["case_id"]]
        caption_relative = indexed_case["assets"].get("caption")
        if (
            not isinstance(caption_relative, str)
            or not caption_relative.endswith("/caption.json")
            or caption_relative in caption_paths
        ):
            raise ValueError(f"invalid caption path for {case['case_id']}")
        caption_paths.add(caption_relative)
        caption_document = load_json(DATASETS_ROOT / caption_relative)
        if set(caption_document) != {"case_id", "scene_id", "caption"}:
            raise ValueError(f"caption is not minimal for {case['case_id']}")
        relative = case["assets"].get("physics_annotation")
        if not isinstance(relative, str) or not relative.endswith("/physics.json"):
            raise ValueError(f"invalid physics path for {case['case_id']}")
        if "physics.v" in relative or relative in physics_paths:
            raise ValueError(f"versioned or duplicate physics path: {relative}")
        physics_paths.add(relative)
        physics_document = load_json(DATASETS_ROOT / relative)
        if set(physics_document) != {"case_id", "scene_id", "physics"}:
            raise ValueError(f"physics document is not minimal for {case['case_id']}")
        prompt = case["text"]["prompt"]
        symbols: set[str] = set()
        for path, _, quantity in iter_physics_quantities(case):
            quantity_count += 1
            value = quantity["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"invalid quantity {case['case_id']}/{path}")
            symbol = quantity["symbol"]
            if not isinstance(symbol, str) or not symbol or symbol in symbols:
                raise ValueError(f"invalid symbol {case['case_id']}/{path}")
            symbols.add(symbol)
            if not _symbol_in_prompt(symbol, prompt):
                raise ValueError(f"prompt-symbol mismatch {case['case_id']}/{path}")
    if (
        len(caption_paths) != 799
        or len(physics_paths) != 799
        or quantity_count != 3959
    ):
        raise ValueError("V12 physics coverage mismatch")
    if list(DATASETS_ROOT.glob("assets/*/*/physics.v11.json")):
        raise ValueError("versioned physics documents remain")
    if len(list(DATASETS_ROOT.glob("assets/*/*/physics.json"))) != 799:
        raise ValueError("V12 does not have exactly 799 physics.json files")
    if len(list(DATASETS_ROOT.glob("assets/*/*/caption.json"))) != 799:
        raise ValueError("V12 does not have exactly 799 caption.json files")

    if (
        len(provenance_cases) != 799
        or {item["case_id"] for item in provenance_cases} != set(loaded_by_id)
        or any("temporal_metadata" not in item for item in provenance_cases)
    ):
        raise ValueError("V12 provenance Case coverage mismatch")

    def forbidden_audit_keys(value: Any) -> list[str]:
        if isinstance(value, list):
            return [
                key
                for child in value
                for key in forbidden_audit_keys(child)
            ]
        if not isinstance(value, dict):
            return []
        return [
            key for key in value if "sha256" in key.lower() or "digest" in key.lower()
        ] + [
            key
            for child in value.values()
            for key in forbidden_audit_keys(child)
        ]

    if forbidden_audit_keys(provenance_cases):
        raise ValueError("V12 provenance contains hash metadata")

    mask_manifests = []
    for indexed_case in indexed_cases:
        relative = indexed_case["assets"].get("first_frame_mask_manifest")
        if relative is None:
            continue
        manifest = load_json(DATASETS_ROOT / relative)
        loaded = loaded_by_id[indexed_case["case_id"]]
        physics = loaded["physics"]
        for index, instance in enumerate(manifest["instances"], 1):
            object_id = f"object_{index}"
            if instance.get("object_id") != object_id:
                raise ValueError(
                    f"mask object ordering mismatch for {indexed_case['case_id']}"
                )
            if indexed_case["scene_id"] == "push_bottle":
                expected_keys = sorted(physics)
            else:
                expected_keys = sorted(
                    f"objects.{object_id}.{name}"
                    for name in physics["objects"][object_id]
                )
            if instance.get("physics_keys") != expected_keys:
                raise ValueError(
                    f"mask physics binding mismatch for {indexed_case['case_id']}"
                )
        mask_manifests.append(manifest)
    if len(mask_manifests) != 797 or forbidden_audit_keys(mask_manifests):
        raise ValueError("V12 mask manifests contain invalid audit metadata")

    evidence = load_json(PROVENANCE_ROOT / "migration.json")
    if evidence["counts"] != {
        "caption_documents_created": 799,
        "cases": 799,
        "inline_members_removed": 1598,
        "physics_documents_renamed": 799,
        "media_changes": 0,
    }:
        raise ValueError("V12 migration evidence counts mismatch")
    report = {
        "schema_version": "1.0",
        "status": "valid",
        "dataset_id": snapshot.dataset_id,
        "cases": 799,
        "caption_documents": 799,
        "physics_documents": 799,
        "provenance_cases": 799,
        "mask_manifests": 797,
        "quantities": 3959,
        "media_changes": 0,
    }
    if write_report:
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(PROVENANCE_ROOT / "validation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=V12_DATASET)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = validate_v12(args.dataset, write_report=args.write_report)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
