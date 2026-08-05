#!/usr/bin/env python3
"""Independently validate the single active Dataset 12.0.0."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

from physbench.datasets import load_dataset
from physbench.io import canonical_sha256, load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
V12_RELEASE_ROOT = DATASETS_ROOT / "releases" / "12.0.0"
V12_DATASET = V12_RELEASE_ROOT / "dataset.json"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "12.0.0"
EXPECTED_ENTRIES = {
    "README.md",
    "dataset.json",
    "release.json",
    "cases.jsonl",
    "assets.lock.json",
    "scenes",
    "views",
}


def _symbol_in_prompt(symbol: str, prompt: str) -> bool:
    boundary = r"[A-Za-z0-9_]"
    return re.search(
        rf"(?<!{boundary}){re.escape(symbol)}(?!{boundary})", prompt
    ) is not None


def _non_physics_files(lock: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in lock["files"]
        if "physics_annotation" not in item["roles"]
    ]


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
    snapshot = load_dataset(dataset_path, check_asset_hashes=True)
    if snapshot.dataset_id != "physics_video_six_scene_v12":
        raise ValueError(f"unexpected Dataset ID: {snapshot.dataset_id}")
    if snapshot.descriptor["schema_version"] != "5.0":
        raise ValueError("V12 requires Dataset schema 5.0")
    if len(snapshot.cases) != 799 or len(snapshot.scene_configs) != 6:
        raise ValueError("V12 Case/Scene count mismatch")
    if len(snapshot.asset_lock["files"]) != 6038:
        raise ValueError("V12 asset-lock count mismatch")

    physics_paths: set[str] = set()
    quantity_count = 0
    for case in snapshot.cases:
        relative = case["assets"].get("physics_annotation")
        if not isinstance(relative, str) or not relative.endswith("/physics.json"):
            raise ValueError(f"invalid physics path for {case['case_id']}")
        if "physics.v" in relative or relative in physics_paths:
            raise ValueError(f"versioned or duplicate physics path: {relative}")
        physics_paths.add(relative)
        prompt = case["text"]["prompt"]
        symbols: set[str] = set()
        for name, quantity in case["physics"].items():
            quantity_count += 1
            value = quantity["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"invalid quantity {case['case_id']}/{name}")
            symbol = quantity["symbol"]
            if not isinstance(symbol, str) or not symbol or symbol in symbols:
                raise ValueError(f"invalid symbol {case['case_id']}/{name}")
            symbols.add(symbol)
            present = _symbol_in_prompt(symbol, prompt)
            if bool(quantity["annotated"]) != present:
                raise ValueError(f"prompt-symbol mismatch {case['case_id']}/{name}")
    if len(physics_paths) != 799 or quantity_count != 5286:
        raise ValueError("V12 physics coverage mismatch")
    if list(DATASETS_ROOT.glob("assets/*/*/physics.v11.json")):
        raise ValueError("versioned physics documents remain")
    if len(list(DATASETS_ROOT.glob("assets/*/*/physics.json"))) != 799:
        raise ValueError("V12 does not have exactly 799 physics.json files")

    evidence = load_json(PROVENANCE_ROOT / "migration.json")
    if evidence["counts"] != {
        "cases": 799,
        "physics_documents_renamed": 799,
        "locked_assets": 6038,
        "media_changes": 0,
    }:
        raise ValueError("V12 migration evidence counts mismatch")
    non_physics_digest = canonical_sha256(
        _non_physics_files(snapshot.asset_lock)
    )
    if non_physics_digest != evidence["non_physics_asset_records_sha256"]:
        raise ValueError("V12 non-physics asset evidence mismatch")

    release = load_json(V12_RELEASE_ROOT / "release.json")
    if release["dataset_digest"] != snapshot.digest:
        raise ValueError("V12 release digest mismatch")
    report = {
        "schema_version": "1.0",
        "status": "valid",
        "dataset_id": snapshot.dataset_id,
        "dataset_digest": snapshot.digest,
        "asset_files_digest": release["asset_files_digest"],
        "cases": 799,
        "physics_documents": 799,
        "quantities": 5286,
        "locked_assets": 6038,
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
