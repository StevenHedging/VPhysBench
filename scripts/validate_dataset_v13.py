#!/usr/bin/env python3
"""Independently validate the single active Dataset 13.0.0 release."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import re
from typing import Any

from physbench.datasets import load_dataset
from physbench.datasets.physics import (
    is_scalar_quantity,
    is_time_series_quantity,
    iter_physics_quantities,
)
from physbench.io import load_json, load_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
RELEASE_ROOT = DATASETS_ROOT / "releases" / "13.0.0"
DATASET = RELEASE_ROOT / "dataset.json"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "13.0.0"
IMPORT_AUDIT = (
    DATASETS_ROOT
    / "provenance/imports/vertical_spring_oscillator_20260806_import_audit.jsonl"
)
PUSH_FORCE_SOURCE = (
    DATASETS_ROOT
    / "provenance/source_docs/20260804_push_bottle/normalized_annotations.json"
)
EXPECTED_RELEASE_ENTRIES = {"dataset.json", "cases.jsonl", "scenes", "views"}
EXPECTED_SCENES = {
    "collision_1d",
    "inclined_plane_slide",
    "parabolic_motion",
    "pendulum",
    "push_bottle",
    "uniform_circular_motion",
    "vertical_spring_oscillator",
}
SPRING_EVENT = (
    "first return to the release-side turning point after one complete oscillation"
)


def _symbol_in_prompt(symbol: str, prompt: str) -> bool:
    boundary = r"[A-Za-z0-9_]"
    return re.search(
        rf"(?<!{boundary}){re.escape(symbol)}(?!{boundary})", prompt
    ) is not None


def _forbidden_audit_keys(value: Any) -> list[str]:
    if isinstance(value, list):
        return [key for child in value for key in _forbidden_audit_keys(child)]
    if not isinstance(value, dict):
        return []
    return [
        key for key in value if "sha256" in key.lower() or "digest" in key.lower()
    ] + [
        key for child in value.values() for key in _forbidden_audit_keys(child)
    ]


def validate_v13(
    dataset_path: Path = DATASET,
    *,
    write_report: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    if dataset_path != DATASET.resolve():
        raise ValueError(f"V13 validator requires {DATASET}")
    active_releases = {
        path.name
        for path in (DATASETS_ROOT / "releases").iterdir()
        if path.is_dir()
    }
    if active_releases != {"13.0.0"}:
        raise ValueError(f"unexpected active Dataset releases: {sorted(active_releases)}")
    if {path.name for path in RELEASE_ROOT.iterdir()} != EXPECTED_RELEASE_ENTRIES:
        raise ValueError("V13 runtime Release tree is not minimal")

    snapshot = load_dataset(dataset_path, check_assets=True)
    if snapshot.dataset_id != "physics_video_seven_scene_v13":
        raise ValueError(f"unexpected Dataset ID: {snapshot.dataset_id}")
    if snapshot.descriptor.get("release") != "13.0.0":
        raise ValueError("V13 descriptor release mismatch")
    if snapshot.descriptor.get("schema_version") != "5.0":
        raise ValueError("V13 requires Dataset schema 5.0")
    if len(snapshot.cases) != 916 or set(snapshot.scene_configs) != EXPECTED_SCENES:
        raise ValueError("V13 Case/Scene coverage mismatch")
    if snapshot.asset_lock is not None:
        raise ValueError("V13 must not use an asset lock")

    indexed_cases = load_jsonl(RELEASE_ROOT / "cases.jsonl")
    provenance_cases = load_jsonl(PROVENANCE_ROOT / "cases.jsonl")
    loaded_by_id = {str(case["case_id"]): case for case in snapshot.cases}
    if (
        len(indexed_cases) != 916
        or len(provenance_cases) != 916
        or {str(item["case_id"]) for item in provenance_cases} != set(loaded_by_id)
    ):
        raise ValueError("V13 index/provenance Case coverage mismatch")
    if _forbidden_audit_keys(provenance_cases):
        raise ValueError("V13 provenance contains asset hash metadata")

    caption_paths: set[str] = set()
    physics_paths: set[str] = set()
    quantity_count = 0
    scalar_quantity_count = 0
    time_series_quantity_count = 0
    mask_manifest_count = 0
    for indexed in indexed_cases:
        if set(indexed) != {
            "appearance",
            "assets",
            "case_id",
            "scene_id",
            "temporal",
        }:
            raise ValueError(f"non-runtime field in Case index: {indexed['case_id']}")
        if set(indexed["temporal"]) != {"encoded_to_physical_speed"}:
            raise ValueError(f"invalid temporal index: {indexed['case_id']}")
        case = loaded_by_id[str(indexed["case_id"])]
        caption_relative = indexed["assets"].get("caption")
        physics_relative = indexed["assets"].get("physics_annotation")
        if (
            not isinstance(caption_relative, str)
            or not caption_relative.endswith("/caption.json")
            or caption_relative in caption_paths
        ):
            raise ValueError(f"invalid caption path: {case['case_id']}")
        if (
            not isinstance(physics_relative, str)
            or not physics_relative.endswith("/physics.json")
            or physics_relative in physics_paths
            or "physics.v" in physics_relative
        ):
            raise ValueError(f"invalid physics path: {case['case_id']}")
        caption_paths.add(caption_relative)
        physics_paths.add(physics_relative)
        caption = load_json(DATASETS_ROOT / caption_relative)
        physics = load_json(DATASETS_ROOT / physics_relative)
        if set(caption) != {"caption", "case_id", "scene_id"}:
            raise ValueError(f"non-minimal Caption: {case['case_id']}")
        if set(physics) != {"case_id", "physics", "scene_id"}:
            raise ValueError(f"non-minimal physics document: {case['case_id']}")
        symbols: set[str] = set()
        for path, _, quantity in iter_physics_quantities(case):
            quantity_count += 1
            if is_scalar_quantity(quantity):
                scalar_quantity_count += 1
                values = [quantity["value"]]
            elif is_time_series_quantity(quantity):
                time_series_quantity_count += 1
                values = [sample.get("value") for sample in quantity["samples"]]
            else:
                raise ValueError(f"invalid quantity shape: {case['case_id']}/{path}")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                for value in values
            ):
                raise ValueError(f"invalid quantity value: {case['case_id']}/{path}")
            symbol = quantity["symbol"]
            if not isinstance(symbol, str) or not symbol or symbol in symbols:
                raise ValueError(f"invalid quantity symbol: {case['case_id']}/{path}")
            symbols.add(symbol)
            if not _symbol_in_prompt(symbol, case["text"]["prompt"]):
                raise ValueError(f"prompt-symbol mismatch: {case['case_id']}/{path}")
        manifest_relative = indexed["assets"].get("first_frame_mask_manifest")
        if manifest_relative is not None:
            manifest = load_json(DATASETS_ROOT / manifest_relative)
            if _forbidden_audit_keys(manifest):
                raise ValueError(f"mask manifest contains hash: {case['case_id']}")
            mask_manifest_count += 1

    if (
        len(caption_paths) != 916
        or len(physics_paths) != 916
        or quantity_count != 4520
        or scalar_quantity_count != 4379
        or time_series_quantity_count != 141
        or mask_manifest_count != 914
    ):
        raise ValueError("V13 document/physics/mask counts mismatch")
    if len(list(DATASETS_ROOT.glob("assets/*/*/caption.json"))) != 916:
        raise ValueError("V13 does not have exactly 916 Caption documents")
    if len(list(DATASETS_ROOT.glob("assets/*/*/physics.json"))) != 916:
        raise ValueError("V13 does not have exactly 916 physics documents")

    provenance_by_id = {str(item["case_id"]): item for item in provenance_cases}
    force_records = load_json(PUSH_FORCE_SOURCE).get("records")
    if not isinstance(force_records, list) or len(force_records) != 141:
        raise ValueError("push-bottle force source coverage mismatch")
    force_by_locator = {}
    for record in force_records:
        annotation = record.get("annotation", {})
        locator = (
            annotation.get("source_workbook_member"),
            annotation.get("source_sheet"),
        )
        samples = record.get("force_annotation", {}).get("force_samples")
        if (
            None in locator
            or locator in force_by_locator
            or not isinstance(samples, list)
            or not samples
            or record.get("force_annotation", {}).get("force_sample_count")
            != len(samples)
        ):
            raise ValueError("invalid push-bottle force source record")
        force_by_locator[locator] = record
    verified_force_series = 0
    for case in snapshot.cases:
        if case["scene_id"] != "push_bottle":
            continue
        locator = provenance_by_id[str(case["case_id"])].get("source_locator", {})
        source_key = (locator.get("annotation_workbook"), locator.get("sheet"))
        record = force_by_locator.get(source_key)
        if record is None:
            raise ValueError(f"unmatched force source: {case['case_id']}")
        expected = [
            {"time": sample["time_s"], "value": max(sample["force_n"], 0.0)}
            for sample in record["force_annotation"]["force_samples"]
        ]
        actual = case["physics"]["objects"]["object_1"]["applied_force"]
        if (
            actual.get("time_unit") != "s"
            or actual.get("unit") != "N"
            or actual.get("symbol") != "F(t)"
            or actual.get("samples") != expected
        ):
            raise ValueError(f"force-series mismatch: {case['case_id']}")
        verified_force_series += 1
    if verified_force_series != 141:
        raise ValueError("push-bottle force-series Case coverage mismatch")

    audits = load_jsonl(IMPORT_AUDIT)
    spring_cases = {
        str(case["case_id"]): case
        for case in snapshot.cases
        if case["scene_id"] == "vertical_spring_oscillator"
    }
    audits_by_id = {str(item["case_id"]): item for item in audits}
    if len(audits) != 117 or set(audits_by_id) != set(spring_cases):
        raise ValueError("spring import audit coverage mismatch")
    if _forbidden_audit_keys(audits):
        raise ValueError("spring import audit contains asset hash metadata")
    for case_id, case in spring_cases.items():
        audit = audits_by_id[case_id]
        provenance = provenance_by_id[case_id]
        review = audit.get("review", {})
        if review.get("status") != "approved" or not all(
            review.get("checks", {}).values()
        ):
            raise ValueError(f"unapproved spring Case: {case_id}")
        if audit.get("alignment", {}).get("canonical_first_frame_event") != SPRING_EVENT:
            raise ValueError(f"spring alignment mismatch: {case_id}")
        if provenance.get("alignment", {}).get("canonical_first_frame_event") != SPRING_EVENT:
            raise ValueError(f"spring release provenance mismatch: {case_id}")
        displacement = case["physics"]["objects"]["object_1"][
            "initial_displacement"
        ]["value"]
        if displacement <= 0:
            raise ValueError(f"non-positive spring x_0: {case_id}")
        direction = audit["direction"]
        if f" {direction} equilibrium" not in case["text"]["prompt"]:
            raise ValueError(f"spring Caption direction mismatch: {case_id}")
        manifest = load_json(
            DATASETS_ROOT / case["assets"]["first_frame_mask_manifest"]
        )
        if (
            len(manifest.get("instances", [])) != 1
            or manifest["instances"][0].get("object_id") != "object_1"
        ):
            raise ValueError(f"spring mask instance mismatch: {case_id}")

    spring_view = snapshot.views["view_a"]["scenes"]["vertical_spring_oscillator"]
    if len(spring_view["train"]) != 97 or len(spring_view["test"]) != 20:
        raise ValueError("spring View A split size mismatch")
    test_directions = Counter(
        audits_by_id[case_id]["direction"] for case_id in spring_view["test"]
    )
    if test_directions != {"above": 10, "below": 10}:
        raise ValueError("spring View A test direction balance mismatch")
    train_groups = {audits_by_id[case_id]["source_group"] for case_id in spring_view["train"]}
    test_groups = {audits_by_id[case_id]["source_group"] for case_id in spring_view["test"]}
    if train_groups & test_groups:
        raise ValueError("spring source group spans train and test")

    migration = load_json(PROVENANCE_ROOT / "migration.json")
    if migration.get("counts") != {
        "base_cases_preserved": 799,
        "cases": 916,
        "media_changes": 0,
        "spring_cases_added": 117,
    }:
        raise ValueError("V13 migration evidence mismatch")
    report = {
        "schema_version": "1.0",
        "status": "valid",
        "dataset_id": snapshot.dataset_id,
        "cases": 916,
        "caption_documents": 916,
        "physics_documents": 916,
        "provenance_cases": 916,
        "mask_manifests": 914,
        "quantities": 4520,
        "scalar_quantities": 4379,
        "time_series_quantities": 141,
        "spring_cases": 117,
        "source_verified_force_series": verified_force_series,
        "media_changes": 0,
    }
    if write_report:
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(PROVENANCE_ROOT / "validation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = validate_v13(args.dataset, write_report=args.write_report)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
