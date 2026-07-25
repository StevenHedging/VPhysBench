#!/usr/bin/env python3
"""Create the prompt-free Dataset v2 bundle from the immutable v1 manifest."""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.io import canonical_sha256, load_json, load_jsonl, write_json, write_jsonl  # noqa: E402
from physbench.data_layout import (  # noqa: E402
    V1_CASES,
    V1_VIEW_A,
    V1_VIEW_B,
    V2_RELEASE_ROOT,
)


def dataset_relative(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    parts = list(path.parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    return Path(*parts).as_posix()


def migrate_case(case: dict[str, Any]) -> dict[str, Any]:
    assets = {
        key: dataset_relative(value)
        for key, value in case["assets"].items()
    }
    alignment = copy.deepcopy(case.get("alignment"))
    if alignment and alignment.get("audit_record"):
        alignment["audit_record"] = dataset_relative(alignment["audit_record"])
    provenance = copy.deepcopy(case["provenance"])
    generator = provenance.get("generator")
    if isinstance(generator, dict) and "prompt" in generator:
        generator["generation_instruction"] = generator.pop("prompt")
    if isinstance(generator, dict) and "source_metadata" in generator:
        # The old path points into a model-specific project.  The immutable
        # first-frame asset and v1 source manifest retain provenance; v2 must
        # not depend on a Baseline workspace being present.
        generator.pop("source_metadata")
        generator["metadata_migrated_from_v1"] = True
    value = {
        "schema_version": "2.0",
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "assets": assets,
        "physics": case["physical_parameters"],
        "appearance": case["appearance"],
        "temporal": case.get("temporal", {
            "encoded_to_physical_speed": 1.0,
            "time_scale": "real_time",
            "annotation_source": "v1 default",
        }),
        "provenance": provenance,
        "ood": case["ood"],
        "has_real_reference_video": case["has_real_reference_video"],
    }
    if alignment is not None:
        value["alignment"] = alignment
    return value


def migrate_view(path: Path) -> dict[str, Any]:
    value = load_json(path)
    value["schema_version"] = "2.0"
    value["view_id"] = "view_a" if value["view"] == "A" else "view_b"
    value.pop("view", None)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=V1_CASES)
    parser.add_argument("--view-a", type=Path, default=V1_VIEW_A)
    parser.add_argument(
        "--view-b", type=Path, default=V1_VIEW_B,
    )
    parser.add_argument("--output", type=Path, default=V2_RELEASE_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty Dataset bundle: {output}")
    (output / "views").mkdir(parents=True, exist_ok=True)
    (output / "scenes").mkdir(parents=True, exist_ok=True)
    v1_cases = load_jsonl(args.manifest)
    cases = [migrate_case(case) for case in v1_cases]
    write_jsonl(output / "cases.jsonl", cases)
    write_json(output / "views" / "view_a.json", migrate_view(args.view_a))
    write_json(output / "views" / "view_b.json", migrate_view(args.view_b))
    for source in sorted((ROOT / "configs" / "scenes").glob("*.json")):
        shutil.copy2(source, output / "scenes" / source.name)
    descriptor = {
        "schema_version": "2.0",
        "dataset_id": "physics_video_three_scene_v2",
        "release": "2.0.0",
        "cases": "cases.jsonl",
        "asset_root": "../..",
        "asset_lock": "assets.lock.json",
        "scene_catalog": "scenes",
        "views": {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        },
    }
    write_json(output / "dataset.json", descriptor)
    write_json(output / "migration_audit.json", {
        "schema_version": "2.0",
        "source_manifest": "../1.0.0/cases.jsonl",
        "source_view_a": "../1.0.0/views/view_a.json",
        "source_view_b": "../1.0.0/views/view_b_seed42_g5.json",
        "case_count": len(cases),
        "case_ids_sha256": canonical_sha256(sorted(case["case_id"] for case in cases)),
        "removed_case_fields": [
            "input_views",
            "physical_parameters",
            "text",
            "view_a_split",
        ],
        "renamed_fields": {"physical_parameters": "physics"},
        "assets_copied": False,
        "source_assets_mutated": False,
        "asset_root_migrated_to": "../../assets",
    })
    print(output / "dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
