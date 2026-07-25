#!/usr/bin/env python3
"""Import the five reviewed pendulum appearance-OOD first frames.

The images remain first-frame-only samples.  Their physical reference videos are
borrowed from an ID parent with exactly identical physical parameters, while
``reference_video`` stays null so visual metrics cannot mistake the parent video
for a real continuation of the changed appearance.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE_ROOT = WORKSPACE / "wan22_pendulum_pipeline" / "data" / "ood_eval"

sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import (  # noqa: E402
    PHYSICS_VIDEO_ASSETS,
    PHYSICS_VIDEO_PROVENANCE,
    V1_CASES as MANIFEST,
    V1_VIEW_A as VIEW_A,
    V1_VIEW_B as VIEW_B,
)
from physbench.io import load_jsonl, write_json, write_jsonl  # noqa: E402
from physbench.splitters import build_view_a, build_view_b  # noqa: E402


AUDIT = PHYSICS_VIDEO_PROVENANCE / "imports" / "pendulum_ood_import_audit.jsonl"

FACTOR_MAP = {
    "background_shift": "background",
    "bob_material_shift": "bob_material",
    "support_shift": "support",
}


def _parameters(metadata: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metadata["string_length_mm"] / 1000.0,
        metadata["bob_radius_mm"] / 1000.0,
        metadata["pendulum_length_mm"] / 1000.0,
        float(metadata["initial_angle_deg"]),
    )


def _case_parameters(case: dict[str, Any]) -> tuple[float, float, float, float]:
    values = case["physical_parameters"]
    return (
        float(values["string_length"]["value"]),
        float(values["bob_radius"]["value"]),
        float(values["pendulum_length"]["value"]),
        float(values["initial_angle"]["value"]),
    )


def main() -> int:
    cases = load_jsonl(MANIFEST)
    metadata_paths = sorted(SOURCE_ROOT.glob("*/*/metadata.json"))
    if len(metadata_paths) != 5:
        raise RuntimeError(f"expected 5 reviewed pendulum OOD samples, found {len(metadata_paths)}")

    incoming_ids = {
        json.loads(path.read_text(encoding="utf-8"))["sample_id"] for path in metadata_paths
    }
    retained = [case for case in cases if case["case_id"] not in incoming_ids]
    id_parents = [
        case for case in retained
        if case["scene_id"] == "pendulum" and case["ood"]["level"] == "id"
    ]
    imported: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_parameters = _parameters(metadata)
        matches = [case for case in id_parents if _case_parameters(case) == target_parameters]
        if len(matches) != 1:
            raise RuntimeError(
                f"{metadata['sample_id']} requires one exact ID parent, found {len(matches)}"
            )
        parent = matches[0]
        category = metadata["ood_category"]
        factor = FACTOR_MAP[category]
        source_frame = WORKSPACE / "wan22_pendulum_pipeline" / metadata["first_frame"]
        suffix = source_frame.suffix.lower()
        destination = (
            PHYSICS_VIDEO_ASSETS / "pendulum" / metadata["sample_id"] / "canonical"
            / f"first_frame{suffix}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_frame, destination)
        frame_asset = os.path.relpath(destination, MANIFEST.parent)
        prompt = parent["text"]["prompt"]
        appearance = copy.deepcopy(parent["appearance"])
        appearance.update({
            "ood_category": category,
            "changed_factor": factor,
            "appearance_annotation_source": "user-provided first frame",
        })
        case = {
            "schema_version": "1.0",
            "case_id": metadata["sample_id"],
            "scene_id": "pendulum",
            "view_a_split": "test_ood1",
            "physical_parameters": copy.deepcopy(parent["physical_parameters"]),
            "appearance": appearance,
            "temporal": copy.deepcopy(parent["temporal"]),
            "text": {
                "description": f"单摆外观 OOD：仅改变 {factor}，物理参数与父 ID case 相同。",
                "prompt": prompt,
            },
            "assets": {
                "first_frame": frame_asset,
                "reference_video": None,
                "physics_reference_video": parent["assets"]["physics_reference_video"],
                "subject_mask": None,
            },
            "has_real_reference_video": False,
            "ood": {"level": "ood1", "factors": [factor]},
            "provenance": {
                "source_kind": "synthetic_first_frame",
                "parent_case_id": parent["case_id"],
                "generator": {
                    "kind": "user_provided_appearance_ood_first_frame",
                    "source_metadata": str(metadata_path.relative_to(WORKSPACE)),
                },
            },
            "input_views": {
                "t2v": {"prompt": prompt},
                "i2v": {"prompt": prompt, "first_frame": frame_asset},
                "ti2v": {"prompt": prompt, "first_frame": frame_asset},
            },
        }
        imported.append(case)
        audit.append({
            "case_id": case["case_id"],
            "ood_factor": factor,
            "parent_case_id": parent["case_id"],
            "source_first_frame": str(source_frame),
            "destination_first_frame": str(destination.relative_to(ROOT)),
            "has_real_reference_video": False,
            "physics_reference_video": case["assets"]["physics_reference_video"],
        })

    merged = sorted(retained + imported, key=lambda case: case["case_id"])
    write_jsonl(MANIFEST, merged)
    write_jsonl(AUDIT, audit)
    write_json(VIEW_A, build_view_a(merged))
    write_json(VIEW_B, build_view_b(merged, groups=5, seed=42))
    print(f"imported={len(imported)} total_cases={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
