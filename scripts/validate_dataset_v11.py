#!/usr/bin/env python3
"""Independently validate immutable Dataset 11.0.0 and its evidence."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

from physbench.datasets import load_dataset
from physbench.io import canonical_sha256, load_json, load_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
V10_RELEASE_ROOT = DATASETS_ROOT / "releases" / "10.0.0"
V11_RELEASE_ROOT = DATASETS_ROOT / "releases" / "11.0.0"
V11_DATASET = V11_RELEASE_ROOT / "dataset.json"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "11.0.0"
EXPECTED_RELEASE_ENTRIES = {
    "README.md",
    "dataset.json",
    "release.json",
    "cases.jsonl",
    "assets.lock.json",
    "scenes",
    "views",
}

SYMBOLS = {
    ("collision_1d", "ball_1_initial_velocity"): "v_1",
    ("collision_1d", "ball_1_mass"): "m_1",
    ("collision_1d", "ball_1_radius"): "r_1",
    ("collision_1d", "ball_2_initial_velocity"): "v_2",
    ("collision_1d", "ball_2_mass"): "m_2",
    ("collision_1d", "ball_2_radius"): "r_2",
    ("collision_1d", "ball_3_initial_velocity"): "v_3",
    ("collision_1d", "ball_3_mass"): "m_3",
    ("collision_1d", "ball_3_radius"): "r_3",
    ("collision_1d", "striker_initial_velocity"): "v_s",
    ("inclined_plane_slide", "block_length"): "l",
    ("inclined_plane_slide", "block_mass"): "m",
    ("inclined_plane_slide", "calibration_length"): "l_cal",
    ("inclined_plane_slide", "friction_force"): "F_f",
    ("inclined_plane_slide", "gravity_acceleration"): "g",
    ("inclined_plane_slide", "incline_angle"): "θ",
    ("inclined_plane_slide", "initial_velocity"): "v_0",
    ("inclined_plane_slide", "kinetic_friction_coefficient"): "μ_k",
    ("inclined_plane_slide", "theoretical_acceleration"): "a_th",
    ("parabolic_motion", "ball_mass"): "m",
    ("parabolic_motion", "ball_radius"): "r",
    ("parabolic_motion", "initial_horizontal_velocity"): "v_0",
    ("parabolic_motion", "launch_height"): "h",
    ("parabolic_motion", "photogate_block_time"): "Δt_gate",
    ("parabolic_motion", "photogate_distance_before_launch"): "d_gate",
    ("parabolic_motion", "ramp_angle"): "α",
    ("parabolic_motion", "release_distance"): "s_release",
    ("pendulum", "bob_mass"): "m",
    ("pendulum", "bob_radius"): "r",
    ("pendulum", "initial_angle"): "θ_0",
    ("pendulum", "pendulum_length"): "l",
    ("pendulum", "string_length"): "l_s",
    ("push_bottle", "bottle_height"): "h",
    ("push_bottle", "bottle_mass"): "m",
    ("push_bottle", "mean_applied_force"): "F_mean",
    ("push_bottle", "peak_applied_force"): "F_peak",
    ("uniform_circular_motion", "angular_velocity"): "ω",
    ("uniform_circular_motion", "angular_velocity_rad_s"): "ω_rad",
    ("uniform_circular_motion", "object_1_orbit_radius"): "r_1",
    ("uniform_circular_motion", "object_2_orbit_radius"): "r_2",
}

DEMOTED = {
    ("collision_1d", "striker_initial_velocity"),
    ("inclined_plane_slide", "calibration_length"),
    ("inclined_plane_slide", "friction_force"),
    ("inclined_plane_slide", "theoretical_acceleration"),
    ("pendulum", "pendulum_length"),
}


def _expected_prompt(case: dict[str, Any]) -> str:
    scene = case["scene_id"]
    if scene == "collision_1d":
        structure = case["appearance"]["collision_structure"]
        if structure == "two_ball_single_incident":
            return (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, "
                "while the right ball has mass m_2, radius r_2, and initial speed "
                "v_2. The left ball is initially stationary, the right ball moves "
                "left, and they undergo a one-dimensional central collision."
            )
        if structure == "two_ball_opposed_incident":
            return (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, "
                "while the right ball has mass m_2, radius r_2, and initial speed "
                "v_2. The left ball moves right, the right ball moves left, and they "
                "undergo a one-dimensional central collision."
            )
        if structure == "three_ball_single_incident":
            return (
                "The left ball has mass m_1, radius r_1, and initial speed v_1, the "
                "middle ball has mass m_2, radius r_2, and initial speed v_2, and the "
                "right ball has mass m_3, radius r_3, and initial speed v_3. The left "
                "ball moves right while the middle and right balls are initially "
                "stationary, and the left ball first undergoes a one-dimensional "
                "central collision with the middle ball."
            )
        raise ValueError(f"unknown collision structure: {structure}")
    if scene == "inclined_plane_slide":
        return (
            "A block of mass m and length l starts from rest and slides down an "
            "incline of angle θ. The kinetic friction coefficient between the "
            "block and incline is μ_k, and the gravitational acceleration is g."
        )
    if scene == "parabolic_motion":
        return (
            "A ball of mass m and radius r is launched horizontally to the left "
            "with initial speed v_0 from a vertical height h, then follows a "
            "downward parabolic path."
        )
    if scene == "pendulum":
        if "bob_mass" in case["physics"]:
            return (
                "A pendulum bob of mass m and radius r is released from rest at "
                "initial angle θ_0 on a string of length l_s, then swings back and "
                "forth about the fixed pivot."
            )
        return (
            "A pendulum bob of radius r is released from rest at initial angle "
            "θ_0 on a string of length l_s, then swings back and forth about the "
            "fixed pivot."
        )
    if scene == "push_bottle":
        return (
            "An upright bottle of mass m and height h is pushed near its top by a "
            "force with mean magnitude F_mean and peak magnitude F_peak, then "
            "tips and falls onto its side."
        )
    if scene == "uniform_circular_motion":
        if "object_2_orbit_radius" in case["physics"]:
            return (
                "Two objects follow circular orbits of radii r_1 and r_2 with angular "
                "speed ω."
            )
        return "An object follows a circular orbit of radius r_1 with angular speed ω."
    raise ValueError(f"unsupported Scene: {scene}")


def _expected_symbol_digest() -> str:
    return canonical_sha256(
        [
            {"scene_id": scene, "parameter": name, "symbol": symbol}
            for (scene, name), symbol in sorted(SYMBOLS.items())
        ]
    )


def validate_v11(
    dataset_path: Path = V11_DATASET,
    *,
    write_report: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    release_root = dataset_path.parent
    if {path.name for path in release_root.iterdir()} != EXPECTED_RELEASE_ENTRIES:
        raise ValueError("V11 release tree is not the minimal seven-entry snapshot")
    descriptor = load_json(dataset_path)
    if descriptor.get("schema_version") != "5.0":
        raise ValueError("V11 Dataset must use schema 5.0")
    if descriptor.get("dataset_id") != "physics_video_six_scene_v11":
        raise ValueError("unexpected V11 Dataset ID")
    if descriptor.get("release") != "11.0.0":
        raise ValueError("unexpected V11 release identity")

    old_cases = load_jsonl(V10_RELEASE_ROOT / "cases.jsonl")
    new_cases = load_jsonl(release_root / "cases.jsonl")
    if len(old_cases) != 799 or len(new_cases) != 799:
        raise ValueError("V10 and V11 must each contain 799 Cases")
    asset_root = (release_root / descriptor["asset_root"]).resolve()
    physics_paths: set[str] = set()
    value_changes = 0
    flag_changes = 0
    for old, new in zip(old_cases, new_cases, strict=True):
        if old["case_id"] != new["case_id"] or new["schema_version"] != "5.0":
            raise ValueError("V11 changed Case identity or has wrong schema")
        old_other = copy.deepcopy(old)
        new_other = copy.deepcopy(new)
        for value in (old_other, new_other):
            value.pop("schema_version")
            value.pop("text")
            value.pop("physics")
            value["assets"].pop("physics_annotation")
        if old_other != new_other:
            raise ValueError(f"V11 changed a non-symbolic Case fact: {new['case_id']}")
        expected_text = {
            "schema_version": "1.0",
            "prompt": _expected_prompt(old),
            "language": "en",
            "annotation_source": "symbolic_physics_prompt_v1",
        }
        if new["text"] != expected_text:
            raise ValueError(f"V11 prompt mismatch: {new['case_id']}")
        if set(old["physics"]) != set(new["physics"]):
            raise ValueError(f"V11 physics keys changed: {new['case_id']}")
        seen_symbols: set[str] = set()
        for name, old_quantity in old["physics"].items():
            quantity = new["physics"][name]
            if set(quantity) != {"value", "unit", "annotated", "symbol"}:
                raise ValueError(f"invalid V11 quantity fields: {new['case_id']}/{name}")
            symbol = SYMBOLS.get((new["scene_id"], name))
            if quantity["symbol"] != symbol or symbol in seen_symbols:
                raise ValueError(f"invalid or duplicate symbol: {new['case_id']}/{name}")
            seen_symbols.add(symbol)
            if quantity["unit"] != old_quantity["unit"]:
                raise ValueError(f"V11 changed unit: {new['case_id']}/{name}")
            expected_value = abs(old_quantity["value"])
            if quantity["value"] != expected_value:
                raise ValueError(f"V11 changed value incorrectly: {new['case_id']}/{name}")
            if old_quantity["value"] != expected_value:
                if (new["scene_id"], name) not in {
                    ("collision_1d", "ball_2_initial_velocity"),
                    ("collision_1d", "striker_initial_velocity"),
                }:
                    raise ValueError("V11 normalized an unauthorized negative value")
                value_changes += 1
            expected_flag = old_quantity["annotated"] and (
                new["scene_id"], name
            ) not in DEMOTED
            if quantity["annotated"] != expected_flag:
                raise ValueError(f"V11 annotation flag mismatch: {new['case_id']}/{name}")
            if old_quantity["annotated"] != expected_flag:
                flag_changes += 1
            if (
                isinstance(quantity["value"], bool)
                or not isinstance(quantity["value"], (int, float))
                or not math.isfinite(float(quantity["value"]))
                or quantity["value"] < 0
            ):
                raise ValueError(f"V11 quantity is not a non-negative finite number")

        relative = new["assets"].get("physics_annotation")
        if not isinstance(relative, str) or not relative.endswith("/physics.v11.json"):
            raise ValueError(f"V11 physics path mismatch: {new['case_id']}")
        if relative in physics_paths:
            raise ValueError(f"duplicate V11 physics path: {relative}")
        physics_paths.add(relative)
        document = load_json(asset_root / relative)
        if document != {
            "schema_version": "2.0",
            "case_id": new["case_id"],
            "scene_id": new["scene_id"],
            "physics": new["physics"],
        }:
            raise ValueError(f"invalid V11 physics document: {relative}")
    if value_changes != 494 or flag_changes != 715 or len(physics_paths) != 799:
        raise ValueError("V11 migration counts do not match the audited contract")

    for old_path in sorted((V10_RELEASE_ROOT / "views").glob("*.json")):
        if old_path.read_bytes() != (release_root / "views" / old_path.name).read_bytes():
            raise ValueError(f"V11 changed View bytes: {old_path.name}")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for case in new_cases:
        by_scene.setdefault(case["scene_id"], []).append(case)
    for old_path in sorted((V10_RELEASE_ROOT / "scenes").glob("*.json")):
        old_scene = load_json(old_path)
        new_scene = load_json(release_root / "scenes" / old_path.name)
        old_other = copy.deepcopy(old_scene)
        new_other = copy.deepcopy(new_scene)
        for scene in (old_other, new_other):
            scene.pop("structured_physics_parameters")
            scene.pop("non_conditionable_physics_parameters")
        if old_other != new_other:
            raise ValueError(f"V11 changed non-classification Scene facts: {old_path.name}")
        cases = by_scene[new_scene["scene_id"]]
        expected_true = sorted({
            name for case in cases for name, q in case["physics"].items() if q["annotated"]
        })
        expected_false = sorted({
            name for case in cases for name, q in case["physics"].items() if not q["annotated"]
        })
        if new_scene["structured_physics_parameters"] != expected_true:
            raise ValueError(f"V11 structured parameter list mismatch: {old_path.name}")
        if new_scene["non_conditionable_physics_parameters"] != expected_false:
            raise ValueError(f"V11 audit parameter list mismatch: {old_path.name}")

    old_lock = load_json(V10_RELEASE_ROOT / "assets.lock.json")
    new_lock = load_json(release_root / "assets.lock.json")
    old_nonphysics = {
        item["path"]: item
        for item in old_lock["files"]
        if not item["path"].endswith("/physics.json")
    }
    new_nonphysics = {
        item["path"]: item
        for item in new_lock["files"]
        if not item["path"].endswith("/physics.v11.json")
    }
    if len(old_nonphysics) != 5239 or old_nonphysics != new_nonphysics:
        raise ValueError("V11 changed a non-physics locked asset")
    if len(new_lock["files"]) != 6038:
        raise ValueError("V11 must lock exactly 6,038 assets")

    migration = load_json(PROVENANCE_ROOT / "migration.json")
    expected_counts = {
        "cases": 799,
        "physics_documents": 799,
        "locked_assets": 6038,
        "value_changes": 494,
        "annotation_flag_changes": 715,
        "prompt_changes": 799,
        "media_changes": 0,
    }
    if migration.get("counts") != expected_counts:
        raise ValueError("V11 migration evidence counts mismatch")
    if migration.get("symbol_table_sha256") != _expected_symbol_digest():
        raise ValueError("V11 symbol table evidence mismatch")
    expected_view_digests = {
        path.name: canonical_sha256(load_json(path))
        for path in sorted((V10_RELEASE_ROOT / "views").glob("*.json"))
    }
    if migration.get("view_digests") != expected_view_digests:
        raise ValueError("V11 View digest evidence mismatch")

    snapshot = load_dataset(dataset_path, check_asset_hashes=True)
    release = load_json(release_root / "release.json")
    if snapshot.digest != release["dataset_digest"]:
        raise ValueError("V11 release manifest Dataset digest mismatch")
    if migration["output"]["dataset_digest"] != snapshot.digest:
        raise ValueError("V11 migration output Dataset digest mismatch")
    report = {
        "schema_version": "1.0",
        "status": "valid",
        "dataset_id": snapshot.dataset_id,
        "release": descriptor["release"],
        "dataset_digest": snapshot.digest,
        "asset_files_digest": new_lock["files_digest"],
        "symbol_table_sha256": _expected_symbol_digest(),
        "counts": expected_counts,
        "checks": {
            "minimal_release_tree": True,
            "v10_non_symbolic_facts_preserved": True,
            "symbolic_prompts_exact": True,
            "physics_documents_equal_inline": True,
            "views_byte_identical": True,
            "non_physics_assets_preserved": True,
            "asset_sha256_verified": True,
        },
    }
    if write_report:
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(PROVENANCE_ROOT / "validation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=V11_DATASET)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = validate_v11(args.dataset, write_report=args.write_report)
    print(
        f"status={report['status']} cases={report['counts']['cases']} "
        f"physics={report['counts']['physics_documents']} "
        f"locked_assets={report['counts']['locked_assets']} "
        f"dataset_digest={report['dataset_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
