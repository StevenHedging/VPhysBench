#!/usr/bin/env python3
"""Build Dataset 11.0.0 symbolic physics metadata from immutable V10."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from physbench.datasets import load_dataset
from physbench.io import (
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)
from scripts.build_dataset_asset_lock import rebuild_asset_lock


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
BASE_RELEASE_ROOT = DATASETS_ROOT / "releases" / "10.0.0"
OUTPUT_RELEASE_ROOT = DATASETS_ROOT / "releases" / "11.0.0"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "11.0.0"
OUTPUT_DATASET_ID = "physics_video_six_scene_v11"
OUTPUT_RELEASE = "11.0.0"
EXPECTED_CASES = 799
EXPECTED_LOCKED_ASSETS = 6038
EXPECTED_BASE_DIGEST = (
    "199f84da728b45208c06fc8aabd2cfe9ecf1dfe3ce7f622725df3a102f03a335"
)
RUNTIME_ENTRIES = {
    "README.md",
    "dataset.json",
    "release.json",
    "cases.jsonl",
    "assets.lock.json",
    "scenes",
    "views",
}


SYMBOLS: dict[tuple[str, str], str] = {
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

AUDIT_ONLY = {
    ("collision_1d", "striker_initial_velocity"),
    ("inclined_plane_slide", "calibration_length"),
    ("inclined_plane_slide", "friction_force"),
    ("inclined_plane_slide", "theoretical_acceleration"),
    ("pendulum", "pendulum_length"),
}

FORBIDDEN_PROMPT_WORDS = {
    "background",
    "camera",
    "viewpoint",
    "laboratory",
    "crop",
    "photogate",
    "resolution",
    "capture",
    "frame",
    "color",
    "colour",
}


@dataclass(frozen=True)
class PhysicsWrite:
    case_id: str
    relative_path: str
    absolute_path: Path
    payload: bytes


def _case_directory(case: dict[str, Any]) -> Path:
    directories: set[Path] = set()
    for role in ("first_frame", "reference_video", "physics_reference_video"):
        value = case.get("assets", {}).get(role)
        if value is None:
            continue
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 5
            or path.parts[0] != "assets"
            or path.parts[1] != case.get("scene_id")
            or path.parts[-2] != "canonical"
        ):
            raise ValueError(
                f"case {case.get('case_id')} assets.{role} does not identify "
                "assets/<scene>/<case>/canonical/<file>"
            )
        directories.add(path.parent.parent)
    if len(directories) != 1:
        raise ValueError(
            f"case {case.get('case_id')} does not resolve to one Case directory"
        )
    return next(iter(directories))


def _velocity(case: dict[str, Any], name: str) -> float:
    quantity = case["physics"].get(name)
    if quantity is None:
        raise ValueError(f"case {case['case_id']} lacks collision velocity {name}")
    value = quantity.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"case {case['case_id']} has invalid collision velocity {name}")
    return float(value)


def _direction_error(case: dict[str, Any]) -> ValueError:
    return ValueError(
        f"case {case['case_id']} signed velocity contradicts audited direction"
    )


def _validate_collision_direction(case: dict[str, Any]) -> str:
    appearance = case.get("appearance", {})
    structure = appearance.get("collision_structure")
    striker = appearance.get("striker_ball_index")
    opposing = appearance.get("opposing_striker_ball_index")
    alias = _velocity(case, "striker_initial_velocity")
    v1 = _velocity(case, "ball_1_initial_velocity")
    v2 = _velocity(case, "ball_2_initial_velocity")

    if structure == "two_ball_single_incident":
        valid = (
            striker == 2
            and opposing is None
            and v1 == 0
            and v2 < 0
            and math.isclose(alias, v2, rel_tol=0, abs_tol=1e-12)
        )
    elif structure == "two_ball_opposed_incident":
        valid = (
            striker == 1
            and opposing == 2
            and v1 > 0
            and v2 < 0
            and math.isclose(alias, v1, rel_tol=0, abs_tol=1e-12)
        )
    elif structure == "three_ball_single_incident":
        v3 = _velocity(case, "ball_3_initial_velocity")
        valid = (
            striker == 1
            and opposing is None
            and v1 > 0
            and v2 == 0
            and v3 == 0
            and math.isclose(alias, v1, rel_tol=0, abs_tol=1e-12)
        )
    else:
        raise ValueError(
            f"case {case['case_id']} has unknown collision structure {structure!r}"
        )
    if not valid:
        raise _direction_error(case)
    return structure


def _collision_prompt(structure: str) -> str:
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
    raise AssertionError(structure)


def _prompt(case: dict[str, Any], collision_structure: str | None) -> str:
    scene = case["scene_id"]
    if scene == "collision_1d":
        if collision_structure is None:
            raise AssertionError("collision structure was not audited")
        return _collision_prompt(collision_structure)
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
    raise ValueError(f"case {case['case_id']} has unsupported Scene {scene}")


def symbol_in_prompt(symbol: str, prompt: str) -> bool:
    boundary = r"[A-Za-z0-9_]"
    return re.search(
        rf"(?<!{boundary}){re.escape(symbol)}(?!{boundary})", prompt
    ) is not None


def validate_prompt(case: dict[str, Any]) -> None:
    prompt = case["text"]["prompt"]
    lower = prompt.lower()
    present_symbols = [
        quantity["symbol"]
        for quantity in case["physics"].values()
        if symbol_in_prompt(quantity["symbol"], prompt)
    ]
    for name, quantity in case["physics"].items():
        present = symbol_in_prompt(quantity["symbol"], prompt)
        if present != quantity["annotated"]:
            role = "independent" if quantity["annotated"] else "audit-only"
            raise ValueError(
                f"case {case['case_id']} {role} symbol mismatch for physics.{name}"
            )
    scrubbed = prompt
    for symbol in sorted(present_symbols, key=len, reverse=True):
        scrubbed = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
            "",
            scrubbed,
        )
    if re.search(r"\d", scrubbed):
        raise ValueError(f"case {case['case_id']} prompt contains a numeric value")
    forbidden = sorted(word for word in FORBIDDEN_PROMPT_WORDS if word in lower)
    if forbidden:
        raise ValueError(
            f"case {case['case_id']} prompt contains forbidden hints: {forbidden}"
        )


def migrate_case(case: dict[str, Any]) -> dict[str, Any]:
    scene = case.get("scene_id")
    collision_structure = (
        _validate_collision_direction(case) if scene == "collision_1d" else None
    )
    output = copy.deepcopy(case)
    output["schema_version"] = "5.0"
    seen_symbols: set[str] = set()
    for name, quantity in output["physics"].items():
        key = (scene, name)
        if key not in SYMBOLS:
            raise ValueError(f"case {case['case_id']} has unknown physics parameter {key}")
        value = quantity.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"case {case['case_id']} physics.{name} is not finite")
        if value < 0:
            if scene != "collision_1d" or name not in {
                "ball_2_initial_velocity",
                "striker_initial_velocity",
            }:
                raise ValueError(
                    f"case {case['case_id']} has unexpected negative physics.{name}"
                )
            quantity["value"] = abs(value)
        symbol = SYMBOLS[key]
        if symbol in seen_symbols:
            raise ValueError(f"case {case['case_id']} has duplicate symbol {symbol}")
        seen_symbols.add(symbol)
        quantity["symbol"] = symbol
        if key in AUDIT_ONLY:
            quantity["annotated"] = False
    output["text"] = {
        "schema_version": "1.0",
        "prompt": _prompt(output, collision_structure),
        "language": "en",
        "annotation_source": "symbolic_physics_prompt_v1",
    }
    output["assets"]["physics_annotation"] = (
        _case_directory(output) / "physics.v11.json"
    ).as_posix()
    validate_prompt(output)
    return output


def migrate_scene(
    scene: dict[str, Any], migrated_cases: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    cases = list(migrated_cases)
    if any(case["scene_id"] != scene["scene_id"] for case in cases):
        raise ValueError(f"scene migration received a foreign Case: {scene['scene_id']}")
    output = copy.deepcopy(scene)
    output["structured_physics_parameters"] = sorted(
        {
            name
            for case in cases
            for name, quantity in case["physics"].items()
            if quantity["annotated"]
        }
    )
    output["non_conditionable_physics_parameters"] = sorted(
        {
            name
            for case in cases
            for name, quantity in case["physics"].items()
            if not quantity["annotated"]
        }
    )
    return output


def _physics_payload(case: dict[str, Any]) -> bytes:
    document = {
        "schema_version": "2.0",
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "physics": copy.deepcopy(case["physics"]),
    }
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def prepare_v11(
    cases: Iterable[dict[str, Any]],
    scenes: Iterable[dict[str, Any]],
    datasets_root: Path = DATASETS_ROOT,
) -> tuple[list[PhysicsWrite], list[dict[str, Any]]]:
    base_cases = list(cases)
    migrated = [migrate_case(case) for case in base_cases]
    known_scenes = {scene["scene_id"] for scene in scenes}
    actual_scenes = {case["scene_id"] for case in migrated}
    if actual_scenes != known_scenes:
        raise ValueError(
            f"Case/Scene catalog mismatch: cases={sorted(actual_scenes)} "
            f"catalog={sorted(known_scenes)}"
        )
    paths: set[str] = set()
    writes: list[PhysicsWrite] = []
    for case in migrated:
        relative = case["assets"]["physics_annotation"]
        if relative in paths:
            raise ValueError(f"multiple Cases resolve to physics document {relative}")
        paths.add(relative)
        writes.append(
            PhysicsWrite(
                case_id=case["case_id"],
                relative_path=relative,
                absolute_path=(datasets_root / relative).resolve(),
                payload=_physics_payload(case),
            )
        )
    return writes, migrated


def materialize_physics_documents(writes: Iterable[PhysicsWrite]) -> None:
    pending: list[PhysicsWrite] = []
    for item in writes:
        if item.absolute_path.exists():
            if not item.absolute_path.is_file() or item.absolute_path.read_bytes() != item.payload:
                raise ValueError(f"conflicting physics document: {item.absolute_path}")
        else:
            pending.append(item)
    for item in pending:
        item.absolute_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{item.absolute_path.name}.", dir=item.absolute_path.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(item.payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(item.absolute_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": "5.0",
        "dataset_id": OUTPUT_DATASET_ID,
        "release": OUTPUT_RELEASE,
        "cases": "cases.jsonl",
        "asset_root": "../..",
        "asset_lock": "assets.lock.json",
        "release_manifest": "release.json",
        "scene_catalog": "scenes",
        "views": {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        },
    }


def _readme() -> str:
    return """# Physics Video Benchmark Dataset 11.0.0

This is the minimal immutable runtime snapshot for
`physics_video_six_scene_v11`. It preserves all 799 V10 Cases, media assets,
Views, and provenance facts while upgrading structured physics metadata.

Every quantity now has a stable `symbol`. Independent quantities use
`annotated=true` and their symbols appear in the English Case prompt without
numeric values. Derived, calibration, auxiliary, and duplicate-alias
quantities remain available to evaluators with `annotated=false`. Stored scalar
values are finite non-negative magnitudes; motion direction is expressed in the
prompt.

Each Case references a locked `physics.v11.json` document whose physics object
must equal inline `case.physics`. Dataset 10.0.0 and its `physics.json` files
remain immutable. Migration and validation evidence is stored separately at
`datasets/provenance/releases/11.0.0/`.
"""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _symbol_table_digest() -> str:
    return canonical_sha256(
        [
            {"scene_id": scene, "parameter": name, "symbol": symbol}
            for (scene, name), symbol in sorted(SYMBOLS.items())
        ]
    )


def _migration_evidence(
    *,
    base_cases: list[dict[str, Any]],
    output_cases: list[dict[str, Any]],
    lock: dict[str, Any],
    release_manifest: dict[str, Any],
) -> dict[str, Any]:
    value_changes: list[dict[str, Any]] = []
    flag_changes: list[dict[str, Any]] = []
    prompt_changes: list[dict[str, Any]] = []
    for old, new in zip(base_cases, output_cases, strict=True):
        if old["case_id"] != new["case_id"]:
            raise ValueError("V11 changed ordered Case identity")
        for name, old_quantity in old["physics"].items():
            new_quantity = new["physics"][name]
            if old_quantity["value"] != new_quantity["value"]:
                value_changes.append(
                    {
                        "case_id": old["case_id"],
                        "parameter": name,
                        "old_value": old_quantity["value"],
                        "new_value": new_quantity["value"],
                    }
                )
            if old_quantity["annotated"] != new_quantity["annotated"]:
                flag_changes.append(
                    {
                        "case_id": old["case_id"],
                        "parameter": name,
                        "old_annotated": old_quantity["annotated"],
                        "new_annotated": new_quantity["annotated"],
                    }
                )
        prompt_changes.append(
            {
                "case_id": old["case_id"],
                "old_sha256": _sha256_text(old["text"]["prompt"]),
                "new_sha256": _sha256_text(new["text"]["prompt"]),
            }
        )
    if len(value_changes) != 494:
        raise ValueError(f"expected 494 value changes, found {len(value_changes)}")
    if len(flag_changes) != 715:
        raise ValueError(f"expected 715 flag changes, found {len(flag_changes)}")
    if len(prompt_changes) != EXPECTED_CASES:
        raise ValueError("V11 prompt evidence is incomplete")

    base_release = load_json(BASE_RELEASE_ROOT / "release.json")
    return {
        "schema_version": "1.0",
        "migration": "symbolic_physics_v11",
        "base": {
            "dataset_id": base_release["dataset_id"],
            "release": base_release["release"],
            "dataset_digest": base_release["dataset_digest"],
            "asset_files_digest": base_release["asset_files_digest"],
        },
        "output": {
            "dataset_id": release_manifest["dataset_id"],
            "release": release_manifest["release"],
            "dataset_digest": release_manifest["dataset_digest"],
            "asset_files_digest": release_manifest["asset_files_digest"],
        },
        "counts": {
            "cases": len(output_cases),
            "physics_documents": len(output_cases),
            "locked_assets": len(lock["files"]),
            "value_changes": len(value_changes),
            "annotation_flag_changes": len(flag_changes),
            "prompt_changes": len(prompt_changes),
            "media_changes": 0,
        },
        "symbol_table_sha256": _symbol_table_digest(),
        "view_digests": {
            path.name: canonical_sha256(load_json(path))
            for path in sorted((BASE_RELEASE_ROOT / "views").glob("*.json"))
        },
        "value_changes": value_changes,
        "annotation_flag_changes": flag_changes,
        "prompt_changes": prompt_changes,
    }


def _preflight_base() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = load_dataset(
        BASE_RELEASE_ROOT / "dataset.json", check_asset_hashes=True
    )
    if snapshot.dataset_id != "physics_video_six_scene_v10":
        raise ValueError(f"unexpected V10 Dataset ID: {snapshot.dataset_id}")
    if snapshot.digest != EXPECTED_BASE_DIGEST:
        raise ValueError(f"unexpected V10 Dataset digest: {snapshot.digest}")
    cases = load_jsonl(BASE_RELEASE_ROOT / "cases.jsonl")
    scenes = [
        load_json(path)
        for path in sorted((BASE_RELEASE_ROOT / "scenes").glob("*.json"))
    ]
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} V10 Cases, found {len(cases)}")
    return cases, scenes


def _check_existing_release(writes: Iterable[PhysicsWrite]) -> dict[str, Any]:
    for item in writes:
        if item.absolute_path.exists() and item.absolute_path.read_bytes() != item.payload:
            raise ValueError(f"conflicting physics document: {item.absolute_path}")
    if not OUTPUT_RELEASE_ROOT.is_dir():
        return {
            "release": "absent",
            "missing_physics_documents": sum(
                not item.absolute_path.exists() for item in writes
            ),
        }
    from scripts.validate_dataset_v11 import validate_v11

    return validate_v11(OUTPUT_RELEASE_ROOT / "dataset.json")


def build_release(*, check: bool = False) -> dict[str, Any]:
    base_cases, base_scenes = _preflight_base()
    writes, output_cases = prepare_v11(base_cases, base_scenes)
    if len(writes) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} physics documents")
    if check:
        return _check_existing_release(writes)
    if OUTPUT_RELEASE_ROOT.exists():
        raise FileExistsError(
            f"V11 release already exists; use --check: {OUTPUT_RELEASE_ROOT}"
        )

    migrated_by_scene: dict[str, list[dict[str, Any]]] = {}
    for case in output_cases:
        migrated_by_scene.setdefault(case["scene_id"], []).append(case)
    output_scenes = [
        migrate_scene(scene, migrated_by_scene[scene["scene_id"]])
        for scene in base_scenes
    ]

    materialize_physics_documents(writes)
    stage = Path(
        tempfile.mkdtemp(prefix=".11.0.0.build-", dir=OUTPUT_RELEASE_ROOT.parent)
    )
    published = False
    try:
        shutil.copytree(BASE_RELEASE_ROOT / "views", stage / "views")
        (stage / "scenes").mkdir()
        for scene in output_scenes:
            write_json(stage / "scenes" / f"{scene['scene_id']}.json", scene)
        write_jsonl(stage / "cases.jsonl", output_cases)
        write_json(stage / "dataset.json", _descriptor())
        (stage / "README.md").write_text(_readme(), encoding="utf-8")
        lock_path, release_manifest = rebuild_asset_lock(stage / "dataset.json")
        lock = load_json(lock_path)
        if len(lock["files"]) != EXPECTED_LOCKED_ASSETS:
            raise ValueError(
                f"expected {EXPECTED_LOCKED_ASSETS} locked assets, "
                f"found {len(lock['files'])}"
            )
        snapshot = load_dataset(stage / "dataset.json", check_asset_hashes=True)
        if snapshot.digest != release_manifest["dataset_digest"]:
            raise ValueError("staged V11 Dataset digest mismatch")
        if {path.name for path in stage.iterdir()} != RUNTIME_ENTRIES:
            raise ValueError("staged V11 release is not the minimal seven-entry tree")
        evidence = _migration_evidence(
            base_cases=base_cases,
            output_cases=output_cases,
            lock=lock,
            release_manifest=release_manifest,
        )
        stage.replace(OUTPUT_RELEASE_ROOT)
        published = True
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(PROVENANCE_ROOT / "migration.json", evidence)

        from scripts.validate_dataset_v11 import validate_v11

        validation = validate_v11(
            OUTPUT_RELEASE_ROOT / "dataset.json", write_report=True
        )
        return {
            "cases": len(output_cases),
            "physics": len(writes),
            "locked_assets": len(lock["files"]),
            "negatives_normalized": evidence["counts"]["value_changes"],
            "flags_demoted": evidence["counts"]["annotation_flag_changes"],
            "prompts": evidence["counts"]["prompt_changes"],
            "dataset_digest": validation["dataset_digest"],
        }
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build_release(check=args.check)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
