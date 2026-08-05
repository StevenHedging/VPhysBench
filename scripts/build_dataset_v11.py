#!/usr/bin/env python3
"""Build Dataset 11.0.0 symbolic physics metadata from immutable V10."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
BASE_RELEASE_ROOT = DATASETS_ROOT / "releases" / "10.0.0"
OUTPUT_RELEASE_ROOT = DATASETS_ROOT / "releases" / "11.0.0"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "11.0.0"
OUTPUT_DATASET_ID = "physics_video_six_scene_v11"
OUTPUT_RELEASE = "11.0.0"
EXPECTED_CASES = 799
EXPECTED_LOCKED_ASSETS = 6038


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
