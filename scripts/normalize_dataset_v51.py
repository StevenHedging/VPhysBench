#!/usr/bin/env python3
"""Normalize ball specifications, prompts, splits, and asset paths for 5.1.0.

Release 5.0.0 remains immutable.  This migration:

* applies the user-confirmed steel-ball specifications;
* recomputes velocities that were derived from a corrected diameter;
* writes collision prompts from each case's actual left-to-right state;
* marks non-conditionable parabolic setup measurements as unannotated;
* groups identical parabolic physical signatures into one View A partition;
* gives every case a concise, physics-bearing asset directory; and
* materializes the new asset paths as hard links to the frozen media bytes.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_dataset_asset_lock import rebuild_asset_lock  # noqa: E402
from physbench.datasets import load_dataset  # noqa: E402
from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)


DATA_ROOT = ROOT / "datasets" / "physics_video"
ASSET_ROOT = DATA_ROOT / "assets"
RELEASES_ROOT = DATA_ROOT / "releases"
BASE_ROOT = RELEASES_ROOT / "5.0.0"
OUTPUT_ROOT = RELEASES_ROOT / "5.1.0"
DATASET_ID = "physics_video_six_scene_v5p1"
RELEASE = "5.1.0"

PARABOLIC_AUXILIARY_FIELDS = {
    "photogate_block_time",
    "photogate_distance_before_launch",
    "ramp_angle",
    "release_distance",
}
ASSET_ROLE_ORDER = (
    "first_frame",
    "reference_video",
    "physics_reference_video",
    "input_video",
    "source_video",
    "subject_mask",
)
GLOBAL_ASSET_ROLES = {"source_archive"}
BACKGROUND_DIRECTORY_TERMS = re.compile(
    r"(?:^|[-_])(?:bg|background|black|white|oil|foam|lab|tabletop)(?:[-_]|$)",
    re.IGNORECASE,
)


BALL_SPECS: dict[str, dict[str, Any]] = {
    "steel_ball_small_d15mm_m14g": {
        "material": "steel",
        "size_class": "small",
        "diameter_m": 0.015,
        "radius_m": 0.0075,
        "mass_kg": 0.014,
        "directory_token": "steelS-d15mm-m14g",
        "prompt_label": "small steel ball (15 mm diameter, 14.00 g)",
    },
    "steel_ball_medium_d20mm_m33p13g": {
        "material": "steel",
        "size_class": "medium",
        "diameter_m": 0.020,
        "radius_m": 0.010,
        "mass_kg": 0.03313,
        "directory_token": "steelM-d20mm-m33p13g",
        "prompt_label": "medium steel ball (20 mm diameter, 33.13 g)",
    },
    "steel_ball_large_d25mm_m64p77g": {
        "material": "steel",
        "size_class": "large",
        "diameter_m": 0.025,
        "radius_m": 0.0125,
        "mass_kg": 0.06477,
        "directory_token": "steelL-d25mm-m64p77g",
        "prompt_label": "large steel ball (25 mm diameter, 64.77 g)",
    },
    "glass_ball_d13p5mm_m3p46g": {
        "material": "glass",
        "size_class": None,
        "diameter_m": 0.0135,
        "radius_m": 0.00675,
        "mass_kg": 0.00346,
        "directory_token": "glass-d13p5mm-m3p46g",
        "prompt_label": "glass ball (13.5 mm diameter, 3.46 g)",
    },
}

BALL_SPEC_ALIASES = {
    "small_steel": "steel_ball_small_d15mm_m14g",
    "steel_ball_d13p7mm_m14g": "steel_ball_small_d15mm_m14g",
    "steel_ball_d15mm_m14g": "steel_ball_small_d15mm_m14g",
    "medium_steel": "steel_ball_medium_d20mm_m33p13g",
    "steel_ball_d18p7mm_m33g": "steel_ball_medium_d20mm_m33p13g",
    "steel_ball_d20mm_m33g": "steel_ball_medium_d20mm_m33p13g",
    "large_steel": "steel_ball_large_d25mm_m64p77g",
    "steel_ball_d23p7mm_m41p6g": "steel_ball_large_d25mm_m64p77g",
    "steel_ball_d25mm_m65g": "steel_ball_large_d25mm_m64p77g",
    "glass_marble": "glass_ball_d13p5mm_m3p46g",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_file_digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def quantity(
    value: float,
    unit: str,
    *,
    annotated: bool = True,
) -> dict[str, Any]:
    return {
        "value": float(value),
        "unit": unit,
        "annotated": annotated,
    }


def canonical_spec(alias: str) -> tuple[str, dict[str, Any]]:
    spec_id = BALL_SPEC_ALIASES.get(alias, alias)
    if spec_id not in BALL_SPECS:
        raise ValueError(f"unrecognized ball specification ID: {alias}")
    return spec_id, BALL_SPECS[spec_id]


def normalize_collision(case: dict[str, Any]) -> dict[str, Any]:
    before = {
        "ball_sequence": copy.deepcopy(case["appearance"]["ball_sequence"]),
        "balls": [],
        "prompt": case["text"]["prompt"],
    }
    sequence = list(case["appearance"]["ball_sequence"])
    ball_count = len(sequence)
    normalized_sequence: list[str] = []
    materials: list[str] = []
    velocity_corrections: list[dict[str, Any]] = []
    supplement = case["case_id"].startswith("collision_supp_20260729_")

    for index, alias in enumerate(sequence, 1):
        spec_id, spec = canonical_spec(alias)
        mass = case["physics"][f"ball_{index}_mass"]
        radius = case["physics"][f"ball_{index}_radius"]
        velocity = case["physics"][f"ball_{index}_initial_velocity"]
        old_diameter = float(radius["value"]) * 2.0
        old_velocity = float(velocity["value"])
        new_velocity = old_velocity

        # Supplement velocities came from diameter / photogate block time.
        # Preserve that independently measured time when correcting diameter.
        if supplement and old_velocity != 0.0:
            new_velocity = (
                old_velocity * float(spec["diameter_m"]) / old_diameter
            )
        case["physics"][f"ball_{index}_mass"] = quantity(
            spec["mass_kg"],
            "kg",
            annotated=bool(mass["annotated"]),
        )
        case["physics"][f"ball_{index}_radius"] = quantity(
            spec["radius_m"],
            "m",
            annotated=bool(radius["annotated"]),
        )
        case["physics"][f"ball_{index}_initial_velocity"] = quantity(
            new_velocity,
            "m/s",
            annotated=bool(velocity["annotated"]),
        )
        before["balls"].append({
            "index": index,
            "spec_id": alias,
            "mass_kg": float(mass["value"]),
            "radius_m": float(radius["value"]),
            "initial_velocity_m_per_s": old_velocity,
        })
        if new_velocity != old_velocity:
            velocity_corrections.append({
                "ball_index": index,
                "previous_diameter_m": old_diameter,
                "canonical_diameter_m": spec["diameter_m"],
                "previous_velocity_m_per_s": old_velocity,
                "corrected_velocity_m_per_s": new_velocity,
                "method": (
                    "preserve_photogate_block_time_by_diameter_ratio"
                ),
            })
        normalized_sequence.append(spec_id)
        materials.append(spec["material"])

    velocities = [
        float(case["physics"][f"ball_{index}_initial_velocity"]["value"])
        for index in range(1, ball_count + 1)
    ]
    signs = tuple(
        "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
        for value in velocities
    )
    if ball_count == 2 and signs == ("zero", "negative"):
        structure = "two_ball_single_incident"
        striker_indices = [2]
    elif ball_count == 2 and signs == ("positive", "negative"):
        structure = "two_ball_opposed_incident"
        striker_indices = [1, 2]
    elif ball_count == 3 and signs == ("positive", "zero", "zero"):
        structure = "three_ball_single_incident"
        striker_indices = [1]
    else:
        raise ValueError(
            "collision direction/structure needs user review: "
            f"{case['case_id']} velocities={velocities}"
        )

    case["appearance"]["ball_sequence"] = normalized_sequence
    case["appearance"]["ball_materials"] = materials
    case["appearance"]["collision_structure"] = structure
    case["appearance"]["striker_ball_index"] = striker_indices[0]
    if len(striker_indices) == 2:
        case["appearance"]["opposing_striker_ball_index"] = striker_indices[1]
    else:
        case["appearance"].pop("opposing_striker_ball_index", None)
    case["physics"]["striker_initial_velocity"] = copy.deepcopy(
        case["physics"][
            f"ball_{striker_indices[0]}_initial_velocity"
        ]
    )
    case["text"] = {
        "schema_version": "1.0",
        "prompt": collision_prompt(case),
        "language": "en",
        "annotation_source": "collision_case_prompt_v2",
    }
    source_locator = case.setdefault("provenance", {}).setdefault(
        "source_locator", {}
    )
    source_locator["release_5p1_correction_audit"] = (
        "annotation_corrections.json"
    )

    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "previous": before,
        "corrected": {
            "ball_sequence": normalized_sequence,
            "balls": [
                {
                    "index": index,
                    "spec_id": normalized_sequence[index - 1],
                    "mass_kg": case["physics"][
                        f"ball_{index}_mass"
                    ]["value"],
                    "radius_m": case["physics"][
                        f"ball_{index}_radius"
                    ]["value"],
                    "initial_velocity_m_per_s": case["physics"][
                        f"ball_{index}_initial_velocity"
                    ]["value"],
                }
                for index in range(1, ball_count + 1)
            ],
            "collision_structure": structure,
            "prompt": case["text"]["prompt"],
        },
        "velocity_corrections": velocity_corrections,
    }


def collision_prompt(case: dict[str, Any]) -> str:
    sequence = case["appearance"]["ball_sequence"]
    count_word = {2: "two", 3: "three"}[len(sequence)]
    relation = "between" if len(sequence) == 2 else "among"
    states: list[str] = []
    for index, spec_id in enumerate(sequence, 1):
        spec = BALL_SPECS[spec_id]
        velocity = float(
            case["physics"][f"ball_{index}_initial_velocity"]["value"]
        )
        if velocity > 0.0:
            motion = f"moves right at {abs(velocity):.4f} m/s"
        elif velocity < 0.0:
            motion = f"moves left at {abs(velocity):.4f} m/s"
        else:
            motion = "is initially stationary"
        states.append(
            f"ball {index} is a {spec['prompt_label']} and {motion}"
        )
    state_text = "; ".join(states)
    return (
        "A fixed-camera real-world laboratory video of a one-dimensional "
        f"central collision {relation} {count_word} aligned balls. At frame 0, "
        f"all {count_word} balls are fully visible. From left to right, "
        f"{state_text}. The camera and track remain stationary, and the "
        "collision is recorded at the true physical time scale."
    )


def normalize_parabolic(case: dict[str, Any]) -> dict[str, Any]:
    before = {
        name: copy.deepcopy(value)
        for name, value in case["physics"].items()
    }
    size_class = case["appearance"]["ball_size_class"]
    if size_class == "medium":
        spec_id = "steel_ball_medium_d20mm_m33p13g"
    elif size_class == "large":
        spec_id = "steel_ball_large_d25mm_m64p77g"
    else:
        raise ValueError(
            f"unknown parabolic ball class in {case['case_id']}: {size_class}"
        )
    spec = BALL_SPECS[spec_id]
    block_time = float(case["physics"]["photogate_block_time"]["value"])
    corrected_velocity = float(spec["diameter_m"]) / block_time

    case["physics"]["ball_mass"] = quantity(spec["mass_kg"], "kg")
    case["physics"]["ball_radius"] = quantity(spec["radius_m"], "m")
    case["physics"]["initial_horizontal_velocity"] = quantity(
        corrected_velocity,
        "m/s",
    )
    for field in PARABOLIC_AUXILIARY_FIELDS:
        if field in case["physics"]:
            case["physics"][field]["annotated"] = False
    case["appearance"]["ball_spec_id"] = spec_id
    case["provenance"]["source_locator"]["release_5p1_correction_audit"] = (
        "annotation_corrections.json"
    )

    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "previous": before,
        "corrected": copy.deepcopy(case["physics"]),
        "ball_spec_id": spec_id,
        "velocity_correction": {
            "method": "canonical_ball_diameter_divided_by_photogate_block_time",
            "canonical_diameter_m": spec["diameter_m"],
            "photogate_block_time_s": block_time,
            "corrected_velocity_m_per_s": corrected_velocity,
        },
    }


def parabolic_signature(case: dict[str, Any]) -> tuple[str, float, float]:
    return (
        case["appearance"]["ball_spec_id"],
        round(float(case["physics"]["launch_height"]["value"]), 9),
        round(
            float(case["physics"]["initial_horizontal_velocity"]["value"]),
            12,
        ),
    )


def grouped_parabolic_split(
    cases: list[dict[str, Any]],
    base_view: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    view = copy.deepcopy(base_view)
    groups = view["scenes"]["parabolic_motion"]
    previous_partition = {
        case_id: partition
        for partition, case_ids in groups.items()
        for case_id in case_ids
    }
    grouped: dict[tuple[str, float, float], list[str]] = defaultdict(list)
    for case in cases:
        if case["scene_id"] == "parabolic_motion":
            grouped[parabolic_signature(case)].append(case["case_id"])
    ordered = [
        (signature, sorted(case_ids))
        for signature, case_ids in sorted(grouped.items())
    ]

    # Exact 70/27 grouped holdout with the fewest changes from the frozen
    # source split.  Lexicographic choices provide a deterministic tie break.
    target_test_count = len(groups["test_id"])
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for _, case_ids in ordered:
        train_cost = sum(
            previous_partition[case_id] == "test_id"
            for case_id in case_ids
        )
        test_cost = sum(
            previous_partition[case_id] == "train"
            for case_id in case_ids
        )
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for count, (cost, choices) in states.items():
            candidates = (
                (count, (cost + train_cost, choices + (0,))),
                (
                    count + len(case_ids),
                    (cost + test_cost, choices + (1,)),
                ),
            )
            for next_count, candidate in candidates:
                if next_count > target_test_count:
                    continue
                previous = next_states.get(next_count)
                if previous is None or candidate < previous:
                    next_states[next_count] = candidate
        states = next_states
    if target_test_count not in states:
        raise ValueError(
            "cannot preserve the parabolic test count under signature grouping"
        )

    _, choices = states[target_test_count]
    train: list[str] = []
    test: list[str] = []
    signature_records: list[dict[str, Any]] = []
    for (signature, case_ids), choice in zip(ordered, choices):
        partition = "test_id" if choice else "train"
        (test if choice else train).extend(case_ids)
        signature_records.append({
            "signature": {
                "ball_spec_id": signature[0],
                "launch_height_m": signature[1],
                "initial_horizontal_velocity_m_per_s": signature[2],
            },
            "partition": partition,
            "case_ids": case_ids,
        })
    train.sort()
    test.sort()
    view["scenes"]["parabolic_motion"] = {
        "train": train,
        "test_id": test,
        "test_ood1": [],
    }
    moved = sorted(
        {
            case_id: {
                "from": previous_partition[case_id],
                "to": partition,
            }
            for partition, case_ids in (("train", train), ("test_id", test))
            for case_id in case_ids
            if previous_partition[case_id] != partition
        }.items()
    )
    selected = sorted(
        case_id
        for scene_groups in view["scenes"].values()
        for case_ids in scene_groups.values()
        for case_id in case_ids
    )
    view["selection_policy"] = (
        "pure_numeric_id_environment_ood1_with_parabolic_"
        "grouped_physical_signature_holdout_v1"
    )
    view["case_set_sha256"] = canonical_sha256(selected)
    audit = {
        "schema_version": "1.0",
        "scene_id": "parabolic_motion",
        "policy": "grouped_physical_signature_holdout_v1",
        "signature_fields": [
            "appearance.ball_spec_id",
            "physics.launch_height",
            "physics.initial_horizontal_velocity",
        ],
        "background_in_signature": False,
        "target_counts": {"train": len(train), "test_id": len(test)},
        "signature_group_count": len(signature_records),
        "moved_case_count": len(moved),
        "moved_cases": [
            {"case_id": case_id, **change}
            for case_id, change in moved
        ],
        "groups": signature_records,
    }
    return view, audit


def decimal_token(value: float, decimals: int = 4) -> str:
    rendered = f"{Decimal(str(value)):.{decimals}f}".rstrip("0").rstrip(".")
    if rendered == "-0":
        rendered = "0"
    return rendered.replace("-", "neg").replace(".", "p")


def identity_token(case_id: str) -> str:
    match = re.search(r"(?:^|_)img_(\d+)(?:_|$)", case_id)
    if match:
        return f"img{match.group(1)}"
    return f"id{hashlib.sha256(case_id.encode()).hexdigest()[:8]}"


def collision_directory(case: dict[str, Any]) -> tuple[str, list[str]]:
    structure = {
        "two_ball_single_incident": "n2-single",
        "two_ball_opposed_incident": "n2-opposed",
        "three_ball_single_incident": "n3-single",
    }[case["appearance"]["collision_structure"]]
    tokens = [structure]
    for index, spec_id in enumerate(case["appearance"]["ball_sequence"], 1):
        velocity = case["physics"][
            f"ball_{index}_initial_velocity"
        ]["value"]
        token = (
            f"b{index}-{BALL_SPECS[spec_id]['directory_token']}-"
            f"v{decimal_token(float(velocity))}mps"
        )
        tokens.append(token)
    name = "v51_collision_" + "_".join(tokens + [identity_token(
        case["case_id"]
    )])
    return name, tokens


def descriptive_directory(case: dict[str, Any]) -> tuple[str, list[str]]:
    scene_id = case["scene_id"]
    physics = case["physics"]
    identity = identity_token(case["case_id"])
    if scene_id == "collision_1d":
        return collision_directory(case)
    if scene_id == "parabolic_motion":
        spec_id = case["appearance"]["ball_spec_id"]
        tokens = [
            BALL_SPECS[spec_id]["directory_token"],
            f"h{decimal_token(physics['launch_height']['value'])}m",
            (
                "v"
                + decimal_token(
                    physics["initial_horizontal_velocity"]["value"]
                )
                + "mps"
            ),
        ]
        return f"v51_projectile_{'_'.join(tokens)}_{identity}", tokens
    if scene_id == "free_fall":
        tokens = [
            f"r{decimal_token(physics['ball_radius']['value'] * 1000)}mm",
            f"m{decimal_token(physics['ball_mass']['value'] * 1000)}g",
            f"h{decimal_token(physics['initial_height']['value'])}m",
            f"v{decimal_token(physics['initial_velocity']['value'])}mps",
        ]
        return f"v51_freefall_{'_'.join(tokens)}_{identity}", tokens
    if scene_id == "inclined_plane_slide":
        tokens = [
            f"a{decimal_token(physics['incline_angle']['value'])}deg",
            f"m{decimal_token(physics['block_mass']['value'] * 1000)}g",
            (
                "mu"
                + decimal_token(
                    physics["kinetic_friction_coefficient"]["value"]
                )
            ),
        ]
        return f"v51_incline_{'_'.join(tokens)}_{identity}", tokens
    if scene_id == "uniform_circular_motion":
        tokens = [
            f"w{decimal_token(physics['angular_velocity']['value'])}dps",
            (
                "r1-"
                + decimal_token(
                    physics["object_1_orbit_radius"]["value"] * 1000
                )
                + "mm"
            ),
        ]
        if "object_2_orbit_radius" in physics:
            tokens.append(
                "r2-"
                + decimal_token(
                    physics["object_2_orbit_radius"]["value"] * 1000
                )
                + "mm"
            )
        return f"v51_circular_{'_'.join(tokens)}_{identity}", tokens
    if scene_id == "pendulum":
        tokens = [
            f"l{decimal_token(physics['pendulum_length']['value'] * 1000)}mm",
            f"ls{decimal_token(physics['string_length']['value'] * 1000)}mm",
            f"r{decimal_token(physics['bob_radius']['value'] * 1000)}mm",
            f"a{decimal_token(physics['initial_angle']['value'])}deg",
        ]
        return f"v51_pendulum_{'_'.join(tokens)}_{identity}", tokens
    raise ValueError(f"unsupported scene for asset naming: {scene_id}")


def link_verified(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source asset is not a regular file: {source}")
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"asset target is not a regular file: {target}")
        if not os.path.samefile(source, target):
            raise ValueError(
                f"refusing non-hard-linked pre-existing target: {target}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.link-{os.getpid()}")
    try:
        os.link(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if not os.path.samefile(source, target):
        raise ValueError(f"hard-link verification failed: {target}")


def target_filename(role: str, source: Path) -> tuple[str, str]:
    suffix = source.suffix.lower()
    if role == "source_video":
        return "source", f"reference{suffix}"
    names = {
        "first_frame": "first_frame",
        "reference_video": "reference",
        "physics_reference_video": "physics_reference",
        "input_video": "input",
        "subject_mask": "subject_mask",
    }
    if role not in names:
        return "canonical", f"{role}{suffix}"
    return "canonical", f"{names[role]}{suffix}"


def materialize_descriptive_assets(
    case: dict[str, Any],
) -> dict[str, Any]:
    directory, physical_tokens = descriptive_directory(case)
    if len(directory.encode("utf-8")) > 240:
        raise ValueError(f"asset directory name is too long: {directory}")
    if BACKGROUND_DIRECTORY_TERMS.search(directory):
        raise ValueError(f"background token leaked into directory: {directory}")
    new_root = ASSET_ROOT / case["scene_id"] / directory
    previous_assets = copy.deepcopy(case["assets"])
    source_to_target: dict[str, str] = {}
    path_records: dict[str, dict[str, Any]] = {}

    ordered_roles = list(ASSET_ROLE_ORDER)
    ordered_roles.extend(
        sorted(set(case["assets"]) - set(ordered_roles))
    )
    for role in ordered_roles:
        value = case["assets"].get(role)
        if value is None or role in GLOBAL_ASSET_ROLES:
            continue
        source = (DATA_ROOT / value).resolve()
        source_key = str(source)
        if source_key in source_to_target:
            new_value = source_to_target[source_key]
            target = DATA_ROOT / new_value
        else:
            subdirectory, filename = target_filename(role, source)
            target = new_root / subdirectory / filename
            # A distinct physics reference may coexist with reference_video.
            if target.exists() and not os.path.samefile(source, target):
                filename = f"{role}{source.suffix.lower()}"
                target = new_root / subdirectory / filename
            link_verified(source, target)
            new_value = str(target.relative_to(DATA_ROOT))
            source_to_target[source_key] = new_value
        case["assets"][role] = new_value
        path_records[role] = {
            "previous": value,
            "corrected": new_value,
            "same_inode": os.path.samefile(source, target),
            "sha256": sha256(source),
        }
    case["assets"].update({
        role: value
        for role, value in previous_assets.items()
        if role in GLOBAL_ASSET_ROLES
    })
    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "previous_case_directories": sorted({
            "/".join(Path(value).parts[:3])
            for role, value in previous_assets.items()
            if (
                role not in GLOBAL_ASSET_ROLES
                and isinstance(value, str)
                and value.startswith("assets/")
                and len(Path(value).parts) >= 3
            )
        }),
        "corrected_case_directory": str(new_root.relative_to(DATA_ROOT)),
        "physical_tokens": physical_tokens,
        "background_token_included": False,
        "paths": path_records,
    }


def validate_normalized_cases(cases: list[dict[str, Any]]) -> None:
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs changed or became duplicated")
    canonical_ids = set(BALL_SPECS)
    for case in cases:
        if any("background" in key.lower() for key in case["physics"]):
            raise ValueError(
                f"background leaked into physics: {case['case_id']}"
            )
        local_paths = [
            value
            for role, value in case["assets"].items()
            if value is not None and role not in GLOBAL_ASSET_ROLES
        ]
        directories = {
            "/".join(Path(value).parts[:3])
            for value in local_paths
        }
        if len(directories) != 1:
            raise ValueError(
                f"case assets are not aligned to one directory: "
                f"{case['case_id']} {directories}"
            )
        directory = next(iter(directories))
        if BACKGROUND_DIRECTORY_TERMS.search(directory):
            raise ValueError(
                f"background token in asset directory: {directory}"
            )
        if case["scene_id"] == "collision_1d":
            sequence = case["appearance"]["ball_sequence"]
            if not set(sequence) <= canonical_ids:
                raise ValueError(
                    f"non-canonical collision spec: {case['case_id']}"
                )
            for index, spec_id in enumerate(sequence, 1):
                spec = BALL_SPECS[spec_id]
                if case["physics"][f"ball_{index}_mass"]["value"] != (
                    spec["mass_kg"]
                ):
                    raise ValueError(f"collision mass mismatch: {case['case_id']}")
                if case["physics"][f"ball_{index}_radius"]["value"] != (
                    spec["radius_m"]
                ):
                    raise ValueError(
                        f"collision radius mismatch: {case['case_id']}"
                    )
                if spec["prompt_label"] not in case["text"]["prompt"]:
                    raise ValueError(
                        f"collision prompt omits ball spec: {case['case_id']}"
                    )
        if case["scene_id"] == "parabolic_motion":
            spec = BALL_SPECS[case["appearance"]["ball_spec_id"]]
            block_time = case["physics"]["photogate_block_time"]["value"]
            expected = spec["diameter_m"] / block_time
            actual = case["physics"][
                "initial_horizontal_velocity"
            ]["value"]
            if abs(expected - actual) > 1e-12:
                raise ValueError(
                    f"parabolic velocity mismatch: {case['case_id']}"
                )
            for field in PARABOLIC_AUXILIARY_FIELDS:
                if (
                    field in case["physics"]
                    and case["physics"][field]["annotated"]
                ):
                    raise ValueError(
                        f"parabolic auxiliary field remains conditionable: "
                        f"{case['case_id']} {field}"
                    )


def build_release() -> tuple[Path, str]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable release: {OUTPUT_ROOT}"
        )
    descriptor = load_json(BASE_ROOT / "dataset.json")
    if (
        descriptor.get("dataset_id") != "physics_video_six_scene_v5"
        or descriptor.get("release") != "5.0.0"
    ):
        raise ValueError("release 5.1.0 requires the frozen 5.0.0 base")
    base_files_before = release_file_digests(BASE_ROOT)
    base_release = load_json(BASE_ROOT / "release.json")
    base_cases = load_jsonl(BASE_ROOT / descriptor["cases"])
    cases = copy.deepcopy(base_cases)
    corrections: list[dict[str, Any]] = []
    for case in cases:
        if case["scene_id"] == "collision_1d":
            corrections.append(normalize_collision(case))
        elif case["scene_id"] == "parabolic_motion":
            corrections.append(normalize_parabolic(case))

    base_view_a = load_json(BASE_ROOT / "views/view_a.json")
    view_a, split_audit = grouped_parabolic_split(cases, base_view_a)
    asset_mappings = [
        materialize_descriptive_assets(case)
        for case in cases
    ]
    asset_directories = [
        item["corrected_case_directory"]
        for item in asset_mappings
    ]
    if len(asset_directories) != len(set(asset_directories)):
        raise ValueError("descriptive asset directories are not case-unique")
    validate_normalized_cases(cases)

    train_signatures = {
        parabolic_signature(case)
        for case in cases
        if (
            case["case_id"]
            in view_a["scenes"]["parabolic_motion"]["train"]
        )
    }
    test_signatures = {
        parabolic_signature(case)
        for case in cases
        if (
            case["case_id"]
            in view_a["scenes"]["parabolic_motion"]["test_id"]
        )
    }
    if train_signatures & test_signatures:
        raise ValueError("parabolic physical signatures still cross partitions")
    split_audit["train_test_signature_overlap_count"] = 0

    stage = Path(
        tempfile.mkdtemp(prefix=".5.1.0.normalize-", dir=RELEASES_ROOT)
    )
    try:
        (stage / "views").mkdir()
        shutil.copytree(BASE_ROOT / "scenes", stage / "scenes")
        write_jsonl(stage / "cases.jsonl", cases)
        write_json(stage / "views/view_a.json", view_a)
        shutil.copy2(
            BASE_ROOT / "views/view_b.json",
            stage / "views/view_b.json",
        )
        write_json(
            stage / "dataset.json",
            {
                "schema_version": "3.0",
                "dataset_id": DATASET_ID,
                "release": RELEASE,
                "cases": "cases.jsonl",
                "asset_root": "../..",
                "asset_lock": "assets.lock.json",
                "release_manifest": "release.json",
                "scene_catalog": "scenes",
                "views": {
                    "view_a": "views/view_a.json",
                    "view_b": "views/view_b.json",
                },
            },
        )
        write_json(
            stage / "ball_spec_catalog.json",
            {
                "schema_version": "1.0",
                "authority": "user_confirmed_20260731",
                "specifications": BALL_SPECS,
                "aliases_replaced": BALL_SPEC_ALIASES,
            },
        )
        write_json(
            stage / "annotation_corrections.json",
            {
                "schema_version": "1.0",
                "base_release": "../5.0.0/dataset.json",
                "records": sorted(
                    corrections,
                    key=lambda item: item["case_id"],
                ),
            },
        )
        write_json(
            stage / "asset_directory_mapping.json",
            {
                "schema_version": "1.0",
                "method": "per_case_descriptive_physical_directory_hardlinks_v1",
                "media_payload_copied_bytes": 0,
                "background_fields_used": False,
                "records": sorted(
                    asset_mappings,
                    key=lambda item: item["case_id"],
                ),
            },
        )
        write_json(stage / "split_audit.json", split_audit)
        shutil.copy2(
            BASE_ROOT / "expansion_audit.json",
            stage / "source_expansion_audit_5.0.0.json",
        )

        scene_counts = Counter(case["scene_id"] for case in cases)
        collision_corrections = [
            record
            for record in corrections
            if record["scene_id"] == "collision_1d"
        ]
        parabolic_corrections = [
            record
            for record in corrections
            if record["scene_id"] == "parabolic_motion"
        ]
        write_json(
            stage / "migration_audit.json",
            {
                "schema_version": "1.0",
                "base_release": "../5.0.0/dataset.json",
                "base_dataset_id": descriptor["dataset_id"],
                "base_dataset_digest": base_release["dataset_digest"],
                "base_release_files_digest": canonical_sha256(
                    base_files_before
                ),
                "case_count": len(cases),
                "case_ids_sha256": canonical_sha256(
                    [case["case_id"] for case in cases]
                ),
                "scene_case_counts": dict(sorted(scene_counts.items())),
                "collision_case_correction_count": len(
                    collision_corrections
                ),
                "collision_velocity_correction_count": sum(
                    len(record["velocity_corrections"])
                    for record in collision_corrections
                ),
                "parabolic_case_correction_count": len(
                    parabolic_corrections
                ),
                "canonical_ball_spec_ids": sorted(BALL_SPECS),
                "asset_directory_mapping_count": len(asset_mappings),
                "asset_materialization": "hard_links_only",
                "media_payload_copied_bytes": 0,
                "background_in_physics": False,
                "background_in_asset_directory_names": False,
                "parabolic_split_policy": split_audit["policy"],
                "parabolic_train_test_signature_overlap_count": 0,
            },
        )
        (stage / "README.md").write_text(
            """# Physics Video Dataset 5.1.0

该 release 从不可变的 5.0.0 派生，case ID 和媒体字节保持稳定。

- 使用用户确认的钢球规格统一碰撞和平抛标注；
- 由错误直径派生的光电门速度已按原遮挡时间重新计算；
- 碰撞 prompt 逐 case 描述球数、从左到右规格、方向和初速度；
- 平抛辅助装置量保留作 provenance，但不再作为可条件化物理输入；
- 平抛 View A 按球规格、发射高度和初速度成组划分，无 train/test_id
  物理 signature 重叠；
- 六个 scene 的每个 case 均使用简短的结构化物理目录名，背景不参与命名；
- 新资产路径全部是指向 5.0.0 冻结媒体的硬链接，不复制媒体 payload。

详见 `ball_spec_catalog.json`、`annotation_corrections.json`、
`asset_directory_mapping.json`、`split_audit.json` 和
`migration_audit.json`。
""",
            encoding="utf-8",
        )
        _, release_manifest = rebuild_asset_lock(stage / "dataset.json")
        snapshot = load_dataset(
            stage / "dataset.json",
            check_assets=True,
            check_asset_hashes=True,
        )
        if snapshot.digest != release_manifest["dataset_digest"]:
            raise ValueError("staged release digest changed after lock build")
        base_files_after = release_file_digests(BASE_ROOT)
        if base_files_after != base_files_before:
            raise ValueError("frozen release 5.0.0 changed during migration")
        stage.rename(OUTPUT_ROOT)
        return OUTPUT_ROOT, snapshot.digest
    except BaseException:
        if stage.exists() and stage.parent == RELEASES_ROOT:
            shutil.rmtree(stage)
        raise


def main() -> int:
    output, digest = build_release()
    release = load_json(output / "release.json")
    print(
        f"{output / 'dataset.json'} "
        f"cases={len(load_jsonl(output / 'cases.jsonl'))} "
        f"assets={release['asset_files']} dataset_digest={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
