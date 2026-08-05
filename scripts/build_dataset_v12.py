#!/usr/bin/env python3
"""Build the single-file Dataset 12.0.0 metadata from Dataset 11.0.0."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_dataset_v11 import _case_directory, _physics_payload


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
BASE_RELEASE_ROOT = DATASETS_ROOT / "releases" / "11.0.0"
OUTPUT_RELEASE_ROOT = DATASETS_ROOT / "releases" / "12.0.0"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / "12.0.0"
OUTPUT_DATASET_ID = "physics_video_six_scene_v12"
OUTPUT_RELEASE = "12.0.0"
EXPECTED_CASES = 799
EXPECTED_LOCKED_ASSETS = 6038


@dataclass(frozen=True)
class PhysicsWrite:
    case_id: str
    relative_path: str
    absolute_path: Path
    payload: bytes


def migrate_case(case: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(case)
    output["assets"]["physics_annotation"] = (
        _case_directory(output) / "physics.json"
    ).as_posix()
    return output


def prepare_v12(
    cases: list[dict[str, Any]],
    datasets_root: Path = DATASETS_ROOT,
) -> tuple[list[PhysicsWrite], list[dict[str, Any]]]:
    migrated = [migrate_case(case) for case in cases]
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
