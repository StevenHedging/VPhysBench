#!/usr/bin/env python3
"""Build Dataset 10.0.0 with one locked physics document per active Case."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
BASE_RELEASE_ROOT = DATASETS_ROOT / "releases" / "9.0.0"
OUTPUT_RELEASE_ROOT = DATASETS_ROOT / "releases" / "10.0.0"


from physbench.io import load_jsonl as read_jsonl  # noqa: E402
from physbench.io import sha256_file  # noqa: E402


@dataclass(frozen=True)
class PhysicsWrite:
    case_id: str
    relative_path: str
    absolute_path: Path
    payload: bytes


@dataclass(frozen=True)
class LegacyRemoval:
    case_id: str
    relative_directory: str
    absolute_directory: Path
    relative_file: str
    size_bytes: int
    sha256: str


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


def _physics_payload(case: dict[str, Any]) -> bytes:
    document = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "physics": copy.deepcopy(case["physics"]),
    }
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def prepare_v10_cases(
    cases: Iterable[dict[str, Any]],
    datasets_root: Path = DATASETS_ROOT,
) -> tuple[list[PhysicsWrite], list[dict[str, Any]]]:
    writes: list[PhysicsWrite] = []
    output_cases: list[dict[str, Any]] = []
    relative_paths: set[str] = set()
    for case in cases:
        case_directory = _case_directory(case)
        relative_path = (case_directory / "physics.json").as_posix()
        if relative_path in relative_paths:
            raise ValueError(
                f"multiple Cases resolve to physics document {relative_path}"
            )
        relative_paths.add(relative_path)
        writes.append(
            PhysicsWrite(
                case_id=case["case_id"],
                relative_path=relative_path,
                absolute_path=(datasets_root / relative_path).resolve(),
                payload=_physics_payload(case),
            )
        )
        output = copy.deepcopy(case)
        output["assets"]["physics_annotation"] = relative_path
        output_cases.append(output)
    return writes, output_cases


def materialize_physics_documents(writes: Iterable[PhysicsWrite]) -> None:
    pending: list[PhysicsWrite] = []
    for item in writes:
        if item.absolute_path.exists():
            if not item.absolute_path.is_file():
                raise ValueError(
                    f"physics document target is not a file: {item.absolute_path}"
                )
            if item.absolute_path.read_bytes() != item.payload:
                raise ValueError(
                    f"conflicting physics document: {item.absolute_path}"
                )
        else:
            pending.append(item)
    for item in pending:
        item.absolute_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{item.absolute_path.name}.",
            dir=item.absolute_path.parent,
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


def _historical_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for release in (
        "1.0.0",
        "2.0.0",
        "3.0.0",
        "4.0.0",
        "5.0.0",
        "5.1.0",
        "6.0.0",
        "7.0.0",
        "8.0.0",
        "9.0.0",
    ):
        path = DATASETS_ROOT / "releases" / release / "cases.jsonl"
        if path.is_file():
            cases.extend(read_jsonl(path))
    return cases


def preflight_legacy_cleanup(
    cases: Iterable[dict[str, Any]],
    *,
    datasets_root: Path = DATASETS_ROOT,
    historical_cases: Iterable[dict[str, Any]] | None = None,
    expected_count: int = 32,
) -> list[LegacyRemoval]:
    cases = list(cases)
    active_by_id = {case["case_id"]: case for case in cases}
    if len(active_by_id) != len(cases):
        raise ValueError("active Dataset contains duplicate Case IDs")
    active_directories = {_case_directory(case) for case in cases}
    collision_root = datasets_root / "assets" / "collision_1d"
    candidates = sorted(
        path
        for path in collision_root.glob("collision_r2_*")
        if path.is_dir()
        and Path("assets/collision_1d") / path.name not in active_directories
    )
    if len(candidates) != expected_count:
        raise ValueError(
            f"expected {expected_count} legacy collision directories, "
            f"found {len(candidates)}"
        )
    references = {
        value
        for case in (
            list(historical_cases)
            if historical_cases is not None
            else _historical_cases()
        )
        for value in case.get("assets", {}).values()
        if isinstance(value, str)
    }
    removals: list[LegacyRemoval] = []
    for directory in candidates:
        case_id = directory.name
        active = active_by_id.get(case_id)
        if active is None:
            raise ValueError(f"legacy directory has no active Case: {case_id}")
        if _case_directory(active) == Path("assets/collision_1d") / case_id:
            raise ValueError(f"legacy directory is still active: {case_id}")
        members = {
            path.relative_to(directory).as_posix(): path
            for path in directory.rglob("*")
        }
        if set(members) != {
            "source",
            "source/first_frame_source.png",
        }:
            raise ValueError(
                f"legacy directory has unexpected content: {directory}"
            )
        file_path = members["source/first_frame_source.png"]
        if not members["source"].is_dir() or not file_path.is_file():
            raise ValueError(
                f"legacy directory has unexpected content: {directory}"
            )
        relative_file = (
            Path("assets/collision_1d")
            / directory.name
            / "source/first_frame_source.png"
        ).as_posix()
        if relative_file in references:
            raise ValueError(f"legacy file remains referenced: {relative_file}")
        removals.append(
            LegacyRemoval(
                case_id=case_id,
                relative_directory=(
                    Path("assets/collision_1d") / directory.name
                ).as_posix(),
                absolute_directory=directory.resolve(),
                relative_file="source/first_frame_source.png",
                size_bytes=file_path.stat().st_size,
                sha256=sha256_file(file_path),
            )
        )
    return removals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cases = read_jsonl(BASE_RELEASE_ROOT / "cases.jsonl")
    writes, _ = prepare_v10_cases(cases)
    removals = preflight_legacy_cleanup(cases)
    if not args.check:
        raise RuntimeError("V10 release publication is not implemented yet")
    print(
        f"cases={len(cases)} physics={len(writes)} "
        f"legacy_dirs={len(removals)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
