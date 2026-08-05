#!/usr/bin/env python3
"""Build Dataset 10.0.0 with one locked physics document per active Case."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
BASE_RELEASE_ROOT = DATASETS_ROOT / "releases" / "9.0.0"
OUTPUT_RELEASE_ROOT = DATASETS_ROOT / "releases" / "10.0.0"


from physbench.io import load_jsonl as read_jsonl  # noqa: E402
from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from physbench.datasets import load_dataset  # noqa: E402
from scripts.build_dataset_asset_lock import rebuild_asset_lock  # noqa: E402


OUTPUT_DATASET_ID = "physics_video_six_scene_v10"
OUTPUT_RELEASE = "10.0.0"
EXPECTED_CASES = 799
EXPECTED_LOCKED_ASSETS = 6038
EXPECTED_LEGACY_REMOVALS = 32
EXPECTED_PHYSICS_FINGERPRINT = (
    "ca8c593007ce4e3d6d7437656f9aa0c3f32b5ff245a4592c570297ba9f957d1a"
)
PROVENANCE_ROOT = DATASETS_ROOT / "provenance" / "releases" / OUTPUT_RELEASE
RUNTIME_ENTRIES = [
    "README.md",
    "dataset.json",
    "release.json",
    "cases.jsonl",
    "assets.lock.json",
    "scenes/",
    "views/",
]


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


def _physics_fingerprint(cases: Iterable[dict[str, Any]]) -> str:
    return canonical_sha256(
        [(case["case_id"], case["physics"]) for case in cases]
    )


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": "4.0",
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
    return """# Physics Video Benchmark Dataset 10.0.0

This is the minimal immutable runtime snapshot for
`physics_video_six_scene_v10`. It contains 799 Cases from Dataset 9.0.0
without changing their ordered facts. Each Case adds one locked
`assets.physics_annotation` reference to its Case-local `physics.json`.

Runtime entries:

- `dataset.json`: stable Dataset entry point.
- `release.json`: Dataset and asset-lock digests.
- `cases.jsonl`: ordered Case facts and asset references.
- `assets.lock.json`: size and SHA-256 for every referenced asset.
- `scenes/`: Scene and evaluator contracts.
- `views/`: frozen train/test and direct-evaluation memberships.
- `README.md`: this responsibility map.

Inline `case.physics` remains the runtime compatibility API and must equal
the referenced Case-local physics document exactly. Dataset 9.0.0's
`masks.jsonl` is intentionally absent because masks remain reachable through
each Case's mask manifest and object asset roles, and no runtime consumer uses
the release-wide index. Build and validation evidence is stored separately at
`datasets/provenance/releases/10.0.0/`.
"""


def _legacy_evidence(item: LegacyRemoval) -> dict[str, Any]:
    return {
        "case_id": item.case_id,
        "directory": item.relative_directory,
        "file": (
            Path(item.relative_directory) / item.relative_file
        ).as_posix(),
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
    }


def _migration_evidence(
    *,
    base_cases: list[dict[str, Any]],
    output_cases: list[dict[str, Any]],
    lock: dict[str, Any],
    release_manifest: dict[str, Any],
    removals: list[LegacyRemoval],
) -> dict[str, Any]:
    base_release = load_json(BASE_RELEASE_ROOT / "release.json")
    base_fingerprint = _physics_fingerprint(base_cases)
    output_fingerprint = _physics_fingerprint(output_cases)
    if base_fingerprint != EXPECTED_PHYSICS_FINGERPRINT:
        raise ValueError(
            f"unexpected V9 physics fingerprint: {base_fingerprint}"
        )
    if output_fingerprint != base_fingerprint:
        raise ValueError("V10 physics fingerprint differs from V9")
    return {
        "schema_version": "1.0",
        "migration": "case_local_physics_v10",
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
            "case_local_physics_documents": sum(
                "physics_annotation" in case["assets"]
                for case in output_cases
            ),
            "locked_assets": len(lock["files"]),
            "removed_legacy_directories": len(removals),
        },
        "physics_fingerprint": {
            "base": base_fingerprint,
            "output": output_fingerprint,
        },
        "runtime_entries": RUNTIME_ENTRIES,
        "removed_legacy_directories": [
            _legacy_evidence(item) for item in removals
        ],
    }


def _remove_legacy_directories(removals: Iterable[LegacyRemoval]) -> None:
    for item in removals:
        file_path = item.absolute_directory / item.relative_file
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if file_path.stat().st_size != item.size_bytes:
            raise ValueError(f"legacy file size changed: {file_path}")
        if sha256_file(file_path) != item.sha256:
            raise ValueError(f"legacy file SHA-256 changed: {file_path}")
    for item in removals:
        file_path = item.absolute_directory / item.relative_file
        file_path.unlink()
        file_path.parent.rmdir()
        item.absolute_directory.rmdir()


def _check_existing_release(
    writes: Iterable[PhysicsWrite],
) -> dict[str, Any]:
    if not OUTPUT_RELEASE_ROOT.is_dir():
        missing = sum(not item.absolute_path.exists() for item in writes)
        return {"release": "absent", "missing_physics_documents": missing}
    from scripts.validate_dataset_v10 import validate_v10

    return validate_v10(OUTPUT_RELEASE_ROOT / "dataset.json")


def build_release(*, check: bool = False) -> dict[str, Any]:
    base_cases = read_jsonl(BASE_RELEASE_ROOT / "cases.jsonl")
    writes, output_cases = prepare_v10_cases(base_cases)
    if len(writes) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} Cases, found {len(writes)}")

    if check:
        for item in writes:
            if item.absolute_path.exists() and item.absolute_path.read_bytes() != item.payload:
                raise ValueError(
                    f"conflicting physics document: {item.absolute_path}"
                )
        legacy_count = len(
            list(
                (DATASETS_ROOT / "assets" / "collision_1d").glob(
                    "collision_r2_*"
                )
            )
        )
        if legacy_count:
            preflight_legacy_cleanup(base_cases)
        result = _check_existing_release(writes)
        result["legacy_directories_present"] = legacy_count
        return result

    if OUTPUT_RELEASE_ROOT.exists():
        raise FileExistsError(
            f"V10 release already exists; use --check: {OUTPUT_RELEASE_ROOT}"
        )
    removals = preflight_legacy_cleanup(
        base_cases,
        expected_count=EXPECTED_LEGACY_REMOVALS,
    )
    materialize_physics_documents(writes)

    stage = Path(
        tempfile.mkdtemp(prefix=".10.0.0.build-", dir=OUTPUT_RELEASE_ROOT.parent)
    )
    published = False
    try:
        shutil.copytree(BASE_RELEASE_ROOT / "scenes", stage / "scenes")
        shutil.copytree(BASE_RELEASE_ROOT / "views", stage / "views")
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
        snapshot = load_dataset(
            stage / "dataset.json",
            check_asset_hashes=True,
        )
        if snapshot.digest != release_manifest["dataset_digest"]:
            raise ValueError("staged V10 Dataset digest mismatch")
        if {path.name for path in stage.iterdir()} != {
            "README.md",
            "dataset.json",
            "release.json",
            "cases.jsonl",
            "assets.lock.json",
            "scenes",
            "views",
        }:
            raise ValueError("staged V10 release contains unexpected entries")

        migration = _migration_evidence(
            base_cases=base_cases,
            output_cases=output_cases,
            lock=lock,
            release_manifest=release_manifest,
            removals=removals,
        )
        stage.replace(OUTPUT_RELEASE_ROOT)
        published = True
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(PROVENANCE_ROOT / "migration.json", migration)
        _remove_legacy_directories(removals)

        from scripts.validate_dataset_v10 import validate_v10

        validation = validate_v10(
            OUTPUT_RELEASE_ROOT / "dataset.json",
            write_report=True,
        )
        return {
            "cases": len(output_cases),
            "physics": len(writes),
            "locked_assets": len(lock["files"]),
            "removed_legacy_dirs": len(removals),
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
