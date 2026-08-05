#!/usr/bin/env python3
"""Build the single-file Dataset 12.0.0 metadata from Dataset 11.0.0."""

from __future__ import annotations

import copy
import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physbench.datasets import load_dataset
from physbench.io import canonical_sha256, load_json, load_jsonl, write_json, write_jsonl
from scripts.build_dataset_asset_lock import rebuild_asset_lock
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
EXPECTED_BASE_DIGEST = (
    "bd9479d5cde83daa97c3acf4abce094497ef36219ceda8b03614f36b442e7615"
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
        "views": {"view_a": "views/view_a.json", "view_b": "views/view_b.json"},
    }


def _readme() -> str:
    return """# Physics Video Benchmark Dataset 12.0.0

This is the sole active runtime Dataset, `physics_video_six_scene_v12`.
It preserves all 799 V11 Case facts, Views, prompts, symbolic quantities,
media, masks, and provenance bindings.

Each Case references one locked schema-2 `physics.json`. Its physics object is
identical to inline `case.physics`: finite non-negative magnitudes with stable
symbols and corrected independent/audit-only roles. Motion direction remains
in the Case prompt. No version-suffixed Case-local physics file is used.
"""


def _preflight() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = load_dataset(
        BASE_RELEASE_ROOT / "dataset.json", check_asset_hashes=True
    )
    if snapshot.dataset_id != "physics_video_six_scene_v11":
        raise ValueError(f"unexpected base Dataset: {snapshot.dataset_id}")
    if snapshot.digest != EXPECTED_BASE_DIGEST:
        raise ValueError(f"unexpected V11 digest: {snapshot.digest}")
    cases = load_jsonl(BASE_RELEASE_ROOT / "cases.jsonl")
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} Cases, found {len(cases)}")
    return cases, load_json(BASE_RELEASE_ROOT / "release.json")


def _install_documents(writes: list[PhysicsWrite], backup_root: Path) -> None:
    for index, item in enumerate(writes):
        source = item.absolute_path.with_name("physics.v11.json")
        if not source.is_file() or source.read_bytes() != item.payload:
            raise ValueError(f"invalid V11 physics source: {source}")
        if not item.absolute_path.is_file():
            raise ValueError(f"missing historical physics destination: {item.absolute_path}")
        backup = backup_root / f"{index:04d}.json"
        shutil.copy2(item.absolute_path, backup)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".physics.json.", dir=item.absolute_path.parent
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


def _restore_documents(writes: list[PhysicsWrite], backup_root: Path) -> None:
    for index, item in enumerate(writes):
        backup = backup_root / f"{index:04d}.json"
        if backup.is_file():
            shutil.copy2(backup, item.absolute_path)
        versioned = item.absolute_path.with_name("physics.v11.json")
        if not versioned.exists():
            versioned.write_bytes(item.payload)


def _evidence(
    *,
    base_release: dict[str, Any],
    output_release: dict[str, Any],
    base_lock: dict[str, Any],
    output_lock: dict[str, Any],
) -> dict[str, Any]:
    def non_physics(lock: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in lock["files"]
            if "physics_annotation" not in item["roles"]
        ]

    base_non_physics = non_physics(base_lock)
    output_non_physics = non_physics(output_lock)
    if base_non_physics != output_non_physics:
        raise ValueError("V12 changed non-physics asset records")
    return {
        "schema_version": "1.0",
        "migration": "single_current_physics_v12",
        "base": {
            "dataset_id": base_release["dataset_id"],
            "release": base_release["release"],
            "dataset_digest": base_release["dataset_digest"],
            "asset_files_digest": base_release["asset_files_digest"],
        },
        "output": {
            "dataset_id": output_release["dataset_id"],
            "release": output_release["release"],
            "dataset_digest": output_release["dataset_digest"],
            "asset_files_digest": output_release["asset_files_digest"],
        },
        "counts": {
            "cases": EXPECTED_CASES,
            "physics_documents_renamed": EXPECTED_CASES,
            "locked_assets": EXPECTED_LOCKED_ASSETS,
            "media_changes": 0,
        },
        "non_physics_asset_records_sha256": canonical_sha256(output_non_physics),
        "case_facts_sha256": canonical_sha256(
            load_jsonl(OUTPUT_RELEASE_ROOT / "cases.jsonl")
            if OUTPUT_RELEASE_ROOT.exists()
            else []
        ),
    }


def build_release(*, check: bool = False) -> dict[str, Any]:
    if check and OUTPUT_RELEASE_ROOT.is_dir():
        from scripts.validate_dataset_v12 import validate_v12

        return validate_v12(OUTPUT_RELEASE_ROOT / "dataset.json")
    if OUTPUT_RELEASE_ROOT.exists():
        raise FileExistsError(f"V12 already exists; use --check: {OUTPUT_RELEASE_ROOT}")
    base_cases, base_release = _preflight()
    writes, output_cases = prepare_v12(base_cases)
    if len(writes) != EXPECTED_CASES:
        raise ValueError("incomplete V12 physics writes")

    parent = OUTPUT_RELEASE_ROOT.parent
    stage = Path(tempfile.mkdtemp(prefix=".12.0.0.build-", dir=parent))
    backup = Path(tempfile.mkdtemp(prefix=".physics-v10.backup-", dir=parent))
    published = False
    installed = False
    try:
        shutil.copytree(BASE_RELEASE_ROOT / "views", stage / "views")
        shutil.copytree(BASE_RELEASE_ROOT / "scenes", stage / "scenes")
        write_jsonl(stage / "cases.jsonl", output_cases)
        write_json(stage / "dataset.json", _descriptor())
        (stage / "README.md").write_text(_readme(), encoding="utf-8")

        _install_documents(writes, backup)
        installed = True
        lock_path, release_manifest = rebuild_asset_lock(stage / "dataset.json")
        lock = load_json(lock_path)
        if len(lock["files"]) != EXPECTED_LOCKED_ASSETS:
            raise ValueError(f"expected {EXPECTED_LOCKED_ASSETS} locked assets")
        snapshot = load_dataset(stage / "dataset.json", check_asset_hashes=True)
        if snapshot.digest != release_manifest["dataset_digest"]:
            raise ValueError("staged V12 Dataset digest mismatch")
        if {path.name for path in stage.iterdir()} != RUNTIME_ENTRIES:
            raise ValueError("V12 is not the seven-entry runtime tree")
        stage.replace(OUTPUT_RELEASE_ROOT)
        published = True

        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        evidence = _evidence(
            base_release=base_release,
            output_release=release_manifest,
            base_lock=load_json(BASE_RELEASE_ROOT / "assets.lock.json"),
            output_lock=lock,
        )
        evidence["case_facts_sha256"] = canonical_sha256(output_cases)
        write_json(PROVENANCE_ROOT / "migration.json", evidence)

        for item in writes:
            item.absolute_path.with_name("physics.v11.json").unlink()

        from scripts.validate_dataset_v12 import validate_v12

        validation = validate_v12(
            OUTPUT_RELEASE_ROOT / "dataset.json", write_report=True
        )
        return {
            "cases": EXPECTED_CASES,
            "physics": EXPECTED_CASES,
            "locked_assets": EXPECTED_LOCKED_ASSETS,
            "media_changes": 0,
            "dataset_digest": validation["dataset_digest"],
        }
    except BaseException:
        if published:
            shutil.rmtree(OUTPUT_RELEASE_ROOT, ignore_errors=True)
        if installed:
            _restore_documents(writes, backup)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build_release(check=args.check)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
