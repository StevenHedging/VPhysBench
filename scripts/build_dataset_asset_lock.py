#!/usr/bin/env python3
"""Build the immutable asset lock for a Dataset release."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import LATEST_DATASET  # noqa: E402
from physbench.datasets import load_dataset  # noqa: E402
from physbench.io import canonical_sha256, load_json, load_jsonl, write_json  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_path(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the release: {value}")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the release: {value}") from exc
    return relative, path


def _load_scene_configs(directory: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        value = load_json(path)
        scene_id = value.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"scene config lacks scene_id: {path}")
        if scene_id in values:
            raise ValueError(f"duplicate scene config: {scene_id}")
        values[scene_id] = value
    return values


def _copy_into_stage(
    source_root: Path,
    stage_root: Path,
    relative: Path,
) -> None:
    source = source_root / relative
    target = stage_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _validate_candidate_snapshot(
    *,
    descriptor_path: Path,
    descriptor: dict[str, Any],
    cases: list[dict[str, Any]],
    lock: dict[str, Any],
    release_manifest: dict[str, Any],
    expected_digest: str,
) -> None:
    """Validate a candidate lock without mutating the real release."""

    root = descriptor_path.parent
    cases_relative, cases_path = _release_path(
        root, descriptor.get("cases"), label="dataset.cases"
    )
    scenes_relative, scenes_path = _release_path(
        root,
        descriptor.get("scene_catalog"),
        label="dataset.scene_catalog",
    )
    views = descriptor.get("views")
    if not isinstance(views, dict) or not views:
        raise ValueError("dataset.views must be a non-empty object")
    view_paths = {
        view_id: _release_path(
            root, relative, label=f"dataset.views.{view_id}"
        )
        for view_id, relative in views.items()
    }
    lock_relative, _ = _release_path(
        root, descriptor.get("asset_lock"), label="dataset.asset_lock"
    )
    release_relative, _ = _release_path(
        root,
        descriptor.get("release_manifest", "release.json"),
        label="dataset.release_manifest",
    )
    if lock_relative == release_relative:
        raise ValueError(
            "dataset.asset_lock and dataset.release_manifest must differ"
        )
    if not cases_path.is_file():
        raise FileNotFoundError(cases_path)
    if not scenes_path.is_dir():
        raise FileNotFoundError(scenes_path)

    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.asset-lock-validation-",
        dir=root.parent,
    ) as temporary:
        stage_root = Path(temporary)
        descriptor_relative = descriptor_path.relative_to(root)
        _copy_into_stage(root, stage_root, descriptor_relative)
        _copy_into_stage(root, stage_root, cases_relative)
        shutil.copytree(
            scenes_path,
            stage_root / scenes_relative,
        )
        for _, (relative, path) in view_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            _copy_into_stage(root, stage_root, relative)
        write_json(stage_root / lock_relative, lock)
        write_json(stage_root / release_relative, release_manifest)

        snapshot = load_dataset(
            stage_root / descriptor_relative,
            check_assets=False,
        )
        if snapshot.digest != expected_digest:
            raise ValueError(
                "candidate Dataset digest mismatch: "
                f"expected={expected_digest}, actual={snapshot.digest}"
            )
        if snapshot.asset_lock != lock:
            raise ValueError("candidate asset lock changed during validation")

    # Guard against a caller accidentally supplying different case content
    # than the descriptor names. This is cheap and makes the digest inputs
    # explicit at the validation boundary.
    if load_jsonl(cases_path) != cases:
        raise ValueError("dataset cases changed while building the asset lock")


def _publish_pair(
    *,
    root: Path,
    lock_path: Path,
    lock: dict[str, Any],
    release_path: Path,
    release_manifest: dict[str, Any],
) -> None:
    """Replace the lock and release manifest as one rollback-safe operation."""

    if lock_path.is_symlink() or release_path.is_symlink():
        raise ValueError("refusing to replace a symlinked release manifest")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".asset-lock-publish-",
        dir=root,
    ) as temporary:
        transaction = Path(temporary)
        candidate_lock = transaction / "candidate-assets.lock.json"
        candidate_release = transaction / "candidate-release.json"
        backup_lock = transaction / "previous-assets.lock.json"
        backup_release = transaction / "previous-release.json"
        write_json(candidate_lock, lock)
        write_json(candidate_release, release_manifest)

        had_lock = lock_path.is_file()
        had_release = release_path.is_file()
        if lock_path.exists() and not had_lock:
            raise ValueError(f"asset lock target is not a file: {lock_path}")
        if release_path.exists() and not had_release:
            raise ValueError(
                f"release manifest target is not a file: {release_path}"
            )
        if had_lock:
            shutil.copy2(lock_path, backup_lock)
        if had_release:
            shutil.copy2(release_path, backup_release)

        try:
            os.replace(candidate_lock, lock_path)
            os.replace(candidate_release, release_path)
        except BaseException:
            if had_lock:
                os.replace(backup_lock, lock_path)
            else:
                lock_path.unlink(missing_ok=True)
            if had_release:
                os.replace(backup_release, release_path)
            else:
                release_path.unlink(missing_ok=True)
            raise


def rebuild_asset_lock(descriptor_path: Path) -> tuple[Path, dict[str, Any]]:
    descriptor_path = Path(descriptor_path).resolve()
    root = descriptor_path.parent
    descriptor = load_json(descriptor_path)
    if descriptor.get("schema_version") != "3.0":
        raise ValueError("asset-lock builder requires Dataset schema_version=3.0")

    cases_relative, cases_path = _release_path(
        root, descriptor.get("cases"), label="dataset.cases"
    )
    del cases_relative
    cases = load_jsonl(cases_path)
    asset_root_value = descriptor.get("asset_root", ".")
    if not isinstance(asset_root_value, str) or not asset_root_value:
        raise ValueError("dataset.asset_root must be a non-empty path")
    asset_root = (root / asset_root_value).resolve()
    if not asset_root.is_dir():
        raise FileNotFoundError(asset_root)

    roles: dict[str, set[str]] = defaultdict(set)
    case_ids: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("dataset contains a case without case_id")
        assets = case.get("assets")
        if not isinstance(assets, dict):
            raise ValueError(f"case {case_id} assets must be an object")
        for role, value in assets.items():
            if not isinstance(role, str) or not role:
                raise ValueError(f"case {case_id} contains an invalid asset role")
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"case {case_id} assets.{role} must be a path or null"
                )
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"case {case_id} assets.{role} must stay inside asset_root"
                )
            roles[value].add(role)
            case_ids[value].add(case_id)

    files: list[dict[str, Any]] = []
    for relative in sorted(roles):
        path = (asset_root / relative).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as exc:
            raise ValueError(f"asset path escapes asset_root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "roles": sorted(roles[relative]),
            "case_ids": sorted(case_ids[relative]),
        })

    lock = {
        "schema_version": "1.0",
        "dataset_id": descriptor["dataset_id"],
        "release": descriptor.get("release"),
        "files": files,
        "files_digest": canonical_sha256(files),
    }
    lock_relative, output = _release_path(
        root,
        descriptor.get("asset_lock"),
        label="dataset.asset_lock",
    )
    release_relative, release_output = _release_path(
        root,
        descriptor.get("release_manifest", "release.json"),
        label="dataset.release_manifest",
    )
    if lock_relative == release_relative:
        raise ValueError(
            "dataset.asset_lock and dataset.release_manifest must differ"
        )

    scenes_relative, scenes_path = _release_path(
        root,
        descriptor.get("scene_catalog"),
        label="dataset.scene_catalog",
    )
    del scenes_relative
    scene_configs = _load_scene_configs(scenes_path)
    views_value = descriptor.get("views")
    if not isinstance(views_value, dict) or not views_value:
        raise ValueError("dataset.views must be a non-empty object")
    views = {
        view_id: load_json(
            _release_path(
                root,
                relative,
                label=f"dataset.views.{view_id}",
            )[1]
        )
        for view_id, relative in views_value.items()
    }
    dataset_digest = canonical_sha256({
        "descriptor": descriptor,
        "cases": tuple(cases),
        "views": views,
        "scenes": scene_configs,
        "asset_lock": lock,
    })
    release_manifest = {
        "schema_version": "1.0",
        "dataset_id": descriptor["dataset_id"],
        "release": descriptor.get("release"),
        "dataset_digest": dataset_digest,
        "asset_files": len(files),
        "asset_files_digest": lock["files_digest"],
    }

    _validate_candidate_snapshot(
        descriptor_path=descriptor_path,
        descriptor=descriptor,
        cases=cases,
        lock=lock,
        release_manifest=release_manifest,
        expected_digest=dataset_digest,
    )
    _publish_pair(
        root=root,
        lock_path=output,
        lock=lock,
        release_path=release_output,
        release_manifest=release_manifest,
    )
    return output, release_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=LATEST_DATASET)
    args = parser.parse_args()

    output, release_manifest = rebuild_asset_lock(args.dataset)
    lock = load_json(output)
    print(
        f"{output} files={len(lock['files'])} digest={lock['files_digest']} "
        f"dataset_digest={release_manifest['dataset_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
