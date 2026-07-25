from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain import DatasetSnapshot
from ..io import canonical_sha256, load_json, load_jsonl, sha256_file


FORBIDDEN_CASE_KEYS = {
    "input_views",
    "physical_parameters",
    "prompt",
    "prompt_profile_id",
    "text",
    "view_a_split",
}
REQUIRED_CASE_KEYS = {
    "schema_version",
    "case_id",
    "scene_id",
    "assets",
    "physics",
    "appearance",
    "temporal",
    "provenance",
    "ood",
    "has_real_reference_video",
}


def _load_directory_json(directory: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        value = load_json(path)
        key = value.get("scene_id")
        if not isinstance(key, str):
            raise ValueError(f"scene config lacks scene_id: {path}")
        if key in values:
            raise ValueError(f"duplicate scene config: {key}")
        values[key] = value
    return values


def _assert_no_prompt_payload(case: dict[str, Any]) -> None:
    invalid = sorted(FORBIDDEN_CASE_KEYS & set(case))
    if invalid:
        raise ValueError(
            f"dataset v2 case {case.get('case_id')} contains model/task fields: {invalid}"
        )
    stack: list[Any] = [case]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if "prompt" in value:
                raise ValueError(
                    f"dataset v2 case {case.get('case_id')} contains a prompt payload"
                )
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def _validate_case(case: dict[str, Any], known_scenes: set[str]) -> None:
    missing = sorted(REQUIRED_CASE_KEYS - set(case))
    if missing:
        raise ValueError(f"dataset v2 case {case.get('case_id')} missing {missing}")
    if case["schema_version"] != "2.0":
        raise ValueError(f"case {case.get('case_id')} must use schema_version=2.0")
    _assert_no_prompt_payload(case)
    if case["scene_id"] not in known_scenes:
        raise ValueError(f"case {case['case_id']} references unknown scene {case['scene_id']}")
    physics = case["physics"]
    if not isinstance(physics, dict) or not physics:
        raise ValueError(f"case {case['case_id']} requires structured physics")
    for name, quantity in physics.items():
        if not isinstance(quantity, dict):
            raise ValueError(f"case {case['case_id']} physics.{name} must be an object")
        if not {"value", "unit", "annotated"} <= set(quantity):
            raise ValueError(f"case {case['case_id']} physics.{name} is incomplete")


def _validate_views(
    cases: tuple[dict[str, Any], ...], views: dict[str, dict[str, Any]]
) -> None:
    case_ids = {case["case_id"] for case in cases}
    for view_id, view in views.items():
        if view.get("schema_version") != "2.0":
            raise ValueError(f"dataset view {view_id} must use schema_version=2.0")
        ids = [
            case_id
            for groups in view.get("scenes", {}).values()
            for members in groups.values()
            for case_id in members
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(f"dataset view {view_id} contains duplicate case IDs")
        unknown = set(ids) - case_ids
        if unknown:
            raise ValueError(
                f"dataset view {view_id} references unknown cases: {sorted(unknown)}"
            )
        coverage = view.get("coverage", "complete")
        if coverage not in {"complete", "subset"}:
            raise ValueError(
                f"dataset view {view_id} has invalid coverage={coverage}"
            )
        if coverage == "complete" and set(ids) != case_ids:
            raise ValueError(
                f"dataset view {view_id} must cover the complete case set"
            )
        if coverage == "subset" and not ids:
            raise ValueError(f"dataset subset view {view_id} cannot be empty")


def _load_asset_lock(
    root: Path,
    descriptor: dict[str, Any],
    cases: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    relative = descriptor.get("asset_lock")
    if relative is None:
        return None
    lock = load_json(root / relative)
    if lock.get("schema_version") != "1.0":
        raise ValueError("dataset asset lock must use schema_version=1.0")
    if lock.get("dataset_id") != descriptor.get("dataset_id"):
        raise ValueError("dataset asset lock dataset_id mismatch")
    if lock.get("release") != descriptor.get("release"):
        raise ValueError("dataset asset lock release mismatch")
    files = lock.get("files")
    if not isinstance(files, list):
        raise ValueError("dataset asset lock files must be a list")
    paths = [item.get("path") for item in files]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("dataset asset lock contains an invalid path")
    if len(paths) != len(set(paths)):
        raise ValueError("dataset asset lock contains duplicate paths")
    for item in files:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"dataset asset lock path must be relative: {path}")
        if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 0:
            raise ValueError(f"dataset asset lock has invalid size: {path}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"dataset asset lock has invalid SHA-256: {path}")
    if lock.get("files_digest") != canonical_sha256(files):
        raise ValueError("dataset asset lock files_digest mismatch")
    locked = set(paths)
    referenced = {
        value
        for case in cases
        for value in case["assets"].values()
        if value
    }
    missing = sorted(referenced - locked)
    if missing:
        raise ValueError(f"dataset asset lock misses referenced assets: {missing}")
    return lock


def load_dataset_v2(
    path: str | Path,
    *,
    check_assets: bool = False,
    check_asset_hashes: bool = False,
) -> DatasetSnapshot:
    descriptor_path = Path(path).resolve()
    descriptor = load_json(descriptor_path)
    if descriptor.get("schema_version") != "2.0":
        raise ValueError("dataset descriptor must use schema_version=2.0")
    root = descriptor_path.parent
    cases = tuple(load_jsonl(root / descriptor["cases"]))
    scene_configs = _load_directory_json(root / descriptor["scene_catalog"])
    if not scene_configs:
        raise ValueError("dataset scene catalog is empty")
    seen: set[str] = set()
    for case in cases:
        _validate_case(case, set(scene_configs))
        if case["case_id"] in seen:
            raise ValueError(f"duplicate dataset case ID {case['case_id']}")
        seen.add(case["case_id"])
    views = {
        view_id: load_json(root / relative_path)
        for view_id, relative_path in descriptor["views"].items()
    }
    _validate_views(cases, views)
    asset_root = (root / descriptor.get("asset_root", ".")).resolve()
    asset_lock = _load_asset_lock(root, descriptor, cases)
    if check_asset_hashes and asset_lock is None:
        raise ValueError("cannot verify asset hashes without an asset lock")
    locked_by_path = (
        {item["path"]: item for item in asset_lock["files"]}
        if asset_lock is not None
        else {}
    )
    if check_assets or check_asset_hashes:
        for case in cases:
            for key, value in case["assets"].items():
                if not value:
                    continue
                path = (asset_root / value).resolve()
                try:
                    path.relative_to(asset_root)
                except ValueError as exc:
                    raise ValueError(
                        f"case {case['case_id']} assets.{key} escapes asset_root"
                    ) from exc
                if not path.is_file():
                    raise FileNotFoundError(
                        f"case {case['case_id']} missing assets.{key}: {path}"
                    )
                locked = locked_by_path.get(value)
                if locked and path.stat().st_size != locked["size_bytes"]:
                    raise ValueError(f"dataset asset size mismatch: {value}")
                if check_asset_hashes and locked:
                    actual = sha256_file(path)
                    if actual != locked["sha256"]:
                        raise ValueError(
                            f"dataset asset SHA-256 mismatch: {value}"
                        )
    digest = canonical_sha256({
        "descriptor": descriptor,
        "cases": cases,
        "views": views,
        "scenes": scene_configs,
        "asset_lock": asset_lock,
    })
    return DatasetSnapshot(
        root=root,
        descriptor=descriptor,
        cases=cases,
        views=views,
        scene_configs=scene_configs,
        asset_lock=asset_lock,
        digest=digest,
        asset_root=asset_root,
    )
