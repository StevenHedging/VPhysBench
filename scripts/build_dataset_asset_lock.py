#!/usr/bin/env python3
"""Build the immutable asset lock for a Dataset release."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import V2_DATASET  # noqa: E402
from physbench.datasets import load_dataset_v2  # noqa: E402
from physbench.io import canonical_sha256, load_json, load_jsonl, write_json  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=V2_DATASET)
    args = parser.parse_args()

    descriptor_path = args.dataset.resolve()
    root = descriptor_path.parent
    descriptor = load_json(descriptor_path)
    cases = load_jsonl(root / descriptor["cases"])
    asset_root = (root / descriptor.get("asset_root", ".")).resolve()
    roles: dict[str, set[str]] = defaultdict(set)
    case_ids: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        for role, value in case["assets"].items():
            if value:
                roles[value].add(role)
                case_ids[value].add(case["case_id"])

    files = []
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
    output = root / descriptor.get("asset_lock", "assets.lock.json")
    write_json(output, lock)
    dataset = load_dataset_v2(descriptor_path, check_assets=True)
    write_json(root / "release.json", {
        "schema_version": "1.0",
        "dataset_id": dataset.dataset_id,
        "release": descriptor.get("release"),
        "dataset_digest": dataset.digest,
        "asset_files": len(files),
        "asset_files_digest": lock["files_digest"],
    })
    print(
        f"{output} files={len(files)} digest={lock['files_digest']} "
        f"dataset_digest={dataset.digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
