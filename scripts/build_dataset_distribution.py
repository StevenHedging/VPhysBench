#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence

from physbench.dataset_distribution import (
    load_distribution_manifest,
    verify_and_extract_shard,
)
from physbench.datasets import load_dataset
from physbench.io import load_jsonl


DEFAULT_MAX_SHARD_BYTES = 2_000_000_000
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_MODE = stat.S_IFREG | 0o644


@dataclass(frozen=True)
class SourceFile:
    path: str
    source: Path
    size_bytes: int


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _require_asset_path(value: str) -> str:
    if (
        not value.startswith("assets/")
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(
            f"indexed Dataset asset must be an NFC POSIX path under assets/: {value!r}"
        )
    return value


def _indexed_asset_paths(snapshot) -> list[str]:
    indexed_cases = load_jsonl(snapshot.root / snapshot.descriptor["cases"])
    paths: set[str] = set()
    for case in indexed_cases:
        assets = case.get("assets")
        if not isinstance(assets, dict):
            raise ValueError(
                f"indexed Dataset case {case.get('case_id')!r} assets must be an object"
            )
        for value in assets.values():
            if value:
                if not isinstance(value, str):
                    raise ValueError("indexed Dataset asset path must be a string")
                paths.add(_require_asset_path(value))
    return sorted(paths)


def _source_files(snapshot) -> list[SourceFile]:
    asset_root = snapshot.asset_root.resolve(strict=True)
    files: list[SourceFile] = []
    for relative in _indexed_asset_paths(snapshot):
        source = asset_root / PurePosixPath(relative)
        if source.is_symlink():
            raise ValueError(f"indexed Dataset asset must not be a symlink: {relative}")
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(asset_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"indexed Dataset asset is missing or escapes asset_root: {relative}"
            ) from exc
        metadata = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"indexed Dataset asset is not a regular file: {relative}"
            )
        files.append(SourceFile(relative, source, metadata.st_size))
    return files


def _partition_files(
    files: Sequence[SourceFile],
    *,
    max_shard_bytes: int,
) -> list[list[SourceFile]]:
    shards: list[list[SourceFile]] = []
    current: list[SourceFile] = []
    current_bytes = 0
    for file_record in files:
        if file_record.size_bytes > max_shard_bytes:
            raise ValueError(
                f"indexed Dataset asset exceeds max shard bytes: {file_record.path}"
            )
        if current and current_bytes + file_record.size_bytes > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(file_record)
        current_bytes += file_record.size_bytes
    if current:
        shards.append(current)
    return shards


def _open_regular_source(file_record: SourceFile) -> BinaryIO:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(file_record.source, flags)
    except OSError as exc:
        raise ValueError(
            f"indexed Dataset asset changed or is not readable: {file_record.path}"
        ) from exc
    handle = os.fdopen(descriptor, "rb")
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != file_record.size_bytes:
        handle.close()
        raise ValueError(f"indexed Dataset asset changed: {file_record.path}")
    return handle


def _write_member(
    archive: zipfile.ZipFile,
    file_record: SourceFile,
) -> str:
    info = zipfile.ZipInfo(file_record.path, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ZIP_MODE << 16
    info.file_size = file_record.size_bytes
    digest = hashlib.sha256()
    copied = 0
    with _open_regular_source(file_record) as source:
        before = os.fstat(source.fileno())
        with archive.open(info, "w") as destination:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                destination.write(block)
                digest.update(block)
                copied += len(block)
        after = os.fstat(source.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        copied != file_record.size_bytes
        or identity_before != identity_after
    ):
        raise ValueError(
            f"indexed Dataset asset changed while building: {file_record.path}"
        )
    return digest.hexdigest()


def _write_shard(
    path: Path,
    files: Sequence[SourceFile],
) -> list[dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for file_record in files:
            records.append({
                "path": file_record.path,
                "size_bytes": file_record.size_bytes,
                "sha256": _write_member(archive, file_record),
            })
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _verify_output(
    manifest_path: Path,
    *,
    dataset_id: str,
    release: str,
    output_root: Path,
) -> None:
    manifest = load_distribution_manifest(
        manifest_path,
        dataset_id=dataset_id,
        release=release,
    )
    for shard in manifest.shards:
        budget = sum(
            file_record.size_bytes
            for file_record in manifest.files
            if file_record.shard == shard.name
        )
        with tempfile.TemporaryDirectory(
            prefix="vphysbench-distribution-verify-"
        ) as temporary:
            verify_and_extract_shard(
                output_root / shard.path,
                manifest=manifest,
                shard_name=shard.name,
                staging_root=Path(temporary) / "staging",
                max_extracted_bytes=budget,
            )


def build_distribution(
    dataset: str | Path,
    *,
    output_root: str | Path,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> Path:
    if isinstance(max_shard_bytes, bool) or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer")
    descriptor = Path(dataset).resolve()
    snapshot = load_dataset(descriptor, check_assets=True)
    release = snapshot.descriptor.get("release")
    if not isinstance(release, str) or not release:
        raise ValueError("Dataset release must be a non-empty string")
    files = _source_files(snapshot)
    partitions = _partition_files(files, max_shard_bytes=max_shard_bytes)

    root = Path(output_root).resolve()
    version_root = root / "distribution" / "v1"
    shard_root = version_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, object]] = []
    shard_records: list[dict[str, object]] = []
    for index, partition in enumerate(partitions, 1):
        name = f"shard-{index:05d}.zip"
        archive_path = shard_root / name
        records = _write_shard(archive_path, partition)
        file_records.extend({**record, "shard": name} for record in records)
        shard_records.append({
            "name": name,
            "path": f"distribution/v1/shards/{name}",
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
        })

    manifest_value = {
        "schema_version": "1.0",
        "dataset_id": snapshot.dataset_id,
        "release": release,
        "dataset_digest": snapshot.digest,
        "total_files": len(file_records),
        "total_bytes": sum(int(record["size_bytes"]) for record in file_records),
        "shards": shard_records,
        "files": file_records,
    }
    manifest_path = version_root / "manifest.json"
    _write_canonical_json(manifest_path, manifest_value)
    _verify_output(
        manifest_path,
        dataset_id=snapshot.dataset_id,
        release=release,
        output_root=root,
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic VPhysBench Dataset distribution shards."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--max-shard-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_SHARD_BYTES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_distribution(
        args.dataset,
        output_root=args.output_root,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
