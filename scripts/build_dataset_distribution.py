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


DEFAULT_MAX_SHARD_BYTES = 2_000_000_000
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_MODE = stat.S_IFREG | 0o644


@dataclass(frozen=True)
class SourceFile:
    path: str
    size_bytes: int
    identity: tuple[int, int, int, int]


@dataclass
class _DirectoryChain:
    descriptors: list[int]
    links: list[tuple[int, str, int]]

    @property
    def current(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for parent, name, child in self.links:
            linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(child)
            if (
                stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISDIR(linked.st_mode)
                or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ValueError(
                    f"indexed Dataset asset directory changed or is a symlink: {name}"
                )

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _require_secure_source_io() -> None:
    if (
        os.name != "posix"
        or getattr(os, "O_DIRECTORY", 0) == 0
        or getattr(os, "O_NOFOLLOW", 0) == 0
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise RuntimeError(
            "secure Dataset distribution building requires POSIX dir_fd, "
            "O_DIRECTORY, and O_NOFOLLOW support"
        )


def _open_asset_root(path: Path) -> int:
    _require_secure_source_io()
    try:
        descriptor = os.open(path, _directory_flags())
    except OSError as exc:
        raise ValueError(
            f"Dataset asset_root is missing, a symlink, or not a directory: {path}"
        ) from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"Dataset asset_root is not a directory: {path}")
    return descriptor


def _open_source_descriptor(asset_root_descriptor: int, relative: str) -> int:
    components = PurePosixPath(relative).parts
    chain = _DirectoryChain(
        descriptors=[os.dup(asset_root_descriptor)],
        links=[],
    )
    descriptor: int | None = None
    try:
        for component in components[:-1]:
            parent = chain.current
            child = os.open(component, _directory_flags(), dir_fd=parent)
            chain.descriptors.append(child)
            chain.links.append((parent, component, child))
            chain.verify()
        descriptor = os.open(
            components[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=chain.current,
        )
        chain.verify()
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(
                f"indexed Dataset asset is not a regular file: {relative}"
            )
        result = descriptor
        descriptor = None
        return result
    except OSError as exc:
        raise ValueError(
            f"indexed Dataset asset path changed, is a symlink, or is not a directory: "
            f"{relative}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        chain.close()


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
    paths: set[str] = set()
    for case in snapshot.cases:
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


def _source_files(snapshot, asset_root_descriptor: int) -> list[SourceFile]:
    files: list[SourceFile] = []
    for relative in _indexed_asset_paths(snapshot):
        descriptor = _open_source_descriptor(asset_root_descriptor, relative)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
        files.append(
            SourceFile(relative, metadata.st_size, _stat_identity(metadata))
        )
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


def _open_regular_source(
    asset_root_descriptor: int,
    file_record: SourceFile,
) -> BinaryIO:
    descriptor = _open_source_descriptor(asset_root_descriptor, file_record.path)
    handle = os.fdopen(descriptor, "rb")
    metadata = os.fstat(handle.fileno())
    if _stat_identity(metadata) != file_record.identity:
        handle.close()
        raise ValueError(f"indexed Dataset asset changed: {file_record.path}")
    return handle


def _write_member(
    archive: zipfile.ZipFile,
    asset_root_descriptor: int,
    file_record: SourceFile,
) -> str:
    info = zipfile.ZipInfo(file_record.path, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ZIP_MODE << 16
    info.file_size = file_record.size_bytes
    digest = hashlib.sha256()
    copied = 0
    with _open_regular_source(asset_root_descriptor, file_record) as source:
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
    if copied != file_record.size_bytes or not (
        identity_before == identity_after == file_record.identity
    ):
        raise ValueError(
            f"indexed Dataset asset changed while building: {file_record.path}"
        )
    return digest.hexdigest()


def _write_shard(
    path: Path,
    asset_root_descriptor: int,
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
                "sha256": _write_member(
                    archive,
                    asset_root_descriptor,
                    file_record,
                ),
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
    root = Path(output_root).resolve()
    version_root = root / "distribution" / "v1"
    if version_root.exists():
        raise FileExistsError(
            f"Dataset distribution output already exists: {version_root}"
        )
    asset_root_descriptor = _open_asset_root(snapshot.asset_root)
    try:
        files = _source_files(snapshot, asset_root_descriptor)
        partitions = _partition_files(files, max_shard_bytes=max_shard_bytes)

        shard_root = version_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        file_records: list[dict[str, object]] = []
        shard_records: list[dict[str, object]] = []
        for index, partition in enumerate(partitions, 1):
            name = f"shard-{index:05d}.zip"
            archive_path = shard_root / name
            records = _write_shard(
                archive_path,
                asset_root_descriptor,
                partition,
            )
            file_records.extend({**record, "shard": name} for record in records)
            shard_records.append({
                "name": name,
                "path": f"distribution/v1/shards/{name}",
                "size_bytes": archive_path.stat().st_size,
                "sha256": _sha256_file(archive_path),
            })
    finally:
        os.close(asset_root_descriptor)

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
