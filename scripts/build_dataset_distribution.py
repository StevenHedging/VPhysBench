#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
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
from physbench.reference_observations.curation.install import refresh_locked_files


DEFAULT_MAX_SHARD_BYTES = 2_000_000_000
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_MODE = stat.S_IFREG | 0o644
# Worst-case standard-library ZIP overhead. A member may use a 20-byte local
# ZIP64 extra, a 24-byte signed ZIP64 data descriptor, and a 28-byte central
# ZIP64 extra. The archive trailer may contain ZIP64 EOCD + locator + EOCD.
_ZIP_MEMBER_FIXED_OVERHEAD_MAX = 30 + 20 + 24 + 46 + 28
_ZIP_END_OVERHEAD_MAX = 56 + 20 + 22
_RENAME_NOREPLACE = 1


@dataclass(frozen=True)
class SourceFile:
    path: str
    size_bytes: int
    identity: tuple[int, int, int, int, int]


@dataclass
class _OwnedTemporary:
    name: str
    descriptor: int
    identity: tuple[int, int]


@dataclass
class _DirectoryChain:
    descriptors: list[int]
    links: list[tuple[int, str, int]]
    identities: list[tuple[int, int, int]]
    label: str = "directory"

    @property
    def current(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for descriptor, expected in zip(
            self.descriptors,
            self.identities,
            strict=True,
        ):
            opened = _directory_identity(os.fstat(descriptor))
            if opened[:2] != expected[:2]:
                raise ValueError("held directory changed after anchoring")
        for index, (parent, name, child) in enumerate(self.links, 1):
            linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
            expected = self.identities[index]
            linked_identity = _directory_identity(linked)
            if (
                stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISDIR(linked.st_mode)
                or linked_identity[:2] != expected[:2]
            ):
                raise ValueError(
                    f"{self.label} changed or is a symlink: {name}"
                )

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_ctime_ns)


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
        or os.mkdir not in os.supports_dir_fd
        or os.rmdir not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or os.listdir not in os.supports_fd
    ):
        raise RuntimeError(
            "secure Dataset distribution building requires POSIX dir_fd, "
            "O_DIRECTORY, and O_NOFOLLOW support"
        )


def _open_directory_chain(
    path: Path,
    *,
    label: str,
) -> _DirectoryChain:
    _require_secure_source_io()
    if not path.is_absolute():
        raise ValueError("Dataset asset_root anchor must be absolute")
    components = path.parts[1:]
    anchor = os.open(path.anchor, _directory_flags())
    chain = _DirectoryChain(
        descriptors=[anchor],
        links=[],
        identities=[_directory_identity(os.fstat(anchor))],
        label=label,
    )
    try:
        for component in components:
            parent = chain.current
            child = os.open(component, _directory_flags(), dir_fd=parent)
            chain.descriptors.append(child)
            chain.links.append((parent, component, child))
            chain.identities.append(_directory_identity(os.fstat(child)))
            chain.verify()
    except OSError as exc:
        chain.close()
        raise ValueError(
            f"{label} is missing, a symlink, or not a directory: {path}"
        ) from exc
    except BaseException:
        chain.close()
        raise
    return chain


def _open_asset_root(path: Path) -> _DirectoryChain:
    return _open_directory_chain(
        path,
        label="Dataset asset_root directory",
    )


def _open_distribution_root(path: Path) -> _DirectoryChain:
    return _open_directory_chain(
        path,
        label="Dataset distribution parent",
    )


def _open_source_descriptor(asset_root_descriptor: int, relative: str) -> int:
    components = PurePosixPath(relative).parts
    anchor = os.dup(asset_root_descriptor)
    chain = _DirectoryChain(
        descriptors=[anchor],
        links=[],
        identities=[_directory_identity(os.fstat(anchor))],
        label="indexed Dataset asset directory",
    )
    descriptor: int | None = None
    try:
        for component in components[:-1]:
            parent = chain.current
            child = os.open(component, _directory_flags(), dir_fd=parent)
            chain.descriptors.append(child)
            chain.links.append((parent, component, child))
            chain.identities.append(_directory_identity(os.fstat(child)))
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


def _distribution_asset_paths(snapshot) -> list[str]:
    if snapshot.asset_lock is not None:
        return sorted(
            _require_asset_path(record["path"])
            for record in snapshot.asset_lock["files"]
        )
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
    for relative in _distribution_asset_paths(snapshot):
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
    current_bytes = _ZIP_END_OVERHEAD_MAX
    for file_record in files:
        if file_record.size_bytes > max_shard_bytes:
            raise ValueError(
                f"indexed Dataset asset exceeds max shard bytes: {file_record.path}"
            )
        name_bytes = len(file_record.path.encode("utf-8"))
        contribution = (
            file_record.size_bytes
            + _ZIP_MEMBER_FIXED_OVERHEAD_MAX
            + 2 * name_bytes
        )
        if _ZIP_END_OVERHEAD_MAX + contribution > max_shard_bytes:
            raise ValueError(
                "indexed Dataset asset plus stored ZIP metadata exceeds max shard "
                f"bytes: {file_record.path}"
            )
        if current and current_bytes + contribution > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = _ZIP_END_OVERHEAD_MAX
        current.append(file_record)
        current_bytes += contribution
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
    identity_before = _stat_identity(before)
    identity_after = _stat_identity(after)
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
    *,
    max_shard_bytes: int,
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
    if path.stat().st_size > max_shard_bytes:
        raise ValueError(f"final stored ZIP exceeds max shard bytes: {path.name}")
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


def _entry_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _create_owned_temporary(parent_descriptor: int) -> _OwnedTemporary:
    for _ in range(32):
        name = f".v1-build-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        created = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        identity = _entry_identity(created)
        try:
            descriptor = os.open(
                name,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
        except BaseException:
            raise
        opened = os.fstat(descriptor)
        linked = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(linked.st_mode)
            or _entry_identity(opened) != identity
            or _entry_identity(linked) != identity
        ):
            os.close(descriptor)
            raise ValueError("owned Dataset distribution temporary changed at creation")
        return _OwnedTemporary(name, descriptor, identity)
    raise FileExistsError("could not allocate Dataset distribution temporary")


def _verify_owned_temporary(
    parent_descriptor: int,
    temporary: _OwnedTemporary,
) -> None:
    opened = os.fstat(temporary.descriptor)
    try:
        linked = os.stat(
            temporary.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise ValueError("owned Dataset distribution temporary entry changed") from exc
    if (
        not stat.S_ISDIR(linked.st_mode)
        or _entry_identity(opened) != temporary.identity
        or _entry_identity(linked) != temporary.identity
    ):
        raise ValueError("owned Dataset distribution temporary entry changed")


def _rename_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "exclusive Dataset distribution publication requires Linux renameat2"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "exclusive Dataset distribution publication requires renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error,
                f"Dataset distribution output already exists: {destination_name}",
                destination_name,
            )
        if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise RuntimeError(
                "exclusive Dataset distribution publication is unsupported"
            )
        raise OSError(error, os.strerror(error), destination_name)


def _private_move(
    parent_descriptor: int,
    source_name: str,
    *,
    prefix: str,
) -> str:
    for _ in range(32):
        destination_name = f".{prefix}-{secrets.token_hex(16)}"
        try:
            _rename_noreplace(
                parent_descriptor,
                source_name,
                destination_name,
            )
        except FileExistsError:
            continue
        return destination_name
    raise FileExistsError(f"could not allocate private {prefix} name")


def _claim_owned_temporary(
    parent_descriptor: int,
    temporary: _OwnedTemporary,
    *,
    prefix: str,
) -> str:
    source_name = temporary.name
    claim_name = _private_move(
        parent_descriptor,
        source_name,
        prefix=prefix,
    )
    opened = _entry_identity(os.fstat(temporary.descriptor))
    try:
        linked = os.stat(
            claim_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        linked = None
    if (
        linked is None
        or not stat.S_ISDIR(linked.st_mode)
        or opened != temporary.identity
        or _entry_identity(linked) != temporary.identity
    ):
        try:
            _rename_noreplace(
                parent_descriptor,
                claim_name,
                source_name,
            )
        except BaseException as exc:
            raise RuntimeError(
                "owned Dataset distribution claim changed and could not be restored"
            ) from exc
        raise ValueError("owned Dataset distribution temporary claim changed")
    temporary.name = claim_name
    return claim_name


def _withdraw_destination(
    parent_descriptor: int,
    destination_name: str,
) -> None:
    try:
        _private_move(
            parent_descriptor,
            destination_name,
            prefix="v1-quarantine",
        )
    except FileNotFoundError:
        pass
    try:
        os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise RuntimeError(
        "unverified Dataset distribution could not be withdrawn from v1"
    )


def _exclusive_publish(
    parent_descriptor: int,
    temporary: _OwnedTemporary,
    destination_name: str,
) -> None:
    _verify_owned_temporary(parent_descriptor, temporary)
    claim_name = _claim_owned_temporary(
        parent_descriptor,
        temporary,
        prefix="v1-claim",
    )
    _rename_noreplace(parent_descriptor, claim_name, destination_name)
    try:
        linked = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        linked = None
    opened = _entry_identity(os.fstat(temporary.descriptor))
    if (
        linked is None
        or not stat.S_ISDIR(linked.st_mode)
        or opened != temporary.identity
        or _entry_identity(linked) != temporary.identity
    ):
        _withdraw_destination(parent_descriptor, destination_name)
        raise RuntimeError(
            "published Dataset distribution inode changed and was withdrawn"
        )


def _clear_directory_descriptor(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        linked = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(linked.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                opened_identity = _entry_identity(os.fstat(child))
                if opened_identity != _entry_identity(linked):
                    raise RuntimeError("owned temporary child directory changed")
                _clear_directory_descriptor(child)
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if _entry_identity(current) != opened_identity:
                    raise RuntimeError("owned temporary child directory changed")
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _cleanup_owned_temporary(
    parent_descriptor: int,
    temporary: _OwnedTemporary,
) -> None:
    if _entry_identity(os.fstat(temporary.descriptor)) != temporary.identity:
        raise RuntimeError("held owned Dataset distribution temporary changed")
    try:
        _claim_owned_temporary(
            parent_descriptor,
            temporary,
            prefix="v1-tombstone",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "owned Dataset distribution temporary missing; refusing cleanup"
        ) from exc
    _clear_directory_descriptor(temporary.descriptor)


def _verify_output(
    manifest_path: Path,
    *,
    dataset_id: str,
    release: str,
    version_root: Path,
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
                version_root / "shards" / shard.name,
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
    descriptor = Path(os.path.abspath(os.fspath(dataset)))
    root = Path(output_root).resolve()
    distribution_root = root / "distribution"
    version_root = root / "distribution" / "v1"
    if version_root.exists():
        raise FileExistsError(
            f"Dataset distribution output already exists: {version_root}"
        )
    distribution_root.mkdir(parents=True, exist_ok=True)
    distribution = _open_distribution_root(distribution_root)
    asset_root: _DirectoryChain | None = None
    try:
        raw_descriptor = json.loads(descriptor.read_text(encoding="utf-8"))
        if not isinstance(raw_descriptor, dict):
            raise ValueError("Dataset descriptor must be an object")
        configured_asset_root = raw_descriptor.get("asset_root", ".")
        if not isinstance(configured_asset_root, str):
            raise ValueError("Dataset asset_root must be a string")
        lexical_asset_root = Path(os.path.abspath(
            os.fspath(descriptor.parent / configured_asset_root)
        ))
        distribution.verify()
        asset_root = _open_asset_root(lexical_asset_root)
        snapshot = load_dataset(descriptor, check_assets=True)
        asset_root.verify()
        snapshot_asset_root = snapshot.asset_root.stat()
        held_asset_root = os.fstat(asset_root.current)
        if (
            snapshot_asset_root.st_dev,
            snapshot_asset_root.st_ino,
        ) != (
            held_asset_root.st_dev,
            held_asset_root.st_ino,
        ):
            raise ValueError("Dataset asset_root changed during validation")
        release = snapshot.descriptor.get("release")
        if not isinstance(release, str) or not release:
            raise ValueError("Dataset release must be a non-empty string")
        distribution.verify()
        temporary = _create_owned_temporary(distribution.current)
        temporary_root = distribution_root / temporary.name
        published = False
        try:
            _verify_owned_temporary(distribution.current, temporary)
            asset_root.verify()
            files = _source_files(snapshot, asset_root.current)
            asset_root.verify()
            partitions = _partition_files(
                files,
                max_shard_bytes=max_shard_bytes,
            )

            shard_root = temporary_root / "shards"
            shard_root.mkdir()
            file_records: list[dict[str, object]] = []
            shard_records: list[dict[str, object]] = []
            for index, partition in enumerate(partitions, 1):
                name = f"shard-{index:05d}.zip"
                archive_path = shard_root / name
                records = _write_shard(
                    archive_path,
                    asset_root.current,
                    partition,
                    max_shard_bytes=max_shard_bytes,
                )
                asset_root.verify()
                file_records.extend(
                    {**record, "shard": name} for record in records
                )
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
                "total_bytes": sum(
                    int(record["size_bytes"]) for record in file_records
                ),
                "shards": shard_records,
                "files": file_records,
            }
            manifest_path = temporary_root / "manifest.json"
            _write_canonical_json(manifest_path, manifest_value)
            _verify_output(
                manifest_path,
                dataset_id=snapshot.dataset_id,
                release=release,
                version_root=temporary_root,
            )
            _verify_owned_temporary(distribution.current, temporary)
            distribution.verify()
            _exclusive_publish(
                distribution.current,
                temporary,
                version_root.name,
            )
            published = True
            return version_root / "manifest.json"
        finally:
            try:
                if not published:
                    _cleanup_owned_temporary(distribution.current, temporary)
            finally:
                os.close(temporary.descriptor)
    finally:
        if asset_root is not None:
            asset_root.close()
        distribution.close()


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
