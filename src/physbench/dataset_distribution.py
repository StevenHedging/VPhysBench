from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_MANIFEST_FIELDS = {
    "schema_version",
    "dataset_id",
    "release",
    "dataset_digest",
    "total_files",
    "total_bytes",
    "shards",
    "files",
}
_SHARD_FIELDS = {"name", "path", "size_bytes", "sha256"}
_FILE_FIELDS = {"path", "shard", "size_bytes", "sha256"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DistributionFile:
    path: str
    shard: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DistributionShard:
    name: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DistributionManifest:
    schema_version: str
    dataset_id: str
    release: str
    dataset_digest: str
    total_files: int
    total_bytes: int
    shards: tuple[DistributionShard, ...]
    files: tuple[DistributionFile, ...]


def _reject_unknown_fields(
    value: dict[str, object],
    *,
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_fields(
    value: dict[str, object],
    *,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _require_array(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _require_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hex characters")


def _require_posix_relative_file(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError(f"{label} must be an unnormalized POSIX relative file path")
    return value


def load_distribution_manifest(
    path: str | Path,
    *,
    dataset_id: str,
    release: str,
) -> DistributionManifest:
    value = _require_object(
        json.loads(Path(path).read_text(encoding="utf-8")),
        label="distribution manifest",
    )
    _require_fields(
        value,
        required=_MANIFEST_FIELDS,
        label="distribution manifest",
    )
    _reject_unknown_fields(
        value,
        allowed=_MANIFEST_FIELDS,
        label="distribution manifest",
    )
    shard_values = _require_array(
        value["shards"],
        label="distribution manifest shards",
    )
    file_values = _require_array(
        value["files"],
        label="distribution manifest files",
    )
    shards: list[dict[str, object]] = []
    for index, raw_shard in enumerate(shard_values):
        shard = _require_object(
            raw_shard,
            label=f"distribution manifest shard[{index}]",
        )
        _require_fields(
            shard,
            required=_SHARD_FIELDS,
            label=f"distribution manifest shard[{index}]",
        )
        _reject_unknown_fields(
            shard,
            allowed=_SHARD_FIELDS,
            label=f"distribution manifest shard[{index}]",
        )
        shards.append(shard)
    files: list[dict[str, object]] = []
    for index, raw_file in enumerate(file_values):
        file_record = _require_object(
            raw_file,
            label=f"distribution manifest file[{index}]",
        )
        _require_fields(
            file_record,
            required=_FILE_FIELDS,
            label=f"distribution manifest file[{index}]",
        )
        _reject_unknown_fields(
            file_record,
            allowed=_FILE_FIELDS,
            label=f"distribution manifest file[{index}]",
        )
        files.append(file_record)

    shard_names: list[str] = []
    shard_paths: list[str] = []
    for index, shard in enumerate(shards):
        shard_names.append(_require_posix_relative_file(
            shard["name"],
            label=f"distribution manifest shard[{index}].name",
        ))
        shard_paths.append(_require_posix_relative_file(
            shard["path"],
            label=f"distribution manifest shard[{index}].path",
        ))
    file_paths: list[str] = []
    for index, file_record in enumerate(files):
        file_paths.append(_require_posix_relative_file(
            file_record["path"],
            label=f"distribution manifest file[{index}].path",
        ))
        if not isinstance(file_record["shard"], str):
            raise ValueError(
                f"distribution manifest file[{index}].shard must be a string"
            )
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("distribution manifest contains a duplicate file path")
    if len(shard_names) != len(set(shard_names)):
        raise ValueError("distribution manifest contains a duplicate shard name")
    if len(shard_paths) != len(set(shard_paths)):
        raise ValueError("distribution manifest contains a duplicate shard path")
    if value["schema_version"] != "1.0":
        raise ValueError("distribution manifest requires schema_version=1.0")
    if value["dataset_id"] != dataset_id:
        raise ValueError("distribution manifest dataset_id does not match")
    if value["release"] != release:
        raise ValueError("distribution manifest release does not match")
    _require_sha256(
        value["dataset_digest"],
        label="distribution manifest dataset_digest",
    )
    for index, shard in enumerate(shards):
        shard_path = shard_paths[index]
        shard_name = shard_names[index]
        if (
            not shard_path.startswith("distribution/v1/shards/")
            or len(PurePosixPath(shard_path).parts) != 4
        ):
            raise ValueError(
                f"distribution manifest shard[{index}].path must be under "
                "distribution/v1/shards/"
            )
        if "/" in shard_name or PurePosixPath(shard_path).name != shard_name:
            raise ValueError(
                f"distribution manifest shard[{index}] name must match its shard path"
            )
        _require_sha256(
            shard["sha256"],
            label=f"distribution manifest shard[{index}].sha256",
        )
        _require_nonnegative_integer(
            shard["size_bytes"],
            label=f"distribution manifest shard[{index}].size_bytes",
        )
    shard_name_set = set(shard_names)
    for index, file_record in enumerate(files):
        file_path = file_paths[index]
        if not file_path.startswith("assets/"):
            raise ValueError(
                f"distribution manifest file[{index}].path must be under assets/"
            )
        if file_record["shard"] not in shard_name_set:
            raise ValueError(
                f"distribution manifest file[{index}] references an unknown shard"
            )
        _require_sha256(
            file_record["sha256"],
            label=f"distribution manifest file[{index}].sha256",
        )
        _require_nonnegative_integer(
            file_record["size_bytes"],
            label=f"distribution manifest file[{index}].size_bytes",
        )
    total_files = _require_nonnegative_integer(
        value["total_files"],
        label="distribution manifest total_files",
    )
    total_bytes = _require_nonnegative_integer(
        value["total_bytes"],
        label="distribution manifest total_bytes",
    )
    if total_files != len(files):
        raise ValueError("distribution manifest total_files is inconsistent")
    if total_bytes != sum(int(item["size_bytes"]) for item in files):
        raise ValueError("distribution manifest total_bytes is inconsistent")
    return DistributionManifest(
        schema_version=str(value["schema_version"]),
        dataset_id=str(value["dataset_id"]),
        release=str(value["release"]),
        dataset_digest=str(value["dataset_digest"]),
        total_files=total_files,
        total_bytes=total_bytes,
        shards=tuple(
            DistributionShard(
                name=str(item["name"]),
                path=str(item["path"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in shards
        ),
        files=tuple(
            DistributionFile(
                path=str(item["path"]),
                shard=str(item["shard"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in files
        ),
    )


def verify_and_extract_shard(
    archive_path: str | Path,
    *,
    manifest: DistributionManifest,
    shard_name: str,
    staging_root: str | Path,
) -> list[Path]:
    matching_shards = [
        shard for shard in manifest.shards if shard.name == shard_name
    ]
    if len(matching_shards) != 1:
        raise ValueError(f"distribution manifest has no shard named {shard_name!r}")
    shard = matching_shards[0]
    archive_file = Path(archive_path)
    if not archive_file.is_file():
        raise FileNotFoundError(f"Dataset shard does not exist: {archive_file}")
    if archive_file.stat().st_size != shard.size_bytes:
        raise ValueError(f"Dataset shard size mismatch: {shard_name}")

    archive_digest = hashlib.sha256()
    with archive_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            archive_digest.update(block)
    if archive_digest.hexdigest() != shard.sha256:
        raise ValueError(f"Dataset shard sha256 mismatch: {shard_name}")

    declared_files = {
        file_record.path: file_record
        for file_record in manifest.files
        if file_record.shard == shard_name
    }
    try:
        archive = zipfile.ZipFile(archive_file)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"invalid ZIP archive for Dataset shard {shard_name}") from exc

    with archive:
        infos = archive.infolist()
        member_names: list[str] = []
        for info in infos:
            member_name = _require_posix_relative_file(
                info.filename,
                label="ZIP member path",
            )
            if not member_name.startswith("assets/"):
                raise ValueError("ZIP member path must be under assets/")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"ZIP member is a symlink: {member_name}")
            file_type = stat.S_IFMT(unix_mode)
            if info.is_dir() or file_type not in {0, stat.S_IFREG}:
                raise ValueError(f"ZIP member is not a regular file: {member_name}")
            member_names.append(member_name)

        if len(member_names) != len(set(member_names)):
            raise ValueError("Dataset shard ZIP contains duplicate member names")
        actual_names = set(member_names)
        declared_names = set(declared_files)
        undeclared = sorted(actual_names - declared_names)
        missing = sorted(declared_names - actual_names)
        if undeclared:
            raise ValueError(
                f"Dataset shard ZIP contains undeclared members: {undeclared}"
            )
        if missing:
            raise ValueError(f"Dataset shard ZIP is missing declared members: {missing}")

        root = Path(staging_root)
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve()
        info_by_name = {info.filename: info for info in infos}
        extracted: list[Path] = []
        for member_name, file_record in declared_files.items():
            target = root / PurePosixPath(member_name)
            try:
                target.resolve(strict=False).relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"ZIP member resolves outside staging root: {member_name}"
                ) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as output:
                    with archive.open(info_by_name[member_name], "r") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if size != file_record.size_bytes:
                    raise ValueError(
                        f"extracted file size mismatch: {member_name}"
                    )
                if digest.hexdigest() != file_record.sha256:
                    raise ValueError(
                        f"extracted file sha256 mismatch: {member_name}"
                    )
                os.replace(temporary, target)
            except BaseException:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                raise
            extracted.append(target)
        return extracted
