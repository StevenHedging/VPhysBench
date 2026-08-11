from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import unicodedata
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
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_REQUIRED_DIR_FD_FUNCTIONS = (os.open, os.mkdir, os.stat, os.rename, os.unlink)
_SECURE_POSIX_IO_SUPPORTED = (
    os.name == "posix"
    and _O_NOFOLLOW != 0
    and _O_DIRECTORY != 0
    and all(function in os.supports_dir_fd for function in _REQUIRED_DIR_FD_FUNCTIONS)
    and os.stat in os.supports_follow_symlinks
)


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
                    f"staging directory component changed or is a symlink: {name}"
                )

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)


def _require_secure_posix_io() -> None:
    if not _SECURE_POSIX_IO_SUPPORTED:
        raise RuntimeError(
            "secure shard extraction requires POSIX dir_fd, O_DIRECTORY, "
            "and O_NOFOLLOW support"
        )


def _directory_flags() -> int:
    return os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC


def _open_directory_component(
    chain: _DirectoryChain,
    name: str,
) -> None:
    try:
        os.mkdir(name, mode=0o700, dir_fd=chain.current)
    except FileExistsError:
        pass
    parent = chain.current
    try:
        child = os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as exc:
        raise ValueError(
            f"staging directory component is a symlink or not a directory: {name}"
        ) from exc
    chain.descriptors.append(child)
    chain.links.append((parent, name, child))
    chain.verify()


def _open_staging_root(path: str | Path) -> _DirectoryChain:
    _require_secure_posix_io()
    staging = Path(path)
    components = list(staging.parts)
    if staging.is_absolute():
        anchor = staging.anchor
        components = components[1:]
    else:
        anchor = "."
    if not components or any(component in {"", ".", ".."} for component in components):
        raise ValueError("staging_root must name a concrete directory without '..'")
    chain = _DirectoryChain(
        descriptors=[os.open(anchor, _directory_flags())],
        links=[],
    )
    try:
        for component in components:
            _open_directory_component(chain, component)
        chain.verify()
        if hasattr(os, "geteuid") and os.fstat(chain.current).st_uid != os.geteuid():
            raise ValueError("staging_root must be owned by the current user")
        return chain
    except BaseException:
        chain.close()
        raise


def _open_member_parent(root: _DirectoryChain, components: tuple[str, ...]) -> _DirectoryChain:
    chain = _DirectoryChain(descriptors=[os.dup(root.current)], links=[])
    try:
        for component in components:
            _open_directory_component(chain, component)
        root.verify()
        chain.verify()
        return chain
    except BaseException:
        chain.close()
        raise


def _create_staging_temporary(parent_fd: int, leaf: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
    for _ in range(32):
        name = f".{leaf}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=parent_fd), name
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique staging temporary file")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


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
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{label} must already use Unicode NFC")
    return value


def _reject_path_collisions(paths: list[str], *, label: str) -> None:
    normalized = [unicodedata.normalize("NFC", path) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"distribution manifest contains a duplicate {label}")
    component_keys = {PurePosixPath(path).parts for path in normalized}
    for components in component_keys:
        if any(components[:end] in component_keys for end in range(1, len(components))):
            raise ValueError(
                f"distribution manifest contains a strict {label} prefix collision"
            )


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
        shard_reference = _require_posix_relative_file(
            file_record["shard"],
            label=f"distribution manifest file[{index}].shard",
        )
        if "/" in shard_reference:
            raise ValueError(
                f"distribution manifest file[{index}].shard must be a shard name"
            )
    _reject_path_collisions(file_paths, label="file path")
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
    max_extracted_bytes: int | None = None,
) -> list[Path]:
    _require_secure_posix_io()
    if max_extracted_bytes is not None:
        _require_nonnegative_integer(
            max_extracted_bytes,
            label="max_extracted_bytes",
        )
    matching_shards = [
        shard for shard in manifest.shards if shard.name == shard_name
    ]
    if len(matching_shards) != 1:
        raise ValueError(f"distribution manifest has no shard named {shard_name!r}")
    shard = matching_shards[0]
    archive_file = Path(archive_path)
    try:
        archive_descriptor = os.open(
            archive_file,
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
        )
    except OSError as exc:
        raise ValueError(
            f"Dataset shard path is missing, a symlink, or not readable: {archive_file}"
        ) from exc

    with os.fdopen(archive_descriptor, "rb") as archive_handle:
        before_hash = os.fstat(archive_handle.fileno())
        if not stat.S_ISREG(before_hash.st_mode):
            raise ValueError(f"Dataset shard is not a regular file: {archive_file}")
        if before_hash.st_size != shard.size_bytes:
            raise ValueError(f"Dataset shard size mismatch: {shard_name}")
        archive_digest = hashlib.sha256()
        for block in iter(lambda: archive_handle.read(1024 * 1024), b""):
            archive_digest.update(block)
        after_hash = os.fstat(archive_handle.fileno())
        if _stat_identity(before_hash) != _stat_identity(after_hash):
            raise ValueError(
                f"Dataset shard changed during verification: {shard_name}"
            )
        if archive_digest.hexdigest() != shard.sha256:
            raise ValueError(f"Dataset shard sha256 mismatch: {shard_name}")
        archive_handle.seek(0)

        declared_files = {
            file_record.path: file_record
            for file_record in manifest.files
            if file_record.shard == shard_name
        }
        try:
            archive = zipfile.ZipFile(archive_handle)
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise ValueError(
                f"invalid ZIP archive for Dataset shard {shard_name}"
            ) from exc

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
                    raise ValueError(
                        f"ZIP member is not a regular file: {member_name}"
                    )
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
                raise ValueError(
                    f"Dataset shard ZIP is missing declared members: {missing}"
                )

            info_by_name = {info.filename: info for info in infos}
            member_total = 0
            declared_total = sum(
                file_record.size_bytes for file_record in declared_files.values()
            )
            for member_name, file_record in declared_files.items():
                member_size = info_by_name[member_name].file_size
                if member_size != file_record.size_bytes:
                    raise ValueError(
                        f"ZIP member file_size differs from manifest: {member_name}"
                    )
                member_total += member_size
            if member_total != declared_total:
                raise ValueError("Dataset shard uncompressed byte total is inconsistent")
            if (
                max_extracted_bytes is not None
                and member_total > max_extracted_bytes
            ):
                raise ValueError(
                    "Dataset shard uncompressed bytes exceed extraction budget"
                )

            root = _open_staging_root(staging_root)
            try:
                extracted: list[Path] = []
                for member_name, file_record in declared_files.items():
                    member_path = PurePosixPath(member_name)
                    parent = _open_member_parent(root, member_path.parts[:-1])
                    temporary_name: str | None = None
                    try:
                        descriptor, temporary_name = _create_staging_temporary(
                            parent.current,
                            member_path.name,
                        )
                        digest = hashlib.sha256()
                        size = 0
                        with os.fdopen(descriptor, "wb") as output:
                            with archive.open(
                                info_by_name[member_name],
                                "r",
                            ) as source:
                                while True:
                                    remaining = file_record.size_bytes - size
                                    block = source.read(min(1024 * 1024, remaining + 1))
                                    if not block:
                                        break
                                    if len(block) > remaining:
                                        raise ValueError(
                                            "extracted file exceeds declared size: "
                                            f"{member_name}"
                                        )
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
                        root.verify()
                        parent.verify()
                        os.rename(
                            temporary_name,
                            member_path.name,
                            src_dir_fd=parent.current,
                            dst_dir_fd=parent.current,
                        )
                        temporary_name = None
                        root.verify()
                        parent.verify()
                    finally:
                        if temporary_name is not None:
                            try:
                                os.unlink(temporary_name, dir_fd=parent.current)
                            except FileNotFoundError:
                                pass
                        parent.close()
                    extracted.append(Path(staging_root) / member_path)
                return extracted
            finally:
                root.close()
