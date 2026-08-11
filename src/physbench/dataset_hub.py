from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from .dataset_distribution import (
    DistributionManifest,
    DistributionShard,
    load_distribution_manifest,
    verify_and_extract_shard,
)
from .datasets import load_dataset
from .huggingface_binding import load_huggingface_dataset_binding


@dataclass(frozen=True)
class Diagnostic:
    name: str
    status: str
    detail: str


Runner = Callable[..., object]
_MANIFEST_REMOTE_PATH = "distribution/v1/manifest.json"
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass
class _AssetTree:
    parent_descriptor: int
    descriptor: int
    name: str
    identity: tuple[int, int]

    def verify(self) -> None:
        opened = _directory_identity(os.fstat(self.descriptor))
        try:
            linked = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Dataset assets root changed after validation") from exc
        if (
            not stat.S_ISDIR(linked.st_mode)
            or opened != self.identity
            or _directory_identity(linked) != self.identity
        ):
            raise RuntimeError("Dataset assets root changed after validation")

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_descriptor)


def _directory_flags() -> int:
    return os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_secure_hub_io() -> None:
    if (
        os.name != "posix"
        or _O_DIRECTORY == 0
        or _O_NOFOLLOW == 0
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.listdir not in os.supports_fd
    ):
        raise RuntimeError(
            "secure Dataset pull requires POSIX dir_fd, O_DIRECTORY, and "
            "O_NOFOLLOW support"
        )


def _open_asset_tree(root: Path) -> _AssetTree:
    _require_secure_hub_io()
    parent = os.open(root, _directory_flags())
    try:
        descriptor = os.open("assets", _directory_flags(), dir_fd=parent)
        linked = os.stat("assets", dir_fd=parent, follow_symlinks=False)
        opened = os.fstat(descriptor)
        identity = _directory_identity(opened)
        if not stat.S_ISDIR(linked.st_mode) or _directory_identity(linked) != identity:
            raise RuntimeError("Dataset assets root changed while opening")
        return _AssetTree(parent, descriptor, "assets", identity)
    except BaseException:
        os.close(parent)
        raise


def _sha256_descriptor(descriptor: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def _inventory_directory(
    descriptor: int,
    prefix: PurePosixPath,
    *,
    files: dict[str, tuple[int, str]],
    directories: set[str],
) -> None:
    for name in sorted(os.listdir(descriptor)):
        relative = prefix / name
        linked = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(linked.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                before = _directory_identity(os.fstat(child))
                if before != _directory_identity(linked):
                    raise RuntimeError(
                        f"staged Dataset directory changed: {relative}"
                    )
                directories.add(relative.as_posix())
                _inventory_directory(
                    child,
                    relative,
                    files=files,
                    directories=directories,
                )
                after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(after.st_mode)
                    or _directory_identity(after) != before
                ):
                    raise RuntimeError(
                        f"staged Dataset directory changed: {relative}"
                    )
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(linked.st_mode):
            raise RuntimeError(
                f"unsafe special file in staged Dataset inventory: {relative}"
            )
        file_descriptor = os.open(
            name,
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=descriptor,
        )
        try:
            before = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or _file_identity(before) != _file_identity(linked)
            ):
                raise RuntimeError(f"staged Dataset file changed: {relative}")
            size, digest = _sha256_descriptor(file_descriptor)
            after = os.fstat(file_descriptor)
            if _file_identity(before) != _file_identity(after):
                raise RuntimeError(f"staged Dataset file changed: {relative}")
            files[relative.as_posix()] = (size, digest)
        finally:
            os.close(file_descriptor)


def _validate_asset_tree(
    tree: _AssetTree,
    manifest: DistributionManifest,
    *,
    trusted_files: dict[str, tuple[int, str]] | None = None,
) -> None:
    tree.verify()
    actual_files: dict[str, tuple[int, str]] = {}
    actual_directories = {"assets"}
    _inventory_directory(
        tree.descriptor,
        PurePosixPath("assets"),
        files=actual_files,
        directories=actual_directories,
    )
    expected_files = {
        record.path: (record.size_bytes, record.sha256)
        for record in manifest.files
    }
    expected_files.update(trusted_files or {})
    expected_directories = {"assets"}
    for path in expected_files:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            expected_directories.add(PurePosixPath(*parts[:index]).as_posix())
    extras = sorted(set(actual_files) - set(expected_files))
    missing = sorted(set(expected_files) - set(actual_files))
    extra_directories = sorted(actual_directories - expected_directories)
    missing_directories = sorted(expected_directories - actual_directories)
    if extras or missing or extra_directories or missing_directories:
        raise RuntimeError(
            "staged Dataset inventory differs from distribution manifest; "
            f"extra={extras[:3]} missing={missing[:3]} "
            f"extra_directories={extra_directories[:3]} "
            f"missing_directories={missing_directories[:3]}"
        )
    for path, expected in expected_files.items():
        if actual_files[path] != expected:
            raise RuntimeError(f"staged Dataset size or hash mismatch: {path}")
    tree.verify()


def _active_tree_is_valid(
    destination: Path,
    descriptor: Path,
    manifest: DistributionManifest,
    *,
    trusted_files: dict[str, tuple[int, str]],
) -> bool:
    tree: _AssetTree | None = None
    try:
        tree = _open_asset_tree(destination)
        _validate_asset_tree(tree, manifest, trusted_files=trusted_files)
        snapshot = load_dataset(descriptor, check_assets=True)
        if snapshot.digest != manifest.dataset_digest:
            return False
        _validate_asset_tree(tree, manifest, trusted_files=trusted_files)
        return True
    except (OSError, ValueError, RuntimeError):
        return False
    finally:
        if tree is not None:
            tree.close()


def _trusted_asset_scaffold(destination: Path) -> dict[str, tuple[int, str]]:
    readme = destination / "assets" / "README.md"
    try:
        descriptor = os.open(
            readme,
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
        )
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RuntimeError("trusted Dataset assets README is unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("trusted Dataset assets README is not regular")
        size, digest = _sha256_descriptor(descriptor)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("trusted Dataset assets README changed")
        return {"assets/README.md": (size, digest)}
    finally:
        os.close(descriptor)


def _trusted_manifest_extras(
    manifest: DistributionManifest,
    trusted_files: dict[str, tuple[int, str]],
) -> dict[str, tuple[int, str]]:
    declared = {
        record.path: (record.size_bytes, record.sha256)
        for record in manifest.files
    }
    extras: dict[str, tuple[int, str]] = {}
    for path, trusted in trusted_files.items():
        if path in declared:
            if trusted != declared[path]:
                raise RuntimeError(
                    "trusted Dataset scaffold differs from manifest: "
                    f"{path}"
                )
            continue
        extras[path] = trusted
    return extras


def _download_file(
    *,
    executable: str,
    repo_id: str,
    revision: str,
    remote_path: str,
    cache_root: Path,
    runner: Runner,
) -> Path:
    print(
        f"Dataset object start: {remote_path}",
        file=sys.stderr,
        flush=True,
    )
    command = [
        executable,
        "download",
        repo_id,
        remote_path,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(cache_root),
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        if exc.errno == getattr(os, "ENOSPC", 28):
            raise RuntimeError(
                f"Hugging Face Dataset download ran out of disk space: {exc}"
            ) from exc
        raise RuntimeError(
            f"could not start Hugging Face Dataset download: {exc}"
        ) from exc
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        stderr = str(getattr(completed, "stderr", "") or "").strip()
        lowered = stderr.lower()
        if (
            "429" in lowered
            or "rate limit" in lowered
            or "too many requests" in lowered
        ):
            category = "rate limit"
        elif any(
            item in lowered
            for item in ("401", "403", "unauthorized", "forbidden")
        ):
            category = "authentication or Dataset access"
        elif "no space left" in lowered or "disk quota" in lowered:
            category = "disk space"
        else:
            category = "transport"
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(
            "Hugging Face Dataset "
            f"{category} failure while downloading {remote_path} "
            f"(exit code {return_code}){detail}"
        )
    downloaded = cache_root / Path(remote_path)
    if not downloaded.is_file():
        raise RuntimeError(
            "Hugging Face Dataset download reported success without "
            f"{remote_path}"
        )
    print(
        f"Dataset object complete: {remote_path}",
        file=sys.stderr,
        flush=True,
    )
    return downloaded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_shard_is_valid(path: Path, shard: DistributionShard) -> bool:
    try:
        return (
            path.stat().st_size == shard.size_bytes
            and _sha256_file(path) == shard.sha256
        )
    except (FileNotFoundError, OSError):
        return False


def _prepare_staged_metadata(
    *,
    destination: Path,
    staging_root: Path,
    release: str,
    trusted_files: dict[str, tuple[int, str]],
) -> Path:
    source = destination / "releases" / release
    target = staging_root / "releases" / release
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    if "assets/README.md" in trusted_files:
        readme = destination / "assets" / "README.md"
        staged_readme = staging_root / "assets" / "README.md"
        shutil.copyfile(readme, staged_readme)
    return target / "dataset.json"


def _renameat2(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    flags: int,
) -> None:
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise RuntimeError("atomic Dataset promotion requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("atomic Dataset promotion requires renameat2") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_descriptor,
        os.fsencode(source_name),
        destination_parent_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise RuntimeError("atomic Dataset promotion is unsupported")
    raise OSError(error, os.strerror(error), destination_name)


def _linked_directory_identity(parent_descriptor: int, name: str) -> tuple[int, int]:
    linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(linked.st_mode):
        raise RuntimeError(f"Dataset assets entry is not a directory: {name}")
    return _directory_identity(linked)


def _same_inode(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return left == right


def _quarantine_active(parent_descriptor: int) -> None:
    for _ in range(32):
        quarantine = f".assets-quarantine-{secrets.token_hex(16)}"
        try:
            _renameat2(
                parent_descriptor,
                "assets",
                parent_descriptor,
                quarantine,
                flags=_RENAME_NOREPLACE,
            )
        except FileExistsError:
            continue
        except FileNotFoundError:
            break
        else:
            break
    try:
        os.stat("assets", dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError("unverified Dataset assets could not be quarantined")


def _promote_assets(
    *,
    destination: Path,
    staged_tree: _AssetTree,
    manifest: DistributionManifest,
    trusted_files: dict[str, tuple[int, str]],
) -> None:
    destination_descriptor = os.open(destination, _directory_flags())
    old_descriptor: int | None = None
    exchanged = False
    target_existed = False
    try:
        try:
            old_descriptor = os.open(
                "assets",
                _directory_flags(),
                dir_fd=destination_descriptor,
            )
        except FileNotFoundError:
            target_existed = False
        else:
            target_existed = True
        old_identity = (
            _directory_identity(os.fstat(old_descriptor))
            if old_descriptor is not None
            else None
        )
        if target_existed:
            if _linked_directory_identity(destination_descriptor, "assets") != old_identity:
                raise RuntimeError("active Dataset assets changed before exchange")
            _renameat2(
                staged_tree.parent_descriptor,
                staged_tree.name,
                destination_descriptor,
                "assets",
                flags=_RENAME_EXCHANGE,
            )
        else:
            _renameat2(
                staged_tree.parent_descriptor,
                staged_tree.name,
                destination_descriptor,
                "assets",
                flags=_RENAME_NOREPLACE,
            )
        exchanged = True
        if not _same_inode(
            _linked_directory_identity(destination_descriptor, "assets"),
            staged_tree.identity,
        ):
            raise RuntimeError("published Dataset assets root changed")
        published_descriptor = os.open(
            "assets",
            _directory_flags(),
            dir_fd=destination_descriptor,
        )
        published_identity = _directory_identity(os.fstat(published_descriptor))
        if not _same_inode(published_identity, staged_tree.identity):
            os.close(published_descriptor)
            raise RuntimeError("published Dataset assets root changed")
        published = _AssetTree(
            os.dup(destination_descriptor),
            published_descriptor,
            "assets",
            published_identity,
        )
        try:
            _validate_asset_tree(
                published,
                manifest,
                trusted_files=trusted_files,
            )
        finally:
            published.close()
    except BaseException as exc:
        if exchanged:
            restored = False
            if target_existed and old_identity is not None:
                try:
                    if _same_inode(
                        _linked_directory_identity(
                            staged_tree.parent_descriptor,
                            staged_tree.name,
                        ),
                        old_identity,
                    ):
                        _renameat2(
                            staged_tree.parent_descriptor,
                            staged_tree.name,
                            destination_descriptor,
                            "assets",
                            flags=_RENAME_EXCHANGE,
                        )
                        restored = _same_inode(
                            _linked_directory_identity(
                                destination_descriptor,
                                "assets",
                            ),
                            old_identity,
                        )
                except (OSError, RuntimeError):
                    restored = False
            if not restored:
                _quarantine_active(destination_descriptor)
            if restored:
                raise RuntimeError(
                    f"Dataset publish validation failed; restored old active tree: {exc}"
                ) from exc
            raise RuntimeError(
                f"Dataset publish validation failed; bad active tree quarantined: {exc}"
            ) from exc
        raise
    finally:
        if old_descriptor is not None:
            os.close(old_descriptor)
        os.close(destination_descriptor)


def pull_dataset(
    binding_path: str | Path,
    *,
    local_dir: str | Path,
    hf_executable: str | None = None,
    runner: Runner = subprocess.run,
    check_assets: bool = True,
) -> Path:
    binding_file = Path(binding_path).resolve()
    destination = Path(local_dir).resolve()
    binding = load_huggingface_dataset_binding(binding_file)
    descriptor = (
        destination
        / "releases"
        / binding.release
        / "dataset.json"
    )
    if not descriptor.is_file():
        raise FileNotFoundError(
            "the Git release is missing its bound Dataset descriptor: "
            f"{descriptor}"
        )
    binding = load_huggingface_dataset_binding(
        binding_file,
        dataset_path=descriptor,
    )
    executable = hf_executable or shutil.which("hf")
    if executable is None:
        raise FileNotFoundError(
            "Hugging Face CLI `hf` is unavailable; install the `hub` extra "
            "and run `hf auth login`"
        )
    destination.mkdir(parents=True, exist_ok=True)
    cache_root = destination / ".vphysbench" / "cache" / binding.revision
    staging_parent = (
        destination / ".vphysbench" / "staging" / binding.revision
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    staging_parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _download_file(
        executable=executable,
        repo_id=binding.repo_id,
        revision=binding.revision,
        remote_path=_MANIFEST_REMOTE_PATH,
        cache_root=cache_root,
        runner=runner,
    )
    try:
        manifest = load_distribution_manifest(
            manifest_path,
            dataset_id=binding.dataset_id,
            release=binding.release,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Dataset distribution manifest: {exc}") from exc

    trusted_files = _trusted_manifest_extras(
        manifest,
        _trusted_asset_scaffold(destination),
    )
    if check_assets and _active_tree_is_valid(
        destination,
        descriptor,
        manifest,
        trusted_files=trusted_files,
    ):
        return descriptor

    staging_root = staging_parent / f"attempt-{secrets.token_hex(16)}"
    staging_root.mkdir(mode=0o700)

    for shard in manifest.shards:
        archive = cache_root / shard.path
        if _cached_shard_is_valid(archive, shard):
            print(
                f"Dataset object start: {shard.path}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Dataset object cache reuse: {shard.path}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Dataset object complete: {shard.path}",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                archive.unlink()
            except FileNotFoundError:
                pass
            archive = _download_file(
                executable=executable,
                repo_id=binding.repo_id,
                revision=binding.revision,
                remote_path=shard.path,
                cache_root=cache_root,
                runner=runner,
            )
        try:
            shard_budget = sum(
                record.size_bytes
                for record in manifest.files
                if record.shard == shard.name
            )
            verify_and_extract_shard(
                archive,
                manifest=manifest,
                shard_name=shard.name,
                staging_root=staging_root,
                max_extracted_bytes=shard_budget,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError):
                category = "secure runtime"
            else:
                message = str(exc)
                category = "unsafe archive" if any(
                    marker in message.lower()
                    for marker in (
                        "symlink",
                        "member",
                        "path",
                        "zip",
                        "undeclared",
                    )
                ) else "checksum or size"
            raise RuntimeError(
                f"Dataset shard {category} verification failed for {shard.name}: {exc}"
            ) from exc

    staged_descriptor = _prepare_staged_metadata(
        destination=destination,
        staging_root=staging_root,
        release=binding.release,
        trusted_files=trusted_files,
    )
    staged_tree: _AssetTree | None = None
    try:
        staged_tree = _open_asset_tree(staging_root)
        _validate_asset_tree(
            staged_tree,
            manifest,
            trusted_files=trusted_files,
        )
        try:
            snapshot = load_dataset(staged_descriptor, check_assets=True)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"final Dataset validation failed: {exc}") from exc
        if snapshot.digest != manifest.dataset_digest:
            raise RuntimeError(
                "final Dataset validation failed: Dataset digest differs from "
                "distribution manifest"
            )
        _validate_asset_tree(
            staged_tree,
            manifest,
            trusted_files=trusted_files,
        )
        if not check_assets:
            return descriptor
        try:
            _promote_assets(
                destination=destination,
                staged_tree=staged_tree,
                manifest=manifest,
                trusted_files=trusted_files,
            )
        except OSError as exc:
            category = (
                "disk space"
                if exc.errno == getattr(os, "ENOSPC", 28)
                else "publish"
            )
            raise RuntimeError(
                f"Dataset {category} failed while promoting verified assets: {exc}"
            ) from exc
    finally:
        if staged_tree is not None:
            staged_tree.close()
    return descriptor


def diagnose_project(
    project_root: str | Path,
    *,
    level: str = "metadata",
) -> list[Diagnostic]:
    if level not in {"metadata", "evaluation"}:
        raise ValueError("doctor level must be metadata or evaluation")
    root = Path(project_root).resolve()
    datasets_root = root / "datasets"
    binding_path = datasets_root / "huggingface.json"
    diagnostics = [Diagnostic(
        "python",
        "ok" if sys.version_info >= (3, 11) else "error",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )]
    executable = shutil.which("hf")
    diagnostics.append(Diagnostic(
        "hf_cli",
        "ok" if executable else "warning",
        executable or "install the `hub` extra before downloading data",
    ))
    try:
        binding = load_huggingface_dataset_binding(binding_path)
        descriptor = (
            datasets_root / "releases" / binding.release / "dataset.json"
        )
        load_huggingface_dataset_binding(binding_path, dataset_path=descriptor)
        diagnostics.append(Diagnostic(
            "dataset_binding",
            "ok",
            f"{binding.repo_id}@{binding.revision}",
        ))
    except (FileNotFoundError, ValueError) as exc:
        diagnostics.append(Diagnostic("dataset_binding", "error", str(exc)))
        return diagnostics

    try:
        load_dataset(descriptor, check_assets=True)
    except (FileNotFoundError, ValueError) as exc:
        status = "warning" if level == "metadata" else "error"
        diagnostics.append(Diagnostic(
            "dataset_assets",
            status,
            f"not ready: {exc}; run `physbench dataset pull`",
        ))
    else:
        diagnostics.append(Diagnostic(
            "dataset_assets",
            "ok",
            "all referenced assets are present",
        ))
    return diagnostics


def diagnostics_succeeded(items: Sequence[Diagnostic]) -> bool:
    return all(item.status != "error" for item in items)
