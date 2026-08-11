from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
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


def _download_file(
    *,
    executable: str,
    repo_id: str,
    revision: str,
    remote_path: str,
    cache_root: Path,
    runner: Runner,
) -> Path:
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


def _validate_staged_inventory(
    staging_root: Path,
    manifest: DistributionManifest,
) -> None:
    expected = {record.path for record in manifest.files}
    assets = staging_root / "assets"
    actual: set[str] = set()
    if assets.is_dir():
        for path in assets.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(
                    f"unsafe symlink in staged Dataset assets: {path}"
                )
            if path.is_file():
                actual.add(path.relative_to(staging_root).as_posix())
    extras = sorted(actual - expected - {"assets/README.md"})
    missing = sorted(expected - actual)
    if extras or missing:
        raise RuntimeError(
            "staged Dataset inventory differs from distribution manifest; "
            f"extra={extras[:3]} missing={missing[:3]}"
        )


def _prepare_staged_metadata(
    *,
    destination: Path,
    staging_root: Path,
    release: str,
) -> Path:
    source = destination / "releases" / release
    target = staging_root / "releases" / release
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    readme = destination / "assets" / "README.md"
    if readme.is_file() and not readme.is_symlink():
        staged_readme = staging_root / "assets" / "README.md"
        staged_readme.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(readme, staged_readme)
    return target / "dataset.json"


def _promote_assets(
    *,
    destination: Path,
    staging_root: Path,
    revision: str,
) -> None:
    source = staging_root / "assets"
    target = destination / "assets"
    backup_root = destination / ".vphysbench" / "backup" / revision
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"assets-{secrets.token_hex(8)}"
    moved_existing = False
    try:
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError(
                    "Dataset assets destination is not a regular directory"
                )
            os.replace(target, backup)
            moved_existing = True
        os.replace(source, target)
    except BaseException:
        if moved_existing and not target.exists() and backup.exists():
            os.replace(backup, target)
        raise


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
    if check_assets:
        try:
            load_dataset(descriptor, check_assets=True)
        except (FileNotFoundError, ValueError):
            pass
        else:
            return descriptor
    executable = hf_executable or shutil.which("hf")
    if executable is None:
        raise FileNotFoundError(
            "Hugging Face CLI `hf` is unavailable; install the `hub` extra "
            "and run `hf auth login`"
        )
    destination.mkdir(parents=True, exist_ok=True)
    cache_root = destination / ".vphysbench" / "cache" / binding.revision
    staging_root = destination / ".vphysbench" / "staging" / binding.revision
    cache_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
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

    for shard in manifest.shards:
        archive = cache_root / shard.path
        if not _cached_shard_is_valid(archive, shard):
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
            verify_and_extract_shard(
                archive,
                manifest=manifest,
                shard_name=shard.name,
                staging_root=staging_root,
                max_extracted_bytes=manifest.total_bytes,
            )
        except (OSError, ValueError) as exc:
            message = str(exc)
            category = "unsafe archive" if any(
                marker in message.lower()
                for marker in ("symlink", "member", "path", "zip", "undeclared")
            ) else "checksum or size"
            raise RuntimeError(
                f"Dataset shard {category} verification failed for {shard.name}: {exc}"
            ) from exc

    _validate_staged_inventory(staging_root, manifest)
    staged_descriptor = _prepare_staged_metadata(
        destination=destination,
        staging_root=staging_root,
        release=binding.release,
    )
    if check_assets:
        try:
            snapshot = load_dataset(staged_descriptor, check_assets=True)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"final Dataset validation failed: {exc}") from exc
        if snapshot.digest != manifest.dataset_digest:
            raise RuntimeError(
                "final Dataset validation failed: Dataset digest differs from "
                "distribution manifest"
            )
    try:
        _promote_assets(
            destination=destination,
            staging_root=staging_root,
            revision=binding.revision,
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
