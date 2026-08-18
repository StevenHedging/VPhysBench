from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from .bundle import CandidateBundle, validate_candidate_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_candidate_bundle(
    manifest_path: str | Path,
    canonical_root: str | Path,
) -> CandidateBundle:
    """Validate then transactionally replace exactly the declared target files."""
    candidate = validate_candidate_bundle(manifest_path)
    root = Path(canonical_root).resolve(strict=True)
    lock_path = root.parent / f".{root.name}.reference-observation-curation.lock"
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path | None]] = []
    try:
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                for record in candidate.files:
                    target = (root / record.target_path).resolve()
                    try:
                        target.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(
                            "candidate target escapes canonical root"
                        ) from exc
                    target.parent.mkdir(parents=True, exist_ok=True)
                    descriptor, temporary_value = tempfile.mkstemp(
                        prefix=f".{target.name}.curation-new-", dir=target.parent
                    )
                    temporary = Path(temporary_value)
                    try:
                        with (
                            os.fdopen(descriptor, "wb") as output,
                            record.source.open("rb") as source,
                        ):
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                            output.flush()
                            os.fsync(output.fileno())
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                    if (
                        temporary.stat().st_size != record.size_bytes
                        or _sha256(temporary) != record.sha256
                    ):
                        temporary.unlink(missing_ok=True)
                        raise ValueError(
                            f"staged candidate digest mismatch: {record.target_path}"
                        )
                    staged.append((temporary, target))

                try:
                    for temporary, target in staged:
                        backup: Path | None = None
                        if target.exists():
                            descriptor, backup_value = tempfile.mkstemp(
                                prefix=f".{target.name}.curation-old-",
                                dir=target.parent,
                            )
                            os.close(descriptor)
                            backup = Path(backup_value)
                            backup.unlink()
                            os.replace(target, backup)
                        backups.append((target, backup))
                        os.replace(temporary, target)
                        _fsync_directory(target.parent)
                except BaseException:
                    for target, backup in reversed(backups):
                        target.unlink(missing_ok=True)
                        if backup is not None:
                            os.replace(backup, target)
                            _fsync_directory(target.parent)
                    raise
                for _, backup in backups:
                    if backup is not None:
                        backup.unlink(missing_ok=True)
            finally:
                for temporary, _ in staged:
                    temporary.unlink(missing_ok=True)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock_path.unlink(missing_ok=True)
    return candidate


def refresh_locked_files(
    lock_path: str | Path,
    asset_root: str | Path,
    paths: Iterable[str],
) -> None:
    """Refresh only selected existing asset-lock records, preserving all others."""
    manifest_path = Path(lock_path)
    root = Path(asset_root).resolve(strict=True)
    selected = set(paths)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = value.get("files")
    if not isinstance(records, list):
        raise ValueError("asset lock files are invalid")
    indexed = {record.get("path"): record for record in records if isinstance(record, dict)}
    missing = selected - set(indexed)
    if missing:
        raise ValueError(f"asset lock lacks requested paths: {sorted(missing)}")
    for relative in selected:
        target = (root / relative).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"asset lock path escapes asset root: {relative}") from exc
        record = indexed[relative]
        record["size_bytes"] = target.stat().st_size
        record["sha256"] = _sha256(target)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", dir=manifest_path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
        _fsync_directory(manifest_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
