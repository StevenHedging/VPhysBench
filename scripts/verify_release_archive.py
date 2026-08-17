#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from scripts.release_audit import audit_release


REQUIRED_FILES = {
    "README.md",
    "RELEASE_MANIFEST.json",
    "datasets/huggingface.json",
    "datasets/releases/14.0.0/dataset.json",
    "docs/GETTING_STARTED.md",
    "docs/CUSTOM_BASELINE_QUICKSTART.md",
    "run/README.md",
    "baselines/README.md",
    "src/physbench/cli.py",
}


def _safe_members(archive: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], list[str]]:
    members = archive.getmembers()
    issues: list[str] = []
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            issues.append(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            issues.append(f"archive contains link: {member.name}")
    return members, issues


def verify_archive(repository: Path, temporary_root: Path) -> list[str]:
    repository = repository.resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    archive_path = temporary_root / "vphysbench-release.tar"
    extracted = temporary_root / "checkout"
    extracted.mkdir()

    completed = subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive_path), "HEAD"],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        return [f"git archive failed: {completed.stderr.strip()}"]

    with tarfile.open(archive_path) as archive:
        members, issues = _safe_members(archive)
        if issues:
            return issues
        archive.extractall(extracted)

    tracked = sorted(
        member.name
        for member in members
        if member.isfile() and member.name != ".git_archival.txt"
    )
    missing = sorted(REQUIRED_FILES.difference(tracked))
    issues = [f"archive is missing required file: {path}" for path in missing]
    issues.extend(audit_release(extracted, tracked_files=tracked))
    if issues:
        return sorted(set(issues))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted / "src")
    smoke = subprocess.run(
        [sys.executable, "-m", "physbench", "--help"],
        cwd=extracted,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if smoke.returncode:
        issues.append(f"archived CLI smoke failed: {smoke.stderr.strip()}")
    return issues


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="vphysbench-release-") as directory:
        issues = verify_archive(repository, Path(directory))
    if issues:
        for issue in issues:
            print(f"release_archive_error={issue}")
        return 1
    print("release_archive=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
