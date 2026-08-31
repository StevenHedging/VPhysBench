#!/usr/bin/env python3
from __future__ import annotations

import json
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
    "constraints/evaluation-cu128.txt",
    "constraints/metadata.txt",
    "datasets/huggingface.json",
    "datasets/releases/14.0.0/dataset.json",
    "docs/GETTING_STARTED.md",
    "docs/CUSTOM_BASELINE_QUICKSTART.md",
    "pyproject.toml",
    "run/README.md",
    "baselines/README.md",
    "scripts/bootstrap_env.sh",
    "src/physbench/bootstrap.py",
    "src/physbench/cli.py",
}
DOCTOR_JSON_FIELDS = {
    "schema_version",
    "level",
    "ready",
    "summary",
    "checks",
}
EXPECTED_METADATA_CHECKS = {
    "python": {"ok"},
    "hf_cli": {"ok", "warning"},
    "dataset_binding": {"ok"},
    "dataset_assets": {"ok", "warning"},
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
    extracted = temporary_root / "relocated checkout"
    outside_checkout = temporary_root / "outside checkout"
    extracted.mkdir()
    outside_checkout.mkdir()

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
    bootstrap_environment = environment.copy()
    bootstrap_environment["VPHYSBENCH_BOOTSTRAP_PYTHON"] = sys.executable
    bootstrap = _run_archived(
        [
            "/bin/bash",
            str(extracted / "scripts" / "bootstrap_env.sh"),
            "--profile",
            "metadata",
            "--dry-run",
        ],
        cwd=outside_checkout,
        environment=bootstrap_environment,
    )
    if bootstrap.returncode:
        issues.append(
            "archived metadata bootstrap dry-run failed: "
            f"{bootstrap.stderr.strip()}"
        )
    if (extracted / ".venv").exists():
        issues.append("archived metadata bootstrap dry-run created .venv")

    help_smoke = _run_archived(
        [sys.executable, "-m", "physbench", "--help"],
        cwd=outside_checkout,
        environment=environment,
    )
    if help_smoke.returncode:
        issues.append(f"archived CLI help failed: {help_smoke.stderr.strip()}")

    doctor = _run_archived(
        [
            sys.executable,
            "-m",
            "physbench",
            "doctor",
            "--level",
            "metadata",
            "--json",
            "--project-root",
            str(extracted),
        ],
        cwd=outside_checkout,
        environment=environment,
    )
    issues.extend(_metadata_doctor_issues(doctor))

    baseline_list = _run_archived(
        [
            sys.executable,
            "-m",
            "physbench",
            "baseline",
            "list",
            "--root",
            str(extracted / "baselines"),
        ],
        cwd=outside_checkout,
        environment=environment,
    )
    if baseline_list.returncode:
        issues.append(
            "archived Baseline discovery failed: "
            f"{baseline_list.stderr.strip()}"
        )
    else:
        try:
            listed = json.loads(baseline_list.stdout)
        except json.JSONDecodeError as exc:
            issues.append(f"archived Baseline list emitted invalid JSON: {exc}")
        else:
            if not isinstance(listed, list):
                issues.append("archived Baseline list did not emit a JSON list")

    interface_smoke = _run_archived(
        [
            sys.executable,
            str(extracted / "scripts" / "smoke_custom_baseline.py"),
            "--metadata-only",
        ],
        cwd=outside_checkout,
        environment=environment,
    )
    if interface_smoke.returncode:
        issues.append(
            "archived metadata interface smoke failed: "
            f"{interface_smoke.stderr.strip()}"
        )
    return issues


def _run_archived(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _metadata_doctor_issues(
    completed: subprocess.CompletedProcess[str],
) -> list[str]:
    if completed.returncode != 0:
        return [
            "archived metadata doctor was not ready: "
            f"exit {completed.returncode}; {completed.stderr.strip()}"
        ]
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return [f"archived metadata doctor emitted invalid JSON: {exc}"]
    if not isinstance(report, dict) or not DOCTOR_JSON_FIELDS.issubset(report):
        return ["archived metadata doctor emitted an incomplete JSON report"]
    if report["level"] != "metadata":
        return ["archived metadata doctor reported the wrong readiness level"]
    if not isinstance(report["ready"], bool) or not isinstance(report["checks"], list):
        return ["archived metadata doctor emitted malformed readiness fields"]
    if not report["ready"]:
        return ["archived metadata doctor reported ready=false"]
    summary = report["summary"]
    if (
        not isinstance(summary, dict)
        or set(summary) != {"ok", "warnings", "errors"}
        or any(type(value) is not int for value in summary.values())
        or any(value < 0 for value in summary.values())
    ):
        return ["archived metadata doctor emitted a malformed summary"]
    check_statuses: dict[str, str] = {}
    for check in report["checks"]:
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("name"), str)
            or check.get("status") not in {"ok", "warning", "error"}
            or not isinstance(check.get("detail"), str)
        ):
            return ["archived metadata doctor emitted a malformed check"]
        name = check["name"]
        if name in check_statuses:
            return [f"archived metadata doctor repeated check: {name}"]
        check_statuses[name] = check["status"]
    missing = sorted(EXPECTED_METADATA_CHECKS.keys() - check_statuses.keys())
    if missing:
        return [
            "archived metadata doctor omitted expected checks: "
            + ", ".join(missing)
        ]
    invalid = sorted(
        name
        for name, statuses in EXPECTED_METADATA_CHECKS.items()
        if check_statuses[name] not in statuses
    )
    if invalid:
        return [
            "archived metadata doctor reported invalid expected check statuses: "
            + ", ".join(invalid)
        ]
    observed_summary = {
        "ok": sum(status == "ok" for status in check_statuses.values()),
        "warnings": sum(status == "warning" for status in check_statuses.values()),
        "errors": sum(status == "error" for status in check_statuses.values()),
    }
    if summary != observed_summary:
        return [
            "archived metadata doctor summary does not match check statuses: "
            f"expected {observed_summary}, got {summary}"
        ]
    if summary["errors"]:
        return ["archived metadata doctor reported required check errors"]
    return []


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
