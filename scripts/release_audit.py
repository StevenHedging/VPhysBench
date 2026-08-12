#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
FORBIDDEN_ROOTS = {"cache", "results"}
FORBIDDEN_MODEL_MARKERS = {
    "wan22",
    "cosmos3",
    "causal_forcing",
    "causal forcing",
    "quantity_embedding",
    "symbol_value_cross_attention",
}
LOCAL_ONLY_NAMES = {"baseline.local.json"}
OFFICIAL_TASKS = {
    "tasks/official/six_scene_direct_eval_v1.json",
    "tasks/official/six_scene_train_six_scene_eval_v1.json",
}
PUBLIC_PROTOCOLS = {
    "configs/evaluation/protocols/scene_default_v1.json",
}
CONTENT_POLICY_FILES = {
    "scripts/release_audit.py",
    "tests/test_release_audit.py",
    "tests/test_release_documentation.py",
    "tests/test_repository_portability.py",
}
_CREDENTIAL = re.compile(
    r"hf_[A-Za-z0-9]{20,}|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"
)
_ABSOLUTE_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:root|mnt)/")


def _git_tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def audit_release(
    root: str | Path,
    *,
    tracked_files: Iterable[str] | None = None,
) -> list[str]:
    repository = Path(root).resolve()
    tracked = sorted(
        set(tracked_files if tracked_files is not None else _git_tracked_files(repository))
    )
    issues: list[str] = []
    existing = {
        relative
        for relative in tracked
        if (repository / relative).is_file()
    }
    if "RELEASE_MANIFEST.json" in existing:
        observed_tasks = {
            relative
            for relative in existing
            if relative.startswith("tasks/") and relative.endswith(".json")
        }
        if observed_tasks != OFFICIAL_TASKS:
            issues.append(
                "release Task inventory mismatch: "
                f"{sorted(observed_tasks)}"
            )
        observed_protocols = {
            relative
            for relative in existing
            if relative.startswith("configs/evaluation/protocols/")
            and relative.endswith(".json")
        }
        if observed_protocols != PUBLIC_PROTOCOLS:
            issues.append(
                "release evaluation protocol inventory mismatch: "
                f"{sorted(observed_protocols)}"
            )
    for relative in tracked:
        normalized = Path(relative).as_posix()
        parts = Path(normalized).parts
        lowered = normalized.casefold()
        path = repository / normalized

        if not parts:
            continue
        if parts[0] in FORBIDDEN_ROOTS:
            issues.append(f"forbidden tracked output root: {normalized}")
        if parts[0] == "run" and normalized != "run/README.md":
            issues.append(f"tracked runtime output: {normalized}")
        if (
            parts[0] == "baselines"
            and len(parts) > 1
            and Path(normalized).name == "baseline.json"
        ):
            issues.append(f"tracked baseline bundle: {normalized}")
        if Path(normalized).name in LOCAL_ONLY_NAMES:
            issues.append(f"local-only file is tracked: {normalized}")
        if Path(normalized).suffix.casefold() in WEIGHT_SUFFIXES:
            issues.append(f"model weight or checkpoint is tracked: {normalized}")
        checks_model_markers = normalized not in CONTENT_POLICY_FILES
        if checks_model_markers and any(
            marker in lowered for marker in FORBIDDEN_MODEL_MARKERS
        ):
            issues.append(f"concrete model marker in tracked path: {normalized}")

        content = _read_text(path)
        if content is None:
            continue
        if normalized in CONTENT_POLICY_FILES:
            continue
        content_lowered = content.casefold()
        if checks_model_markers and any(
            marker in content_lowered for marker in FORBIDDEN_MODEL_MARKERS
        ):
            issues.append(f"concrete model marker in tracked content: {normalized}")
        if _ABSOLUTE_LOCAL_PATH.search(content):
            issues.append(f"machine-local absolute path in tracked content: {normalized}")
        if _CREDENTIAL.search(content):
            issues.append(f"credential-like content in tracked file: {normalized}")
    return sorted(set(issues))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(arguments[0] if arguments else ".")
    issues = audit_release(root)
    if issues:
        for issue in issues:
            print(f"release_audit_error={issue}")
        return 1
    print("release_audit=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
