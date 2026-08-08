from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .datasets import load_dataset
from .huggingface_binding import load_huggingface_dataset_binding


@dataclass(frozen=True)
class Diagnostic:
    name: str
    status: str
    detail: str


Runner = Callable[..., object]


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
    descriptor = (
        destination
        / "releases"
        / load_huggingface_dataset_binding(binding_file).release
        / "dataset.json"
    )
    binding = load_huggingface_dataset_binding(
        binding_file,
        dataset_path=descriptor if descriptor.is_file() else None,
    )
    executable = hf_executable or shutil.which("hf")
    if executable is None:
        raise FileNotFoundError(
            "Hugging Face CLI `hf` is unavailable; install the `hub` extra "
            "and run `hf auth login`"
        )
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "download",
        binding.repo_id,
        "--repo-type",
        "dataset",
        "--revision",
        binding.revision,
        "--local-dir",
        str(destination),
    ]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return_code = int(getattr(completed, "returncode", 1))
    if return_code != 0:
        raise RuntimeError(
            "Hugging Face Dataset download failed with exit code "
            f"{return_code}; verify `hf auth login` and Dataset access"
        )
    if not descriptor.is_file():
        raise FileNotFoundError(
            f"download completed without the bound Dataset descriptor: {descriptor}"
        )
    load_huggingface_dataset_binding(binding_file, dataset_path=descriptor)
    if check_assets:
        load_dataset(descriptor, check_assets=True)
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
