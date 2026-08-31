"""Read-only environment diagnostics for a VPhysBench checkout."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAM31_CHECKPOINT_ENV = "VPHYSBENCH_SAM31_CHECKPOINT"
SAM31_CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
SCENE_EVALUATION_DEPENDENCIES = (
    "cv2",
    "matplotlib",
    "numpy",
    "sam2",
    "scipy",
    "torch",
)
SCENE_EVALUATION_IMPORTS = tuple(
    "sam2.sam2_video_predictor" if name == "sam2" else name
    for name in SCENE_EVALUATION_DEPENDENCIES
)


@dataclass(frozen=True)
class Diagnostic:
    """One actionable, side-effect-free readiness check."""

    name: str
    status: str
    detail: str
    hint: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "warning", "error"}:
            raise ValueError("diagnostic status must be ok, warning, or error")


Which = Callable[[str], str | None]
FindSpec = Callable[[str], object | None]
TorchProbe = Callable[[], Mapping[str, Any]]
Importer = Callable[[str], object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnose_checkpoint(
    *,
    environ: Mapping[str, str] = os.environ,
    expected_sha256: str = SAM31_CHECKPOINT_SHA256,
) -> Diagnostic:
    """Validate the official SAM3.1 checkpoint path and frozen digest."""

    value = environ.get(SAM31_CHECKPOINT_ENV, "").strip()
    hint = (
        f"export {SAM31_CHECKPOINT_ENV}=SAM31_CHECKPOINT_ABSOLUTE_PATH"
    )
    if not value:
        return Diagnostic(
            "sam31_checkpoint",
            "error",
            f"{SAM31_CHECKPOINT_ENV} is not set",
            hint,
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        return Diagnostic(
            "sam31_checkpoint",
            "error",
            f"checkpoint path must be absolute: {path}",
            hint,
        )
    if not path.is_file():
        return Diagnostic(
            "sam31_checkpoint",
            "error",
            f"checkpoint file does not exist: {path}",
            hint,
        )
    try:
        actual = _sha256(path)
    except OSError as exc:
        return Diagnostic(
            "sam31_checkpoint",
            "error",
            f"cannot read checkpoint: {exc}",
            "check the checkpoint path and file permissions",
        )
    if actual != expected_sha256:
        return Diagnostic(
            "sam31_checkpoint",
            "error",
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}",
            "use the protocol-pinned SAM3.1 checkpoint",
        )
    return Diagnostic(
        "sam31_checkpoint",
        "ok",
        f"{path} (SHA-256 verified)",
    )


def _default_torch_probe() -> Mapping[str, Any]:
    import torch

    return {
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        ),
    }


def diagnose_imports(
    name: str,
    modules: tuple[str, ...],
    *,
    find_spec: FindSpec | None = None,
    importer: Importer | None = None,
    install_hint: str,
) -> Diagnostic:
    """Import each named module and report missing or broken dependencies."""

    find_spec = find_spec or importlib.util.find_spec
    importer = importer or importlib.import_module
    missing: list[str] = []
    broken: list[str] = []
    for module in modules:
        try:
            available = find_spec(module) is not None
        except Exception as exc:
            broken.append(f"{module} ({type(exc).__name__}: {exc})")
            continue
        if not available:
            missing.append(module)
            continue
        try:
            importer(module)
        except Exception as exc:
            broken.append(f"{module} ({type(exc).__name__}: {exc})")
    if missing:
        detail = "missing modules: " + ", ".join(missing)
    else:
        detail = ""
    if broken:
        if detail:
            detail += "; "
        detail += "failed imports: " + ", ".join(broken)
    if missing or broken:
        return Diagnostic(name, "error", detail, install_hint)
    return Diagnostic(name, "ok", "available: " + ", ".join(modules))


def diagnose_runtime(
    *,
    which: Which = shutil.which,
    find_spec: FindSpec = importlib.util.find_spec,
    environ: Mapping[str, str] = os.environ,
    torch_probe: TorchProbe = _default_torch_probe,
    importer: Importer = importlib.import_module,
    include_checkpoint: bool = True,
) -> list[Diagnostic]:
    """Diagnose the system and model runtime needed by official evaluation."""

    diagnostics: list[Diagnostic] = []
    for executable, hint in (
        ("git", "install Git and ensure it is on PATH"),
        ("ffmpeg", "install ffmpeg and ensure it is on PATH"),
        ("ffprobe", "install ffmpeg/ffprobe and ensure both are on PATH"),
    ):
        resolved = which(executable)
        diagnostics.append(Diagnostic(
            executable,
            "ok" if resolved else "error",
            resolved or "not found on PATH",
            None if resolved else hint,
        ))

    diagnostics.append(diagnose_imports(
        "scene_evaluation_dependencies",
        SCENE_EVALUATION_IMPORTS,
        find_spec=find_spec,
        importer=importer,
        install_hint='python -m pip install -e ".[scene-evaluation]"',
    ))
    diagnostics.append(diagnose_imports(
        "sam31_dependency",
        ("sam3.model_builder",),
        find_spec=find_spec,
        importer=importer,
        install_hint='python -m pip install -e ".[sam31-evaluation]"',
    ))

    try:
        torch_info = dict(torch_probe())
    except Exception as exc:  # diagnostic boundary: report, never crash
        diagnostics.append(Diagnostic(
            "cuda",
            "error",
            f"PyTorch runtime probe failed: {type(exc).__name__}: {exc}",
            "install a CUDA-compatible PyTorch build and verify the driver",
        ))
    else:
        device_count = int(torch_info.get("device_count", 0))
        available = bool(torch_info.get("cuda_available")) and device_count > 0
        detail = (
            f"torch={torch_info.get('version', 'unknown')} "
            f"cuda={torch_info.get('cuda_version') or 'unavailable'} "
            f"devices={device_count}"
        )
        diagnostics.append(Diagnostic(
            "cuda",
            "ok" if available else "error",
            detail,
            None if available else (
                "install a CUDA-compatible PyTorch build and expose a GPU"
            ),
        ))

    if include_checkpoint:
        diagnostics.append(diagnose_checkpoint(environ=environ))
    try:
        importer("physbench.evaluation.common.csti.metric")
    except Exception as exc:  # diagnostic boundary: preserve the actual cause
        diagnostics.append(Diagnostic(
            "evaluator_smoke",
            "error",
            f"CSTI evaluator import failed: {type(exc).__name__}: {exc}",
            "reinstall the evaluator extras, then rerun `physbench doctor`",
        ))
    else:
        diagnostics.append(Diagnostic(
            "evaluator_smoke",
            "ok",
            "CSTI evaluator import smoke passed",
        ))
    return diagnostics


__all__ = [
    "Diagnostic",
    "SAM31_CHECKPOINT_ENV",
    "SAM31_CHECKPOINT_SHA256",
    "SCENE_EVALUATION_DEPENDENCIES",
    "diagnose_checkpoint",
    "diagnose_imports",
    "diagnose_runtime",
]
