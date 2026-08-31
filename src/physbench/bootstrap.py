"""Deterministic, dependency-free environment bootstrap for VPhysBench."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CUDA_128_INDEX = "https://download.pytorch.org/whl/cu128"


class BootstrapError(RuntimeError):
    """Raised when a requested bootstrap cannot safely be performed."""


@dataclass(frozen=True)
class Command:
    """One exact command in a bootstrap plan."""

    args: tuple[str, ...]

    def render(self) -> str:
        return shlex.join(self.args)


@dataclass(frozen=True)
class ExistingVenvPreflight:
    """Runtime validation required before reusing an existing environment."""

    profile: str
    interpreter: Path
    configured_version: tuple[int, int]


@dataclass(frozen=True)
class BootstrapPlan:
    """The side-effect-free plan for a selected dependency profile."""

    profile: str
    venv: Path
    actions: tuple[str, ...]
    commands: tuple[Command, ...]
    preflight: ExistingVenvPreflight | None = None


@dataclass(frozen=True)
class _Profile:
    minimum_python: tuple[int, int]
    constraints_name: str
    extras: str
    doctor_level: str
    cuda_torch: bool = False


_PROFILES = {
    "metadata": _Profile(
        minimum_python=(3, 11),
        constraints_name="metadata.txt",
        extras="hub",
        doctor_level="metadata",
    ),
    "evaluation": _Profile(
        minimum_python=(3, 12),
        constraints_name="evaluation-cu128.txt",
        extras="hub,scene-evaluation,sam31-evaluation",
        doctor_level="runtime",
        cuda_torch=True,
    ),
}
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")


def _normalise_root(project_root: Path) -> Path:
    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise BootstrapError(f"project root is not a directory: {root}")
    if not (root / "pyproject.toml").is_file():
        raise BootstrapError(f"project root does not contain pyproject.toml: {root}")
    return root


def _venv_version(venv: Path) -> tuple[int, int]:
    config = venv / "pyvenv.cfg"
    if not config.is_file():
        raise BootstrapError(
            f"existing venv is incompatible (missing pyvenv.cfg): {venv}"
        )
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BootstrapError(f"cannot inspect existing venv {venv}: {exc}") from exc
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "version":
            match = _VERSION_RE.match(value.strip())
            if match:
                return (int(match.group(1)), int(match.group(2)))
    raise BootstrapError(
        f"existing venv is incompatible (missing Python version): {venv}"
    )


def _venv_interpreter_version(interpreter: Path) -> tuple[int, int]:
    """Read the existing venv interpreter's actual major/minor version.

    This intentionally executes the selected executable and therefore belongs
    only to execution preflight, never to side-effect-free planning or dry-run.
    Isolation reduces Python startup hooks, but cannot make a swapped executable
    itself trustworthy.
    """

    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise BootstrapError(
            f"existing venv interpreter is missing or not executable: {interpreter}"
        )
    environment = {
        key: value
        for key in ("SYSTEMROOT", "SystemRoot", "WINDIR")
        if (value := os.environ.get(key))
    }
    try:
        result = subprocess.run(
            [
                str(interpreter),
                "-I",
                "-S",
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BootstrapError(
            f"existing venv interpreter cannot run: {interpreter}"
        ) from exc
    match = _VERSION_RE.match(result.stdout.strip())
    if not match:
        raise BootstrapError(
            f"existing venv interpreter returned an invalid version: {interpreter}"
        )
    return (int(match.group(1)), int(match.group(2)))


def _existing_venv_preflight(
    *, venv: Path, profile: str, selected: _Profile
) -> ExistingVenvPreflight:
    """Perform static, non-executing checks for an existing venv."""

    configured_version = _venv_version(venv)
    interpreter = _venv_python(venv)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise BootstrapError(
            f"existing venv interpreter is missing or not executable: {interpreter}"
        )
    _require_python(
        profile=selected, python_version=configured_version, venv=venv
    )
    return ExistingVenvPreflight(
        profile=profile,
        interpreter=interpreter,
        configured_version=configured_version,
    )


def _validate_existing_venv(preflight: ExistingVenvPreflight) -> None:
    """Validate an existing venv immediately before install commands run."""

    actual_version = _venv_interpreter_version(preflight.interpreter)
    if preflight.configured_version != actual_version:
        raise BootstrapError(
            "existing venv is incompatible: pyvenv.cfg declares Python "
            f"{preflight.configured_version[0]}.{preflight.configured_version[1]}, "
            f"but its interpreter reports Python {actual_version[0]}.{actual_version[1]}"
        )
    _require_python(
        profile=_PROFILES[preflight.profile],
        python_version=actual_version,
        venv=preflight.interpreter.parent.parent,
    )


def _require_python(
    *, profile: _Profile, python_version: tuple[int, int], venv: Path | None = None
) -> None:
    minimum = profile.minimum_python
    if python_version < minimum:
        target = f"Python {minimum[0]}.{minimum[1]}+"
        location = "existing venv" if venv else "bootstrap interpreter"
        raise BootstrapError(
            f"{location} is incompatible with this profile; {target} is required"
        )


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def plan_bootstrap(
    *,
    project_root: Path,
    profile: str,
    python: Path | None = None,
    python_version: tuple[int, int] | None = None,
    venv: Path | None = None,
) -> BootstrapPlan:
    """Return an exact, side-effect-free command plan for *profile*."""

    try:
        selected = _PROFILES[profile]
    except KeyError as exc:
        raise BootstrapError(f"unknown bootstrap profile: {profile}") from exc
    root = _normalise_root(project_root)
    # Keep the selected executable spelling in the emitted plan.  Resolving a
    # venv's ``python`` symlink would make an otherwise identical plan depend
    # on an implementation-specific interpreter installation path.
    interpreter = (python or Path(sys.executable)).expanduser().absolute()
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    _require_python(profile=selected, python_version=version)
    target_venv = (venv or root / ".venv").expanduser().resolve()

    if target_venv.exists():
        preflight = _existing_venv_preflight(
            venv=target_venv, profile=profile, selected=selected
        )
        actions = ("reuse venv", "preflight: validate existing venv interpreter")
        commands: list[Command] = []
    else:
        actions = ("create venv",)
        commands = [Command((str(interpreter), "-m", "venv", str(target_venv)))]
        preflight = None

    environment_python = _venv_python(target_venv)
    constraints = root / "constraints" / selected.constraints_name
    if not constraints.is_file():
        raise BootstrapError(f"constraints file is missing: {constraints}")
    if selected.cuda_torch:
        commands.append(Command((
            str(environment_python), "-m", "pip", "install",
            "--index-url", CUDA_128_INDEX,
            "torch==2.10.0", "torchvision==0.25.0",
        )))
    commands.append(Command((
        str(environment_python), "-m", "pip", "install",
        "--constraint", str(constraints),
        "-e", f"{root}[{selected.extras}]",
    )))
    commands.append(Command((
        str(environment_python), "-m", "physbench", "doctor",
        "--project-root", str(root), "--level", selected.doctor_level,
    )))
    return BootstrapPlan(
        profile=profile,
        venv=target_venv,
        actions=actions,
        commands=tuple(commands),
        preflight=preflight,
    )


def _run_command(command: Command) -> None:
    subprocess.run(command.args, check=True)


def execute_plan(plan: BootstrapPlan) -> None:
    """Execute a plan exactly as rendered by :func:`plan_bootstrap`."""

    if plan.preflight is not None:
        _validate_existing_venv(plan.preflight)
    for command in plan.commands:
        _run_command(command)


def _print_plan(plan: BootstrapPlan) -> None:
    print(f"VPhysBench bootstrap plan ({plan.profile})")
    for action in plan.actions:
        print(f"action: {action}")
    for command in plan.commands:
        print(command.render())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or reuse a VPhysBench environment from a deterministic "
            "command plan."
        )
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    parser.add_argument("--venv", help="venv path (default: PROJECT_ROOT/.venv)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = plan_bootstrap(
            project_root=Path(args.project_root),
            profile=args.profile,
            venv=Path(args.venv) if args.venv else None,
        )
        _print_plan(plan)
        if not args.dry_run:
            execute_plan(plan)
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: bootstrap command failed ({exc.returncode}): {exc}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
