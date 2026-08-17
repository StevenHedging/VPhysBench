"""Fail-closed Dataset path resolution for frozen observation children."""

from __future__ import annotations

from pathlib import Path


def _reject_symlink_components(root: Path, candidate: Path, *, label: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes Dataset asset root") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")


def resolve_dataset_file(
    asset_root: str | Path,
    value: str | Path,
    *,
    label: str,
    allow_absolute: bool = False,
) -> Path:
    """Resolve one existing non-symlink file within a Dataset asset root."""

    root = Path(asset_root).resolve(strict=True)
    try:
        supplied = Path(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a Dataset asset path") from exc
    if supplied.is_absolute():
        if not allow_absolute:
            raise ValueError(f"{label} must be a relative Dataset asset path")
        candidate = supplied.absolute()
    else:
        if ".." in supplied.parts:
            raise ValueError(f"{label} escapes Dataset asset root")
        candidate = (root / supplied).absolute()
    _reject_symlink_components(root, candidate, label=label)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label} escapes Dataset asset root or is missing") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must resolve to a regular file")
    return resolved


__all__ = ["resolve_dataset_file"]
