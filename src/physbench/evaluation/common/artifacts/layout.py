"""Run-owned filesystem layout for evaluation visualizations."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from ....io import canonical_sha256


_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_component(value: object, *, fallback: str) -> str:
    """Return one non-traversing path component with a stable suffix."""

    raw = str(value or "").strip()
    if not raw:
        return fallback
    normalized = _UNSAFE_COMPONENT.sub("_", raw).strip("._")[:96]
    if not normalized:
        normalized = fallback
    if normalized == raw and normalized not in {".", ".."}:
        return normalized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def visualization_bundle_directory(
    request: Any,
    *,
    identity_payload: Any,
    fallback_scene: str = "unknown_scene",
) -> tuple[Path, Path, Path]:
    """Resolve one bundle below the owning evaluation's visualization root.

    Normal AtomicRun evaluation supplies ``request.visualization_root`` as
    ``<evaluation_dir>/visualizations``.  Direct evaluator calls fall back to
    a directory inside their Case artifact directory, so this helper never
    routes output to a machine-global location.
    """

    configured_root = getattr(request, "visualization_root", None)
    root = (
        Path(configured_root)
        if configured_root is not None
        else Path(request.artifact_dir) / "visualizations"
    ).expanduser().resolve()
    scene_id = safe_component(
        request.case.get("scene_id", fallback_scene),
        fallback=fallback_scene,
    )
    case_id = safe_component(
        request.case.get("case_id", "unknown_case"),
        fallback="unknown_case",
    )
    seed = request.job.get("seed")
    label = (
        f"seed{int(seed):06d}"
        if isinstance(seed, int) and not isinstance(seed, bool)
        else "evaluation"
    )
    identity = canonical_sha256(identity_payload)[:16]
    relative = Path(scene_id) / case_id / f"{label}-{identity}"
    return root / relative, relative, root


__all__ = ["safe_component", "visualization_bundle_directory"]
