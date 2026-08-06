from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import CaseEvaluationRequest
from .media import VideoProtocolError


def resolve_asset(asset_root: Path, value: str) -> Path:
    path = (asset_root / value).resolve()
    try:
        path.relative_to(asset_root.resolve())
    except ValueError as exc:
        raise VideoProtocolError(
            "reference_path_escape",
            f"reference asset escapes dataset root: {value}",
        ) from exc
    return path


def resolve_physics_reference(
    request: CaseEvaluationRequest,
) -> tuple[Path, str, str | None]:
    case = request.case
    value = case.get("assets", {}).get("reference_video")
    if not value:
        raise VideoProtocolError(
            "no_trustworthy_physics_reference",
            "case has no reference_video",
        )
    path = resolve_asset(request.asset_root, str(value))
    if not path.is_file():
        raise VideoProtocolError(
            "reference_video_missing", f"reference video not found: {path}"
        )
    return path, "same_case_reference", None
