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
    value = case.get("assets", {}).get("physics_reference_video")
    if value and case.get("has_real_reference_video", False):
        mode = "same_case_reference"
        parent_id = None
    else:
        parent_id = case.get("provenance", {}).get("parent_case_id")
        if not parent_id:
            raise VideoProtocolError(
                "no_trustworthy_physics_reference",
                "case has neither a real reference nor a physics-identical parent",
            )
        parent = request.case_catalog.get(parent_id)
        if parent is None:
            raise VideoProtocolError(
                "ood_parent_missing",
                f"parent case is absent from dataset: {parent_id}",
            )
        if parent.get("physics") != case.get("physics"):
            raise VideoProtocolError(
                "ood_parent_physics_mismatch",
                "case and parent do not have identical physics labels",
            )
        value = parent.get("assets", {}).get("physics_reference_video")
        if not value:
            raise VideoProtocolError(
                "no_trustworthy_physics_reference",
                "physics-identical parent has no reference video",
            )
        mode = "parent_physics_reference"
    path = resolve_asset(request.asset_root, str(value))
    if not path.is_file():
        raise VideoProtocolError(
            "reference_video_missing", f"reference video not found: {path}"
        )
    return path, mode, parent_id
