from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import load_json


REQUIRED_CASE_KEYS = {
    "schema_version", "case_id", "scene_id", "view_a_split", "physical_parameters",
    "appearance", "text", "assets", "has_real_reference_video", "ood", "provenance",
    "input_views",
}


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str
    case_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scene_configs(directory: str | Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(directory).glob("*.json")):
        config = load_json(path)
        scene_id = config.get("scene_id")
        if not isinstance(scene_id, str):
            raise ValueError(f"missing scene_id in {path}")
        if scene_id in configs:
            raise ValueError(f"duplicate scene_id {scene_id}")
        configs[scene_id] = config
    return configs


def _asset_path(value: str, manifest_path: Path, asset_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (asset_root if asset_root is not None else manifest_path.parent) / path


def validate_cases(
    cases: list[dict[str, Any]],
    *,
    manifest_path: str | Path,
    scene_configs: dict[str, dict[str, Any]] | None = None,
    check_assets: bool = False,
    asset_root: str | Path | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    by_id: dict[str, dict[str, Any]] = {}
    manifest = Path(manifest_path)
    root = Path(asset_root) if asset_root else None

    for index, case in enumerate(cases, 1):
        case_id = case.get("case_id") if isinstance(case.get("case_id"), str) else None
        missing = sorted(REQUIRED_CASE_KEYS - case.keys())
        if missing:
            issues.append(Issue("error", "missing_fields", f"row {index}: {missing}", case_id))
            continue
        if case.get("schema_version") != "1.0":
            issues.append(Issue("error", "schema_version", "expected schema_version=1.0", case_id))
        temporal = case.get("temporal", {})
        speed_factor = temporal.get("encoded_to_physical_speed", 1.0)
        if not isinstance(speed_factor, (int, float)) or isinstance(speed_factor, bool) or speed_factor <= 0:
            issues.append(Issue(
                "error", "invalid_time_scale",
                "temporal.encoded_to_physical_speed must be a positive number", case_id,
            ))
        alignment = case.get("alignment")
        if alignment is not None:
            if not isinstance(alignment, dict):
                issues.append(Issue("error", "invalid_alignment", "alignment must be an object", case_id))
            else:
                start_frame = alignment.get("source_start_frame")
                if not isinstance(start_frame, int) or isinstance(start_frame, bool) or start_frame < 0:
                    issues.append(Issue(
                        "error", "invalid_alignment_frame",
                        "alignment.source_start_frame must be a non-negative integer", case_id,
                    ))
                if alignment.get("review_status") != "visually_verified":
                    issues.append(Issue(
                        "error", "unreviewed_alignment",
                        "aligned cases must be visually_verified", case_id,
                    ))
        if case_id in by_id:
            issues.append(Issue("error", "duplicate_case_id", f"duplicate {case_id}", case_id))
        elif case_id is not None:
            by_id[case_id] = case
        else:
            issues.append(Issue("error", "invalid_case_id", "case_id must be a string", None))

        split = case.get("view_a_split")
        level = case.get("ood", {}).get("level")
        factors = case.get("ood", {}).get("factors")
        if split not in {"train", "test_id", "test_ood1"}:
            issues.append(Issue("error", "invalid_split", f"invalid view_a_split={split}", case_id))
        if split in {"train", "test_id"} and (level != "id" or factors != []):
            issues.append(Issue("error", "id_mismatch", "train/test_id must have ood.level=id and no factors", case_id))
        if split == "test_ood1" and (level != "ood1" or not isinstance(factors, list) or not factors):
            issues.append(Issue("error", "ood1_mismatch", "test_ood1 requires non-empty OOD1 factors", case_id))

        assets = case.get("assets", {})
        if alignment is not None:
            if not assets.get("source_video"):
                issues.append(Issue(
                    "error", "missing_alignment_source",
                    "aligned cases require assets.source_video", case_id,
                ))
            if assets.get("reference_video") == assets.get("source_video"):
                issues.append(Issue(
                    "error", "alignment_not_materialized",
                    "aligned reference_video must be distinct from source_video", case_id,
                ))
        has_real = case.get("has_real_reference_video")
        if has_real and not assets.get("reference_video"):
            issues.append(Issue("error", "missing_real_reference", "has_real_reference_video requires assets.reference_video", case_id))
        provenance = case.get("provenance", {})
        if provenance.get("source_kind") == "synthetic_first_frame":
            if not provenance.get("parent_case_id"):
                issues.append(Issue("error", "missing_parent", "synthetic OOD1 requires parent_case_id", case_id))
            if has_real:
                issues.append(Issue("error", "synthetic_marked_real", "synthetic first frame cannot imply a real continuation", case_id))
            if not assets.get("first_frame"):
                issues.append(Issue("error", "missing_first_frame", "synthetic OOD1 requires a first frame", case_id))

        views = case.get("input_views")
        if not isinstance(views, dict) or not views:
            issues.append(Issue("error", "missing_input_views", "at least one input view is required", case_id))
        else:
            # Prompts are run-level conditions resolved from PromptProfiles.  Legacy
            # per-case prompt fields remain accepted, but are no longer required.
            if "i2v" in views and not views["i2v"].get("first_frame"):
                issues.append(Issue("error", "invalid_i2v", "i2v requires first_frame", case_id))
            if "ti2v" in views and not views["ti2v"].get("first_frame"):
                issues.append(Issue("error", "invalid_ti2v", "ti2v requires first_frame", case_id))

        if scene_configs is not None:
            scene = scene_configs.get(case.get("scene_id"))
            if scene is None:
                issues.append(Issue("error", "unknown_scene", f"unknown scene_id={case.get('scene_id')}", case_id))
            else:
                required_parameters = set(scene.get("id_parameters", []))
                actual_parameters = set(case.get("physical_parameters", {}))
                absent = sorted(required_parameters - actual_parameters)
                if absent:
                    issues.append(Issue("error", "missing_physics", f"missing scene parameters {absent}", case_id))
                disallowed = sorted(set(factors or []) - set(scene.get("ood1_factors", [])))
                if disallowed:
                    issues.append(Issue("error", "disallowed_ood1_factor", f"not allowed for scene: {disallowed}", case_id))

        if check_assets:
            for key in ("first_frame", "reference_video", "physics_reference_video", "source_video", "subject_mask"):
                value = assets.get(key)
                if value and not _asset_path(value, manifest, root).is_file():
                    issues.append(Issue("error", "missing_asset", f"{key} does not exist: {value}", case_id))

    for case in cases:
        provenance = case.get("provenance", {})
        parent_id = provenance.get("parent_case_id")
        if not parent_id:
            continue
        case_id = case.get("case_id")
        parent = by_id.get(parent_id)
        if parent is None:
            issues.append(Issue("error", "parent_not_found", f"parent {parent_id} not found", case_id))
            continue
        if parent.get("scene_id") != case.get("scene_id"):
            issues.append(Issue("error", "parent_scene_mismatch", "parent and child must share scene_id", case_id))
        if parent.get("ood", {}).get("level") != "id":
            issues.append(Issue("error", "parent_not_id", "OOD1 parent must be ID", case_id))
        if parent.get("physical_parameters") != case.get("physical_parameters"):
            issues.append(Issue("error", "ood1_physics_changed", "synthetic OOD1 must preserve parent physical parameters", case_id))

    return issues


def errors(issues: list[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.level == "error"]
