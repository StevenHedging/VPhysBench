from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..domain import DatasetSnapshot
from ..identifiers import require_safe_id
from ..io import canonical_sha256, load_json, load_jsonl, sha256_file


FORBIDDEN_CASE_KEYS = {
    "input_views",
    "physical_parameters",
    "prompt_profile_id",
    "view_a_split",
}
REQUIRED_CASE_KEYS_V3 = {
    "schema_version",
    "case_id",
    "scene_id",
    "assets",
    "text",
    "physics",
    "appearance",
    "temporal",
    "provenance",
    "ood",
    "has_real_reference_video",
}
REQUIRED_CASE_KEYS_V4 = REQUIRED_CASE_KEYS_V3 - {"ood"}
GENERALIZATION_REGIMES = {"id", "ood", "mixed"}
GENERALIZATION_FACTOR_CATEGORIES = {
    "physical_parameter",
    "object_composition",
    "interaction_structure",
    "appearance",
    "environment",
    "acquisition",
}


def _load_directory_json(directory: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        value = load_json(path)
        key = value.get("scene_id")
        if not isinstance(key, str):
            raise ValueError(f"scene config lacks scene_id: {path}")
        if key in values:
            raise ValueError(f"duplicate scene config: {key}")
        values[key] = value
    return values


def _validate_scene_v2(scene: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "scene_id",
        "display_name",
        "structured_physics_parameters",
        "non_conditionable_physics_parameters",
        "generalization_factors",
        "constraints",
        "metric_spec",
    }
    if set(scene) != required:
        raise ValueError(
            f"scene {scene.get('scene_id')} schema 2.0 fields must be "
            f"{sorted(required)}"
        )
    if scene["schema_version"] != "2.0":
        raise ValueError(f"scene {scene.get('scene_id')} must use schema 2.0")
    require_safe_id(scene["scene_id"], label="scene.scene_id")
    if (
        not isinstance(scene["display_name"], str)
        or not scene["display_name"].strip()
    ):
        raise ValueError(
            f"scene {scene['scene_id']} display_name must be non-empty"
        )
    for field in (
        "structured_physics_parameters",
        "non_conditionable_physics_parameters",
    ):
        values = scene[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(
                f"scene {scene['scene_id']} {field} must be a string array"
            )
        if len(values) != len(set(values)):
            raise ValueError(f"scene {scene['scene_id']} {field} has duplicates")
    factors = scene["generalization_factors"]
    if not isinstance(factors, list):
        raise ValueError(
            f"scene {scene['scene_id']} generalization_factors must be an array"
        )
    factor_names: list[str] = []
    for index, factor in enumerate(factors):
        if not isinstance(factor, dict) or set(factor) != {
            "name",
            "category",
            "source",
        }:
            raise ValueError(
                f"scene {scene['scene_id']} factor {index} has invalid fields"
            )
        require_safe_id(
            factor["name"],
            label=f"scene.{scene['scene_id']}.generalization_factors[{index}].name",
        )
        if factor["category"] not in GENERALIZATION_FACTOR_CATEGORIES:
            raise ValueError(
                f"scene {scene['scene_id']} factor {factor['name']} has "
                f"invalid category={factor['category']}"
            )
        if (
            not isinstance(factor["source"], str)
            or not factor["source"].strip()
        ):
            raise ValueError(
                f"scene {scene['scene_id']} factor {factor['name']} needs source"
            )
        factor_names.append(factor["name"])
    if len(factor_names) != len(set(factor_names)):
        raise ValueError(
            f"scene {scene['scene_id']} generalization factor names repeat"
        )
    if not isinstance(scene["constraints"], list) or any(
        not isinstance(value, str) or not value.strip()
        for value in scene["constraints"]
    ):
        raise ValueError(f"scene {scene['scene_id']} constraints must be strings")
    if not isinstance(scene["metric_spec"], dict):
        raise ValueError(f"scene {scene['scene_id']} metric_spec must be an object")


def _assert_no_model_payload(case: dict[str, Any]) -> None:
    invalid = sorted(FORBIDDEN_CASE_KEYS & set(case))
    if invalid:
        raise ValueError(
            f"dataset case {case.get('case_id')} contains model/task fields: {invalid}"
        )
    stack: list[tuple[tuple[str, ...], Any]] = [((), case)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, key)
                if key == "prompt" and child_path != ("text", "prompt"):
                    raise ValueError(
                        f"case {case.get('case_id')} contains prompt outside "
                        "case.text.prompt"
                    )
                stack.append((child_path, child))
        elif isinstance(value, list):
            stack.extend(((*path, str(index)), child) for index, child in enumerate(value))


def _validate_case(case: dict[str, Any], known_scenes: set[str]) -> None:
    schema_version = case.get("schema_version")
    if schema_version not in {"3.0", "4.0", "5.0"}:
        raise ValueError(
            f"case {case.get('case_id')} must use schema_version=3.0, 4.0, or 5.0"
        )
    required = (
        REQUIRED_CASE_KEYS_V3
        if schema_version == "3.0"
        else REQUIRED_CASE_KEYS_V4
    )
    missing = sorted(required - set(case))
    if missing:
        raise ValueError(
            f"dataset case {case.get('case_id')} missing {missing}"
        )
    if schema_version in {"4.0", "5.0"} and "ood" in case:
        raise ValueError(
            f"dataset v4+ case {case.get('case_id')} must not contain view-relative ood"
        )
    if schema_version in {"4.0", "5.0"}:
        allowed = REQUIRED_CASE_KEYS_V4 | {"alignment"}
        unknown = sorted(set(case) - allowed)
        if unknown:
            raise ValueError(
                f"dataset v4+ case {case.get('case_id')} has unknown fields: "
                f"{unknown}"
            )
    require_safe_id(case.get("case_id"), label="case.case_id")
    require_safe_id(case.get("scene_id"), label="case.scene_id")
    _assert_no_model_payload(case)
    if case["scene_id"] not in known_scenes:
        raise ValueError(f"case {case['case_id']} references unknown scene {case['scene_id']}")
    text = case["text"]
    if not isinstance(text, dict):
        raise ValueError(f"case {case['case_id']} text must be an object")
    expected_text_fields = {
        "schema_version",
        "prompt",
        "language",
        "annotation_source",
    }
    if set(text) != expected_text_fields:
        raise ValueError(
            f"case {case['case_id']} text fields must be "
            f"{sorted(expected_text_fields)}"
        )
    if text["schema_version"] != "1.0":
        raise ValueError(
            f"case {case['case_id']} text.schema_version must be 1.0"
        )
    for key in ("prompt", "language", "annotation_source"):
        if not isinstance(text[key], str) or not text[key].strip():
            raise ValueError(
                f"case {case['case_id']} text.{key} must be non-empty"
            )
    physics = case["physics"]
    if not isinstance(physics, dict) or not physics:
        raise ValueError(f"case {case['case_id']} requires structured physics")
    for name, quantity in physics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"case {case['case_id']} physics parameter names must be non-empty"
            )
        if not isinstance(quantity, dict):
            raise ValueError(f"case {case['case_id']} physics.{name} must be an object")
        expected_quantity_fields = {"value", "unit", "annotated"}
        if schema_version == "5.0":
            expected_quantity_fields.add("symbol")
        if set(quantity) != expected_quantity_fields:
            raise ValueError(
                f"case {case['case_id']} physics.{name} fields must be "
                f"{sorted(expected_quantity_fields)}"
            )
        value = quantity["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"case {case['case_id']} physics.{name}.value must be a "
                "finite number"
            )
        if schema_version == "5.0" and value < 0:
            raise ValueError(
                f"case {case['case_id']} physics.{name}.value must be non-negative"
            )
        if (
            not isinstance(quantity["unit"], str)
            or not quantity["unit"].strip()
        ):
            raise ValueError(
                f"case {case['case_id']} physics.{name}.unit must be non-empty"
            )
        if not isinstance(quantity["annotated"], bool):
            raise ValueError(
                f"case {case['case_id']} physics.{name}.annotated must be boolean"
            )
        if schema_version == "5.0" and (
            not isinstance(quantity["symbol"], str)
            or not quantity["symbol"].strip()
        ):
            raise ValueError(
                f"case {case['case_id']} physics.{name}.symbol must be non-empty"
            )
    if not isinstance(case["assets"], dict):
        raise ValueError(f"case {case['case_id']} assets must be an object")
    for name, value in case["assets"].items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"case {case['case_id']} asset names must be non-empty"
            )
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(
                f"case {case['case_id']} assets.{name} must be null or "
                "a non-empty relative path"
            )
        if isinstance(value, str):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"case {case['case_id']} assets.{name} must be a "
                    f"relative path inside asset_root: {value}"
                )
    if not isinstance(case["has_real_reference_video"], bool):
        raise ValueError(
            f"case {case['case_id']} has_real_reference_video must be boolean"
        )
    for field in ("appearance", "temporal", "provenance"):
        if not isinstance(case[field], dict):
            raise ValueError(
                f"case {case['case_id']} {field} must be an object"
            )
    alignment = case.get("alignment")
    if alignment is not None and not isinstance(alignment, dict):
        raise ValueError(
            f"case {case['case_id']} alignment must be null or an object"
        )


def _validate_factor_list(
    value: Any,
    *,
    label: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    factors: list[dict[str, str]] = []
    for index, factor in enumerate(value):
        if not isinstance(factor, dict) or set(factor) != {"name", "category"}:
            raise ValueError(
                f"{label}[{index}] must contain exactly name and category"
            )
        name = factor["name"]
        category = factor["category"]
        require_safe_id(name, label=f"{label}[{index}].name")
        if category not in GENERALIZATION_FACTOR_CATEGORIES:
            raise ValueError(
                f"{label}[{index}].category must be one of "
                f"{sorted(GENERALIZATION_FACTOR_CATEGORIES)}"
            )
        factors.append({"name": name, "category": category})
    if len({item["name"] for item in factors}) != len(factors):
        raise ValueError(f"{label} must not repeat factor names")
    return factors


def _validate_view_a_v3(
    view: dict[str, Any],
    *,
    case_ids: set[str],
    case_scene: dict[str, str],
) -> None:
    if view.get("view_id") != "view_a":
        raise ValueError(
            "dataset view key view_a does not match its view_id"
        )
    if view.get("coverage") != "complete":
        raise ValueError("dataset view view_a schema 3.0 must have complete coverage")
    semantics = view.get("split_semantics")
    if not isinstance(semantics, dict):
        raise ValueError("dataset view view_a split_semantics must be an object")
    if semantics.get("primary_partitions") != ["train", "test"]:
        raise ValueError(
            "dataset view view_a primary_partitions must be ['train', 'test']"
        )
    if semantics.get("generalization_regimes") != ["id", "ood", "mixed"]:
        raise ValueError(
            "dataset view view_a generalization_regimes must be "
            "['id', 'ood', 'mixed']"
        )
    if semantics.get("regime_is_relative_to") != "view_a.train":
        raise ValueError(
            "dataset view view_a regimes must be relative to view_a.train"
        )

    scenes = view.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("dataset view view_a scenes must be non-empty")
    train_ids: list[str] = []
    test_ids: list[str] = []
    for scene_id, groups in scenes.items():
        if not isinstance(scene_id, str) or not isinstance(groups, dict):
            raise ValueError("dataset view view_a has an invalid scene bucket")
        if set(groups) != {"train", "test"}:
            raise ValueError(
                f"dataset view view_a scene {scene_id} must contain train/test"
            )
        for partition in ("train", "test"):
            members = groups[partition]
            if not isinstance(members, list) or any(
                not isinstance(case_id, str) or not case_id
                for case_id in members
            ):
                raise ValueError(
                    f"dataset view view_a has invalid {scene_id}/{partition} members"
                )
            for case_id in members:
                actual_scene = case_scene.get(case_id)
                if actual_scene is not None and actual_scene != scene_id:
                    raise ValueError(
                        f"dataset view view_a places case {case_id} from scene "
                        f"{actual_scene} in scene bucket {scene_id}"
                    )
            (train_ids if partition == "train" else test_ids).extend(members)
    all_ids = train_ids + test_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("dataset view view_a contains duplicate case IDs")
    unknown = set(all_ids) - case_ids
    if unknown:
        raise ValueError(
            f"dataset view view_a references unknown cases: {sorted(unknown)}"
        )
    if set(all_ids) != case_ids:
        raise ValueError("dataset view view_a must cover the complete case set")

    annotations = view.get("test_annotations")
    if not isinstance(annotations, dict):
        raise ValueError("dataset view view_a test_annotations must be an object")
    if set(annotations) != set(test_ids):
        raise ValueError(
            "dataset view view_a test_annotations must exactly cover test cases"
        )
    for case_id, annotation in annotations.items():
        if not isinstance(annotation, dict) or set(annotation) != {
            "generalization_regime",
            "ood_factors",
            "co_varying_factors",
            "rationale",
        }:
            raise ValueError(
                f"view_a test annotation {case_id} has invalid fields"
            )
        regime = annotation["generalization_regime"]
        if regime not in GENERALIZATION_REGIMES:
            raise ValueError(
                f"view_a test annotation {case_id} has invalid regime={regime}"
            )
        ood_factors = _validate_factor_list(
            annotation["ood_factors"],
            label=f"view_a.test_annotations.{case_id}.ood_factors",
        )
        co_varying = _validate_factor_list(
            annotation["co_varying_factors"],
            label=f"view_a.test_annotations.{case_id}.co_varying_factors",
        )
        rationale = annotation["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(
                f"view_a test annotation {case_id} rationale must be non-empty"
            )
        if regime == "id" and (ood_factors or co_varying):
            raise ValueError(
                f"view_a ID annotation {case_id} cannot declare held-out factors"
            )
        if regime == "ood" and (not ood_factors or co_varying):
            raise ValueError(
                f"view_a OOD annotation {case_id} requires only OOD factors"
            )
        if regime == "mixed" and (not ood_factors or not co_varying):
            raise ValueError(
                f"view_a mixed annotation {case_id} requires OOD and co-varying factors"
            )

    expected_case_set_digest = canonical_sha256(sorted(all_ids))
    if view.get("case_set_sha256") != expected_case_set_digest:
        raise ValueError("dataset view view_a case_set_sha256 mismatch")


def _validate_views(
    cases: tuple[dict[str, Any], ...], views: dict[str, dict[str, Any]]
) -> None:
    case_ids = {case["case_id"] for case in cases}
    case_scene = {
        case["case_id"]: case["scene_id"]
        for case in cases
    }
    for view_id, view in views.items():
        if view_id == "view_a" and view.get("schema_version") == "3.0":
            _validate_view_a_v3(
                view,
                case_ids=case_ids,
                case_scene=case_scene,
            )
            continue
        if view.get("schema_version") != "2.0":
            raise ValueError(
                f"dataset view {view_id} must use schema_version=2.0, or "
                "view_a schema_version=3.0"
            )
        if view.get("view_id") != view_id:
            raise ValueError(
                f"dataset view key {view_id} does not match "
                f"view_id={view.get('view_id')!r}"
            )
        scenes = view.get("scenes")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError(f"dataset view {view_id} scenes must be non-empty")
        ids: list[str] = []
        for scene_id, groups in scenes.items():
            if not isinstance(scene_id, str) or not isinstance(groups, dict):
                raise ValueError(
                    f"dataset view {view_id} has an invalid scene bucket"
                )
            for group_id, members in groups.items():
                if (
                    not isinstance(group_id, str)
                    or not group_id
                    or not isinstance(members, list)
                    or any(
                        not isinstance(case_id, str) or not case_id
                        for case_id in members
                    )
                ):
                    raise ValueError(
                        f"dataset view {view_id} has an invalid group "
                        f"{scene_id}/{group_id}"
                    )
                for case_id in members:
                    actual_scene = case_scene.get(case_id)
                    if actual_scene is not None and actual_scene != scene_id:
                        raise ValueError(
                            f"dataset view {view_id} places case {case_id} "
                            f"from scene {actual_scene} in scene bucket "
                            f"{scene_id}"
                        )
                ids.extend(members)
        if len(ids) != len(set(ids)):
            raise ValueError(f"dataset view {view_id} contains duplicate case IDs")
        unknown = set(ids) - case_ids
        if unknown:
            raise ValueError(
                f"dataset view {view_id} references unknown cases: {sorted(unknown)}"
            )
        coverage = view.get("coverage", "complete")
        if coverage not in {"complete", "subset"}:
            raise ValueError(
                f"dataset view {view_id} has invalid coverage={coverage}"
            )
        if coverage == "complete" and set(ids) != case_ids:
            raise ValueError(
                f"dataset view {view_id} must cover the complete case set"
            )
        if coverage == "subset" and not ids:
            raise ValueError(f"dataset subset view {view_id} cannot be empty")
        expected_case_set_digest = canonical_sha256(sorted(ids))
        if view.get("case_set_sha256") != expected_case_set_digest:
            raise ValueError(
                f"dataset view {view_id} case_set_sha256 mismatch"
            )


def _validate_v4_scene_case_and_view_contracts(
    cases: tuple[dict[str, Any], ...],
    scenes: dict[str, dict[str, Any]],
    view_a: dict[str, Any],
) -> None:
    by_id = {case["case_id"]: case for case in cases}
    for case in cases:
        scene = scenes[case["scene_id"]]
        conditionable = set(scene["structured_physics_parameters"])
        non_conditionable = set(scene["non_conditionable_physics_parameters"])
        overlap = conditionable & non_conditionable
        if overlap:
            raise ValueError(
                f"scene {case['scene_id']} repeats physics parameters across "
                f"conditionable sets: {sorted(overlap)}"
            )
        for name, quantity in case["physics"].items():
            expected = (
                conditionable
                if quantity["annotated"]
                else non_conditionable
            )
            if name not in expected:
                state = (
                    "structured"
                    if quantity["annotated"]
                    else "non-conditionable"
                )
                raise ValueError(
                    f"case {case['case_id']} physics.{name} is {state} but absent "
                    f"from its scene contract"
                )

    for case_id, annotation in view_a["test_annotations"].items():
        case = by_id[case_id]
        declared = {
            factor["name"]: factor["category"]
            for factor in scenes[case["scene_id"]]["generalization_factors"]
        }
        for factor in annotation["ood_factors"]:
            if declared.get(factor["name"]) != factor["category"]:
                raise ValueError(
                    f"view_a factor {factor['name']} for case {case_id} is not "
                    "declared by the scene with the same category"
                )
        structured = set(
            scenes[case["scene_id"]]["structured_physics_parameters"]
        )
        for factor in annotation["co_varying_factors"]:
            if factor["category"] == "physical_parameter":
                valid = factor["name"] in structured
            else:
                valid = declared.get(factor["name"]) == factor["category"]
            if not valid:
                raise ValueError(
                    f"view_a co-varying factor {factor['name']} for case "
                    f"{case_id} is absent from the scene contract"
                )


def _load_asset_lock(
    root: Path,
    descriptor: dict[str, Any],
    cases: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    relative = descriptor.get("asset_lock")
    if relative is None:
        return None
    lock = load_json(root / relative)
    if lock.get("schema_version") != "1.0":
        raise ValueError("dataset asset lock must use schema_version=1.0")
    if lock.get("dataset_id") != descriptor.get("dataset_id"):
        raise ValueError("dataset asset lock dataset_id mismatch")
    if lock.get("release") != descriptor.get("release"):
        raise ValueError("dataset asset lock release mismatch")
    files = lock.get("files")
    if not isinstance(files, list):
        raise ValueError("dataset asset lock files must be a list")
    paths = [item.get("path") for item in files]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("dataset asset lock contains an invalid path")
    if len(paths) != len(set(paths)):
        raise ValueError("dataset asset lock contains duplicate paths")
    for item in files:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"dataset asset lock path must be relative: {path}")
        if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 0:
            raise ValueError(f"dataset asset lock has invalid size: {path}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"dataset asset lock has invalid SHA-256: {path}")
    if lock.get("files_digest") != canonical_sha256(files):
        raise ValueError("dataset asset lock files_digest mismatch")
    locked = set(paths)
    referenced = {
        value
        for case in cases
        for value in case["assets"].values()
        if value
    }
    missing = sorted(referenced - locked)
    if missing:
        raise ValueError(f"dataset asset lock misses referenced assets: {missing}")
    return lock


def _validate_case_physics_annotation(
    case: dict[str, Any],
    asset_root: Path,
) -> None:
    relative = case["assets"].get("physics_annotation")
    if relative is None:
        return
    path = (asset_root / relative).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise ValueError(
            f"case {case['case_id']} physics annotation escapes asset_root"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"case {case['case_id']} missing assets.physics_annotation: {path}"
        )
    document = load_json(path)
    expected_fields = {"schema_version", "case_id", "scene_id", "physics"}
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError(
            f"case {case['case_id']} physics annotation fields must be "
            f"{sorted(expected_fields)}"
        )
    expected_schema = "2.0" if case.get("schema_version") == "5.0" else "1.0"
    if document["schema_version"] != expected_schema:
        raise ValueError(
            f"case {case['case_id']} physics annotation schema must be "
            f"{expected_schema}"
        )
    if document["case_id"] != case["case_id"]:
        raise ValueError(
            f"case {case['case_id']} physics annotation Case mismatch"
        )
    if document["scene_id"] != case["scene_id"]:
        raise ValueError(
            f"case {case['case_id']} physics annotation Scene mismatch"
        )
    if document["physics"] != case["physics"]:
        raise ValueError(
            f"case {case['case_id']} physics annotation differs from inline "
            "case.physics"
        )


def _case_member_path(
    case: dict[str, Any],
    asset_root: Path,
    role: str,
) -> Path:
    assets = case.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(f"case {case.get('case_id')} assets must be an object")
    relative = assets.get(role)
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(
            f"case {case.get('case_id')} requires assets.{role}"
        )
    path = (asset_root / relative).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise ValueError(
            f"case {case.get('case_id')} {role} escapes asset_root"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"case {case.get('case_id')} missing assets.{role}: {path}"
        )
    return path


def _materialize_v5_case_members(
    indexed_case: dict[str, Any],
    asset_root: Path,
) -> dict[str, Any]:
    case_id = indexed_case.get("case_id")
    scene_id = indexed_case.get("scene_id")
    if "text" in indexed_case or "physics" in indexed_case:
        raise ValueError(
            f"schema 5 case index {case_id} must not inline text or physics"
        )

    caption = load_json(_case_member_path(indexed_case, asset_root, "caption"))
    caption_fields = {
        "schema_version",
        "case_id",
        "scene_id",
        "caption",
        "language",
        "annotation_source",
    }
    if not isinstance(caption, dict) or set(caption) != caption_fields:
        raise ValueError(
            f"case {case_id} caption fields must be {sorted(caption_fields)}"
        )
    if caption["schema_version"] != "1.0":
        raise ValueError(f"case {case_id} caption schema must be 1.0")
    if caption["case_id"] != case_id:
        raise ValueError(f"case {case_id} caption Case mismatch")
    if caption["scene_id"] != scene_id:
        raise ValueError(f"case {case_id} caption Scene mismatch")

    physics_document = load_json(
        _case_member_path(indexed_case, asset_root, "physics_annotation")
    )
    physics_fields = {"schema_version", "case_id", "scene_id", "physics"}
    if (
        not isinstance(physics_document, dict)
        or set(physics_document) != physics_fields
    ):
        raise ValueError(
            f"case {case_id} physics annotation fields must be "
            f"{sorted(physics_fields)}"
        )
    if physics_document["schema_version"] != "2.0":
        raise ValueError(f"case {case_id} physics annotation schema must be 2.0")
    if physics_document["case_id"] != case_id:
        raise ValueError(f"case {case_id} physics annotation Case mismatch")
    if physics_document["scene_id"] != scene_id:
        raise ValueError(f"case {case_id} physics annotation Scene mismatch")

    case = dict(indexed_case)
    case["text"] = {
        "schema_version": caption["schema_version"],
        "prompt": caption["caption"],
        "language": caption["language"],
        "annotation_source": caption["annotation_source"],
    }
    case["physics"] = physics_document["physics"]
    return case


def load_dataset(
    path: str | Path,
    *,
    check_assets: bool = False,
    check_asset_hashes: bool = False,
) -> DatasetSnapshot:
    descriptor_path = Path(path).resolve()
    descriptor = load_json(descriptor_path)
    descriptor_schema = descriptor.get("schema_version")
    if descriptor_schema not in {"3.0", "4.0", "5.0"}:
        raise ValueError(
            "dataset descriptor must use schema_version=3.0, 4.0, or 5.0"
        )
    require_safe_id(
        descriptor.get("dataset_id"),
        label="dataset.dataset_id",
    )
    root = descriptor_path.parent
    indexed_cases = tuple(load_jsonl(root / descriptor["cases"]))
    asset_root = (root / descriptor.get("asset_root", ".")).resolve()
    cases = (
        tuple(
            _materialize_v5_case_members(case, asset_root)
            for case in indexed_cases
        )
        if descriptor_schema == "5.0"
        else indexed_cases
    )
    scene_configs = _load_directory_json(root / descriptor["scene_catalog"])
    if not scene_configs:
        raise ValueError("dataset scene catalog is empty")
    if descriptor_schema in {"4.0", "5.0"}:
        for scene in scene_configs.values():
            _validate_scene_v2(scene)
    seen: set[str] = set()
    for case in cases:
        _validate_case(case, set(scene_configs))
        expected_case_schema = descriptor_schema
        if case.get("schema_version") != expected_case_schema:
            raise ValueError(
                f"dataset descriptor schema {descriptor_schema} cannot contain "
                f"case schema {case.get('schema_version')}"
            )
        if case["case_id"] in seen:
            raise ValueError(f"duplicate dataset case ID {case['case_id']}")
        seen.add(case["case_id"])
    views = {
        view_id: load_json(root / relative_path)
        for view_id, relative_path in descriptor["views"].items()
    }
    _validate_views(cases, views)
    if descriptor_schema in {"4.0", "5.0"}:
        _validate_v4_scene_case_and_view_contracts(
            cases,
            scene_configs,
            views["view_a"],
        )
    if descriptor_schema != "5.0":
        for case in cases:
            _validate_case_physics_annotation(case, asset_root)
    asset_lock = _load_asset_lock(root, descriptor, cases)
    if check_asset_hashes and asset_lock is None:
        raise ValueError("cannot verify asset hashes without an asset lock")
    locked_by_path = (
        {item["path"]: item for item in asset_lock["files"]}
        if asset_lock is not None
        else {}
    )
    if check_assets or check_asset_hashes:
        checked_asset_paths: set[str] = set()
        for case in cases:
            for key, value in case["assets"].items():
                if not value:
                    continue
                if value in checked_asset_paths:
                    continue
                path = (asset_root / value).resolve()
                try:
                    path.relative_to(asset_root)
                except ValueError as exc:
                    raise ValueError(
                        f"case {case['case_id']} assets.{key} escapes asset_root"
                    ) from exc
                if not path.is_file():
                    raise FileNotFoundError(
                        f"case {case['case_id']} missing assets.{key}: {path}"
                    )
                locked = locked_by_path.get(value)
                if locked and path.stat().st_size != locked["size_bytes"]:
                    raise ValueError(f"dataset asset size mismatch: {value}")
                if check_asset_hashes and locked:
                    actual = sha256_file(path)
                    if actual != locked["sha256"]:
                        raise ValueError(
                            f"dataset asset SHA-256 mismatch: {value}"
                        )
                checked_asset_paths.add(value)
    digest = canonical_sha256({
        "descriptor": descriptor,
        "cases": cases,
        "views": views,
        "scenes": scene_configs,
        "asset_lock": asset_lock,
    })
    release_relative = descriptor.get("release_manifest")
    if release_relative is not None:
        release_manifest = load_json(root / release_relative)
        if release_manifest.get("schema_version") != "1.0":
            raise ValueError("dataset release manifest must use schema_version=1.0")
        if release_manifest.get("dataset_id") != descriptor.get("dataset_id"):
            raise ValueError("dataset release manifest dataset_id mismatch")
        if release_manifest.get("release") != descriptor.get("release"):
            raise ValueError("dataset release manifest release mismatch")
        if release_manifest.get("dataset_digest") != digest:
            raise ValueError("dataset release manifest dataset_digest mismatch")
        if asset_lock is not None and (
            release_manifest.get("asset_files_digest")
            != asset_lock.get("files_digest")
        ):
            raise ValueError(
                "dataset release manifest asset_files_digest mismatch"
            )
    return DatasetSnapshot(
        root=root,
        descriptor=descriptor,
        cases=cases,
        views=views,
        scene_configs=scene_configs,
        asset_lock=asset_lock,
        digest=digest,
        asset_root=asset_root,
    )
