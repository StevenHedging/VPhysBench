"""Read-only catalog projection for V14 reference-observation curation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...datasets.loader import load_dataset
from ...io import load_json
from ..paths import resolve_dataset_file
from .identity import (
    EntityIdentity,
    IdentityIssue,
    audit_entity_order,
    audit_physics_key_binding,
)


@dataclass(frozen=True)
class CurationEntity:
    identity: EntityIdentity
    physics_keys: tuple[str, ...]
    anchor_npz_path: Path
    anchor_png_path: Path | None
    mask_tube_path: Path
    trajectory_path: Path
    anchor_record: dict[str, Any]
    observation_record: dict[str, Any]


@dataclass(frozen=True)
class CurationCase:
    case_id: str
    scene_id: str
    asset_root: Path
    caption: str
    physics: dict[str, Any]
    appearance: dict[str, Any]
    temporal: dict[str, Any]
    assets: dict[str, Any]
    first_frame_path: Path
    reference_video_path: Path
    mask_manifest_path: Path
    observation_manifest_path: Path
    visualization_manifest_path: Path
    mask_manifest: dict[str, Any]
    observation_manifest: dict[str, Any]
    timeline: dict[str, Any]
    quality: dict[str, Any]
    review: dict[str, Any]
    visualization_manifest: dict[str, Any]
    entities: tuple[CurationEntity, ...]


def _asset_path(asset_root: Path, value: Any, *, label: str) -> Path:
    return resolve_dataset_file(asset_root, value, label=label)


def _child_path(
    asset_root: Path,
    parent: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError(f"{label} must stay inside its observation bundle")
    candidate = (parent / supplied).absolute()
    return resolve_dataset_file(
        asset_root,
        candidate,
        label=label,
        allow_absolute=True,
    )


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _identity_from_anchor(record: dict[str, Any], *, label: str) -> EntityIdentity:
    try:
        centroid = tuple(float(value) for value in record["centroid_xy"])
        bbox = tuple(float(value) for value in record["bbox_xyxy"])
        object_id = str(record["object_id"])
        mask_id = str(record["mask_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid identity geometry") from exc
    if len(centroid) != 2 or len(bbox) != 4:
        raise ValueError(f"{label} has invalid identity geometry")
    return EntityIdentity(
        object_id=object_id,
        mask_id=mask_id,
        centroid_xy=(centroid[0], centroid[1]),
        bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )


def _load_case(
    value: dict[str, Any],
    *,
    asset_root: Path,
) -> CurationCase:
    case_id = str(value["case_id"])
    scene_id = str(value["scene_id"])
    assets = _require_dict(value["assets"], label=f"case {case_id} assets")
    first_frame_path = _asset_path(
        asset_root,
        assets["first_frame"],
        label=f"case {case_id} first frame",
    )
    reference_video_path = _asset_path(
        asset_root,
        assets["reference_video"],
        label=f"case {case_id} reference video",
    )
    mask_manifest_path = _asset_path(
        asset_root,
        assets["first_frame_mask_manifest"],
        label=f"case {case_id} mask manifest",
    )
    observation_manifest_path = _asset_path(
        asset_root,
        assets["reference_observation_manifest"],
        label=f"case {case_id} observation manifest",
    )
    visualization_manifest_path = _asset_path(
        asset_root,
        assets["reference_observation_visualization_manifest"],
        label=f"case {case_id} visualization manifest",
    )
    mask_manifest = _require_dict(
        load_json(mask_manifest_path),
        label=f"case {case_id} mask manifest",
    )
    observation_manifest = _require_dict(
        load_json(observation_manifest_path),
        label=f"case {case_id} observation manifest",
    )
    observation_root = observation_manifest_path.parent
    timeline = _require_dict(
        load_json(
            _child_path(
                asset_root,
                observation_root,
                observation_manifest["timeline"]["path"],
                label=f"case {case_id} observation timeline",
            )
        ),
        label=f"case {case_id} observation timeline",
    )
    quality = _require_dict(
        load_json(
            _child_path(
                asset_root,
                observation_root,
                observation_manifest["quality"]["path"],
                label=f"case {case_id} observation quality",
            )
        ),
        label=f"case {case_id} observation quality",
    )
    review_path = _child_path(
        asset_root,
        observation_root,
        "review.json",
        label=f"case {case_id} observation review",
    )
    review = _require_dict(
        load_json(review_path),
        label=f"case {case_id} observation review",
    )
    visualization_manifest = _require_dict(
        load_json(visualization_manifest_path),
        label=f"case {case_id} visualization manifest",
    )

    observation_by_object = {
        str(record.get("object_id")): record
        for record in observation_manifest.get("entities", [])
        if isinstance(record, dict)
    }
    entities: list[CurationEntity] = []
    for index, raw_anchor in enumerate(mask_manifest.get("instances", [])):
        anchor = _require_dict(
            raw_anchor,
            label=f"case {case_id} anchor instance {index}",
        )
        identity = _identity_from_anchor(
            anchor,
            label=f"case {case_id} anchor instance {index}",
        )
        observation = observation_by_object.get(identity.object_id)
        if not isinstance(observation, dict):
            raise ValueError(
                f"case {case_id} has no observation entity for "
                f"{identity.object_id}"
            )
        anchor_png = anchor.get("asset")
        entities.append(
            CurationEntity(
                identity=identity,
                physics_keys=tuple(str(key) for key in anchor.get("physics_keys", [])),
                anchor_npz_path=_asset_path(
                    asset_root,
                    anchor["npz_asset"],
                    label=f"case {case_id} {identity.object_id} anchor NPZ",
                ),
                anchor_png_path=(
                    _asset_path(
                        asset_root,
                        anchor_png,
                        label=f"case {case_id} {identity.object_id} anchor PNG",
                    )
                    if anchor_png
                    else None
                ),
                mask_tube_path=_child_path(
                    asset_root,
                    observation_root,
                    observation["mask_tube"]["path"],
                    label=f"case {case_id} {identity.object_id} mask tube",
                ),
                trajectory_path=_child_path(
                    asset_root,
                    observation_root,
                    observation["trajectory"]["path"],
                    label=f"case {case_id} {identity.object_id} trajectory",
                ),
                anchor_record=dict(anchor),
                observation_record=dict(observation),
            )
        )
    return CurationCase(
        case_id=case_id,
        scene_id=scene_id,
        asset_root=asset_root,
        caption=str(value["text"]["prompt"]),
        physics=_require_dict(value["physics"], label=f"case {case_id} physics"),
        appearance=_require_dict(
            value["appearance"], label=f"case {case_id} appearance"
        ),
        temporal=_require_dict(value["temporal"], label=f"case {case_id} temporal"),
        assets=dict(assets),
        first_frame_path=first_frame_path,
        reference_video_path=reference_video_path,
        mask_manifest_path=mask_manifest_path,
        observation_manifest_path=observation_manifest_path,
        visualization_manifest_path=visualization_manifest_path,
        mask_manifest=mask_manifest,
        observation_manifest=observation_manifest,
        timeline=timeline,
        quality=quality,
        review=review,
        visualization_manifest=visualization_manifest,
        entities=tuple(entities),
    )


def load_curation_cases(
    dataset_path: str | Path,
    *,
    case_ids: Iterable[str] | None = None,
) -> tuple[CurationCase, ...]:
    snapshot = load_dataset(dataset_path)
    requested = None if case_ids is None else set(case_ids)
    known = {str(case["case_id"]) for case in snapshot.cases}
    if requested is not None:
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown requested Case IDs: {unknown}")
    return tuple(
        _load_case(case, asset_root=snapshot.asset_root)
        for case in snapshot.cases
        if requested is None or case["case_id"] in requested
    )


def _object_ids(case: CurationCase) -> tuple[str, ...]:
    objects = case.physics.get("objects")
    if not isinstance(objects, dict):
        return ()
    return tuple(str(value) for value in objects)


def audit_case_bindings(case: CurationCase) -> tuple[IdentityIssue, ...]:
    issues: list[IdentityIssue] = []
    object_ids = _object_ids(case)
    expected_ids = tuple(f"object_{index}" for index in range(1, len(object_ids) + 1))
    entity_ids = tuple(entity.identity.object_id for entity in case.entities)
    expected_masks = tuple(f"{index:02d}" for index in range(1, len(object_ids) + 1))
    observed_masks = tuple(entity.identity.mask_id for entity in case.entities)
    for code, expected, observed in (
        ("physics_object_ids_non_contiguous", expected_ids, object_ids),
        ("anchor_entity_set_mismatch", object_ids, entity_ids),
        ("anchor_mask_ids_non_contiguous", expected_masks, observed_masks),
    ):
        if expected != observed:
            issues.append(
                IdentityIssue(
                    code=code,
                    severity="error",
                    message=f"case {case.case_id} expected {expected}, observed {observed}",
                    expected_order=expected,
                    observed_order=observed,
                )
            )
    objects = case.physics.get("objects", {})
    for entity in case.entities:
        quantities = objects.get(entity.identity.object_id, {})
        quantity_names = quantities if isinstance(quantities, dict) else ()
        issues.extend(
            audit_physics_key_binding(
                object_id=entity.identity.object_id,
                quantity_names=quantity_names,
                physics_keys=entity.physics_keys,
            )
        )
    symbols: list[str] = []
    for group in (case.physics.get("objects", {}), case.physics.get("environment", {})):
        if not isinstance(group, dict):
            continue
        values = group.values()
        if group is case.physics.get("objects"):
            values = (
                quantity
                for object_quantities in group.values()
                if isinstance(object_quantities, dict)
                for quantity in object_quantities.values()
            )
        for quantity in values:
            if isinstance(quantity, dict) and isinstance(quantity.get("symbol"), str):
                symbols.append(quantity["symbol"])
    missing_symbols = tuple(sorted(symbol for symbol in symbols if symbol not in case.caption))
    if missing_symbols:
        issues.append(
            IdentityIssue(
                code="caption_missing_physics_symbols",
                severity="error",
                message=f"case {case.case_id} caption misses symbols {missing_symbols}",
                missing_keys=missing_symbols,
            )
        )
    review_entities = case.review.get("entity_reviews", {})
    review_ids = tuple(review_entities) if isinstance(review_entities, dict) else ()
    if review_ids != object_ids:
        issues.append(
            IdentityIssue(
                code="review_entity_set_mismatch",
                severity="error",
                message=(
                    f"case {case.case_id} review entities {review_ids} differ "
                    f"from physics entities {object_ids}"
                ),
                expected_order=object_ids,
                observed_order=review_ids,
            )
        )
    return tuple(issues)


def audit_visual_order(case: CurationCase) -> tuple[IdentityIssue, ...]:
    return audit_entity_order(
        case.scene_id,
        (entity.identity for entity in case.entities),
    )


__all__ = [
    "CurationCase",
    "CurationEntity",
    "audit_case_bindings",
    "audit_visual_order",
    "load_curation_cases",
]
