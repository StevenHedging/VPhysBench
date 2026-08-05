#!/usr/bin/env python3
"""Create metadata-only Dataset 6.0.0 with train/test View A semantics.

Collision cases are repartitioned with replicate-group leakage protection and
explicit ID/OOD strata. Other scenes preserve the 5.1.0 training membership
and recover every remaining valid Case to test. Train-relative generalization
labels live in View A instead of Case records. Media files are neither copied
nor modified.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from physbench.datasets import load_dataset
from physbench.io import canonical_sha256, load_json, write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASES_ROOT = PROJECT_ROOT / "datasets/releases"
BASE_ROOT = RELEASES_ROOT / "5.1.0"
OUTPUT_ROOT = RELEASES_ROOT / "6.0.0"
COLLISION_CURATION_ROOT = (
    PROJECT_ROOT
    / "datasets/provenance/source_docs/20260730_parabolic_collision/collision"
)
DATASET_ID = "physics_video_six_scene_v6"
RELEASE = "6.0.0"
COLLISION_SPLIT_POLICY = "collision_stratified_replicate_safe_v2"
COLLISION_TEST_ID_FRACTION = 0.20
COLLISION_OOD_STRUCTURE = "two_ball_opposed_incident"

FACTOR_MIGRATIONS = {
    "background": ("background", "environment"),
    "ball_material": ("ball_material", "object_composition"),
    "collision_pair": ("ball_spec_composition", "object_composition"),
    "moving_object_composition": (
        "moving_object_composition",
        "object_composition",
    ),
}

SCENE_FACTOR_SOURCES = {
    "collision_1d": {
        "ball_spec_composition": (
            "object_composition",
            "appearance.ball_sequence",
        ),
        "ball_material": ("object_composition", "appearance.ball_materials"),
        "collision_structure": (
            "interaction_structure",
            "appearance.collision_structure",
        ),
        "background": ("environment", "appearance.background"),
    },
    "free_fall": {
        "ball_material": ("object_composition", "appearance.ball_material"),
        "background": ("environment", "appearance.background"),
    },
    "inclined_plane_slide": {
        "background": ("environment", "appearance.background"),
        "block_appearance": ("appearance", "appearance.block"),
        "track_material": ("environment", "appearance.track_material"),
        "viewpoint": ("acquisition", "appearance.camera"),
    },
    "parabolic_motion": {
        "background": ("environment", "appearance.background"),
        "ball_size": ("physical_parameter", "physics.ball_radius"),
        "ball_material": ("object_composition", "appearance.ball_material"),
    },
    "pendulum": {
        "background": ("environment", "appearance.background"),
        "bob_material": ("object_composition", "appearance.bob_material"),
        "support": ("environment", "appearance.support"),
    },
    "uniform_circular_motion": {
        "background": ("environment", "appearance.background"),
        "disk_color": ("appearance", "appearance.disk_color"),
        "moving_object": ("object_composition", "appearance.moving_objects"),
        "moving_object_composition": (
            "object_composition",
            "appearance.moving_objects",
        ),
        "viewpoint": ("acquisition", "appearance.camera"),
    },
}


def _factor(name: str) -> dict[str, str]:
    try:
        migrated_name, category = FACTOR_MIGRATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unmapped active OOD factor {name!r}") from exc
    return {"name": migrated_name, "category": category}


def _case_annotation(
    case: dict[str, Any],
    *,
    old_partition: str | None,
) -> dict[str, Any]:
    old_ood = case["ood"]
    factors = [_factor(name) for name in old_ood["factors"]]
    if old_ood["level"] == "id":
        return {
            "generalization_regime": "id",
            "ood_factors": [],
            "co_varying_factors": [],
            "rationale": (
                "Independent test trial within factor domains represented by "
                "View A training data."
            ),
        }
    if (
        case["scene_id"] == "inclined_plane_slide"
        and old_partition is None
    ):
        return {
            "generalization_regime": "mixed",
            "ood_factors": factors,
            "co_varying_factors": [
                {"name": "incline_angle", "category": "physical_parameter"}
            ],
            "rationale": (
                "An unseen background co-varies with the held-out numerical "
                "incline-angle condition, so the background effect is not isolated."
            ),
        }
    return {
        "generalization_regime": "ood",
        "ood_factors": factors,
        "co_varying_factors": [],
        "rationale": (
            "Held-out factor value or interaction is absent from View A "
            "training while control domains remain represented."
        ),
    }


def _collision_physics_signature(case: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (name, quantity["value"], quantity["unit"], quantity["annotated"])
        for name, quantity in sorted(case["physics"].items())
    )


def _collision_replicate_components(
    cases: list[dict[str, Any]],
) -> dict[str, str]:
    """Join audited near replicates, excluded duplicate pairs and exact trials."""

    case_ids = {case["case_id"] for case in cases}
    parent = {case_id: case_id for case_id in case_ids}

    def find(case_id: str) -> str:
        while parent[case_id] != case_id:
            parent[case_id] = parent[parent[case_id]]
            case_id = parent[case_id]
        return case_id

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            return
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    audit_path = COLLISION_CURATION_ROOT / "curation_audit.jsonl"
    cluster_members: dict[str, list[str]] = defaultdict(list)
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = record["case_id"]
            cluster = record["curation"].get("near_replicate_cluster")
            if case_id in case_ids and cluster is not None:
                cluster_members[cluster].append(case_id)
    for members in cluster_members.values():
        for case_id in members[1:]:
            union(members[0], case_id)

    curation_summary = load_json(COLLISION_CURATION_ROOT / "curation_summary.json")
    for members in curation_summary["duplicate_components"]:
        for case_id in members[1:]:
            union(members[0], case_id)

    exact_physics: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for case in cases:
        exact_physics[_collision_physics_signature(case)].append(case["case_id"])
    for members in exact_physics.values():
        for case_id in members[1:]:
            union(members[0], case_id)

    components: dict[str, list[str]] = defaultdict(list)
    for case_id in sorted(case_ids):
        components[find(case_id)].append(case_id)
    return {
        case_id: min(members)
        for members in components.values()
        for case_id in members
    }


def _collision_is_ood(case: dict[str, Any]) -> bool:
    appearance = case["appearance"]
    return (
        appearance["collision_structure"] == COLLISION_OOD_STRUCTURE
        or "glass" in appearance["ball_materials"]
    )


def _collision_ood_annotation(case: dict[str, Any]) -> dict[str, Any]:
    appearance = case["appearance"]
    factors = [{
        "name": "ball_spec_composition",
        "category": "object_composition",
    }]
    rationale_parts = [
        "the exact ball-spec composition is absent from collision training"
    ]
    if appearance["collision_structure"] == COLLISION_OOD_STRUCTURE:
        factors.append({
            "name": "collision_structure",
            "category": "interaction_structure",
        })
        rationale_parts.append(
            "two opposed incident balls are withheld as an interaction-structure test"
        )
    if "glass" in appearance["ball_materials"]:
        factors.append({
            "name": "ball_material",
            "category": "object_composition",
        })
        rationale_parts.append("glass balls are absent from collision training")
    return {
        "generalization_regime": "ood",
        "ood_factors": factors,
        "co_varying_factors": [],
        "rationale": "; ".join(rationale_parts) + ".",
    }


def _collision_id_annotation() -> dict[str, Any]:
    return {
        "generalization_regime": "id",
        "ood_factors": [],
        "co_varying_factors": [],
        "rationale": (
            "Independent replicate-group test trial whose steel-ball composition, "
            "single-incident interaction structure and velocity range are represented "
            "by collision training cases."
        ),
    }


def _collision_split(
    cases: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    component_by_case = _collision_replicate_components(cases)
    cases_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_component[component_by_case[case["case_id"]]].append(case)

    for component_cases in cases_by_component.values():
        regimes = {_collision_is_ood(case) for case in component_cases}
        strata = {
            (
                tuple(case["appearance"]["ball_sequence"]),
                case["appearance"]["collision_structure"],
            )
            for case in component_cases
        }
        if len(regimes) != 1 or len(strata) != 1:
            raise ValueError(
                "collision replicate component crosses an OOD boundary or stratum: "
                f"{sorted(case['case_id'] for case in component_cases)}"
            )

    ood_cases = sorted(
        (case for case in cases if _collision_is_ood(case)),
        key=lambda case: case["case_id"],
    )
    common_strata: dict[
        tuple[tuple[str, ...], str],
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        if _collision_is_ood(case):
            continue
        stratum = (
            tuple(case["appearance"]["ball_sequence"]),
            case["appearance"]["collision_structure"],
        )
        common_strata[stratum][component_by_case[case["case_id"]]].append(case)

    train_ids: list[str] = []
    id_test_ids: list[str] = []
    stratum_audit: list[dict[str, Any]] = []
    for stratum, units in sorted(common_strata.items()):
        stratum_count = sum(len(unit_cases) for unit_cases in units.values())
        target = max(1, round(stratum_count * COLLISION_TEST_ID_FRACTION))

        velocity_by_unit = {
            component: sum(
                abs(case["physics"]["striker_initial_velocity"]["value"])
                for case in unit_cases
            ) / len(unit_cases)
            for component, unit_cases in units.items()
        }
        protected = {
            min(velocity_by_unit, key=lambda key: (velocity_by_unit[key], key)),
            max(velocity_by_unit, key=lambda key: (velocity_by_unit[key], key)),
        }
        candidates = sorted(
            (component for component in units if component not in protected),
            key=lambda component: canonical_sha256({
                "policy": COLLISION_SPLIT_POLICY,
                "stratum": stratum,
                "case_ids": sorted(
                    case["case_id"] for case in units[component]
                ),
            }),
        )

        subsets: dict[int, tuple[str, ...]] = {0: ()}
        for component in candidates:
            size = len(units[component])
            for subtotal, selected in sorted(
                tuple(subsets.items()), reverse=True
            ):
                updated = subtotal + size
                if updated <= target and updated not in subsets:
                    subsets[updated] = selected + (component,)
        if target not in subsets:
            raise ValueError(
                f"cannot meet collision ID target {target} for stratum {stratum} "
                "without splitting a replicate component"
            )
        test_components = set(subsets[target])
        stratum_train = sorted(
            case["case_id"]
            for component, unit_cases in units.items()
            if component not in test_components
            for case in unit_cases
        )
        stratum_test = sorted(
            case["case_id"]
            for component, unit_cases in units.items()
            if component in test_components
            for case in unit_cases
        )
        train_ids.extend(stratum_train)
        id_test_ids.extend(stratum_test)
        stratum_audit.append({
            "ball_sequence": list(stratum[0]),
            "collision_structure": stratum[1],
            "total": stratum_count,
            "replicate_components": len(units),
            "train": len(stratum_train),
            "test_id": len(stratum_test),
            "test_id_fraction": len(stratum_test) / stratum_count,
            "velocity_endpoint_components_forced_to_train": sorted(protected),
        })

    ood_ids = [case["case_id"] for case in ood_cases]
    annotations = {
        case_id: _collision_id_annotation()
        for case_id in id_test_ids
    }
    annotations.update({
        case["case_id"]: _collision_ood_annotation(case)
        for case in ood_cases
    })
    component_partitions: dict[str, set[str]] = defaultdict(set)
    for case_id in train_ids:
        component_partitions[component_by_case[case_id]].add("train")
    for case_id in id_test_ids + ood_ids:
        component_partitions[component_by_case[case_id]].add("test")
    leaking_components = {
        component: sorted(partitions)
        for component, partitions in component_partitions.items()
        if len(partitions) != 1
    }
    if leaking_components:
        raise ValueError(
            f"collision replicate components cross train/test: {leaking_components}"
        )
    audit = {
        "policy": COLLISION_SPLIT_POLICY,
        "test_id_fraction_within_common_strata": COLLISION_TEST_ID_FRACTION,
        "replicate_group_sources": [
            "curation_audit.near_replicate_cluster",
            "curation_summary.duplicate_components",
            "exact_structured_physics_signature",
        ],
        "replicate_component_count": len(cases_by_component),
        "train_test_replicate_component_overlap": 0,
        "velocity_endpoints_forced_to_train": True,
        "ood_policy": {
            "interaction_structure": [COLLISION_OOD_STRUCTURE],
            "ball_material": ["glass"],
        },
        "strata": stratum_audit,
        "counts": {
            "train": len(train_ids),
            "test_id": len(id_test_ids),
            "test_ood": len(ood_ids),
            "test": len(id_test_ids) + len(ood_ids),
        },
    }
    return (
        sorted(train_ids),
        sorted(id_test_ids + ood_ids),
        dict(sorted(annotations.items())),
        audit,
    )


def _build_view_a(
    cases: list[dict[str, Any]],
    old_view: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old_partition = {
        case_id: partition
        for groups in old_view["scenes"].values()
        for partition, case_ids in groups.items()
        for case_id in case_ids
    }
    old_train = {
        case_id
        for case_id, partition in old_partition.items()
        if partition == "train"
    }
    scenes: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"train": [], "test": []}
    )
    annotations: dict[str, dict[str, Any]] = {}
    collision_cases = [
        case for case in cases if case["scene_id"] == "collision_1d"
    ]
    (
        collision_train,
        collision_test,
        collision_annotations,
        collision_audit,
    ) = _collision_split(collision_cases)
    scenes["collision_1d"] = {
        "train": collision_train,
        "test": collision_test,
    }
    annotations.update(collision_annotations)
    for case in cases:
        case_id = case["case_id"]
        scene_id = case["scene_id"]
        if scene_id == "collision_1d":
            continue
        if case_id in old_train:
            scenes[scene_id]["train"].append(case_id)
            continue
        scenes[scene_id]["test"].append(case_id)
        annotations[case_id] = _case_annotation(
            case,
            old_partition=old_partition.get(case_id),
        )
    frozen_scenes = {
        scene_id: {
            partition: sorted(case_ids)
            for partition, case_ids in groups.items()
        }
        for scene_id, groups in sorted(scenes.items())
    }
    all_ids = sorted(case["case_id"] for case in cases)
    view = {
        "schema_version": "3.0",
        "view_id": "view_a",
        "coverage": "complete",
        "selection_policy": (
            "collision_repartition_other_scenes_v5p1_train_complete_test_v2"
        ),
        "split_semantics": {
            "primary_partitions": ["train", "test"],
            "generalization_regimes": ["id", "ood", "mixed"],
            "regime_is_relative_to": "view_a.train",
            "factor_categories": [
                "physical_parameter",
                "object_composition",
                "interaction_structure",
                "appearance",
                "environment",
                "acquisition",
            ],
        },
        "case_set_sha256": canonical_sha256(all_ids),
        "scenes": frozen_scenes,
        "test_annotations": dict(sorted(annotations.items())),
    }
    counts = {
        scene_id: {
            "total": len(groups["train"]) + len(groups["test"]),
            "train": len(groups["train"]),
            "test": len(groups["test"]),
            "test_regimes": dict(sorted(Counter(
                annotations[case_id]["generalization_regime"]
                for case_id in groups["test"]
            ).items())),
        }
        for scene_id, groups in frozen_scenes.items()
    }
    new_partition = {
        case_id: partition
        for groups in frozen_scenes.values()
        for partition, case_ids in groups.items()
        for case_id in case_ids
    }
    movement: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        case_id = case["case_id"]
        movement[case["scene_id"]][
            f"{old_partition.get(case_id, 'previously_unselected')}->{new_partition[case_id]}"
        ] += 1
    audit = {
        "collision": collision_audit,
        "partition_movement_from_5.1.0": {
            scene_id: dict(sorted(counter.items()))
            for scene_id, counter in sorted(movement.items())
        },
    }
    return view, counts, audit


def _migrate_scene(
    scene: dict[str, Any],
    scene_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    parameter_flags: dict[str, set[bool]] = defaultdict(set)
    for case in scene_cases:
        for name, quantity in case["physics"].items():
            parameter_flags[name].add(bool(quantity["annotated"]))
    factors = []
    for name, (category, source) in SCENE_FACTOR_SOURCES[scene["scene_id"]].items():
        factors.append({
            "name": name,
            "category": category,
            "source": source,
        })
    constraints = [
        constraint
        for constraint in scene.get("constraints", [])
        if "OOD1" not in constraint
        and "test_id" not in constraint
        and "excluded from View A" not in constraint
        and "ID evaluation" not in constraint
    ]
    return {
        "schema_version": "2.0",
        "scene_id": scene["scene_id"],
        "display_name": scene["display_name"],
        "structured_physics_parameters": sorted(
            name for name, flags in parameter_flags.items() if True in flags
        ),
        "non_conditionable_physics_parameters": sorted(
            name for name, flags in parameter_flags.items() if flags == {False}
        ),
        "generalization_factors": factors,
        "constraints": constraints,
        "metric_spec": copy.deepcopy(scene["metric_spec"]),
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {OUTPUT_ROOT}")
    base = load_dataset(BASE_ROOT / "dataset.json")
    old_cases = [copy.deepcopy(case) for case in base.cases]
    view_a, counts, split_details = _build_view_a(
        old_cases,
        base.views["view_a"],
    )
    new_cases = []
    for case in old_cases:
        migrated = copy.deepcopy(case)
        migrated["schema_version"] = "4.0"
        migrated.pop("ood")
        new_cases.append(migrated)

    stage = Path(tempfile.mkdtemp(prefix=".6.0.0.migrate-", dir=RELEASES_ROOT))
    try:
        (stage / "views").mkdir()
        (stage / "scenes").mkdir()
        write_jsonl(stage / "cases.jsonl", new_cases)
        write_json(stage / "views/view_a.json", view_a)
        write_json(stage / "views/view_b.json", base.views["view_b"])

        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in old_cases:
            by_scene[case["scene_id"]].append(case)
        for scene_id, scene in base.scene_configs.items():
            write_json(
                stage / "scenes" / f"{scene_id}.json",
                _migrate_scene(scene, by_scene[scene_id]),
            )

        for name in (
            "annotation_corrections.json",
            "asset_directory_mapping.json",
            "ball_spec_catalog.json",
        ):
            shutil.copy2(BASE_ROOT / name, stage / name)
        shutil.copy2(
            BASE_ROOT / "split_audit.json",
            stage / "legacy_split_audit_5.1.0.json",
        )
        factor_counts = Counter(
            f"{factor['category']}/{factor['name']}"
            for annotation in view_a["test_annotations"].values()
            for factor in annotation["ood_factors"]
        )
        regime_counts = Counter(
            annotation["generalization_regime"]
            for annotation in view_a["test_annotations"].values()
        )
        write_json(stage / "split_audit.json", {
            "schema_version": "2.0",
            "policy": view_a["selection_policy"],
            "base_release": "5.1.0",
            "base_membership_policy": {
                "collision_1d": (
                    "repartitioned from all valid cases with stratified, "
                    "replicate-safe ID/OOD rules"
                ),
                "other_scenes": (
                    "preserve 5.1.0 train members and recover all other valid "
                    "cases to test"
                ),
            },
            "primary_partitions": ["train", "test"],
            "coverage": "complete",
            "scene_counts": counts,
            "test_regime_counts": dict(sorted(regime_counts.items())),
            "ood_factor_counts": dict(sorted(factor_counts.items())),
            "split_details": split_details,
            "mixed_policy": (
                "OOD factors that co-vary with another held-out factor are "
                "retained in test and labeled mixed."
            ),
            "legacy_split_audit": "legacy_split_audit_5.1.0.json",
        })

        asset_lock = copy.deepcopy(base.asset_lock)
        if asset_lock is None:
            raise ValueError("6.0.0 requires the frozen 5.1.0 asset lock")
        asset_lock["dataset_id"] = DATASET_ID
        asset_lock["release"] = RELEASE
        write_json(stage / "assets.lock.json", asset_lock)

        descriptor = {
            "schema_version": "4.0",
            "dataset_id": DATASET_ID,
            "release": RELEASE,
            "cases": "cases.jsonl",
            "asset_root": "../..",
            "asset_lock": "assets.lock.json",
            "release_manifest": "release.json",
            "scene_catalog": "scenes",
            "views": {
                "view_a": "views/view_a.json",
                "view_b": "views/view_b.json",
            },
        }
        write_json(stage / "dataset.json", descriptor)
        scene_configs = {
            path.stem: load_json(path)
            for path in sorted((stage / "scenes").glob("*.json"))
        }
        views = {
            "view_a": view_a,
            "view_b": base.views["view_b"],
        }
        dataset_digest = canonical_sha256({
            "descriptor": descriptor,
            "cases": tuple(new_cases),
            "views": views,
            "scenes": scene_configs,
            "asset_lock": asset_lock,
        })
        write_json(stage / "release.json", {
            "schema_version": "1.0",
            "dataset_id": DATASET_ID,
            "release": RELEASE,
            "dataset_digest": dataset_digest,
            "asset_files": len(asset_lock["files"]),
            "asset_files_digest": asset_lock["files_digest"],
        })
        write_json(stage / "migration_audit.json", {
            "schema_version": "1.0",
            "base_release": "5.1.0",
            "base_dataset_id": base.dataset_id,
            "base_dataset_digest": base.digest,
            "output_release": RELEASE,
            "output_dataset_id": DATASET_ID,
            "output_dataset_digest": dataset_digest,
            "case_count": len(new_cases),
            "removed_case_ood_fields": len(new_cases),
            "view_a_counts": counts,
            "view_a_coverage": "complete",
            "view_a_split_policy": view_a["selection_policy"],
            "collision_repartitioned": True,
            "media_files_copied": 0,
            "media_files_modified": 0,
            "asset_files_digest_unchanged": (
                asset_lock["files_digest"] == base.asset_lock["files_digest"]
            ),
        })
        (stage / "README.md").write_text(
            """# Physics Video Dataset 6.0.0

这是从5.1.0生成的纯元数据release。View A使用完整且互斥的`train/test`主划分；
碰撞scene基于全部有效Case重新执行分层、replicate-safe划分，其它scene保留5.1.0
train成员。ID/OOD/mixed是相对于`view_a.train`的test annotation，不再是Case字段或
顶层partition。View B继续覆盖全部Case用于direct evaluation。没有媒体字节被复制或修改。

旧平抛划分审计保留在`legacy_split_audit_5.1.0.json`；新的`split_audit.json`
描述当前View A。完整数据导入规范见`docs/DATASET_INGESTION.md`。
""",
            encoding="utf-8",
        )
        os.replace(stage, OUTPUT_ROOT)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
