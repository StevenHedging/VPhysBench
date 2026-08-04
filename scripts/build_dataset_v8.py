#!/usr/bin/env python3
"""Publish Dataset 8.0.0 with two new imports and ID-only train/test splits."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from physbench.datasets import load_dataset
from physbench.io import canonical_sha256, load_json, load_jsonl, write_json, write_jsonl

from scripts.migrate_dataset_v6 import _collision_replicate_components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets/physics_video"
RELEASES_ROOT = DATA_ROOT / "releases"
BASE_ROOT = RELEASES_ROOT / "7.0.0"
OUTPUT_ROOT = RELEASES_ROOT / "8.0.0"
PENDULUM_DRAFTS = (
    DATA_ROOT
    / "provenance/source_docs/20260804_pendulum_supplement/case_drafts.jsonl"
)
PUSH_DRAFTS = (
    DATA_ROOT / "provenance/source_docs/20260804_push_bottle/case_drafts.jsonl"
)
DATASET_ID = "physics_video_six_scene_v8"
RELEASE = "8.0.0"
PENDULUM_PROMPT = (
    "A pendulum bob is released from rest at the first frame and swings back "
    "and forth about the fixed pivot."
)
TEST_TARGETS = {
    "collision_1d": 20,
    "inclined_plane_slide": 15,
    "parabolic_motion": 15,
    "pendulum": 20,
    "uniform_circular_motion": 6,
    "push_bottle": 14,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_annotation() -> dict[str, Any]:
    return {
        "generalization_regime": "id",
        "ood_factors": [],
        "co_varying_factors": [],
        "rationale": (
            "Independent held-out trial whose scene, interaction form, object "
            "domain, and physical-parameter domain are represented in View A training."
        ),
    }


def _hash_order(label: str, value: Any) -> str:
    return canonical_sha256({"policy": "id_only_stratified_v1", "label": label, "value": value})


def _old_id_tests(base: Any, scene_id: str) -> list[str]:
    view = base.views["view_a"]
    return sorted(
        case_id
        for case_id in view["scenes"][scene_id]["test"]
        if view["test_annotations"][case_id]["generalization_regime"] == "id"
    )


def _physics_signature(case: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (name, value["value"], value["unit"], value["annotated"])
        for name, value in sorted(case["physics"].items())
    )


def _subset_groups_exact(
    groups: list[list[str]], target: int, *, label: str
) -> list[str]:
    ordered = sorted(
        groups,
        key=lambda members: (
            _hash_order(label, sorted(members)),
            sorted(members),
        ),
    )
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for index, members in enumerate(ordered):
        size = len(members)
        for subtotal, selected in sorted(tuple(choices.items()), reverse=True):
            updated = subtotal + size
            if updated <= target and updated not in choices:
                choices[updated] = (*selected, index)
    if target not in choices:
        raise ValueError(f"cannot select exactly {target} grouped cases for {label}")
    return sorted(
        case_id
        for index in choices[target]
        for case_id in ordered[index]
    )


def _collision_stratum(case: dict[str, Any]) -> tuple[Any, ...]:
    appearance = case["appearance"]
    return (
        appearance["collision_structure"],
        tuple(appearance["ball_materials"]),
        tuple(appearance["ball_sequence"]),
    )


def _collision_test(cases: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    component_by_case = _collision_replicate_components(cases)
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_component[component_by_case[case["case_id"]]].append(case)
    by_stratum: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for component, members in by_component.items():
        strata = {_collision_stratum(case) for case in members}
        if len(strata) != 1:
            raise ValueError(f"collision replicate component crosses strata: {component}")
        by_stratum[next(iter(strata))].append(component)

    selected: set[str] = set()
    selected_components: set[str] = set()
    protected_components: set[str] = set()
    stratum_audit: list[dict[str, Any]] = []
    for stratum, components in sorted(by_stratum.items()):
        if len(components) < 2:
            raise ValueError(f"collision stratum has no train/test choice: {stratum}")
        velocity = {
            component: sum(
                abs(float(case["physics"]["striker_initial_velocity"]["value"]))
                for case in by_component[component]
            ) / len(by_component[component])
            for component in components
        }
        protected = {
            min(components, key=lambda key: (velocity[key], key)),
            max(components, key=lambda key: (velocity[key], key)),
        }
        protected_components.update(protected)
        candidates = [
            component
            for component in components
            if component not in protected and len(by_component[component]) == 1
        ]
        if not candidates:
            candidates = [
                component for component in components if component not in protected
            ]
        chosen = min(
            candidates,
            key=lambda component: (
                len(by_component[component]),
                _hash_order("collision_seed", sorted(
                    case["case_id"] for case in by_component[component]
                )),
            ),
        )
        selected_components.add(chosen)
        selected.update(case["case_id"] for case in by_component[chosen])
        stratum_audit.append({
            "collision_structure": stratum[0],
            "ball_materials": list(stratum[1]),
            "ball_sequence": list(stratum[2]),
            "total_cases": sum(len(by_component[item]) for item in components),
            "replicate_components": len(components),
            "seed_test_component": chosen,
            "protected_velocity_endpoint_components": sorted(protected),
        })

    target = TEST_TARGETS["collision_1d"]
    while len(selected) < target:
        candidates = []
        for stratum, components in by_stratum.items():
            total = sum(len(by_component[item]) for item in components)
            current = sum(
                len(by_component[item])
                for item in components
                if item in selected_components
            )
            for component in components:
                size = len(by_component[component])
                if (
                    component in selected_components
                    or component in protected_components
                    or len(selected) + size > target
                    or len(components) - sum(
                        item in selected_components for item in components
                    ) <= 1
                ):
                    continue
                candidates.append((
                    (current + size) / total,
                    size,
                    _hash_order("collision_fill", sorted(
                        case["case_id"] for case in by_component[component]
                    )),
                    component,
                ))
        if not candidates:
            raise ValueError("cannot reach 20 collision test cases without leakage")
        component = min(candidates)[-1]
        selected_components.add(component)
        selected.update(case["case_id"] for case in by_component[component])
    if len(selected) != target:
        raise ValueError("collision ID test target mismatch")
    partitions: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        partitions[component_by_case[case["case_id"]]].add(
            "test" if case["case_id"] in selected else "train"
        )
    overlap = [key for key, values in partitions.items() if len(values) != 1]
    if overlap:
        raise ValueError(f"collision replicate leakage: {overlap}")
    return sorted(selected), {
        "policy": "all_13_strata_then_fraction_balanced_fill",
        "replicate_group_sources": [
            "curation near-replicate clusters",
            "curation duplicate components",
            "exact structured-physics signatures",
        ],
        "replicate_component_overlap": 0,
        "velocity_endpoint_components_forced_to_train": True,
        "strata": stratum_audit,
    }


def _new_pendulum_test(cases: list[dict[str, Any]], target: int) -> list[str]:
    by_stratum: dict[tuple[float, float], list[str]] = defaultdict(list)
    for case in cases:
        physics = case["physics"]
        by_stratum[(
            float(physics["pendulum_length"]["value"]),
            float(physics["initial_angle"]["value"]),
        )].append(case["case_id"])
    eligible = {
        key: sorted(values)
        for key, values in by_stratum.items()
        if len(values) >= 2
    }
    selected = [
        min(values, key=lambda case_id: _hash_order("pendulum_stratum", case_id))
        for _, values in sorted(eligible.items())
    ]
    candidates = sorted(
        (
            case_id
            for values in eligible.values()
            for case_id in values
            if case_id not in selected
        ),
        key=lambda case_id: _hash_order("pendulum_fill", case_id),
    )
    selected.extend(candidates[: target - len(selected)])
    if len(selected) != target:
        raise ValueError("new pendulum ID test target mismatch")
    return sorted(selected)


def _push_test(cases: list[dict[str, Any]]) -> list[str]:
    by_mass: dict[float, list[str]] = defaultdict(list)
    for case in cases:
        by_mass[float(case["physics"]["bottle_mass"]["value"])].append(
            case["case_id"]
        )
    if len(by_mass) != 7:
        raise ValueError("push-bottle import must contain seven mass groups")
    selected = []
    for mass, members in sorted(by_mass.items()):
        ordered = sorted(
            members,
            key=lambda case_id: _hash_order(f"push_bottle_mass_{mass}", case_id),
        )
        if len(ordered) < 3:
            raise ValueError(f"push-bottle mass group too small: {mass}")
        selected.extend(ordered[:2])
    return sorted(selected)


def _build_view_a(
    cases: list[dict[str, Any]], base: Any, new_pendulum_ids: set[str], push_ids: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id = {case["case_id"]: case for case in cases}
    for case in cases:
        by_scene[case["scene_id"]].append(case)

    tests: dict[str, list[str]] = {}
    tests["collision_1d"], collision_audit = _collision_test(
        by_scene["collision_1d"]
    )
    tests["inclined_plane_slide"] = _old_id_tests(
        base, "inclined_plane_slide"
    )
    parabolic_candidates = set(_old_id_tests(base, "parabolic_motion"))
    signature_groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for case_id in parabolic_candidates:
        signature_groups[_physics_signature(by_id[case_id])].append(case_id)
    tests["parabolic_motion"] = _subset_groups_exact(
        list(signature_groups.values()),
        TEST_TARGETS["parabolic_motion"],
        label="parabolic_physics_signature",
    )
    old_pendulum = _old_id_tests(base, "pendulum")
    new_pendulum = [by_id[case_id] for case_id in sorted(new_pendulum_ids)]
    tests["pendulum"] = sorted(
        old_pendulum
        + _new_pendulum_test(
            new_pendulum,
            TEST_TARGETS["pendulum"] - len(old_pendulum),
        )
    )
    tests["uniform_circular_motion"] = _old_id_tests(
        base, "uniform_circular_motion"
    )
    tests["push_bottle"] = _push_test(
        [by_id[case_id] for case_id in sorted(push_ids)]
    )

    for scene_id, target in TEST_TARGETS.items():
        if len(tests[scene_id]) != target:
            raise ValueError(
                f"{scene_id} test target mismatch: {len(tests[scene_id])} != {target}"
            )
    scenes = {}
    annotations = {}
    for scene_id, scene_cases in sorted(by_scene.items()):
        all_ids = sorted(case["case_id"] for case in scene_cases)
        test_ids = sorted(tests[scene_id])
        test_set = set(test_ids)
        train_ids = [case_id for case_id in all_ids if case_id not in test_set]
        if not train_ids or not test_ids:
            raise ValueError(f"scene requires nonempty train and test: {scene_id}")
        scenes[scene_id] = {"train": train_ids, "test": test_ids}
        annotations.update({case_id: _id_annotation() for case_id in test_ids})
    view = {
        "schema_version": "3.0",
        "view_id": "view_a",
        "coverage": "complete",
        "selection_policy": "global_train_id_test_only_stratified_v1",
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
        "case_set_sha256": canonical_sha256(sorted(by_id)),
        "scenes": scenes,
        "test_annotations": dict(sorted(annotations.items())),
    }
    audit = {
        "schema_version": "3.0",
        "policy": view["selection_policy"],
        "base_release": "7.0.0",
        "primary_partitions": ["train", "test"],
        "test_semantics": "ID only; no current OOD or mixed test subsets",
        "test_count_cap_per_scene": 20,
        "scene_counts": {
            scene_id: {
                "total": len(groups["train"]) + len(groups["test"]),
                "train": len(groups["train"]),
                "test": len(groups["test"]),
                "test_id": len(groups["test"]),
                "test_ood": 0,
                "test_mixed": 0,
            }
            for scene_id, groups in scenes.items()
        },
        "collision_replicate_audit": collision_audit,
        "selection_notes": {
            "inclined_plane_slide": "Retains the 15 previously reviewed ID trials.",
            "parabolic_motion": "Selects 15 complete physical-signature groups from the previous ID pool.",
            "pendulum": "Retains 8 prior ID trials and selects 12 supplementary trials across represented length-angle strata.",
            "uniform_circular_motion": "Retains the 6 previously reviewed ID trials.",
            "push_bottle": "Selects 2 independent trials from each of 7 bottle-mass groups.",
        },
    }
    return view, audit


def _build_view_b(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_scene[case["scene_id"]].append(case["case_id"])
    scenes = {}
    for scene_id, case_ids in sorted(by_scene.items()):
        ordered = sorted(
            case_ids,
            key=lambda case_id: _hash_order(f"view_b_seed42_{scene_id}", case_id),
        )
        groups = {f"group_{index}": [] for index in range(1, 6)}
        for index, case_id in enumerate(ordered):
            groups[f"group_{index % 5 + 1}"].append(case_id)
        scenes[scene_id] = groups
    return {
        "schema_version": "2.0",
        "view_id": "view_b",
        "coverage": "complete",
        "seed": 42,
        "group_count": 5,
        "case_set_sha256": canonical_sha256(sorted(case["case_id"] for case in cases)),
        "scenes": scenes,
    }


def _asset_lock(
    cases: list[dict[str, Any]], base_lock: dict[str, Any]
) -> dict[str, Any]:
    base_hashes = {
        item["path"]: (item["size_bytes"], item["sha256"])
        for item in base_lock["files"]
    }
    known_hashes = dict(base_hashes)
    for audit_path in (
        DATA_ROOT / "provenance/imports/pendulum_supplement_20260804_import_audit.jsonl",
        DATA_ROOT / "provenance/imports/push_bottle_20260804_import_audit.jsonl",
    ):
        for item in load_jsonl(audit_path):
            case = item["case"]
            known_hashes[case["assets"]["reference_video"]] = (
                Path(DATA_ROOT / case["assets"]["reference_video"]).stat().st_size,
                item["canonical_reference_sha256"],
            )
            known_hashes[case["assets"]["first_frame"]] = (
                Path(DATA_ROOT / case["assets"]["first_frame"]).stat().st_size,
                item["canonical_first_frame_sha256"],
            )
    for summary_path, roles in (
        (
            DATA_ROOT / "provenance/imports/pendulum_supplement_20260804_summary.json",
            [("source_archive", "source_archive_sha256")],
        ),
        (
            DATA_ROOT / "provenance/imports/push_bottle_20260804_summary.json",
            [
                ("source_video_archive", "source_video_archive_sha256"),
                ("source_annotation_archive", "source_annotation_archive_sha256"),
            ],
        ),
    ):
        summary = load_json(summary_path)
        for path_key, hash_key in roles:
            relative = summary[path_key]
            path = DATA_ROOT / relative
            known_hashes[relative] = (path.stat().st_size, summary[hash_key])

    role_map: dict[str, set[str]] = defaultdict(set)
    case_map: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        for role, relative in case["assets"].items():
            if relative is None:
                continue
            role_map[relative].add(role)
            case_map[relative].add(case["case_id"])
    files = []
    for relative in sorted(role_map):
        path = DATA_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        known = known_hashes.get(relative)
        digest = known[1] if known and known[0] == size else sha256(path)
        files.append({
            "path": relative,
            "size_bytes": size,
            "sha256": digest,
            "roles": sorted(role_map[relative]),
            "case_ids": sorted(case_map[relative]),
        })
    return {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "release": RELEASE,
        "files": files,
        "files_digest": canonical_sha256(files),
    }


def _push_scene() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "scene_id": "push_bottle",
        "display_name": "推水瓶",
        "structured_physics_parameters": [
            "bottle_height",
            "bottle_mass",
            "mean_applied_force",
            "peak_applied_force",
        ],
        "non_conditionable_physics_parameters": [],
        "generalization_factors": [
            {
                "name": "interaction",
                "category": "interaction_structure",
                "source": "appearance.interaction",
            },
            {
                "name": "capture_session",
                "category": "acquisition",
                "source": "appearance.capture_session",
            },
        ],
        "constraints": [
            "Canonical frame zero shows an upright bottle before the push develops",
            "Canonical media preserves the complete source video without crop or trim",
        ],
        "metric_spec": {
            "prediction_subjects": ["bottle"],
            "prediction_quantities": [
                "tipping onset",
                "angular motion",
                "fall direction",
                "final toppled state",
            ],
            "common_sense_checks": [
                "bottle begins upright",
                "push initiates tipping",
                "continuous rigid-body fall",
                "no spontaneous topology change",
            ],
            "visual_attributes": [
                "bottle integrity",
                "support contact",
                "push interaction",
            ],
        },
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {OUTPUT_ROOT}")
    base = load_dataset(BASE_ROOT / "dataset.json")
    old_cases = [copy.deepcopy(case) for case in base.cases]
    pendulum_drafts = load_jsonl(PENDULUM_DRAFTS)
    push_drafts = load_jsonl(PUSH_DRAFTS)
    if len(pendulum_drafts) != 65 or len(push_drafts) != 141:
        raise ValueError("staged import counts changed")

    for case in old_cases:
        if case["scene_id"] == "pendulum":
            case["text"] = {
                "schema_version": "1.0",
                "prompt": PENDULUM_PROMPT,
                "language": "en",
                "annotation_source": "pendulum_release_semantics_v8",
            }
            alignment = case.get("alignment") or {}
            alignment["canonical_first_frame_event"] = "initial release point"
            case["alignment"] = alignment
    pendulum_archive = (
        "assets/source_archives/20260804_new_data/pendulum_supplement.zip"
    )
    for case in pendulum_drafts:
        case["assets"]["source_archive"] = pendulum_archive
        case["provenance"]["source_locator"] = {
            "archive": pendulum_archive,
            "member": case["provenance"]["source_member"],
            "annotation_workbook": (
                "provenance/source_docs/20260804_pendulum_supplement/钟摆实验.xlsx"
            ),
            "row": case["provenance"]["source_workbook_row"],
            "normalized_annotation": (
                "provenance/source_docs/20260804_pendulum_supplement/"
                "normalized_annotations.json"
            ),
        }
    cases = sorted(
        old_cases + pendulum_drafts + push_drafts,
        key=lambda case: case["case_id"],
    )
    if len(cases) != 799 or len({case["case_id"] for case in cases}) != 799:
        raise ValueError("Dataset 8.0.0 must contain 799 unique cases")
    new_pendulum_ids = {case["case_id"] for case in pendulum_drafts}
    push_ids = {case["case_id"] for case in push_drafts}
    view_a, split_audit = _build_view_a(
        cases, base, new_pendulum_ids, push_ids
    )
    view_b = _build_view_b(cases)

    stage = Path(tempfile.mkdtemp(prefix=".8.0.0.build-", dir=RELEASES_ROOT))
    try:
        (stage / "views").mkdir()
        (stage / "scenes").mkdir()
        write_jsonl(stage / "cases.jsonl", cases)
        write_json(stage / "views/view_a.json", view_a)
        write_json(stage / "views/view_b.json", view_b)
        for scene_id, config in base.scene_configs.items():
            updated = copy.deepcopy(config)
            if scene_id == "pendulum":
                updated["structured_physics_parameters"] = sorted(
                    set(updated["structured_physics_parameters"]) | {"bob_mass"}
                )
                updated["constraints"] = [
                    "Canonical frame zero is the Dataset-defined initial release point",
                    "Supplementary-video trimming is only for removing the person's hand",
                ]
            write_json(stage / "scenes" / f"{scene_id}.json", updated)
        write_json(stage / "scenes/push_bottle.json", _push_scene())
        for name in (
            "annotation_corrections.json",
            "asset_directory_mapping.json",
            "ball_spec_catalog.json",
        ):
            shutil.copy2(BASE_ROOT / name, stage / name)
        shutil.copy2(BASE_ROOT / "split_audit.json", stage / "legacy_split_audit_7.0.0.json")
        write_json(stage / "split_audit.json", split_audit)
        asset_lock = _asset_lock(cases, base.asset_lock)
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
        dataset_digest = canonical_sha256({
            "descriptor": descriptor,
            "cases": tuple(cases),
            "views": {"view_a": view_a, "view_b": view_b},
            "scenes": scene_configs,
            "asset_lock": asset_lock,
        })
        release_manifest = {
            "schema_version": "1.0",
            "dataset_id": DATASET_ID,
            "release": RELEASE,
            "dataset_digest": dataset_digest,
            "asset_files": len(asset_lock["files"]),
            "asset_files_digest": asset_lock["files_digest"],
        }
        write_json(stage / "release.json", release_manifest)
        write_json(stage / "migration_audit.json", {
            "schema_version": "1.0",
            "base_release": "7.0.0",
            "base_dataset_id": base.dataset_id,
            "base_dataset_digest": base.digest,
            "output_release": RELEASE,
            "output_dataset_id": DATASET_ID,
            "output_dataset_digest": dataset_digest,
            "old_case_count": len(old_cases),
            "pendulum_supplement_cases_added": len(pendulum_drafts),
            "push_bottle_cases_added": len(push_drafts),
            "output_case_count": len(cases),
            "scene_count": 6,
            "pendulum_text_records_unified": sum(
                case["scene_id"] == "pendulum" for case in cases
            ),
            "view_a_policy": view_a["selection_policy"],
            "all_test_annotations_id": all(
                item["generalization_regime"] == "id"
                for item in view_a["test_annotations"].values()
            ),
            "media_policy": {
                "old_assets_modified": 0,
                "pendulum_supplement": "trimmed prefix only to remove hand; retained frames and source FPS preserved",
                "push_bottle": "byte-preserving source video; no trim, crop, resampling, or re-encode",
            },
        })
        (stage / "README.md").write_text(
            """# Physics Video Dataset 8.0.0

当前 release 在 7.0.0 的 593 条 Case 上新增 65 条补充单摆和 141 条推水瓶，
共 799 条、6 个 scene。View A 对每个 scene 仅使用互斥的 `train/test`，全部 test
均为 ID，单个 scene 的 test 不超过 20 条；当前 release 不再构造 OOD/mixed 测试集。

补充单摆的 canonical 第 0 帧按 Dataset 语义定义为初始释放点，时间裁剪唯一目的为
去除人手。推水瓶视频按原字节保留，不做裁剪、剪辑、重采样或重编码。完整来源与
排除记录位于 `datasets/physics_video/provenance/`。
""",
            encoding="utf-8",
        )
        staged_snapshot = load_dataset(stage / "dataset.json", check_assets=True)
        if staged_snapshot.digest != dataset_digest:
            raise ValueError("staged Dataset digest mismatch")
        os.replace(stage, OUTPUT_ROOT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({
        "dataset_id": DATASET_ID,
        "release": RELEASE,
        "cases": len(cases),
        "assets": len(asset_lock["files"]),
        "dataset_digest": dataset_digest,
        "splits": split_audit["scene_counts"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
