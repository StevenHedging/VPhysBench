from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from .io import canonical_sha256
from .prompts import DEFAULT_EVAL_PROFILES, DEFAULT_TRAIN_PROFILE


def validate_task(task: dict[str, Any]) -> None:
    if task.get("schema_version") != "1.0":
        raise ValueError("task schema_version must be 1.0")
    if task.get("ood2", {}).get("enabled") is not False:
        raise ValueError("OOD2 is intentionally unsupported in benchmark v1")
    mode, view = task.get("mode"), task.get("view")
    if (mode, view) not in {("finetune_and_eval", "A"), ("zero_shot_eval", "B")}:
        raise ValueError("finetune_and_eval requires view A; zero_shot_eval requires view B")


def _selected_scenes(selection: dict[str, Any], available: set[str]) -> list[str]:
    requested = selection.get("scene_ids", "all")
    if requested == "all":
        return sorted(available)
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"task requests unavailable scenes: {sorted(unknown)}")
    return sorted(requested)


def plan_task(
    task: dict[str, Any],
    cases: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    train_preview_per_scene: int | None = None,
    train_preview_seed: int | None = None,
    train_prompt_profile: str | None = None,
    eval_prompt_profiles: list[str] | None = None,
) -> dict[str, Any]:
    validate_task(task)
    if split.get("view") != task["view"]:
        raise ValueError("task view and split view differ")
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case IDs")
    expected_fingerprint = canonical_sha256(sorted(by_id))
    if split.get("case_set_sha256") != expected_fingerprint:
        raise ValueError("split fingerprint does not match the manifest case set")
    split_ids = [
        case_id
        for scene_groups in split.get("scenes", {}).values()
        for group_ids in scene_groups.values()
        for case_id in group_ids
    ]
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(by_id):
        raise ValueError("split must contain every manifest case exactly once")
    available_scenes = set(split.get("scenes", {}))
    selection = task.get("selection", {})
    scenes = _selected_scenes(selection, available_scenes)
    train_ids: list[str] = []
    eval_entries: list[tuple[str, str]] = []
    train_ids_by_scene: dict[str, list[str]] = {scene: [] for scene in scenes}

    if task["view"] == "A":
        train_splits = selection.get("train_splits", ["train"])
        eval_splits = selection.get("eval_splits", ["test_id", "test_ood1"])
        for scene in scenes:
            groups = split["scenes"][scene]
            for split_name in train_splits:
                selected_train_ids = groups.get(split_name, [])
                train_ids.extend(selected_train_ids)
                train_ids_by_scene[scene].extend(selected_train_ids)
            for split_name in eval_splits:
                for case_id in groups.get(split_name, []):
                    eval_entries.append((case_id, split_name))
        allowed_factors = selection.get("ood1_factors", "all_available")
        if allowed_factors != "all_available":
            allowed = set(allowed_factors)
            eval_entries = [
                entry for entry in eval_entries
                if by_id[entry[0]]["ood"]["level"] == "id"
                or bool(set(by_id[entry[0]]["ood"]["factors"]) & allowed)
            ]
    else:
        explicit = selection.get("case_ids", [])
        if explicit:
            missing = set(explicit) - by_id.keys()
            if missing:
                raise ValueError(f"unknown explicit case IDs: {sorted(missing)}")
            eval_entries = [(case_id, "explicit") for case_id in explicit]
        else:
            requested_groups = selection.get("groups", "all")
            for scene in scenes:
                groups = split["scenes"][scene]
                group_names = sorted(groups) if requested_groups == "all" else requested_groups
                for group_name in group_names:
                    if group_name not in groups:
                        raise ValueError(f"unknown group {scene}/{group_name}")
                    eval_entries.extend((case_id, group_name) for case_id in groups[group_name])

    preview_config = task.get("train_preview", {})
    preview_per_scene = (
        int(preview_config.get("per_scene", 0))
        if train_preview_per_scene is None
        else int(train_preview_per_scene)
    )
    preview_seed = (
        int(preview_config.get("seed", 42))
        if train_preview_seed is None
        else int(train_preview_seed)
    )
    if preview_per_scene < 0:
        raise ValueError("train preview per-scene count must be non-negative")
    if preview_per_scene and task["view"] != "A":
        raise ValueError("train previews are only supported for view A finetune_and_eval tasks")

    preview_ids_by_scene: dict[str, list[str]] = {}
    if preview_per_scene:
        for scene in scenes:
            candidates = sorted(set(train_ids_by_scene[scene]))
            scene_seed = int.from_bytes(
                hashlib.sha256(f"{preview_seed}:{scene}".encode("utf-8")).digest()[:8],
                "big",
            )
            selected = sorted(
                random.Random(scene_seed).sample(candidates, min(preview_per_scene, len(candidates)))
            )
            preview_ids_by_scene[scene] = selected
            eval_entries.extend((case_id, "train_seen") for case_id in selected)

    prompt_config = task.get("prompts", {})
    configured_train_profile = prompt_config.get(
        "train_profile",
        DEFAULT_TRAIN_PROFILE if task["mode"] == "finetune_and_eval" else None,
    )
    selected_train_profile = (
        train_prompt_profile if train_prompt_profile is not None else configured_train_profile
    )
    configured_eval_profiles = prompt_config.get(
        "eval_profiles",
        [selected_train_profile] if selected_train_profile else list(DEFAULT_EVAL_PROFILES),
    )
    selected_eval_profiles = (
        list(eval_prompt_profiles)
        if eval_prompt_profiles is not None
        else list(configured_eval_profiles)
    )
    selected_eval_profiles = list(dict.fromkeys(selected_eval_profiles))
    if train_ids and not selected_train_profile:
        raise ValueError("finetuning tasks require a train prompt profile")
    if not selected_eval_profiles:
        raise ValueError("at least one eval prompt profile is required")
    for profile_id in [
        *selected_eval_profiles,
        *([selected_train_profile] if selected_train_profile else []),
    ]:
        if not isinstance(profile_id, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", profile_id):
            raise ValueError(f"invalid prompt profile ID {profile_id!r}")

    for case_id in train_ids + [entry[0] for entry in eval_entries]:
        if case_id not in by_id:
            raise ValueError(f"split references missing case {case_id}")

    seeds = task.get("seeds", [42])
    jobs = []
    for case_id, evaluation_partition in sorted(set(eval_entries)):
        for profile_id in selected_eval_profiles:
            for seed in seeds:
                jobs.append({
                    "job_id": (
                        f"{task['task_id']}__{case_id}__prompt-{profile_id}"
                        f"__seed{int(seed):06d}"
                    ),
                    "case_id": case_id,
                    "scene_id": by_id[case_id]["scene_id"],
                    "evaluation_partition": evaluation_partition,
                    "prompt_profile_id": profile_id,
                    "seed": int(seed),
                })
    return {
        "schema_version": "1.0",
        "task_id": task["task_id"],
        "mode": task["mode"],
        "view": task["view"],
        "scene_ids": scenes,
        "train_case_ids": sorted(set(train_ids)),
        "prompt_profiles": {
            "train": selected_train_profile,
            "eval": selected_eval_profiles,
        },
        "train_preview": {
            "enabled": bool(preview_per_scene),
            "per_scene": preview_per_scene,
            "seed": preview_seed,
            "selected_case_ids_by_scene": preview_ids_by_scene,
        },
        "jobs": jobs,
    }
