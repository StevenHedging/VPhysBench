from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain import AtomicPlan, DatasetSnapshot, TaskSpec
from ..io import canonical_sha256, load_json


FAMILIES = {"finetune_eval", "direct_eval"}
CONDITIONING = {"generic", "physics"}


def load_task_v2(path: str | Path) -> TaskSpec:
    task_path = Path(path).resolve()
    value = load_json(task_path)
    if value.get("schema_version") != "2.0":
        raise ValueError("task must use schema_version=2.0")
    if value.get("family") not in FAMILIES:
        raise ValueError(f"unsupported task family {value.get('family')}")
    if value.get("conditioning") not in CONDITIONING:
        raise ValueError(f"unsupported conditioning {value.get('conditioning')}")
    expected_view = "view_a" if value["family"] == "finetune_eval" else "view_b"
    if value.get("dataset_view") != expected_view:
        raise ValueError(f"{value['family']} requires dataset_view={expected_view}")
    seeds = value.get("seeds", {})
    if not seeds.get("inference"):
        raise ValueError("task requires at least one inference seed")
    if value["family"] == "finetune_eval" and not seeds.get("training"):
        raise ValueError("finetune_eval requires at least one training seed")
    if value["family"] == "finetune_eval" and len(seeds["training"]) != 1:
        raise ValueError(
            "an AtomicTask must contain exactly one training seed; expand multiple "
            "training seeds into separate AtomicRuns"
        )
    return TaskSpec(task_path, value, canonical_sha256(value))


def _selected_scenes(requested: Any, available: set[str]) -> list[str]:
    if requested == "all":
        return sorted(available)
    if not isinstance(requested, list) or not requested:
        raise ValueError("selection.scene_ids must be 'all' or a non-empty list")
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"task requests unavailable scenes: {sorted(unknown)}")
    return sorted(requested)


def _view_a_plan(
    task: TaskSpec, dataset: DatasetSnapshot
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    view = dataset.views["view_a"]
    available = set(view["scenes"])
    selection = task.value["selection"]
    scenes = _selected_scenes(selection.get("scene_ids", "all"), available)
    train_ids: list[str] = []
    entries: list[tuple[str, str]] = []
    for scene_id in scenes:
        groups = view["scenes"][scene_id]
        train_ids.extend(groups.get("train", []))
        for partition in selection.get("eval_partitions", ["test_id", "test_ood1"]):
            entries.extend((case_id, partition) for case_id in groups.get(partition, []))
    ood2 = task.value.get("ood2", {"enabled": False})
    if ood2.get("enabled"):
        heldout = ood2.get("heldout_scenes", [])
        overlap = set(heldout) & set(scenes)
        if overlap:
            raise ValueError(f"OOD2 held-out scenes leak into training: {sorted(overlap)}")
        for scene_id in heldout:
            if scene_id not in available:
                raise ValueError(f"OOD2 held-out scene unavailable: {scene_id}")
            groups = view["scenes"][scene_id]
            for members in groups.values():
                entries.extend((case_id, "test_ood2") for case_id in members)
    return sorted(set(train_ids)), sorted(set(entries)), scenes


def _view_b_plan(
    task: TaskSpec, dataset: DatasetSnapshot
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    view = dataset.views["view_b"]
    selection = task.value["selection"]
    scenes = _selected_scenes(selection.get("scene_ids", "all"), set(view["scenes"]))
    explicit = selection.get("case_ids", [])
    if explicit:
        known = {case["case_id"] for case in dataset.cases}
        missing = set(explicit) - known
        if missing:
            raise ValueError(f"unknown explicit case IDs: {sorted(missing)}")
        by_id = {case["case_id"]: case for case in dataset.cases}
        outside = [
            case_id for case_id in explicit if by_id[case_id]["scene_id"] not in scenes
        ]
        if outside:
            raise ValueError(f"explicit cases outside selected scenes: {outside}")
        return [], [(case_id, "explicit") for case_id in sorted(set(explicit))], scenes
    requested_groups = selection.get("groups", "all")
    entries: list[tuple[str, str]] = []
    for scene_id in scenes:
        groups = view["scenes"][scene_id]
        names = sorted(groups) if requested_groups == "all" else list(requested_groups)
        for name in names:
            if name not in groups:
                raise ValueError(f"unknown View B group {scene_id}/{name}")
            entries.extend((case_id, name) for case_id in groups[name])
    return [], sorted(set(entries)), scenes


def plan_atomic_task(task: TaskSpec, dataset: DatasetSnapshot) -> AtomicPlan:
    if task.value["dataset_id"] != dataset.dataset_id:
        raise ValueError(
            f"task dataset {task.value['dataset_id']} != loaded {dataset.dataset_id}"
        )
    if task.family == "finetune_eval":
        train_ids, entries, scenes = _view_a_plan(task, dataset)
    else:
        train_ids, entries, scenes = _view_b_plan(task, dataset)
    by_id = {case["case_id"]: case for case in dataset.cases}
    jobs = []
    for case_id, partition in entries:
        for seed in task.value["seeds"]["inference"]:
            jobs.append({
                "job_id": (
                    f"{task.task_id}__{case_id}__{task.conditioning}"
                    f"__seed{int(seed):06d}"
                ),
                "case_id": case_id,
                "scene_id": by_id[case_id]["scene_id"],
                "evaluation_partition": partition,
                "conditioning": task.conditioning,
                "seed": int(seed),
            })
    train_seed = (
        int(task.value["seeds"]["training"][0])
        if task.family == "finetune_eval"
        else None
    )
    return AtomicPlan({
        "schema_version": "2.0",
        "task_id": task.task_id,
        "family": task.family,
        "conditioning": task.conditioning,
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest,
        "scene_ids": scenes,
        "train_case_ids": train_ids,
        "training_seed": train_seed,
        "jobs": jobs,
    })
