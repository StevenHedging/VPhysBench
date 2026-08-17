from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain import AtomicPlan, DatasetSnapshot, TaskSpec
from ..identifiers import require_safe_id
from ..io import canonical_sha256, load_json


FAMILIES = {"finetune_eval", "direct_eval"}
GENERALIZATION_REGIMES = {"id", "ood", "mixed"}
TASK_SCHEMA_VERSION = "1.0"
TASK_FIELDS = {
    "$schema",
    "schema_version",
    "task_id",
    "family",
    "dataset_id",
    "selection",
    "seeds",
    "evaluation",
}


def _require_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _reject_unknown_fields(
    value: dict[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")


def _require_fields(
    value: dict[str, Any],
    *,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _validate_safe_id_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a" if allow_empty else "a non-empty"
        raise ValueError(f"{label} must be {qualifier} array")
    identifiers = [
        require_safe_id(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} must contain unique identifiers")
    return identifiers


def _validate_id_selector(value: Any, *, label: str) -> None:
    if value == "all":
        return
    _validate_safe_id_list(value, label=label, allow_empty=False)


def _validate_seed_list(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    if len(value) < minimum or (
        maximum is not None and len(value) > maximum
    ):
        if minimum == maximum:
            raise ValueError(
                f"{label} must contain exactly {minimum} seed(s)"
            )
        raise ValueError(
            f"{label} must contain at least {minimum} seed(s)"
        )
    if any(
        isinstance(seed, bool) or not isinstance(seed, int)
        for seed in value
    ):
        raise ValueError(f"{label} must contain only non-boolean integers")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain unique seeds")
    return value


def _validate_selection(
    value: Any,
    *,
    family: str,
) -> None:
    selection = _require_object(value, label="task.selection")
    if family == "finetune_eval":
        allowed = {
            "training_scene_ids",
            "evaluation_scene_ids",
            "test_regimes",
        }
        _require_fields(
            selection,
            required=allowed,
            label="task.selection",
        )
        _reject_unknown_fields(
            selection,
            allowed=allowed,
            label="task.selection",
        )
        _validate_id_selector(
            selection["training_scene_ids"],
            label="task.selection.training_scene_ids",
        )
        _validate_id_selector(
            selection["evaluation_scene_ids"],
            label="task.selection.evaluation_scene_ids",
        )
        regimes = selection["test_regimes"]
        if regimes == "all":
            return
        if not isinstance(regimes, list) or not regimes:
            raise ValueError(
                "task.selection.test_regimes must be 'all' or a "
                "non-empty array"
            )
        if any(
            not isinstance(regime, str)
            or regime not in GENERALIZATION_REGIMES
            for regime in regimes
        ):
            raise ValueError(
                "task.selection.test_regimes may only contain "
                f"{sorted(GENERALIZATION_REGIMES)}"
            )
        if len(regimes) != len(set(regimes)):
            raise ValueError(
                "task.selection.test_regimes must contain unique values"
            )
        return

    allowed = {"evaluation_scene_ids", "groups", "case_ids"}
    _require_fields(
        selection,
        required={"evaluation_scene_ids", "groups"},
        label="task.selection",
    )
    _reject_unknown_fields(
        selection,
        allowed=allowed,
        label="task.selection",
    )
    _validate_id_selector(
        selection["evaluation_scene_ids"],
        label="task.selection.evaluation_scene_ids",
    )
    _validate_id_selector(
        selection["groups"],
        label="task.selection.groups",
    )
    if "case_ids" in selection:
        _validate_safe_id_list(
            selection["case_ids"],
            label="task.selection.case_ids",
            allow_empty=True,
        )


def _validate_seeds(value: Any, *, family: str) -> None:
    seeds = _require_object(value, label="task.seeds")
    _require_fields(
        seeds,
        required={"training", "inference"},
        label="task.seeds",
    )
    _reject_unknown_fields(
        seeds,
        allowed={"training", "inference"},
        label="task.seeds",
    )
    _validate_seed_list(
        seeds["training"],
        label="task.seeds.training",
        minimum=1 if family == "finetune_eval" else 0,
        maximum=1 if family == "finetune_eval" else 0,
    )
    _validate_seed_list(
        seeds["inference"],
        label="task.seeds.inference",
        minimum=1,
    )


def _validate_reporting(value: Any) -> None:
    reporting = _require_object(value, label="task.evaluation.reporting")
    allowed = {
        "primary_score",
        "breakdowns",
        "minimum_subgroup_jobs",
    }
    _require_fields(reporting, required=allowed, label="task.evaluation.reporting")
    _reject_unknown_fields(
        reporting,
        allowed=allowed,
        label="task.evaluation.reporting",
    )
    if reporting["primary_score"] != "overall_test":
        raise ValueError(
            "task.evaluation.reporting.primary_score must be overall_test"
        )
    breakdowns = reporting["breakdowns"]
    allowed_breakdowns = {"generalization_regime", "ood_factor"}
    if not isinstance(breakdowns, list) or any(
        not isinstance(item, str) or item not in allowed_breakdowns
        for item in breakdowns
    ):
        raise ValueError(
            "task.evaluation.reporting.breakdowns may only contain "
            f"{sorted(allowed_breakdowns)}"
        )
    if len(breakdowns) != len(set(breakdowns)):
        raise ValueError(
            "task.evaluation.reporting.breakdowns must contain unique values"
        )
    minimum = reporting["minimum_subgroup_jobs"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError(
            "task.evaluation.reporting.minimum_subgroup_jobs must be a "
            "positive integer"
        )


def _validate_task_document(value: Any) -> dict[str, Any]:
    task = _require_object(value, label="task")
    schema_version = task.get("schema_version")
    if schema_version != TASK_SCHEMA_VERSION:
        raise ValueError(
            f"task must use schema_version={TASK_SCHEMA_VERSION}"
        )
    required = TASK_FIELDS - {"$schema"}
    _require_fields(task, required=required, label="task")
    unknown_fields = sorted(set(task) - TASK_FIELDS)
    if unknown_fields:
        raise ValueError(
            "Task contains fields outside the model-agnostic contract: "
            f"{unknown_fields}; physics use is a Baseline input-policy property"
        )
    require_safe_id(task["task_id"], label="task_id")
    require_safe_id(task["dataset_id"], label="task.dataset_id")
    family = task["family"]
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError(f"unsupported task family {family}")
    _validate_selection(
        task["selection"],
        family=family,
    )
    _validate_seeds(task["seeds"], family=family)
    evaluation = _require_object(
        task["evaluation"],
        label="task.evaluation",
    )
    evaluation_fields = {"protocol", "reporting"}
    _require_fields(
        evaluation,
        required=evaluation_fields,
        label="task.evaluation",
    )
    _reject_unknown_fields(
        evaluation,
        allowed=evaluation_fields,
        label="task.evaluation",
    )
    require_safe_id(
        evaluation["protocol"],
        label="task.evaluation.protocol",
    )
    _validate_reporting(evaluation["reporting"])
    return task


def load_task(path: str | Path) -> TaskSpec:
    task_path = Path(path).resolve()
    value = _validate_task_document(load_json(task_path))
    return TaskSpec(task_path, value, canonical_sha256(value))


def _selected_scenes(
    requested: Any,
    available: set[str],
    *,
    label: str,
) -> list[str]:
    if requested == "all":
        return sorted(available)
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"{label} must be 'all' or a non-empty list")
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"{label} contains unavailable scenes: {sorted(unknown)}")
    return sorted(requested)


def _view_a_plan(
    task: TaskSpec, dataset: DatasetSnapshot
) -> tuple[
    list[str],
    list[tuple[str, str]],
    list[str],
    list[str],
    dict[str, dict[str, Any]],
]:
    view = dataset.views["view_a"]
    if view.get("schema_version") != "3.0":
        raise ValueError("Task v1 finetune_eval requires View A schema 3.0")
    available = set(view["scenes"])
    selection = task.value["selection"]
    training_scenes = _selected_scenes(
        selection["training_scene_ids"],
        available,
        label="task.selection.training_scene_ids",
    )
    evaluation_scenes = _selected_scenes(
        selection["evaluation_scene_ids"],
        available,
        label="task.selection.evaluation_scene_ids",
    )
    requested = selection["test_regimes"]
    regimes = GENERALIZATION_REGIMES if requested == "all" else set(requested)
    train_ids: list[str] = []
    entries: list[tuple[str, str]] = []
    annotations: dict[str, dict[str, Any]] = {}
    view_annotations = view["test_annotations"]
    for scene_id in training_scenes:
        groups = view["scenes"][scene_id]
        train_ids.extend(groups["train"])
    for scene_id in evaluation_scenes:
        groups = view["scenes"][scene_id]
        for case_id in groups["test"]:
            annotation = view_annotations[case_id]
            if annotation["generalization_regime"] not in regimes:
                continue
            entries.append((case_id, "test"))
            annotations[case_id] = annotation
    return (
        sorted(set(train_ids)),
        sorted(set(entries)),
        evaluation_scenes,
        training_scenes,
        annotations,
    )


def _view_b_plan(
    task: TaskSpec, dataset: DatasetSnapshot
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    view = dataset.views["view_b"]
    selection = task.value["selection"]
    scenes = _selected_scenes(
        selection["evaluation_scene_ids"],
        set(view["scenes"]),
        label="task.selection.evaluation_scene_ids",
    )
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
    _validate_task_document(task.value)
    task_schema = task.value["schema_version"]
    dataset_schema = dataset.descriptor["schema_version"]
    if dataset_schema not in {"3.0", "4.0", "5.0", "6.0"}:
        raise ValueError(
            f"task schema {task_schema} is incompatible with dataset schema "
            f"{dataset_schema}"
        )
    if task.value["dataset_id"] != dataset.dataset_id:
        raise ValueError(
            f"task dataset {task.value['dataset_id']} != loaded {dataset.dataset_id}"
        )
    require_safe_id(dataset.dataset_id, label="dataset.dataset_id")
    seen_case_ids: set[str] = set()
    for index, case in enumerate(dataset.cases):
        case_id = require_safe_id(
            case.get("case_id"),
            label=f"dataset.cases[{index}].case_id",
        )
        require_safe_id(
            case.get("scene_id"),
            label=f"dataset.cases[{index}].scene_id",
        )
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate dataset case ID {case_id}")
        seen_case_ids.add(case_id)
    annotations_by_case: dict[str, dict[str, Any]] = {}
    if task.family == "finetune_eval":
        (
            train_ids,
            entries,
            scenes,
            training_scenes,
            annotations_by_case,
        ) = _view_a_plan(
            task,
            dataset,
        )
    else:
        train_ids, entries, scenes = _view_b_plan(task, dataset)
        training_scenes = []
    by_id = {case["case_id"]: case for case in dataset.cases}
    # ``scene_ids`` is the aggregation/evaluation universe; training has its
    # own explicit universe in ``training_scene_ids``.
    scenes = sorted(
        set(scenes)
        | {
            by_id[case_id]["scene_id"]
            for case_id, _ in entries
        }
    )
    jobs = []
    evaluation_annotations: dict[str, dict[str, Any]] = {}
    for case_id, partition in entries:
        for seed in task.value["seeds"]["inference"]:
            job_id = f"{task.task_id}__{case_id}__seed{seed:06d}"
            jobs.append({
                "job_id": job_id,
                "case_id": case_id,
                "scene_id": by_id[case_id]["scene_id"],
                "evaluation_partition": partition,
                "seed": int(seed),
            })
            if case_id in annotations_by_case:
                evaluation_annotations[job_id] = annotations_by_case[case_id]
    train_seed = (
        task.value["seeds"]["training"][0]
        if task.family == "finetune_eval"
        else None
    )
    plan = {
        "schema_version": task.value["schema_version"],
        "task_id": task.task_id,
        "family": task.family,
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest,
        "training_scene_ids": training_scenes,
        "scene_ids": scenes,
        "train_case_ids": train_ids,
        "training_seed": train_seed,
        "jobs": jobs,
    }
    plan["reporting_policy"] = task.value["evaluation"]["reporting"]
    if task.family == "finetune_eval":
        plan["evaluation_annotations"] = evaluation_annotations
    return AtomicPlan(plan)
