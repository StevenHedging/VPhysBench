from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..identifiers import SAFE_ID_PATTERN


_TOP_LEVEL_FIELDS = frozenset({
    "schema_version",
    "instance_id",
    "instance_digest",
    "identity",
    "semantics",
    "canonical_plan",
    "source",
    "adaptations",
    "training",
    "inference",
    "execution_graph",
    "cache_bindings",
    "baseline_payload",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_JOB_FIELDS = frozenset({
    "job_id",
    "case_id",
    "scene_id",
    "evaluation_partition",
    "seed",
})


def _invalid(path: str, message: str) -> ValueError:
    return ValueError(f"Invalid BaselineTaskInstance at {path}: {message}")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(path, "must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid(path, "must be an array")
    return value


def _require_fields(
    value: dict[str, Any],
    required: set[str] | frozenset[str],
    path: str,
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise _invalid(path, f"missing required field(s): {', '.join(missing)}")


def _require_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(path, "must be a non-empty string")
    return value


def _require_safe_identifier(value: Any, path: str) -> str:
    result = _require_non_empty_string(value, path)
    if SAFE_ID_PATTERN.fullmatch(result) is None:
        raise _invalid(
            path,
            "must contain only path-safe identifier characters",
        )
    return result


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _invalid(path, "must be a lowercase 64-character SHA-256 hex digest")
    return value


def _require_seed(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(path, "must be a non-boolean integer")
    return value


def _validate_identity(identity_value: Any) -> None:
    identity = _require_object(identity_value, "identity")
    _require_fields(
        identity,
        {
            "dataset",
            "task",
            "baseline",
            "task_builder",
            "data_adapter",
            "canonical_plan_digest",
        },
        "identity",
    )
    dataset = _require_object(identity["dataset"], "identity.dataset")
    _require_fields(dataset, {"dataset_id", "digest"}, "identity.dataset")
    _require_safe_identifier(
        dataset["dataset_id"], "identity.dataset.dataset_id"
    )
    _require_sha256(dataset["digest"], "identity.dataset.digest")

    task = _require_object(identity["task"], "identity.task")
    _require_fields(task, {"task_id", "digest"}, "identity.task")
    _require_safe_identifier(task["task_id"], "identity.task.task_id")
    _require_sha256(task["digest"], "identity.task.digest")

    task_builder = _require_object(
        identity["task_builder"], "identity.task_builder"
    )
    _require_fields(
        task_builder,
        {"type", "fingerprint"},
        "identity.task_builder",
    )
    _require_non_empty_string(
        task_builder["type"], "identity.task_builder.type"
    )
    _require_sha256(
        task_builder["fingerprint"],
        "identity.task_builder.fingerprint",
    )

    adapter = _require_object(
        identity["data_adapter"], "identity.data_adapter"
    )
    _require_fields(
        adapter,
        {"fingerprint", "materialization_fingerprint"},
        "identity.data_adapter",
    )
    _require_sha256(
        adapter["fingerprint"], "identity.data_adapter.fingerprint"
    )
    _require_sha256(
        adapter["materialization_fingerprint"],
        "identity.data_adapter.materialization_fingerprint",
    )

    baseline = _require_object(identity["baseline"], "identity.baseline")
    _require_fields(
        baseline,
        {
            "baseline_id",
            "baseline_version",
            "digest",
            "deployment_digest",
        },
        "identity.baseline",
    )
    _require_safe_identifier(
        baseline["baseline_id"], "identity.baseline.baseline_id"
    )
    _require_non_empty_string(
        baseline["baseline_version"], "identity.baseline.baseline_version"
    )
    _require_sha256(baseline["digest"], "identity.baseline.digest")
    _require_sha256(
        baseline["deployment_digest"],
        "identity.baseline.deployment_digest",
    )
    _require_sha256(
        identity["canonical_plan_digest"],
        "identity.canonical_plan_digest",
    )


def _validate_semantics(semantics_value: Any) -> None:
    semantics = _require_object(semantics_value, "semantics")
    _require_fields(
        semantics,
        {"family", "training_scene_ids", "scene_ids"},
        "semantics",
    )
    _require_non_empty_string(semantics["family"], "semantics.family")
    for field in ("training_scene_ids", "scene_ids"):
        scene_ids = _require_list(semantics[field], f"semantics.{field}")
        for index, scene_id in enumerate(scene_ids):
            _require_safe_identifier(
                scene_id, f"semantics.{field}[{index}]"
            )
        if len(scene_ids) != len(set(scene_ids)):
            raise _invalid(
                f"semantics.{field}",
                "must not contain duplicates",
            )


def _validate_source(source_value: Any) -> set[str]:
    source = _require_object(source_value, "source")
    _require_fields(source, {"asset_root", "cases"}, "source")
    _require_non_empty_string(source["asset_root"], "source.asset_root")
    cases = _require_list(source["cases"], "source.cases")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        case = _require_object(case, f"source.cases[{index}]")
        case_id = _require_safe_identifier(
            case.get("case_id"), f"source.cases[{index}].case_id"
        )
        if "scene_id" in case:
            _require_safe_identifier(
                case["scene_id"],
                f"source.cases[{index}].scene_id",
            )
        if case_id in case_ids:
            raise _invalid(
                f"source.cases[{index}].case_id",
                f"duplicate case_id {case_id!r}",
            )
        case_ids.add(case_id)
    return case_ids


def _validate_adaptations(
    adaptations_value: Any,
) -> dict[str, dict[str, Any]]:
    adaptations = _require_list(adaptations_value, "adaptations")
    by_id: dict[str, dict[str, Any]] = {}
    for index, adaptation_value in enumerate(adaptations):
        path = f"adaptations[{index}]"
        adaptation = _require_object(adaptation_value, path)
        _require_fields(
            adaptation,
            {"adaptation_id", "case_id", "role", "native_inputs"},
            path,
        )
        adaptation_id = _require_safe_identifier(
            adaptation["adaptation_id"], f"{path}.adaptation_id"
        )
        if adaptation_id in by_id:
            raise _invalid(
                f"{path}.adaptation_id",
                f"duplicate adaptation_id {adaptation_id!r}",
            )
        _require_safe_identifier(
            adaptation["case_id"], f"{path}.case_id"
        )
        if adaptation["role"] not in {"train", "eval"}:
            raise _invalid(
                f"{path}.role",
                "must be train or eval",
            )
        _require_object(
            adaptation["native_inputs"], f"{path}.native_inputs"
        )
        by_id[adaptation_id] = adaptation
    return by_id


def _validate_inference(
    inference_value: Any,
    adaptations: dict[str, dict[str, Any]],
    case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    inference = _require_object(inference_value, "inference")
    _require_fields(inference, {"predictor", "jobs"}, "inference")
    _require_object(inference["predictor"], "inference.predictor")

    jobs = _require_list(inference["jobs"], "inference.jobs")
    by_id: dict[str, dict[str, Any]] = {}
    for index, job_value in enumerate(jobs):
        path = f"inference.jobs[{index}]"
        job = _require_object(job_value, path)
        _require_fields(
            job,
            {
                *_CANONICAL_JOB_FIELDS,
                "adaptation_id",
                "native_inputs",
            },
            path,
        )
        job_id = _require_safe_identifier(
            job["job_id"], f"{path}.job_id"
        )
        if job_id in by_id:
            raise _invalid(
                f"{path}.job_id",
                f"duplicate job_id {job_id!r}",
            )
        by_id[job_id] = job

        case_id = _require_safe_identifier(
            job["case_id"], f"{path}.case_id"
        )
        _require_safe_identifier(
            job["scene_id"], f"{path}.scene_id"
        )
        _require_non_empty_string(
            job["evaluation_partition"],
            f"{path}.evaluation_partition",
        )
        _require_seed(job["seed"], f"{path}.seed")
        if case_id not in case_ids:
            raise _invalid(
                f"{path}.case_id",
                f"references unknown source case_id {case_id!r}",
            )

        adaptation_id = _require_safe_identifier(
            job["adaptation_id"], f"{path}.adaptation_id"
        )
        if adaptation_id not in adaptations:
            raise _invalid(
                f"{path}.adaptation_id",
                f"references unknown adaptation_id {adaptation_id!r}",
            )
        adaptation = adaptations[adaptation_id]
        if adaptation["case_id"] != case_id:
            raise _invalid(
                f"{path}.adaptation_id",
                "adaptation case_id does not match inference job case_id",
            )
        if adaptation["role"] != "eval":
            raise _invalid(
                f"{path}.adaptation_id",
                "inference job must reference an eval adaptation",
            )
        native_inputs = _require_object(
            job["native_inputs"], f"{path}.native_inputs"
        )
        if native_inputs != adaptation["native_inputs"]:
            raise _invalid(
                f"{path}.native_inputs",
                "must equal the referenced adaptation.native_inputs",
            )
    return by_id


def _validate_plan_identity(
    canonical_plan: dict[str, Any],
    identity: dict[str, Any],
    semantics: dict[str, Any],
) -> None:
    _require_fields(
        canonical_plan,
        {
            "task_id",
            "family",
            "dataset_id",
            "dataset_digest",
            "training_scene_ids",
            "scene_ids",
            "train_case_ids",
            "training_seed",
            "jobs",
        },
        "canonical_plan",
    )
    _require_safe_identifier(
        canonical_plan["task_id"],
        "canonical_plan.task_id",
    )
    _require_safe_identifier(
        canonical_plan["dataset_id"],
        "canonical_plan.dataset_id",
    )
    for field in ("training_scene_ids", "scene_ids"):
        scene_ids = _require_list(
            canonical_plan[field],
            f"canonical_plan.{field}",
        )
        for index, scene_id in enumerate(scene_ids):
            _require_safe_identifier(
                scene_id,
                f"canonical_plan.{field}[{index}]",
            )
        if len(scene_ids) != len(set(scene_ids)):
            raise _invalid(
                f"canonical_plan.{field}",
                "must not contain duplicates",
            )
    comparisons = {
        "task_id": identity["task"]["task_id"],
        "dataset_id": identity["dataset"]["dataset_id"],
        "dataset_digest": identity["dataset"]["digest"],
        "family": semantics["family"],
        "training_scene_ids": semantics["training_scene_ids"],
        "scene_ids": semantics["scene_ids"],
    }
    mismatched = sorted(
        field
        for field, expected in comparisons.items()
        if canonical_plan.get(field) != expected
    )
    if mismatched:
        raise _invalid(
            "canonical_plan",
            "identity/semantics mismatch for field(s): "
            f"{', '.join(mismatched)}",
        )


def _validate_plan_jobs(
    canonical_plan: dict[str, Any],
    inference_jobs: dict[str, dict[str, Any]],
) -> set[str]:
    plan_jobs = _require_list(
        canonical_plan["jobs"], "canonical_plan.jobs"
    )
    by_id: dict[str, dict[str, Any]] = {}
    for index, job_value in enumerate(plan_jobs):
        path = f"canonical_plan.jobs[{index}]"
        job = _require_object(job_value, path)
        _require_fields(job, _CANONICAL_JOB_FIELDS, path)
        job_id = _require_safe_identifier(
            job["job_id"], f"{path}.job_id"
        )
        _require_safe_identifier(
            job["case_id"], f"{path}.case_id"
        )
        _require_safe_identifier(
            job["scene_id"], f"{path}.scene_id"
        )
        _require_non_empty_string(
            job["evaluation_partition"],
            f"{path}.evaluation_partition",
        )
        _require_seed(job["seed"], f"{path}.seed")
        if job_id in by_id:
            raise _invalid(
                f"{path}.job_id",
                f"duplicate job_id {job_id!r}",
            )
        by_id[job_id] = job

    if set(by_id) != set(inference_jobs):
        raise _invalid(
            "inference.jobs",
            "job IDs must exactly match canonical_plan.jobs",
        )
    for job_id, plan_job in by_id.items():
        inference_job = inference_jobs[job_id]
        mismatched = sorted(
            field
            for field in _CANONICAL_JOB_FIELDS
            if inference_job.get(field) != plan_job[field]
        )
        if mismatched:
            raise _invalid(
                f"inference.jobs[{job_id!r}]",
                "does not match canonical_plan job field(s): "
                f"{', '.join(mismatched)}",
            )
    return set(by_id)


def _validate_training(
    training_value: Any,
    canonical_plan: dict[str, Any],
    adaptations: dict[str, dict[str, Any]],
    source_case_ids: set[str],
    inference_jobs: dict[str, dict[str, Any]],
) -> None:
    train_case_values = _require_list(
        canonical_plan["train_case_ids"],
        "canonical_plan.train_case_ids",
    )
    train_case_ids = [
        _require_safe_identifier(
            case_id,
            f"canonical_plan.train_case_ids[{index}]",
        )
        for index, case_id in enumerate(train_case_values)
    ]
    if len(train_case_ids) != len(set(train_case_ids)):
        raise _invalid(
            "canonical_plan.train_case_ids",
            "must not contain duplicates",
        )
    unknown = sorted(set(train_case_ids) - source_case_ids)
    if unknown:
        raise _invalid(
            "canonical_plan.train_case_ids",
            f"reference unknown source case IDs: {unknown}",
        )
    overlap = sorted(
        set(train_case_ids)
        & {job["case_id"] for job in inference_jobs.values()}
    )
    if overlap:
        raise _invalid(
            "canonical_plan",
            f"training and inference cases overlap: {overlap}",
        )

    family = canonical_plan["family"]
    if family == "direct_eval":
        if train_case_ids:
            raise _invalid(
                "canonical_plan.train_case_ids",
                "must be empty for direct_eval",
            )
        if canonical_plan.get("training_seed") is not None:
            raise _invalid(
                "canonical_plan.training_seed",
                "must be null for direct_eval",
            )
        if training_value is not None:
            raise _invalid("training", "must be null for direct_eval")
        return
    if family != "finetune_eval":
        raise _invalid(
            "canonical_plan.family",
            "must be direct_eval or finetune_eval",
        )
    _require_seed(
        canonical_plan.get("training_seed"),
        "canonical_plan.training_seed",
    )

    training = _require_object(training_value, "training")
    _require_fields(
        training,
        {"case_ids", "adaptation_ids", "seed"},
        "training",
    )
    training_case_ids = _require_list(
        training["case_ids"], "training.case_ids"
    )
    if training_case_ids != train_case_ids:
        raise _invalid(
            "training.case_ids",
            "must exactly match canonical_plan.train_case_ids",
        )
    if training["seed"] != canonical_plan.get("training_seed"):
        raise _invalid(
            "training.seed",
            "must match canonical_plan.training_seed",
        )
    _require_seed(training["seed"], "training.seed")
    adaptation_ids = _require_list(
        training["adaptation_ids"], "training.adaptation_ids"
    )
    if (
        len(adaptation_ids) != len(train_case_ids)
        or len(set(adaptation_ids)) != len(adaptation_ids)
    ):
        raise _invalid(
            "training.adaptation_ids",
            "must contain one unique adaptation per training case",
        )
    adaptation_case_ids: list[str] = []
    for index, adaptation_id_value in enumerate(adaptation_ids):
        path = f"training.adaptation_ids[{index}]"
        adaptation_id = _require_safe_identifier(
            adaptation_id_value, path
        )
        adaptation = adaptations.get(adaptation_id)
        if adaptation is None:
            raise _invalid(
                path,
                f"references unknown adaptation_id {adaptation_id!r}",
            )
        if adaptation["role"] != "train":
            raise _invalid(
                path,
                "training must reference a train adaptation",
            )
        adaptation_case_ids.append(adaptation["case_id"])
    if adaptation_case_ids != train_case_ids:
        raise _invalid(
            "training.adaptation_ids",
            "adaptation case IDs must match training.case_ids in order",
        )


def _validate_reference_list(
    value: Any,
    path: str,
    known_ids: set[str],
    reference_name: str,
    *,
    owner_id: str | None = None,
) -> None:
    references = _require_list(value, path)
    for index, reference_value in enumerate(references):
        reference_path = f"{path}[{index}]"
        reference = _require_non_empty_string(reference_value, reference_path)
        if owner_id is not None and reference == owner_id:
            raise _invalid(
                reference_path,
                f"operation {owner_id!r} must not depend on itself",
            )
        if reference not in known_ids:
            raise _invalid(
                reference_path,
                f"references unknown {reference_name} {reference!r}",
            )


def _validate_execution_graph(
    execution_graph_value: Any,
    job_ids: set[str],
) -> None:
    execution_graph = _require_object(
        execution_graph_value, "execution_graph"
    )
    _require_fields(execution_graph, {"operations"}, "execution_graph")
    operations = _require_list(
        execution_graph["operations"], "execution_graph.operations"
    )

    operation_ids: set[str] = set()
    parsed_operations: list[tuple[str, dict[str, Any]]] = []
    job_coverage = {"infer": set(), "evaluate": set()}
    for index, operation_value in enumerate(operations):
        path = f"execution_graph.operations[{index}]"
        operation = _require_object(operation_value, path)
        _require_fields(operation, {"operation_id"}, path)
        operation_id = _require_non_empty_string(
            operation["operation_id"], f"{path}.operation_id"
        )
        if operation_id in operation_ids:
            raise _invalid(
                f"{path}.operation_id",
                f"duplicate operation_id {operation_id!r}",
            )
        operation_ids.add(operation_id)
        parsed_operations.append((path, operation))
        kind = operation.get("kind")
        if kind in job_coverage:
            if "job_ids" not in operation:
                raise _invalid(
                    path,
                    f"{kind} operation requires job_ids",
                )
            references = _require_list(
                operation["job_ids"], f"{path}.job_ids"
            )
            for reference in references:
                if isinstance(reference, str):
                    job_coverage[kind].add(reference)

    for path, operation in parsed_operations:
        operation_id = operation["operation_id"]
        if "depends_on" in operation:
            _validate_reference_list(
                operation["depends_on"],
                f"{path}.depends_on",
                operation_ids,
                "operation_id",
                owner_id=operation_id,
            )
        if "job_ids" in operation:
            _validate_reference_list(
                operation["job_ids"],
                f"{path}.job_ids",
                job_ids,
                "inference job_id",
            )

    dependencies = {
        operation["operation_id"]: list(operation.get("depends_on", []))
        for _, operation in parsed_operations
    }
    active: set[str] = set()
    complete: set[str] = set()

    def visit(operation_id: str) -> None:
        if operation_id in complete:
            return
        if operation_id in active:
            raise _invalid(
                "execution_graph.operations",
                f"dependency cycle includes operation {operation_id!r}",
            )
        active.add(operation_id)
        for dependency in dependencies[operation_id]:
            visit(dependency)
        active.remove(operation_id)
        complete.add(operation_id)

    for operation_id in dependencies:
        visit(operation_id)

    for kind, covered in job_coverage.items():
        if covered != job_ids:
            raise _invalid(
                "execution_graph.operations",
                f"{kind} operations must cover every inference job",
            )


def validate_task_instance_document(document: dict) -> None:
    """Validate a schema-v3 BaselineTaskInstance runtime document.

    The validation deliberately leaves each adaptation's baseline-owned payload
    opaque. In particular, ``input_contract`` is not required so that existing
    command-v3 Baselines remain loadable.
    """

    root = _require_object(document, "$")
    actual_fields = set(root)
    missing = sorted(_TOP_LEVEL_FIELDS - actual_fields)
    unexpected = sorted(actual_fields - _TOP_LEVEL_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing required field(s): {', '.join(missing)}")
        if unexpected:
            details.append(
                f"unexpected field(s): {', '.join(unexpected)}"
            )
        raise _invalid("$", "; ".join(details))

    if root["schema_version"] != "3.0":
        raise _invalid("schema_version", "must equal '3.0'")
    _require_safe_identifier(root["instance_id"], "instance_id")
    _require_sha256(root["instance_digest"], "instance_digest")
    _validate_identity(root["identity"])
    _validate_semantics(root["semantics"])
    identity = root["identity"]
    semantics = root["semantics"]
    canonical_plan = _require_object(
        root["canonical_plan"], "canonical_plan"
    )
    if (
        _canonical_sha256(canonical_plan)
        != root["identity"]["canonical_plan_digest"]
    ):
        raise _invalid(
            "identity.canonical_plan_digest",
            "does not match canonical_plan",
        )
    _validate_plan_identity(canonical_plan, identity, semantics)
    case_ids = _validate_source(root["source"])

    adaptations = _validate_adaptations(root["adaptations"])
    unknown_adaptation_cases = sorted({
        item["case_id"]
        for item in adaptations.values()
        if item["case_id"] not in case_ids
    })
    if unknown_adaptation_cases:
        raise _invalid(
            "adaptations",
            "reference unknown source case IDs: "
            f"{unknown_adaptation_cases}",
        )
    inference_jobs = _validate_inference(
        root["inference"],
        adaptations,
        case_ids,
    )
    _validate_training(
        root["training"],
        canonical_plan,
        adaptations,
        case_ids,
        inference_jobs,
    )
    job_ids = _validate_plan_jobs(canonical_plan, inference_jobs)
    _validate_execution_graph(root["execution_graph"], job_ids)

    _require_list(root["cache_bindings"], "cache_bindings")
    _require_object(root["baseline_payload"], "baseline_payload")


__all__ = ["validate_task_instance_document"]
