from __future__ import annotations

import re
from typing import Any


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


def _invalid(path: str, message: str) -> ValueError:
    return ValueError(f"Invalid BaselineTaskInstance at {path}: {message}")


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


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _invalid(path, "must be a lowercase 64-character SHA-256 hex digest")
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
    _require_non_empty_string(
        dataset["dataset_id"], "identity.dataset.dataset_id"
    )
    _require_sha256(dataset["digest"], "identity.dataset.digest")

    task = _require_object(identity["task"], "identity.task")
    _require_fields(task, {"task_id", "digest"}, "identity.task")
    _require_non_empty_string(task["task_id"], "identity.task.task_id")
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
    _require_non_empty_string(
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
        {"family", "conditioning", "scene_ids"},
        "semantics",
    )
    _require_non_empty_string(semantics["family"], "semantics.family")
    _require_non_empty_string(
        semantics["conditioning"], "semantics.conditioning"
    )
    scene_ids = _require_list(semantics["scene_ids"], "semantics.scene_ids")
    for index, scene_id in enumerate(scene_ids):
        _require_non_empty_string(
            scene_id, f"semantics.scene_ids[{index}]"
        )
    if len(scene_ids) != len(set(scene_ids)):
        raise _invalid("semantics.scene_ids", "must not contain duplicates")


def _validate_source(source_value: Any) -> set[str]:
    source = _require_object(source_value, "source")
    _require_fields(source, {"asset_root", "cases"}, "source")
    _require_non_empty_string(source["asset_root"], "source.asset_root")
    cases = _require_list(source["cases"], "source.cases")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        case = _require_object(case, f"source.cases[{index}]")
        case_id = _require_non_empty_string(
            case.get("case_id"), f"source.cases[{index}].case_id"
        )
        if case_id in case_ids:
            raise _invalid(
                f"source.cases[{index}].case_id",
                f"duplicate case_id {case_id!r}",
            )
        case_ids.add(case_id)
    return case_ids


def _validate_adaptations(adaptations_value: Any) -> set[str]:
    adaptations = _require_list(adaptations_value, "adaptations")
    adaptation_ids: set[str] = set()
    for index, adaptation_value in enumerate(adaptations):
        path = f"adaptations[{index}]"
        adaptation = _require_object(adaptation_value, path)
        _require_fields(adaptation, {"adaptation_id"}, path)
        adaptation_id = _require_non_empty_string(
            adaptation["adaptation_id"], f"{path}.adaptation_id"
        )
        if adaptation_id in adaptation_ids:
            raise _invalid(
                f"{path}.adaptation_id",
                f"duplicate adaptation_id {adaptation_id!r}",
            )
        adaptation_ids.add(adaptation_id)
    return adaptation_ids


def _validate_inference(
    inference_value: Any,
    adaptation_ids: set[str],
    case_ids: set[str],
) -> set[str]:
    inference = _require_object(inference_value, "inference")
    _require_fields(inference, {"predictor", "jobs"}, "inference")
    _require_object(inference["predictor"], "inference.predictor")

    jobs = _require_list(inference["jobs"], "inference.jobs")
    job_ids: set[str] = set()
    for index, job_value in enumerate(jobs):
        path = f"inference.jobs[{index}]"
        job = _require_object(job_value, path)
        _require_fields(
            job,
            {"job_id", "case_id", "adaptation_id"},
            path,
        )
        job_id = _require_non_empty_string(job["job_id"], f"{path}.job_id")
        if job_id in job_ids:
            raise _invalid(
                f"{path}.job_id",
                f"duplicate job_id {job_id!r}",
            )
        job_ids.add(job_id)

        case_id = _require_non_empty_string(
            job["case_id"], f"{path}.case_id"
        )
        if case_id not in case_ids:
            raise _invalid(
                f"{path}.case_id",
                f"references unknown source case_id {case_id!r}",
            )

        adaptation_id = _require_non_empty_string(
            job["adaptation_id"], f"{path}.adaptation_id"
        )
        if adaptation_id not in adaptation_ids:
            raise _invalid(
                f"{path}.adaptation_id",
                f"references unknown adaptation_id {adaptation_id!r}",
            )
    return job_ids


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


def validate_task_instance_document(document: dict) -> None:
    """Validate a schema-v2.1 BaselineTaskInstance runtime document.

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

    if root["schema_version"] != "2.1":
        raise _invalid("schema_version", "must equal '2.1'")
    _require_non_empty_string(root["instance_id"], "instance_id")
    _require_sha256(root["instance_digest"], "instance_digest")
    _validate_identity(root["identity"])
    _validate_semantics(root["semantics"])
    _require_object(root["canonical_plan"], "canonical_plan")
    case_ids = _validate_source(root["source"])

    adaptation_ids = _validate_adaptations(root["adaptations"])
    training = root["training"]
    if training is not None:
        _require_object(training, "training")
    job_ids = _validate_inference(
        root["inference"],
        adaptation_ids,
        case_ids,
    )
    _validate_execution_graph(root["execution_graph"], job_ids)

    _require_list(root["cache_bindings"], "cache_bindings")
    _require_object(root["baseline_payload"], "baseline_payload")


__all__ = ["validate_task_instance_document"]
