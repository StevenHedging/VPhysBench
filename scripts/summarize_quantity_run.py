#!/usr/bin/env python3
"""Create a reproducible, read-only summary of a completed AtomicRun."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from physbench.artifacts import (
    PREDICTION_STATUSES,
    validate_prediction_records,
)
from physbench.baselines.wan22_quantity import Wan22QuantityLoraAdapter
from physbench.baselines.wan22_quantity_model import (
    verified_quantity_checkpoint_bytes,
)
from physbench.domain import BaselineTaskInstance
from physbench.evaluation.contracts import (
    CASE_STATUSES,
    CaseEvaluationResult,
)
from physbench.evaluation.task_evaluator import aggregate_task_results


BOOTSTRAP_SEED = 20260728
BOOTSTRAP_ITERATIONS = 10_000
OUTPUT_JSON = "quantity_run_summary.json"
OUTPUT_MARKDOWN = "quantity_run_summary.md"
REPORTABLE_RUN_STATUSES = {
    "complete",
    "inference_incomplete",
    "failed",
}
EXPECTED_LORA_TENSOR_COUNT = 600
EXPECTED_QUANTITY_ENCODER_TENSOR_COUNT = 19
EXPECTED_COMBINED_TENSOR_COUNT = (
    EXPECTED_LORA_TENSOR_COUNT
    + EXPECTED_QUANTITY_ENCODER_TENSOR_COUNT
)
LEGACY_CHECKPOINT_INVENTORY_FIELDS = (
    "tensor_count",
    "parameter_count",
    "lora_tensor_count",
    "quantity_encoder_tensor_count",
    "dtype_tensor_counts",
)
HARDENED_CHECKPOINT_INVENTORY_FIELDS = (
    "tensor_count",
    "parameter_count",
    "lora_tensor_count",
    "lora_pair_count",
    "lora_rank",
    "lora_target_topology",
    "quantity_encoder_tensor_count",
    "dtype_tensor_counts",
    "safetensors_layout_verified",
    "finite_payload_verified",
)
CHECKPOINT_INVENTORY_PROFILES = {
    "1.0.0": {
        "profile_id": "wan22_quantity_checkpoint_inventory_v1_legacy",
        "required_fields": LEGACY_CHECKPOINT_INVENTORY_FIELDS,
    },
    "1.0.1": {
        "profile_id": "wan22_quantity_checkpoint_inventory_v1_hardened",
        "required_fields": HARDENED_CHECKPOINT_INVENTORY_FIELDS,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required AtomicRun file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required AtomicRun file is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"expected a JSON object at {path}:{line_number}"
            )
        records.append(value)
    return records


def _load_optional_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    return _load_jsonl(path)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _same_number(left: Any, right: Any) -> bool:
    if not _finite_number(left) or not _finite_number(right):
        return left is None and right is None
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    **details: Any,
) -> None:
    issues.append({"code": code, **details})


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _values_equal(left: Any, right: Any) -> bool:
    if _finite_number(left) or _finite_number(right):
        return _same_number(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            set(left) == set(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return (
            len(left) == len(right)
            and all(
                _values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return left == right


def _normalized_object_map(
    value: Any,
    *,
    field: str,
    integrity_issues: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        _issue(
            integrity_issues,
            f"official_{field}_invalid",
            observed_type=type(value).__name__,
        )
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            _issue(
                integrity_issues,
                f"official_{field}_entry_invalid",
                key=repr(key),
                observed_type=type(item).__name__,
            )
            continue
        normalized[key] = copy.deepcopy(item)
    return normalized


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_run(path: Path, run_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _artifact_reference(
    run_dir: Path,
    relative_path: str,
    *,
    json_summary_keys: Sequence[str] = (),
) -> dict[str, Any]:
    path = run_dir / relative_path
    reference: dict[str, Any] = {
        "path": relative_path,
        "available": path.is_file(),
        "sha256": _file_sha256(path) if path.is_file() else None,
    }
    if path.is_file() and json_summary_keys:
        value = _load_json(path)
        reference["summary"] = {
            key: value.get(key) for key in json_summary_keys
        }
    return reference


def _unique_by_job_id(
    records: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"{label}[{index}] has no valid job_id")
        if job_id in indexed:
            raise ValueError(f"{label} contains duplicate job_id {job_id}")
        indexed[job_id] = record
    return indexed


def _validate_record_identity(
    record: dict[str, Any],
    job: dict[str, Any],
    *,
    label: str,
) -> None:
    for field in (
        "case_id",
        "scene_id",
        "evaluation_partition",
        "seed",
    ):
        if record.get(field) != job[field]:
            raise ValueError(
                f"{label} identity mismatch for {job['job_id']}: "
                f"{field}={record.get(field)!r}, expected {job[field]!r}"
            )


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _derived_bootstrap_seed(label: str) -> int:
    payload = f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _score_statistics(
    case_scores: Sequence[float],
    *,
    label: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Summarize case means and bootstrap their arithmetic mean."""
    if not case_scores:
        return {
            "score_count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "ci95": None,
        }
    values = [float(value) for value in case_scores]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite evaluated score")
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values)
    if len(values) == 1:
        lower = upper = values[0]
    else:
        rng = random.Random(_derived_bootstrap_seed(label))
        count = len(values)
        bootstrap_means = [
            statistics.fmean(values[rng.randrange(count)] for _ in range(count))
            for _ in range(iterations)
        ]
        bootstrap_means.sort()
        lower = _linear_quantile(bootstrap_means, 0.025)
        upper = _linear_quantile(bootstrap_means, 0.975)
    return {
        "score_count": len(values),
        "mean": mean,
        "median": median,
        "std": std,
        "ci95": {"low": lower, "high": upper},
    }


def _summarize_jobs(
    jobs: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    prediction_statuses: Counter[str] = Counter()
    evaluation_statuses: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    completed = 0
    evaluated = 0
    failed_ids: set[str] = set()
    unavailable_ids: set[str] = set()
    other_terminal_ids: set[str] = set()
    missing_prediction_ids: set[str] = set()
    missing_evaluation_ids: set[str] = set()
    missing_prediction = 0
    missing_evaluation = 0
    other_terminal = 0
    scores_by_case: dict[str, list[float]] = defaultdict(list)
    evaluated_job_scores: list[float] = []

    for job in jobs:
        job_id = job["job_id"]
        prediction = predictions.get(job_id)
        evaluation = evaluations.get(job_id)

        if prediction is None:
            missing_prediction += 1
            missing_prediction_ids.add(job_id)
        else:
            status = str(prediction.get("status", "unknown"))
            prediction_statuses[status] += 1
            if status == "complete":
                completed += 1
            elif status in {"error", "failed"}:
                failed_ids.add(job_id)

        if evaluation is None:
            missing_evaluation += 1
            missing_evaluation_ids.add(job_id)
            continue
        status = str(evaluation.get("status", "unknown"))
        evaluation_statuses[status] += 1
        reason_code = evaluation.get("reason_code")
        if isinstance(reason_code, str) and reason_code:
            reason_codes[reason_code] += 1
        if status == "evaluated":
            score = evaluation.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError(
                    f"evaluated job {job_id} has no valid score in [0,1]"
                )
            evaluated += 1
            numeric_score = float(score)
            scores_by_case[job["case_id"]].append(numeric_score)
            evaluated_job_scores.append(numeric_score)
        elif status in {"error", "failed"}:
            failed_ids.add(job_id)
        elif status in {"unavailable", "unsupported"}:
            if job_id not in failed_ids:
                unavailable_ids.add(job_id)
        else:
            other_terminal += 1
            other_terminal_ids.add(job_id)

    case_means = [
        statistics.fmean(scores_by_case[case_id])
        for case_id in sorted(scores_by_case)
    ]
    statistics_value = _score_statistics(case_means, label=label)
    expected = len(jobs)
    return {
        "expected": expected,
        "completed": completed,
        "evaluated": evaluated,
        "failed": len(failed_ids),
        "unavailable": len(unavailable_ids - failed_ids),
        "missing_prediction": missing_prediction,
        "missing_evaluation": missing_evaluation,
        "other_terminal": other_terminal,
        "coverage": evaluated / expected if expected else None,
        **statistics_value,
        "job_micro_mean": (
            statistics.fmean(evaluated_job_scores)
            if evaluated_job_scores
            else None
        ),
        "score_unit": "case_mean_across_evaluated_inference_seeds",
        "aggregation_policy": (
            "case_macro_after_within_case_inference_seed_mean"
        ),
        "prediction_status_counts": dict(sorted(prediction_statuses.items())),
        "evaluation_status_counts": dict(sorted(evaluation_statuses.items())),
        "reason_code_counts": dict(sorted(reason_codes.items())),
        "failed_job_ids": sorted(failed_ids),
        "unavailable_job_ids": sorted(unavailable_ids - failed_ids),
        "missing_prediction_job_ids": sorted(missing_prediction_ids),
        "missing_evaluation_job_ids": sorted(missing_evaluation_ids),
        "other_terminal_job_ids": sorted(other_terminal_ids),
    }


def _checkpoint_inventory_profile(
    run_dir: Path,
    integrity_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    run = _load_optional_json(run_dir / "run.json")
    frozen_baseline = _load_optional_json(
        run_dir / "frozen" / "baseline.json"
    )
    sealed_instance = _load_optional_json(
        run_dir / "task_instance" / "manifest.json"
    )
    sealed_identity = (
        sealed_instance.get("identity")
        if isinstance(sealed_instance, dict)
        else None
    )
    sealed_baseline = (
        sealed_identity.get("baseline")
        if isinstance(sealed_identity, dict)
        else None
    )
    identity_documents = {
        "run": run,
        "frozen_baseline": frozen_baseline,
        "sealed_task_instance": sealed_baseline,
    }
    identity_sources: dict[str, dict[str, Any]] = {}
    invalid_sources: list[str] = []
    for source, document in identity_documents.items():
        baseline_id = (
            document.get("baseline_id")
            if isinstance(document, dict)
            else None
        )
        baseline_version = (
            document.get("baseline_version")
            if isinstance(document, dict)
            else None
        )
        identity_sources[source] = {
            "baseline_id": baseline_id,
            "baseline_version": baseline_version,
        }
        if (
            not isinstance(baseline_id, str)
            or not baseline_id
            or not isinstance(baseline_version, str)
            or not baseline_version
        ):
            invalid_sources.append(source)

    source_pairs = {
        (
            identity["baseline_id"],
            identity["baseline_version"],
        )
        for identity in identity_sources.values()
        if (
            isinstance(identity["baseline_id"], str)
            and identity["baseline_id"]
            and isinstance(identity["baseline_version"], str)
            and identity["baseline_version"]
        )
    }
    identity_verified = not invalid_sources and len(source_pairs) == 1
    if not identity_verified:
        _issue(
            integrity_issues,
            "checkpoint_inventory_profile_identity_mismatch",
            invalid_sources=invalid_sources,
            identity_sources=identity_sources,
        )

    baseline_id: str | None = None
    baseline_version: str | None = None
    if identity_verified:
        baseline_id, baseline_version = next(iter(source_pairs))
    selected = (
        CHECKPOINT_INVENTORY_PROFILES.get(baseline_version)
        if baseline_version is not None
        else None
    )
    if identity_verified and selected is None:
        _issue(
            integrity_issues,
            "checkpoint_inventory_profile_unsupported",
            baseline_id=baseline_id,
            baseline_version=baseline_version,
            supported_versions=sorted(CHECKPOINT_INVENTORY_PROFILES),
        )

    return {
        "profile_id": (
            selected["profile_id"] if selected is not None else None
        ),
        "baseline_id": baseline_id,
        "baseline_version": baseline_version,
        "version_sources": {
            source: identity["baseline_version"]
            for source, identity in identity_sources.items()
        },
        "baseline_id_sources": {
            source: identity["baseline_id"]
            for source, identity in identity_sources.items()
        },
        "identity_sources": identity_sources,
        "required_fields": (
            list(selected["required_fields"])
            if selected is not None
            else []
        ),
        "verified": identity_verified and selected is not None,
    }


def _checkpoint_inventory_declaration(
    *,
    profile: dict[str, Any],
    declared_inventory: dict[str, Any] | None,
    actual_inventory: dict[str, Any] | None,
    integrity_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    required_fields = list(profile["required_fields"])
    declared_fields = (
        sorted(declared_inventory)
        if declared_inventory is not None
        else []
    )
    observed_fields = (
        sorted(actual_inventory)
        if actual_inventory is not None
        else []
    )
    derived_not_declared = (
        sorted(set(actual_inventory) - set(declared_inventory))
        if actual_inventory is not None
        and declared_inventory is not None
        else []
    )

    missing_declared_fields: list[str] = []
    missing_observed_fields: list[str] = []
    undeclared_observed_fields: list[str] = []
    mismatched_fields: list[str] = []
    if declared_inventory is not None:
        for field in required_fields:
            if field not in declared_inventory:
                missing_declared_fields.append(field)
                _issue(
                    integrity_issues,
                    "checkpoint_inventory_declaration_missing_field",
                    profile_id=profile["profile_id"],
                    field=field,
                )
    if actual_inventory is not None:
        missing_observed_fields = sorted(
            set(required_fields) - set(actual_inventory)
        )
    if (
        declared_inventory is not None
        and actual_inventory is not None
    ):
        undeclared_observed_fields = sorted(
            set(declared_inventory) - set(actual_inventory)
        )
        mismatched_fields = sorted(
            field
            for field in set(declared_inventory) & set(actual_inventory)
            if not _values_equal(
                declared_inventory[field],
                actual_inventory[field],
            )
        )
        if (
            missing_observed_fields
            or undeclared_observed_fields
            or mismatched_fields
        ):
            _issue(
                integrity_issues,
                "checkpoint_declared_inventory_mismatch",
                missing_observed_required_fields=missing_observed_fields,
                declared_fields_absent_from_observed=(
                    undeclared_observed_fields
                ),
                mismatched_fields=mismatched_fields,
                declared=declared_inventory,
                actual=actual_inventory,
            )

    verified = (
        profile.get("verified") is True
        and declared_inventory is not None
        and actual_inventory is not None
        and not missing_declared_fields
        and not missing_observed_fields
        and not undeclared_observed_fields
        and not mismatched_fields
    )
    return {
        "required_fields": required_fields,
        "declared_fields": declared_fields,
        "observed_fields": observed_fields,
        "derived_not_declared": derived_not_declared,
        "verified": verified,
    }


def _checkpoint_provenance(
    run_dir: Path,
    integrity_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_count_before = len(integrity_issues)
    inventory_profile = _checkpoint_inventory_profile(
        run_dir,
        integrity_issues,
    )
    relative_manifest = "artifacts/wan22/checkpoint.json"
    manifest_path = run_dir / relative_manifest
    reference = _artifact_reference(run_dir, relative_manifest)
    expected_inventory = {
        "tensor_count": EXPECTED_COMBINED_TENSOR_COUNT,
        "lora_tensor_count": EXPECTED_LORA_TENSOR_COUNT,
        "quantity_encoder_tensor_count": (
            EXPECTED_QUANTITY_ENCODER_TENSOR_COUNT
        ),
    }
    if not manifest_path.is_file():
        _issue(
            integrity_issues,
            "checkpoint_manifest_missing",
            path=relative_manifest,
        )
        return {
            "digest": None,
            "path": None,
            "available": False,
            "digest_verified": None,
            "byte_binding": None,
            "acceptance": {
                "passed": False,
                "expected_inventory": expected_inventory,
                "observed_inventory": None,
            },
            "inventory_profile": inventory_profile,
            "inventory_declaration": _checkpoint_inventory_declaration(
                profile=inventory_profile,
                declared_inventory=None,
                actual_inventory=None,
                integrity_issues=integrity_issues,
            ),
            "manifest": reference,
        }
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "2.0":
        _issue(
            integrity_issues,
            "checkpoint_manifest_schema_invalid",
            observed=manifest.get("schema_version"),
        )
    checkpoint_value = manifest.get("checkpoint")
    checkpoint_path: Path | None = None
    if isinstance(checkpoint_value, str) and checkpoint_value:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
    elif checkpoint_value is not None:
        _issue(
            integrity_issues,
            "checkpoint_path_invalid",
            value=repr(checkpoint_value),
        )
    available = checkpoint_path is not None and checkpoint_path.is_file()
    declared = manifest.get("checkpoint_sha256")
    declared_valid = _valid_sha256(declared)
    if not declared_valid:
        _issue(
            integrity_issues,
            (
                "checkpoint_digest_missing"
                if declared is None
                else "checkpoint_digest_invalid"
            ),
            declared_sha256=declared,
        )
    actual: str | None = None
    actual_size: int | None = None
    actual_inventory: dict[str, Any] | None = None
    checkpoint_byte_binding: dict[str, Any] | None = None
    if checkpoint_path is not None and not available:
        _issue(
            integrity_issues,
            "checkpoint_file_missing",
            path=_relative_to_run(checkpoint_path, run_dir),
        )
    elif checkpoint_path is None:
        _issue(
            integrity_issues,
            "checkpoint_path_missing",
            manifest=relative_manifest,
        )
    elif Path(_relative_to_run(checkpoint_path, run_dir)).is_absolute():
        _issue(
            integrity_issues,
            "checkpoint_outside_atomic_run",
            path=str(checkpoint_path.resolve()),
        )

    status = manifest.get("status")
    if status != "complete":
        _issue(
            integrity_issues,
            "checkpoint_status_not_complete",
            status=status,
        )
    baseline_id = manifest.get("baseline_id")
    frozen_baseline = _load_optional_json(
        run_dir / "frozen" / "baseline.json"
    ) or {}
    expected_baseline_id = frozen_baseline.get("baseline_id")
    if baseline_id != expected_baseline_id:
        _issue(
            integrity_issues,
            "checkpoint_baseline_identity_mismatch",
            expected=expected_baseline_id,
            observed=baseline_id,
        )

    declared_size = manifest.get("checkpoint_size")

    declared_inventory = manifest.get("inventory")
    if not isinstance(declared_inventory, dict):
        _issue(
            integrity_issues,
            "checkpoint_inventory_missing",
            observed=declared_inventory,
        )
        declared_inventory = None

    if available:
        try:
            with verified_quantity_checkpoint_bytes(
                checkpoint_path,
                manifest_path,
            ) as (checkpoint_bytes, byte_binding):
                actual_inventory = (
                    Wan22QuantityLoraAdapter._checkpoint_inventory_bytes(
                        checkpoint_bytes
                    )
                )
            checkpoint_byte_binding = byte_binding
            actual = byte_binding["checkpoint_sha256"]
            actual_size = byte_binding["checkpoint_size"]
        except (
            KeyError,
            MemoryError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            _issue(
                integrity_issues,
                "checkpoint_inventory_read_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
    digest = declared if declared_valid else actual
    verified = (
        actual == declared
        if actual is not None and declared_valid
        else None
    )
    if verified is False:
        _issue(
            integrity_issues,
            "checkpoint_digest_mismatch",
            declared_sha256=declared,
            actual_sha256=actual,
        )
    if actual_size is not None and declared_size != actual_size:
        _issue(
            integrity_issues,
            "checkpoint_size_mismatch",
            declared=declared_size,
            actual=actual_size,
        )
    if actual_inventory is not None:
        for field, expected in expected_inventory.items():
            observed = actual_inventory.get(field)
            if observed != expected:
                _issue(
                    integrity_issues,
                    "checkpoint_inventory_mismatch",
                    field=field,
                    expected=expected,
                    observed=observed,
                )
    inventory_declaration = _checkpoint_inventory_declaration(
        profile=inventory_profile,
        declared_inventory=declared_inventory,
        actual_inventory=actual_inventory,
        integrity_issues=integrity_issues,
    )

    sampling = _load_optional_json(
        run_dir / "artifacts" / "wan22" / "training_sampling_plan.json"
    )
    expected_steps = (
        sampling.get("expected_total_optimizer_steps")
        if isinstance(sampling, dict)
        else None
    )
    expected_world_size = (
        sampling.get("world_size")
        if isinstance(sampling, dict)
        else None
    )
    trainer_config = (
        frozen_baseline.get("trainer", {}).get("config", {})
        if isinstance(frozen_baseline.get("trainer"), dict)
        else {}
    )
    recovery_required = trainer_config.get("save_optimizer_state") is True
    recovery = manifest.get("training_state")
    recovery_summary: dict[str, Any] = {
        "required": recovery_required,
        "available": isinstance(recovery, dict),
        "passed": not recovery_required and recovery is None,
        "optimizer_scheduler": None,
        "state_manifest": None,
        "rng_states": [],
    }
    if recovery_required and not isinstance(recovery, dict):
        _issue(
            integrity_issues,
            "checkpoint_training_state_missing",
        )
    if isinstance(recovery, dict):
        recovery_start = len(integrity_issues)

        def recovery_path(field: str) -> Path | None:
            value = recovery.get(field)
            if not isinstance(value, str) or not value:
                _issue(
                    integrity_issues,
                    "checkpoint_training_state_path_invalid",
                    field=field,
                    observed=value,
                )
                return None
            path = Path(value)
            if not path.is_absolute():
                path = run_dir / path
            relative = _relative_to_run(path, run_dir)
            if Path(relative).is_absolute():
                _issue(
                    integrity_issues,
                    "checkpoint_training_state_outside_atomic_run",
                    field=field,
                    path=str(path.resolve()),
                )
            return path

        state_directory = recovery_path("directory")
        optimizer_path = recovery_path("optimizer_scheduler")
        state_manifest_path = recovery_path("state_manifest")
        for field, path in (
            ("optimizer_scheduler", optimizer_path),
            ("state_manifest", state_manifest_path),
        ):
            if path is None or not path.is_file():
                _issue(
                    integrity_issues,
                    "checkpoint_training_state_file_missing",
                    field=field,
                    path=(
                        _relative_to_run(path, run_dir)
                        if path is not None
                        else None
                    ),
                )
        optimizer_digest = recovery.get("optimizer_scheduler_sha256")
        optimizer_actual_digest = (
            _file_sha256(optimizer_path)
            if optimizer_path is not None and optimizer_path.is_file()
            else None
        )
        if (
            not _valid_sha256(optimizer_digest)
            or optimizer_digest != optimizer_actual_digest
        ):
            _issue(
                integrity_issues,
                "checkpoint_training_state_digest_mismatch",
                declared=optimizer_digest,
                actual=optimizer_actual_digest,
            )

        state_document = (
            _load_optional_json(state_manifest_path)
            if state_manifest_path is not None
            else None
        )
        if state_document is not None:
            for field, expected in (
                ("status", "complete"),
                ("global_step", expected_steps),
                ("world_size", expected_world_size),
            ):
                if expected is not None and state_document.get(field) != expected:
                    _issue(
                        integrity_issues,
                        "checkpoint_training_state_manifest_mismatch",
                        field=field,
                        expected=expected,
                        observed=state_document.get(field),
                    )
            if (
                checkpoint_path is not None
                and state_document.get("model_checkpoint")
                != checkpoint_path.name
            ):
                _issue(
                    integrity_issues,
                    "checkpoint_training_state_manifest_mismatch",
                    field="model_checkpoint",
                    expected=checkpoint_path.name,
                    observed=state_document.get("model_checkpoint"),
                )

        rng_paths: list[Path] = []
        if (
            state_directory is not None
            and state_directory.is_dir()
            and isinstance(expected_world_size, int)
            and not isinstance(expected_world_size, bool)
            and expected_world_size > 0
        ):
            expected_rng_names = {
                f"rng_rank_{rank:02d}.pt"
                for rank in range(expected_world_size)
            }
            observed_rng_names = {
                path.name
                for path in state_directory.glob("rng_rank_*.pt")
                if path.is_file()
            }
            if observed_rng_names - expected_rng_names:
                _issue(
                    integrity_issues,
                    "checkpoint_rng_state_inventory_extra",
                    filenames=sorted(
                        observed_rng_names - expected_rng_names
                    ),
                )
            for rank in range(expected_world_size):
                rng_path = state_directory / f"rng_rank_{rank:02d}.pt"
                if not rng_path.is_file():
                    _issue(
                        integrity_issues,
                        "checkpoint_rng_state_missing",
                        rank=rank,
                        path=_relative_to_run(rng_path, run_dir),
                    )
                elif rng_path.stat().st_size == 0:
                    _issue(
                        integrity_issues,
                        "checkpoint_rng_state_empty",
                        rank=rank,
                        path=_relative_to_run(rng_path, run_dir),
                    )
                else:
                    rng_paths.append(rng_path)
        elif recovery_required:
            _issue(
                integrity_issues,
                "checkpoint_rng_state_inventory_invalid",
                world_size=expected_world_size,
            )
        recovery_summary = {
            "required": recovery_required,
            "available": True,
            "passed": len(integrity_issues) == recovery_start,
            "optimizer_scheduler": (
                {
                    "path": _relative_to_run(optimizer_path, run_dir),
                    "sha256": optimizer_actual_digest,
                }
                if optimizer_path is not None
                else None
            ),
            "state_manifest": (
                _artifact_reference(
                    run_dir,
                    _relative_to_run(state_manifest_path, run_dir),
                )
                if state_manifest_path is not None
                and not Path(
                    _relative_to_run(state_manifest_path, run_dir)
                ).is_absolute()
                else None
            ),
            "rng_states": [
                {
                    "path": _relative_to_run(path, run_dir),
                    "sha256": _file_sha256(path),
                    "sha256_semantics": (
                        "observed_file_sha256_for_reporting_only;"
                        "no_manifest_digest_anchor"
                    ),
                }
                for path in rng_paths
            ],
        }
    return {
        "digest": digest,
        "declared_sha256": declared,
        "actual_sha256": actual,
        "path": (
            _relative_to_run(checkpoint_path, run_dir)
            if checkpoint_path is not None
            else None
        ),
        "available": available,
        "digest_verified": verified,
        "size": actual_size,
        "status": status,
        "source": manifest.get("source"),
        "byte_binding": checkpoint_byte_binding,
        "inventory": actual_inventory,
        "declared_inventory": declared_inventory,
        "inventory_profile": inventory_profile,
        "inventory_declaration": inventory_declaration,
        "training_state": recovery_summary,
        "acceptance": {
            "passed": len(integrity_issues) == issue_count_before,
            "expected_inventory": expected_inventory,
            "observed_inventory": actual_inventory,
        },
        "manifest": reference,
    }


def _partition_names(
    task: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    if task.get("family") == "finetune_eval":
        selected = task.get("selection", {}).get("eval_partitions")
        if isinstance(selected, list) and all(
            isinstance(item, str) and item for item in selected
        ):
            selected_names = list(dict.fromkeys(selected))
            planned_names = sorted({
                str(job["evaluation_partition"])
                for job in plan["jobs"]
            })
            return [
                *selected_names,
                *[
                    name
                    for name in planned_names
                    if name not in selected_names
                ],
            ]
    return sorted({
        str(job["evaluation_partition"])
        for job in plan["jobs"]
    })


def _validate_plan_jobs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("AtomicRun plan.jobs must be an array")
    indexed: set[str] = set()
    required = {
        "job_id",
        "case_id",
        "scene_id",
        "evaluation_partition",
        "seed",
    }
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or not required.issubset(job):
            raise ValueError(f"plan.jobs[{index}] is incomplete")
        job_id = job["job_id"]
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"plan.jobs[{index}].job_id is invalid")
        if job_id in indexed:
            raise ValueError(f"plan contains duplicate job_id {job_id}")
        indexed.add(job_id)
    return jobs


def _validate_sealed_run_identity(
    *,
    run_dir: Path,
    run: dict[str, Any],
    plan: dict[str, Any],
    task: dict[str, Any],
    component_fingerprints: dict[str, Any],
    integrity_issues: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    manifest_path = run_dir / "task_instance" / "manifest.json"
    manifest = _load_optional_json(manifest_path)
    sealed: BaselineTaskInstance | None = None
    if manifest is None:
        _issue(
            integrity_issues,
            "sealed_task_instance_missing",
            path="task_instance/manifest.json",
        )
    else:
        try:
            sealed = BaselineTaskInstance.from_document(manifest)
            sealed.verify()
        except (KeyError, TypeError, ValueError) as exc:
            _issue(
                integrity_issues,
                "sealed_task_instance_invalid",
                error=f"{type(exc).__name__}: {exc}",
            )

    instance = sealed.value if sealed is not None else None
    summary = {
        "available": manifest is not None,
        "verified": sealed is not None,
        "instance_id": (
            instance.get("instance_id")
            if instance is not None
            else manifest.get("instance_id") if manifest is not None else None
        ),
        "digest": sealed.digest if sealed is not None else None,
        "manifest": _artifact_reference(
            run_dir,
            "task_instance/manifest.json",
        ),
    }
    if instance is None:
        return None, summary

    canonical_plan = instance["canonical_plan"]
    if canonical_plan != plan:
        _issue(
            integrity_issues,
            "sealed_canonical_plan_mismatch",
            sealed_digest=_canonical_sha256(canonical_plan),
            plan_digest=_canonical_sha256(plan),
        )
    projected_plan = _load_optional_json(
        run_dir / "task_instance" / "canonical_plan.json"
    )
    if projected_plan is None:
        _issue(
            integrity_issues,
            "sealed_plan_projection_missing",
            path="task_instance/canonical_plan.json",
        )
    elif projected_plan != canonical_plan:
        _issue(
            integrity_issues,
            "sealed_plan_projection_mismatch",
        )

    identity = instance["identity"]
    semantics = instance["semantics"]
    component_expectations = {
        "dataset": identity["dataset"]["digest"],
        "task": identity["task"]["digest"],
        "baseline": identity["baseline"]["digest"],
        "baseline_deployment": identity["baseline"]["deployment_digest"],
        "task_builder": identity["task_builder"]["fingerprint"],
        "data_adapter": identity["data_adapter"]["fingerprint"],
        "data_adapter_materialization": identity["data_adapter"][
            "materialization_fingerprint"
        ],
        "task_instance": sealed.digest,
    }
    for field, expected in component_expectations.items():
        observed = component_fingerprints.get(field)
        if observed != expected:
            _issue(
                integrity_issues,
                "component_fingerprint_identity_mismatch",
                component=field,
                expected=expected,
                observed=observed,
            )

    run_expectations = {
        "dataset_id": identity["dataset"]["dataset_id"],
        "dataset_digest": identity["dataset"]["digest"],
        "task_id": identity["task"]["task_id"],
        "task_family": semantics["family"],
        "baseline_id": identity["baseline"]["baseline_id"],
        "baseline_version": identity["baseline"]["baseline_version"],
        "baseline_digest": identity["baseline"]["digest"],
        "baseline_deployment_digest": identity["baseline"][
            "deployment_digest"
        ],
        "task_instance_id": instance["instance_id"],
        "task_instance_digest": sealed.digest,
    }
    for field, expected in run_expectations.items():
        observed = run.get(field)
        if observed != expected:
            _issue(
                integrity_issues,
                "run_sealed_identity_mismatch",
                field=field,
                expected=expected,
                observed=observed,
            )

    plan_task_expectations = {
        "task_id": task.get("task_id"),
        "family": task.get("family"),
        "dataset_id": task.get("dataset_id"),
        "scene_ids": semantics.get("scene_ids"),
    }
    for field, expected in plan_task_expectations.items():
        if plan.get(field) != expected:
            _issue(
                integrity_issues,
                "plan_task_identity_mismatch",
                field=field,
                expected=expected,
                observed=plan.get(field),
            )
    computed_task_digest = _canonical_sha256(task)
    if identity["task"]["digest"] != computed_task_digest:
        _issue(
            integrity_issues,
            "sealed_task_digest_mismatch",
            sealed=identity["task"]["digest"],
            actual=computed_task_digest,
        )
    if (
        identity.get("canonical_plan_digest")
        != _canonical_sha256(plan)
    ):
        _issue(
            integrity_issues,
            "sealed_plan_digest_mismatch",
            sealed=identity.get("canonical_plan_digest"),
            actual=_canonical_sha256(plan),
        )

    frozen_baseline = _load_optional_json(
        run_dir / "frozen" / "baseline.json"
    )
    if frozen_baseline is None:
        _issue(
            integrity_issues,
            "frozen_baseline_missing",
            path="frozen/baseline.json",
        )
    else:
        for field in ("baseline_id", "baseline_version"):
            expected = identity["baseline"][field]
            if frozen_baseline.get(field) != expected:
                _issue(
                    integrity_issues,
                    "frozen_baseline_identity_mismatch",
                    field=field,
                    expected=expected,
                    observed=frozen_baseline.get(field),
                )

    frozen_dataset = _load_optional_json(
        run_dir / "frozen" / "dataset.json"
    )
    if frozen_dataset is None:
        _issue(
            integrity_issues,
            "frozen_dataset_missing",
            path="frozen/dataset.json",
        )
    elif (
        frozen_dataset.get("dataset_id")
        != identity["dataset"]["dataset_id"]
    ):
        _issue(
            integrity_issues,
            "frozen_dataset_identity_mismatch",
            expected=identity["dataset"]["dataset_id"],
            observed=frozen_dataset.get("dataset_id"),
        )

    frozen_cases = _load_optional_jsonl(
        run_dir / "frozen" / "cases.jsonl"
    )
    if frozen_cases is None:
        _issue(
            integrity_issues,
            "frozen_cases_missing",
            path="frozen/cases.jsonl",
        )
    else:
        by_case = {
            case.get("case_id"): case
            for case in frozen_cases
            if isinstance(case.get("case_id"), str)
        }
        for source_case in instance["source"]["cases"]:
            frozen_case = by_case.get(source_case["case_id"])
            if frozen_case is None:
                _issue(
                    integrity_issues,
                    "sealed_source_case_missing_from_frozen_dataset",
                    case_id=source_case["case_id"],
                )
            elif (
                source_case.get("scene_id") is not None
                and frozen_case.get("scene_id")
                != source_case.get("scene_id")
            ):
                _issue(
                    integrity_issues,
                    "sealed_source_case_scene_mismatch",
                    case_id=source_case["case_id"],
                    sealed=source_case.get("scene_id"),
                    frozen=frozen_case.get("scene_id"),
                )
    return instance, summary


def _validate_quantity_audit_record(
    *,
    record: dict[str, Any],
    expected_native_inputs: dict[str, Any],
    expected_case_id: str,
    expected_scene_id: str,
    expected_job_id: str | None,
    integrity_issues: list[dict[str, Any]],
    artifact: str,
) -> bool:
    issue_start = len(integrity_issues)
    text = expected_native_inputs.get("text")
    physics = expected_native_inputs.get("physics")
    if not isinstance(text, dict) or not isinstance(physics, dict):
        _issue(
            integrity_issues,
            "sealed_quantity_inputs_invalid",
            artifact=artifact,
            case_id=expected_case_id,
        )
        return False
    expected_registry_id = physics.get("registry_id")
    expected_registry_fingerprint = physics.get("registry_fingerprint")
    expected_quantities = physics.get("quantities")
    if (
        not isinstance(expected_registry_id, str)
        or not _valid_sha256(expected_registry_fingerprint)
        or not isinstance(expected_quantities, list)
        or not expected_quantities
    ):
        _issue(
            integrity_issues,
            "sealed_quantity_payload_invalid",
            artifact=artifact,
            case_id=expected_case_id,
        )
        return False

    registry_id_field = (
        "registry_id" if expected_job_id is not None else "quantity_registry_id"
    )
    registry_fingerprint_field = (
        "registry_fingerprint"
        if expected_job_id is not None
        else "quantity_registry_fingerprint"
    )
    expected_identity = {
        "schema_version": "1.0",
        "case_id": expected_case_id,
        "prompt": text.get("prompt"),
        "audited_prompt": text.get("audited_prompt"),
        registry_id_field: expected_registry_id,
        registry_fingerprint_field: expected_registry_fingerprint,
    }
    if expected_job_id is not None:
        expected_identity["job_id"] = expected_job_id
    else:
        expected_identity["scene_id"] = expected_scene_id
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            _issue(
                integrity_issues,
                "quantity_token_audit_identity_mismatch",
                artifact=artifact,
                field=field,
                expected=expected,
                observed=record.get(field),
            )

    observed_quantities = record.get("quantities")
    expected_by_name = {
        item.get("name"): item
        for item in expected_quantities
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if (
        len(expected_by_name) != len(expected_quantities)
        or not isinstance(observed_quantities, list)
    ):
        _issue(
            integrity_issues,
            "quantity_token_audit_quantities_invalid",
            artifact=artifact,
        )
        return False
    observed_by_name = {
        item.get("name"): item
        for item in observed_quantities
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if (
        len(observed_by_name) != len(observed_quantities)
        or set(observed_by_name) != set(expected_by_name)
    ):
        _issue(
            integrity_issues,
            "quantity_token_audit_quantity_coverage_mismatch",
            artifact=artifact,
            expected_names=sorted(expected_by_name),
            observed_names=sorted(
                name for name in observed_by_name if isinstance(name, str)
            ),
        )
        return False

    token_ids: set[int] = set()
    token_spans: set[tuple[int, int]] = set()
    sentinels: set[str] = set()
    for name, expected in expected_by_name.items():
        observed = observed_by_name[name]
        for field in (
            "dimension",
            "quantity_type",
            "quantity_type_id",
            "sentinel",
            "canonical_si_unit",
        ):
            if observed.get(field) != expected.get(field):
                _issue(
                    integrity_issues,
                    "quantity_token_audit_value_mismatch",
                    artifact=artifact,
                    quantity=name,
                    field=field,
                    expected=expected.get(field),
                    observed=observed.get(field),
                )
        if not _same_number(
            observed.get("si_value"),
            expected.get("si_value"),
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_value_mismatch",
                artifact=artifact,
                quantity=name,
                field="si_value",
                expected=expected.get("si_value"),
                observed=observed.get("si_value"),
            )
        dimension = observed.get("dimension")
        if (
            not isinstance(dimension, list)
            or len(dimension) != 7
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in dimension
            )
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_dimension_invalid",
                artifact=artifact,
                quantity=name,
                observed=dimension,
            )
        type_id = observed.get("quantity_type_id")
        if (
            not isinstance(type_id, int)
            or isinstance(type_id, bool)
            or type_id < 0
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_type_id_invalid",
                artifact=artifact,
                quantity=name,
                observed=type_id,
            )
        token_id = observed.get("sentinel_token_id")
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id in token_ids
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_sentinel_token_invalid",
                artifact=artifact,
                quantity=name,
                observed=token_id,
            )
        else:
            token_ids.add(token_id)
        sentinel = observed.get("sentinel")
        if (
            not isinstance(sentinel, str)
            or not sentinel
            or sentinel in sentinels
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_sentinel_invalid",
                artifact=artifact,
                quantity=name,
                observed=sentinel,
            )
        else:
            sentinels.add(sentinel)
        span = observed.get("token_span")
        span_tuple = (
            tuple(span)
            if isinstance(span, list)
            and len(span) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in span
            )
            else None
        )
        if (
            span_tuple is None
            or span_tuple[0] < 0
            or span_tuple[1] != span_tuple[0] + 1
            or span_tuple in token_spans
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_token_span_invalid",
                artifact=artifact,
                quantity=name,
                observed=span,
            )
        else:
            token_spans.add(span_tuple)
    return len(integrity_issues) == issue_start


def _read_loss_curve(
    path: Path,
    *,
    integrity_issues: list[dict[str, Any]],
) -> tuple[list[int], list[float]] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        _issue(
            integrity_issues,
            "loss_curve_csv_invalid",
            error=f"{type(exc).__name__}: {exc}",
        )
        return None
    steps: list[int] = []
    losses: list[float] = []
    try:
        for row in rows:
            step = int(row["step"])
            loss = float(row["loss"])
            if not math.isfinite(loss):
                raise ValueError("non-finite loss")
            steps.append(step)
            losses.append(loss)
    except (KeyError, TypeError, ValueError) as exc:
        _issue(
            integrity_issues,
            "loss_curve_csv_invalid",
            error=f"{type(exc).__name__}: {exc}",
        )
        return None
    return steps, losses


def _training_evidence(
    run_dir: Path,
    plan: dict[str, Any],
    task_instance: dict[str, Any] | None,
    predictions: dict[str, dict[str, Any]],
    integrity_issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    loss_summary_keys = (
        "recorded_steps",
        "first_step",
        "last_step",
        "expected_total_steps",
        "finite_fraction",
        "mean",
        "median",
        "first_100_mean",
        "last_100_mean",
        "last_over_first_100_mean",
    )
    gradient_summary_keys = (
        "sample_count",
        "positive_quantity_gradient_count",
        "min_positive_quantity_gradient_l2",
        "max_quantity_gradient_l2",
        "text_encoder_gradient_tensor_count_max",
    )
    sampling_summary_keys = (
        "unique_case_count",
        "metadata_row_count",
        "dataset_repeat",
        "num_epochs",
        "world_size",
        "expected_optimizer_steps_per_epoch",
        "expected_total_optimizer_steps",
    )
    evidence = {
        "training_sampling_plan": _artifact_reference(
            run_dir,
            "artifacts/wan22/training_sampling_plan.json",
            json_summary_keys=sampling_summary_keys,
        ),
        "loss_export": _artifact_reference(
            run_dir,
            "training/loss_export.json",
            json_summary_keys=("status", "return_code"),
        ),
        "loss_summary": _artifact_reference(
            run_dir,
            "artifacts/wan22/loss_analysis/loss_summary.json",
            json_summary_keys=loss_summary_keys,
        ),
        "loss_curve_csv": _artifact_reference(
            run_dir,
            "artifacts/wan22/loss_analysis/loss_curve.csv",
        ),
        "loss_curve_png": _artifact_reference(
            run_dir,
            "artifacts/wan22/loss_analysis/loss_curve.png",
        ),
        "gradient_audit": _artifact_reference(
            run_dir,
            "artifacts/wan22/checkpoints/gradient_audit.json",
            json_summary_keys=gradient_summary_keys,
        ),
        "quantity_token_audit": _artifact_reference(
            run_dir,
            "artifacts/wan22/training_quantity_token_audit.jsonl",
        ),
    }
    completed_prediction_ids = sorted(
        job_id
        for job_id, prediction in predictions.items()
        if prediction.get("status") == "complete"
    )
    inference_audit_root = (
        run_dir / "artifacts" / "wan22"
        / "inference_quantity_token_audits"
    )
    inference_audit_files = (
        sorted(inference_audit_root.glob("*.json"))
        if inference_audit_root.is_dir()
        else []
    )
    inference_file_references = [
        {
            "path": _relative_to_run(path, run_dir),
            "sha256": _file_sha256(path),
        }
        for path in inference_audit_files
    ]
    evidence["inference_quantity_token_audits"] = {
        "path": "artifacts/wan22/inference_quantity_token_audits",
        "available": (
            inference_audit_root.is_dir()
            or not completed_prediction_ids
        ),
        "sha256": _canonical_sha256(inference_file_references),
        "summary": {
            "expected_complete_jobs": len(completed_prediction_ids),
            "observed_files": len(inference_audit_files),
        },
        "files": inference_file_references,
    }
    for name, reference in evidence.items():
        if not reference["available"]:
            _issue(
                integrity_issues,
                "training_evidence_missing",
                artifact=name,
                path=reference["path"],
            )

    sampling_start = len(integrity_issues)
    sampling_path = (
        run_dir / "artifacts/wan22/training_sampling_plan.json"
    )
    sampling = _load_optional_json(sampling_path)
    expected_steps: int | None = None
    if sampling is not None:
        value = sampling.get("expected_total_optimizer_steps")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            expected_steps = value
        else:
            _issue(
                integrity_issues,
                "training_expected_steps_invalid",
                observed=value,
            )
        metadata_rows = sampling.get("metadata_row_count")
        dataset_repeat = sampling.get("dataset_repeat")
        world_size = sampling.get("world_size")
        num_epochs = sampling.get("num_epochs")
        steps_per_epoch = sampling.get(
            "expected_optimizer_steps_per_epoch"
        )
        positive_integer_fields = {
            "metadata_row_count": metadata_rows,
            "dataset_repeat": dataset_repeat,
            "world_size": world_size,
            "num_epochs": num_epochs,
        }
        if any(
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value <= 0
            for field_value in positive_integer_fields.values()
        ):
            _issue(
                integrity_issues,
                "training_sampling_dimensions_invalid",
                observed=positive_integer_fields,
            )
        else:
            derived_steps_per_epoch = (
                metadata_rows * dataset_repeat + world_size - 1
            ) // world_size
            derived_total = derived_steps_per_epoch * num_epochs
            if steps_per_epoch != derived_steps_per_epoch:
                _issue(
                    integrity_issues,
                    "training_steps_per_epoch_mismatch",
                    expected=derived_steps_per_epoch,
                    observed=steps_per_epoch,
                )
            if expected_steps != derived_total:
                _issue(
                    integrity_issues,
                    "training_total_steps_mismatch",
                    expected=derived_total,
                    observed=expected_steps,
                )
        train_case_ids = plan.get("train_case_ids")
        if isinstance(train_case_ids, list):
            observed_unique = sampling.get("unique_case_count")
            if observed_unique != len(set(train_case_ids)):
                _issue(
                    integrity_issues,
                    "training_unique_case_count_mismatch",
                    expected=len(set(train_case_ids)),
                    observed=observed_unique,
                )
    sampling_passed = (
        sampling is not None
        and len(integrity_issues) == sampling_start
    )

    loss_start = len(integrity_issues)
    loss_path = (
        run_dir / "artifacts/wan22/loss_analysis/loss_summary.json"
    )
    loss = _load_optional_json(loss_path)
    if loss is not None:
        expected_in_loss = loss.get("expected_total_steps")
        if expected_steps is not None and expected_in_loss != expected_steps:
            _issue(
                integrity_issues,
                "loss_expected_steps_mismatch",
                sampling_plan=expected_steps,
                loss_summary=expected_in_loss,
            )
        recorded = loss.get("recorded_steps")
        if expected_steps is not None and recorded != expected_steps:
            _issue(
                integrity_issues,
                "loss_recorded_steps_mismatch",
                expected=expected_steps,
                observed=recorded,
            )
        expected_first = 1 if expected_steps is not None else None
        if expected_first is not None and loss.get("first_step") != 1:
            _issue(
                integrity_issues,
                "loss_first_step_mismatch",
                expected=1,
                observed=loss.get("first_step"),
            )
        if (
            expected_steps is not None
            and loss.get("last_step") != expected_steps
        ):
            _issue(
                integrity_issues,
                "loss_last_step_mismatch",
                expected=expected_steps,
                observed=loss.get("last_step"),
            )
        if not _same_number(loss.get("finite_fraction"), 1.0):
            _issue(
                integrity_issues,
                "loss_non_finite_or_incomplete",
                finite_fraction=loss.get("finite_fraction"),
            )
        for field in (
            "mean",
            "median",
            "first_100_mean",
            "last_100_mean",
            "last_over_first_100_mean",
        ):
            if not _finite_number(loss.get(field)):
                _issue(
                    integrity_issues,
                    "loss_statistic_invalid",
                    field=field,
                    observed=loss.get(field),
                )
    loss_curve_path = (
        run_dir / "artifacts" / "wan22" / "loss_analysis"
        / "loss_curve.csv"
    )
    curve = (
        _read_loss_curve(
            loss_curve_path,
            integrity_issues=integrity_issues,
        )
        if loss_curve_path.is_file()
        else None
    )
    if curve is not None:
        curve_steps, curve_losses = curve
        expected_curve_steps = (
            list(range(1, expected_steps + 1))
            if expected_steps is not None
            else None
        )
        if (
            expected_curve_steps is None
            or curve_steps != expected_curve_steps
        ):
            _issue(
                integrity_issues,
                "loss_curve_step_coverage_mismatch",
                expected_count=expected_steps,
                observed_count=len(curve_steps),
                first_step=curve_steps[0] if curve_steps else None,
                last_step=curve_steps[-1] if curve_steps else None,
            )
        if curve_losses and loss is not None:
            sample_count = min(100, len(curve_losses))
            curve_statistics = {
                "recorded_steps": len(curve_losses),
                "first_step": curve_steps[0],
                "last_step": curve_steps[-1],
                "mean": statistics.fmean(curve_losses),
                "median": statistics.median(curve_losses),
                "first_100_mean": statistics.fmean(
                    curve_losses[:sample_count]
                ),
                "last_100_mean": statistics.fmean(
                    curve_losses[-sample_count:]
                ),
            }
            first_mean = curve_statistics["first_100_mean"]
            curve_statistics["last_over_first_100_mean"] = (
                curve_statistics["last_100_mean"] / first_mean
                if first_mean != 0
                else None
            )
            for field, expected in curve_statistics.items():
                if not _values_equal(loss.get(field), expected):
                    _issue(
                        integrity_issues,
                        "loss_summary_curve_mismatch",
                        field=field,
                        expected=expected,
                        observed=loss.get(field),
                    )
    loss_png_path = (
        run_dir / "artifacts" / "wan22" / "loss_analysis"
        / "loss_curve.png"
    )
    if loss_png_path.is_file():
        with loss_png_path.open("rb") as handle:
            signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            _issue(
                integrity_issues,
                "loss_curve_png_invalid",
            )
    loss_export = _load_optional_json(run_dir / "training/loss_export.json")
    if loss_export is not None and (
        loss_export.get("status") != "complete"
        or loss_export.get("return_code") != 0
    ):
        _issue(
            integrity_issues,
            "loss_export_failed",
            status=loss_export.get("status"),
            return_code=loss_export.get("return_code"),
        )
    loss_passed = (
        loss is not None
        and evidence["loss_export"]["available"]
        and evidence["loss_curve_csv"]["available"]
        and evidence["loss_curve_png"]["available"]
        and len(integrity_issues) == loss_start
    )

    gradient_start = len(integrity_issues)
    gradient_path = (
        run_dir / "artifacts/wan22/checkpoints/gradient_audit.json"
    )
    gradient = _load_optional_json(gradient_path)
    if gradient is not None:
        sample_count = gradient.get("sample_count")
        positive_count = gradient.get(
            "positive_quantity_gradient_count"
        )
        if expected_steps is not None and sample_count != expected_steps:
            _issue(
                integrity_issues,
                "gradient_sample_count_mismatch",
                expected=expected_steps,
                observed=sample_count,
            )
        if positive_count != sample_count or not (
            isinstance(sample_count, int)
            and not isinstance(sample_count, bool)
            and sample_count > 0
        ):
            _issue(
                integrity_issues,
                "quantity_gradient_not_positive_every_step",
                sample_count=sample_count,
                positive_count=positive_count,
            )
        minimum = gradient.get("min_positive_quantity_gradient_l2")
        maximum = gradient.get("max_quantity_gradient_l2")
        if (
            not _finite_number(minimum)
            or float(minimum) <= 0
            or not _finite_number(maximum)
            or float(maximum) < float(minimum)
        ):
            _issue(
                integrity_issues,
                "quantity_gradient_range_invalid",
                minimum=minimum,
                maximum=maximum,
            )
        if gradient.get("text_encoder_gradient_tensor_count_max") != 0:
            _issue(
                integrity_issues,
                "frozen_text_encoder_received_gradients",
                observed=gradient.get(
                    "text_encoder_gradient_tensor_count_max"
                ),
            )
        samples = gradient.get("samples")
        if not isinstance(samples, list) or len(samples) != sample_count:
            _issue(
                integrity_issues,
                "gradient_samples_incomplete",
                declared=sample_count,
                observed=(
                    len(samples) if isinstance(samples, list) else None
                ),
            )
        elif any(
            sample.get("step") != index
            or not _finite_number(sample.get("quantity_gradient_l2"))
            or float(sample["quantity_gradient_l2"]) <= 0
            or sample.get("quantity_gradient_tensor_count")
            != EXPECTED_QUANTITY_ENCODER_TENSOR_COUNT
            or sample.get("text_encoder_gradient_tensor_count") != 0
            for index, sample in enumerate(samples, start=1)
            if isinstance(sample, dict)
        ) or any(not isinstance(sample, dict) for sample in samples):
            _issue(
                integrity_issues,
                "gradient_sample_acceptance_failed",
                expected_quantity_gradient_tensors=(
                    EXPECTED_QUANTITY_ENCODER_TENSOR_COUNT
                ),
            )
    gradient_passed = (
        gradient is not None
        and len(integrity_issues) == gradient_start
    )

    token_start = len(integrity_issues)
    token_path = (
        run_dir / "artifacts/wan22/training_quantity_token_audit.jsonl"
    )
    token_records = _load_optional_jsonl(token_path)
    expected_train_ids = plan.get("train_case_ids")
    if not isinstance(expected_train_ids, list):
        _issue(
            integrity_issues,
            "quantity_token_audit_plan_train_ids_invalid",
            observed=expected_train_ids,
        )
        expected_train_ids = []
    if token_records is not None:
        observed_ids = [
            record.get("case_id")
            for record in token_records
            if isinstance(record.get("case_id"), str)
        ]
        if (
            len(observed_ids) != len(token_records)
            or len(observed_ids) != len(set(observed_ids))
            or set(observed_ids) != set(expected_train_ids)
        ):
            _issue(
                integrity_issues,
                "quantity_token_audit_case_coverage_mismatch",
                expected_count=len(set(expected_train_ids)),
                observed_count=len(observed_ids),
                unique_observed_count=len(set(observed_ids)),
                missing_case_ids=sorted(
                    set(expected_train_ids) - set(observed_ids)
                ),
                extra_case_ids=sorted(
                    set(observed_ids) - set(expected_train_ids)
                ),
            )
    if task_instance is None:
        _issue(
            integrity_issues,
            "quantity_token_audit_sealed_instance_unavailable",
        )
    elif token_records is not None:
        source_by_case = {
            case["case_id"]: case
            for case in task_instance["source"]["cases"]
        }
        train_adaptations = {
            adaptation["case_id"]: adaptation
            for adaptation in task_instance["adaptations"]
            if adaptation["role"] == "train"
        }
        token_by_case = {
            record.get("case_id"): record
            for record in token_records
            if isinstance(record.get("case_id"), str)
        }
        for case_id in expected_train_ids:
            adaptation = train_adaptations.get(case_id)
            record = token_by_case.get(case_id)
            source_case = source_by_case.get(case_id, {})
            if adaptation is None or record is None:
                continue
            _validate_quantity_audit_record(
                record=record,
                expected_native_inputs=adaptation["native_inputs"],
                expected_case_id=case_id,
                expected_scene_id=str(
                    adaptation.get("scene_id")
                    or source_case.get("scene_id")
                    or ""
                ),
                expected_job_id=None,
                integrity_issues=integrity_issues,
                artifact=(
                    "artifacts/wan22/"
                    "training_quantity_token_audit.jsonl"
                    f"#{case_id}"
                ),
            )

        inference_jobs = {
            job["job_id"]: job
            for job in task_instance["inference"]["jobs"]
        }
        observed_inference_ids = {
            path.stem for path in inference_audit_files
        }
        expected_inference_ids = set(completed_prediction_ids)
        if observed_inference_ids != expected_inference_ids:
            _issue(
                integrity_issues,
                "inference_quantity_token_audit_coverage_mismatch",
                expected_count=len(expected_inference_ids),
                observed_count=len(observed_inference_ids),
                missing_job_ids=sorted(
                    expected_inference_ids - observed_inference_ids
                ),
                extra_job_ids=sorted(
                    observed_inference_ids - expected_inference_ids
                ),
            )
        for job_id in sorted(
            expected_inference_ids & observed_inference_ids
        ):
            job = inference_jobs.get(job_id)
            if job is None:
                _issue(
                    integrity_issues,
                    "inference_quantity_token_audit_unknown_job",
                    job_id=job_id,
                )
                continue
            audit_path = inference_audit_root / f"{job_id}.json"
            record = _load_json(audit_path)
            _validate_quantity_audit_record(
                record=record,
                expected_native_inputs=job["native_inputs"],
                expected_case_id=job["case_id"],
                expected_scene_id=job["scene_id"],
                expected_job_id=job_id,
                integrity_issues=integrity_issues,
                artifact=_relative_to_run(audit_path, run_dir),
            )
    token_passed = (
        token_records is not None
        and task_instance is not None
        and len(integrity_issues) == token_start
    )
    acceptance = {
        "policy": {
            "loss_steps": (
                "recorded loss steps must equal the independently frozen "
                "training sampling plan"
            ),
            "loss_observation": (
                "TensorBoard loss is the rank-0 local batch loss at each "
                "optimizer step, not an all-rank reduced loss"
            ),
            "quantity_gradients": (
                "all optimizer steps must have positive gradients for all "
                f"{EXPECTED_QUANTITY_ENCODER_TENSOR_COUNT} quantity-encoder "
                "tensors"
            ),
            "text_encoder": "UMT5 gradient tensor count must remain zero",
            "token_audit": (
                "every training Case and complete inference job must have "
                "one sealed-registry quantity audit with matching SI values, "
                "dimensions, types, sentinels and unique token spans"
            ),
        },
        "expected_optimizer_steps": expected_steps,
        "sampling_plan_passed": sampling_passed,
        "loss_passed": loss_passed,
        "gradient_passed": gradient_passed,
        "quantity_token_audit_passed": token_passed,
    }
    acceptance["passed"] = all(
        acceptance[field]
        for field in (
            "sampling_plan_passed",
            "loss_passed",
            "gradient_passed",
            "quantity_token_audit_passed",
        )
    )
    return evidence, acceptance


def _validate_prediction_contract(
    *,
    run_dir: Path,
    jobs: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    baseline_id: Any,
    integrity_issues: list[dict[str, Any]],
) -> None:
    if not isinstance(baseline_id, str) or not baseline_id:
        _issue(
            integrity_issues,
            "prediction_baseline_identity_invalid",
            observed=baseline_id,
        )
        return
    selected_jobs = [
        job for job in jobs if job["job_id"] in predictions
    ]
    selected_predictions = [
        predictions[job["job_id"]] for job in selected_jobs
    ]
    for prediction in selected_predictions:
        if prediction.get("status") not in PREDICTION_STATUSES:
            _issue(
                integrity_issues,
                "prediction_status_invalid",
                job_id=prediction.get("job_id"),
                observed=prediction.get("status"),
            )
    if not selected_predictions:
        return
    artifact_manifest_path = (
        run_dir / "artifacts" / "prediction_artifacts.json"
    )
    artifact_manifest = _load_optional_json(artifact_manifest_path)
    if artifact_manifest is None:
        _issue(
            integrity_issues,
            "prediction_artifact_manifest_missing",
            path="artifacts/prediction_artifacts.json",
        )
    try:
        validate_prediction_records(
            selected_predictions,
            jobs=selected_jobs,
            baseline_id=baseline_id,
            run_dir=run_dir,
            expected_artifact_manifest=artifact_manifest,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        _issue(
            integrity_issues,
            "prediction_contract_validation_failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _validate_case_evaluation_contract(
    *,
    jobs: Sequence[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    integrity_issues: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    clean: dict[str, dict[str, Any]] = {}
    valid = True
    by_job = {job["job_id"]: job for job in jobs}
    for job_id, record in evaluations.items():
        job = by_job[job_id]
        record_valid = True
        try:
            _validate_record_identity(
                record,
                job,
                label="case evaluation",
            )
        except ValueError as exc:
            _issue(
                integrity_issues,
                "case_evaluation_identity_mismatch",
                job_id=job_id,
                error=str(exc),
            )
            record_valid = False
        status = record.get("status")
        score = record.get("score")
        if status not in CASE_STATUSES:
            _issue(
                integrity_issues,
                "case_evaluation_status_invalid",
                job_id=job_id,
                observed=status,
            )
            record_valid = False
        if score is not None and (
            not _finite_number(score)
            or not 0.0 <= float(score) <= 1.0
        ):
            _issue(
                integrity_issues,
                "case_evaluation_score_invalid",
                job_id=job_id,
                status=status,
                observed=score,
            )
            record_valid = False
        required_objects = {
            field: record.get(field)
            for field in (
                "evaluator",
                "metrics",
                "quality",
                "artifacts",
                "provenance",
            )
        }
        invalid_objects = sorted(
            field
            for field, value in required_objects.items()
            if not isinstance(value, dict)
        )
        if invalid_objects:
            _issue(
                integrity_issues,
                "case_evaluation_object_fields_invalid",
                job_id=job_id,
                fields=invalid_objects,
            )
            record_valid = False
        if record_valid:
            try:
                CaseEvaluationResult(
                    job_id=record["job_id"],
                    case_id=record["case_id"],
                    scene_id=record["scene_id"],
                    evaluator=record["evaluator"],
                    status=record["status"],
                    score=record.get("score"),
                    reason_code=record.get("reason_code"),
                    reason=record.get("reason"),
                    metrics=record["metrics"],
                    quality=record["quality"],
                    artifacts=record["artifacts"],
                    provenance=record["provenance"],
                )
            except (TypeError, ValueError) as exc:
                _issue(
                    integrity_issues,
                    "case_evaluation_contract_validation_failed",
                    job_id=job_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                record_valid = False
        if record_valid:
            clean[job_id] = record
        else:
            valid = False
            clean[job_id] = {
                **record,
                "status": "invalid",
                "score": None,
            }
    return clean, valid


def _job_records(
    jobs: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job["job_id"]
        prediction = predictions.get(job_id)
        evaluation = evaluations.get(job_id)
        score = evaluation.get("score") if evaluation is not None else None
        if not _finite_number(score):
            score = None
        records.append({
            "job_id": job_id,
            "case_id": job["case_id"],
            "scene_id": job["scene_id"],
            "partition": job["evaluation_partition"],
            "seed": job["seed"],
            "prediction_status": (
                prediction.get("status")
                if prediction is not None
                else None
            ),
            "evaluation_status": (
                evaluation.get("status")
                if evaluation is not None
                else None
            ),
            "score": score,
            "reason_code": (
                evaluation.get("reason_code")
                if evaluation is not None
                else None
            ),
            "reason": (
                evaluation.get("reason")
                if evaluation is not None
                else None
            ),
            "prediction_error": (
                prediction.get("error")
                if prediction is not None
                else None
            ),
            "evaluator": (
                copy.deepcopy(evaluation.get("evaluator"))
                if evaluation is not None
                else None
            ),
            "metrics": (
                copy.deepcopy(evaluation.get("metrics"))
                if evaluation is not None
                else None
            ),
            "quality": (
                copy.deepcopy(evaluation.get("quality"))
                if evaluation is not None
                else None
            ),
            "artifacts": (
                copy.deepcopy(evaluation.get("artifacts"))
                if evaluation is not None
                else None
            ),
            "evaluation_provenance": (
                copy.deepcopy(evaluation.get("provenance"))
                if evaluation is not None
                else None
            ),
            "case_result_path": (
                f"evaluation/cases/{job_id}/result.json"
            ),
        })
    return records


def _official_task_result(
    *,
    run: dict[str, Any],
    plan: dict[str, Any],
    task: dict[str, Any],
    component_fingerprints: dict[str, Any],
    task_result: dict[str, Any] | None,
    evaluations: dict[str, dict[str, Any]],
    evaluations_valid: bool,
    integrity_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if task_result is None:
        _issue(
            integrity_issues,
            "official_task_result_missing",
            path="evaluation/task_result.json",
        )
        return {
            "schema_version": None,
            "task_id": plan.get("task_id"),
            "task_family": plan.get("family"),
            "protocol": None,
            "status": None,
            "expected_jobs": None,
            "evaluated_jobs": None,
            "coverage": None,
            "score": None,
            "observed_mean_score": None,
            "aggregation_policy": None,
            "status_counts": None,
            "integrity_issues": None,
            "by_scene": {},
            "breakdown": {},
            "aggregation_verified": False,
            "protocol_verified": False,
            "result_complete": False,
            "recomputed_aggregation": None,
        }

    if task_result.get("schema_version") != "1.0":
        _issue(
            integrity_issues,
            "official_task_result_schema_invalid",
            observed=task_result.get("schema_version"),
        )
    official_issues = task_result.get("integrity_issues")
    official_integrity_valid = isinstance(official_issues, list)
    if not official_integrity_valid:
        _issue(
            integrity_issues,
            "official_integrity_issues_invalid",
            observed_type=type(official_issues).__name__,
        )
        official_issues = []
    else:
        for index, item in enumerate(official_issues):
            if not isinstance(item, dict):
                _issue(
                    integrity_issues,
                    "official_integrity_issue_entry_invalid",
                    index=index,
                    observed_type=type(item).__name__,
                )
            _issue(
                integrity_issues,
                "official_evaluation_integrity_issue",
                issue=copy.deepcopy(item),
            )

    for field, expected in (
        ("task_id", plan.get("task_id")),
        ("task_family", plan.get("family")),
    ):
        if task_result.get(field) != expected:
            _issue(
                integrity_issues,
                "official_task_identity_mismatch",
                field=field,
                expected=expected,
                observed=task_result.get(field),
            )

    protocol = task_result.get("protocol")
    expected_protocol_id = task.get("evaluation", {}).get("protocol")
    expected_protocol_fingerprint = component_fingerprints.get(
        "evaluation_protocol"
    )
    protocol_verified = True
    if not isinstance(protocol, dict):
        _issue(
            integrity_issues,
            "official_evaluation_protocol_invalid",
            observed_type=type(protocol).__name__,
        )
        protocol = {}
        protocol_verified = False
    protocol_expectations = {
        "id": expected_protocol_id,
        "fingerprint": expected_protocol_fingerprint,
    }
    for field, expected in protocol_expectations.items():
        observed = protocol.get(field)
        if (
            (field == "fingerprint" and not _valid_sha256(expected))
            or observed != expected
        ):
            _issue(
                integrity_issues,
                "official_evaluation_protocol_mismatch",
                field=field,
                expected=expected,
                observed=observed,
            )
            protocol_verified = False
    if not isinstance(protocol.get("path"), str) or not protocol.get("path"):
        _issue(
            integrity_issues,
            "official_evaluation_protocol_path_invalid",
            observed=protocol.get("path"),
        )
        protocol_verified = False

    planned_ids = {job["job_id"] for job in plan["jobs"]}
    aggregation_verified = False
    recomputed: dict[str, Any] | None = None
    if evaluations_valid and set(evaluations) == planned_ids:
        ordered_results = [
            evaluations[job["job_id"]]
            for job in plan["jobs"]
        ]
        recomputed = aggregate_task_results(
            plan=plan,
            case_results=ordered_results,
        )
        aggregate_fields = (
            "status",
            "expected_jobs",
            "evaluated_jobs",
            "coverage",
            "status_counts",
            "score",
            "observed_mean_score",
            "aggregation_policy",
            "by_scene",
            "breakdown",
        )
        aggregation_verified = True
        for field in aggregate_fields:
            expected = recomputed.get(field)
            observed = task_result.get(field)
            if not _values_equal(observed, expected):
                _issue(
                    integrity_issues,
                    "official_aggregation_mismatch",
                    field=field,
                    expected=expected,
                    observed=observed,
                )
                aggregation_verified = False
    else:
        _issue(
            integrity_issues,
            "official_aggregation_recompute_unavailable",
            evaluation_contract_valid=evaluations_valid,
            missing_job_ids=sorted(planned_ids - set(evaluations)),
            extra_job_ids=sorted(set(evaluations) - planned_ids),
        )

    status = task_result.get("status")
    score = task_result.get("score")
    status_score_valid = (
        status == "complete"
        and _same_number(task_result.get("coverage"), 1.0)
        and _finite_number(score)
        and 0.0 <= float(score) <= 1.0
    ) or (
        status == "partial"
        and score is None
        and _finite_number(task_result.get("coverage"))
        and float(task_result["coverage"]) < 1.0
    )
    if not status_score_valid:
        _issue(
            integrity_issues,
            "official_task_status_score_inconsistent",
            status=status,
            coverage=task_result.get("coverage"),
            score=score,
        )
    for run_field, task_field in (
        ("evaluation_status", "status"),
        ("evaluation_coverage", "coverage"),
        ("evaluation_score", "score"),
    ):
        run_value = run.get(run_field)
        task_value = task_result.get(task_field)
        matches = (
            _same_number(run_value, task_value)
            if task_field in {"coverage", "score"}
            else run_value == task_value
        )
        if not matches:
            _issue(
                integrity_issues,
                "run_evaluation_projection_mismatch",
                run_field=run_field,
                run_value=run_value,
                task_result_field=task_field,
                task_result_value=task_value,
            )

    by_scene = _normalized_object_map(
        task_result.get("by_scene"),
        field="by_scene",
        integrity_issues=integrity_issues,
    )
    breakdown = _normalized_object_map(
        task_result.get("breakdown"),
        field="breakdown",
        integrity_issues=integrity_issues,
    )
    result_complete = (
        run.get("status") == "complete"
        and status == "complete"
        and _same_number(task_result.get("coverage"), 1.0)
        and _finite_number(score)
        and 0.0 <= float(score) <= 1.0
        and official_integrity_valid
        and official_issues == []
        and aggregation_verified
        and protocol_verified
    )
    return {
        "schema_version": task_result.get("schema_version"),
        "task_id": task_result.get("task_id"),
        "task_family": task_result.get("task_family"),
        "protocol": copy.deepcopy(protocol),
        "status": status,
        "expected_jobs": task_result.get("expected_jobs"),
        "evaluated_jobs": task_result.get("evaluated_jobs"),
        "coverage": task_result.get("coverage"),
        "score": score,
        "observed_mean_score": task_result.get("observed_mean_score"),
        "aggregation_policy": task_result.get("aggregation_policy"),
        "status_counts": copy.deepcopy(task_result.get("status_counts")),
        "integrity_issues": copy.deepcopy(official_issues),
        "by_scene": by_scene,
        "breakdown": breakdown,
        "aggregation_verified": aggregation_verified,
        "protocol_verified": protocol_verified,
        "result_complete": result_complete,
        "recomputed_aggregation": copy.deepcopy(recomputed),
    }


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Read and summarize a terminal AtomicRun without modifying it."""
    directory = Path(run_dir).resolve()
    run_record = _load_optional_json(directory / "run.json")
    plan = _load_json(directory / "plan.json")
    task = _load_json(directory / "frozen" / "task.json")
    component_fingerprints = _load_json(
        directory / "component_fingerprints.json"
    )
    failed_state_projection = False
    failed_state: dict[str, Any] | None = None
    if run_record is None:
        failed_state = _load_optional_json(directory / "state.json")
        if failed_state is None or failed_state.get("stage") != "failed":
            raise FileNotFoundError(
                "run.json is missing and state.json does not identify a "
                f"failed AtomicRun: {directory}"
            )
        failed_state_projection = True
        frozen_baseline = _load_optional_json(
            directory / "frozen" / "baseline.json"
        ) or {}
        task_instance = _load_optional_json(
            directory / "task_instance" / "manifest.json"
        ) or {}
        instance_identity = task_instance.get("identity")
        if not isinstance(instance_identity, dict):
            instance_identity = {}
        baseline_identity = instance_identity.get("baseline")
        if not isinstance(baseline_identity, dict):
            baseline_identity = {}
        run = {
            "schema_version": None,
            "run_id": directory.name,
            "status": "failed",
            "dataset_id": plan.get("dataset_id"),
            "dataset_digest": component_fingerprints.get("dataset"),
            "task_id": plan.get("task_id"),
            "task_family": plan.get("family"),
            "baseline_id": (
                baseline_identity.get("baseline_id")
                or frozen_baseline.get("baseline_id")
            ),
            "baseline_version": (
                baseline_identity.get("baseline_version")
                or frozen_baseline.get("baseline_version")
            ),
            "baseline_digest": component_fingerprints.get("baseline"),
            "baseline_deployment_digest": component_fingerprints.get(
                "baseline_deployment"
            ),
            "task_instance_id": task_instance.get("instance_id"),
            "task_instance_digest": component_fingerprints.get(
                "task_instance"
            ),
            "evaluation_status": None,
            "evaluation_coverage": None,
            "evaluation_score": None,
            "created_at": failed_state.get("updated_at"),
        }
    else:
        run = run_record
    if run.get("status") not in REPORTABLE_RUN_STATUSES:
        raise ValueError(
            "quantity run summary requires a reportable run.json status in "
            f"{sorted(REPORTABLE_RUN_STATUSES)}; "
            f"got {run.get('status')!r}"
        )
    task_result = _load_optional_json(
        directory / "evaluation" / "task_result.json"
    )
    integrity_issues: list[dict[str, Any]] = []
    if failed_state_projection:
        _issue(
            integrity_issues,
            "run_json_missing_failed_state_projection",
            path="run.json",
            state_path="state.json",
        )
    jobs = _validate_plan_jobs(plan)
    sealed_instance, sealed_identity = _validate_sealed_run_identity(
        run_dir=directory,
        run=run,
        plan=plan,
        task=task,
        component_fingerprints=component_fingerprints,
        integrity_issues=integrity_issues,
    )
    state_document = _load_optional_json(directory / "state.json")
    if state_document is None:
        _issue(
            integrity_issues,
            "atomic_run_state_missing",
            path="state.json",
        )
    elif (
        not failed_state_projection
        and state_document.get("stage") != run.get("status")
    ):
        _issue(
            integrity_issues,
            "atomic_run_state_status_mismatch",
            run_status=run.get("status"),
            state_stage=state_document.get("stage"),
        )
    prediction_values = _load_optional_jsonl(
        directory / "predictions.jsonl"
    )
    if prediction_values is None:
        _issue(
            integrity_issues,
            "atomic_run_artifact_missing",
            path="predictions.jsonl",
        )
        prediction_values = []
    predictions = _unique_by_job_id(
        prediction_values,
        label="predictions",
    )
    evaluation_values = _load_optional_jsonl(
        directory / "evaluation" / "case_results.jsonl"
    )
    if evaluation_values is None:
        _issue(
            integrity_issues,
            "atomic_run_artifact_missing",
            path="evaluation/case_results.jsonl",
        )
        evaluation_values = []
    evaluations = _unique_by_job_id(
        evaluation_values,
        label="case evaluations",
    )
    planned_ids = {job["job_id"] for job in jobs}
    unknown_predictions = sorted(set(predictions) - planned_ids)
    unknown_evaluations = sorted(set(evaluations) - planned_ids)
    if unknown_predictions:
        _issue(
            integrity_issues,
            "predictions_contain_unknown_jobs",
            job_ids=unknown_predictions,
        )
    if unknown_evaluations:
        _issue(
            integrity_issues,
            "case_evaluations_contain_unknown_jobs",
            job_ids=unknown_evaluations,
        )
    predictions = {
        job_id: record
        for job_id, record in predictions.items()
        if job_id in planned_ids
    }
    evaluations = {
        job_id: record
        for job_id, record in evaluations.items()
        if job_id in planned_ids
    }
    missing_prediction_records = sorted(planned_ids - set(predictions))
    missing_evaluation_records = sorted(planned_ids - set(evaluations))
    if missing_prediction_records:
        _issue(
            integrity_issues,
            "prediction_record_coverage_incomplete",
            missing_job_ids=missing_prediction_records,
        )
    if missing_evaluation_records:
        _issue(
            integrity_issues,
            "case_evaluation_record_coverage_incomplete",
            missing_job_ids=missing_evaluation_records,
        )
    by_job = {job["job_id"]: job for job in jobs}
    for job_id, prediction in predictions.items():
        try:
            _validate_record_identity(
                prediction,
                by_job[job_id],
                label="prediction",
            )
        except ValueError as exc:
            _issue(
                integrity_issues,
                "prediction_identity_mismatch",
                job_id=job_id,
                error=str(exc),
            )
        if prediction.get("baseline_id") != run.get("baseline_id"):
            _issue(
                integrity_issues,
                "prediction_baseline_identity_mismatch",
                job_id=job_id,
                expected=run.get("baseline_id"),
                observed=prediction.get("baseline_id"),
            )
    _validate_prediction_contract(
        run_dir=directory,
        jobs=jobs,
        predictions=predictions,
        baseline_id=run.get("baseline_id"),
        integrity_issues=integrity_issues,
    )
    evaluations, evaluations_valid = (
        _validate_case_evaluation_contract(
            jobs=jobs,
            evaluations=evaluations,
            integrity_issues=integrity_issues,
        )
    )
    evaluations_valid = (
        evaluations_valid
        and not missing_evaluation_records
        and not unknown_evaluations
    )
    incomplete_prediction_job_ids = sorted(
        job["job_id"]
        for job in jobs
        if (
            predictions.get(job["job_id"]) is None
            or predictions[job["job_id"]].get("status") != "complete"
        )
    )
    all_planned_predictions_complete = (
        not incomplete_prediction_job_ids
    )
    if (
        run.get("status") == "complete"
        and not all_planned_predictions_complete
    ):
        _issue(
            integrity_issues,
            "complete_run_has_incomplete_predictions",
            job_ids=incomplete_prediction_job_ids,
        )
    evaluated_without_complete_prediction = sorted(
        job_id
        for job_id, evaluation in evaluations.items()
        if (
            evaluation.get("status") == "evaluated"
            and (
                predictions.get(job_id) is None
                or predictions[job_id].get("status") != "complete"
            )
        )
    )
    if evaluated_without_complete_prediction:
        _issue(
            integrity_issues,
            "evaluated_case_without_complete_prediction",
            job_ids=evaluated_without_complete_prediction,
        )
        evaluations_valid = False

    task_digest = component_fingerprints.get("task")
    computed_task_digest = _canonical_sha256(task)
    if task_digest != computed_task_digest:
        integrity_issues.append({
            "code": "task_digest_mismatch",
            "declared_sha256": task_digest,
            "actual_sha256": computed_task_digest,
        })
    for field, fingerprint_name in (
        ("dataset_digest", "dataset"),
        ("baseline_digest", "baseline"),
        ("baseline_deployment_digest", "baseline_deployment"),
        ("task_instance_digest", "task_instance"),
    ):
        declared = run.get(field)
        frozen = component_fingerprints.get(fingerprint_name)
        if declared != frozen:
            integrity_issues.append({
                "code": f"{field}_mismatch",
                "run_value": declared,
                "component_fingerprint": frozen,
            })

    checkpoint = _checkpoint_provenance(directory, integrity_issues)
    scene_ids = plan.get("scene_ids")
    if not isinstance(scene_ids, list) or any(
        not isinstance(scene_id, str) or not scene_id
        for scene_id in scene_ids
    ):
        raise ValueError("AtomicRun plan.scene_ids must be a string array")
    partitions = _partition_names(task, plan)
    scene_partition: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        for partition in partitions:
            selected = [
                job
                for job in jobs
                if job["scene_id"] == scene_id
                and job["evaluation_partition"] == partition
            ]
            scene_partition.append({
                "scene_id": scene_id,
                "partition": partition,
                **_summarize_jobs(
                    selected,
                    predictions,
                    evaluations,
                    label=f"scene_partition:{scene_id}/{partition}",
                ),
            })

    by_scene: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        selected = [job for job in jobs if job["scene_id"] == scene_id]
        by_scene.append({
            "scene_id": scene_id,
            **_summarize_jobs(
                selected,
                predictions,
                evaluations,
                label=f"scene:{scene_id}",
            ),
        })
    overall = _summarize_jobs(
        jobs,
        predictions,
        evaluations,
        label="overall",
    )
    training_evidence, training_acceptance = _training_evidence(
        directory,
        plan,
        sealed_instance,
        predictions,
        integrity_issues,
    )
    training_acceptance["checkpoint_passed"] = checkpoint[
        "acceptance"
    ]["passed"]
    training_acceptance["passed"] = (
        training_acceptance["passed"]
        and training_acceptance["checkpoint_passed"]
    )
    official = _official_task_result(
        run=run,
        plan=plan,
        task=task,
        component_fingerprints=component_fingerprints,
        task_result=task_result,
        evaluations=evaluations,
        evaluations_valid=evaluations_valid,
        integrity_issues=integrity_issues,
    )
    job_records = _job_records(jobs, predictions, evaluations)
    benchmark_score_publishable = (
        official["result_complete"]
        and all_planned_predictions_complete
        and training_acceptance["passed"]
        and not integrity_issues
    )
    sources = {
        "run": _artifact_reference(directory, "run.json"),
        "state": _artifact_reference(directory, "state.json"),
        "plan": _artifact_reference(directory, "plan.json"),
        "task": _artifact_reference(directory, "frozen/task.json"),
        "predictions": _artifact_reference(directory, "predictions.jsonl"),
        "case_results": _artifact_reference(
            directory,
            "evaluation/case_results.jsonl",
        ),
        "task_result": _artifact_reference(
            directory,
            "evaluation/task_result.json",
        ),
        "task_instance": _artifact_reference(
            directory,
            "task_instance/manifest.json",
        ),
        "component_fingerprints": _artifact_reference(
            directory,
            "component_fingerprints.json",
        ),
        "prediction_artifacts": _artifact_reference(
            directory,
            "artifacts/prediction_artifacts.json",
        ),
    }
    return {
        "schema_version": "2.0",
        "summary_type": "quantity_atomic_run_evaluation",
        "source_run_dir": str(directory),
        "reporting_status": {
            "run_status": run.get("status"),
            "partial_report": (
                run.get("status") != "complete"
                or official.get("status") != "complete"
            ),
            "integrity_passed": not integrity_issues,
            "all_planned_predictions_complete": (
                all_planned_predictions_complete
            ),
            "benchmark_score_publishable": (
                benchmark_score_publishable
            ),
        },
        "provenance": {
            "run": {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "record_available": not failed_state_projection,
                "digest": _canonical_sha256(run),
                "digest_semantics": (
                    "canonical_run_json_sha256"
                    if not failed_state_projection
                    else (
                        "canonical_sha256_of_explicit_failed_state_identity_"
                        "projection; run.json was not available"
                    )
                ),
                "failed_state": copy.deepcopy(failed_state),
            },
            "dataset": {
                "dataset_id": run.get("dataset_id"),
                "digest": run.get("dataset_digest"),
            },
            "task": {
                "task_id": run.get("task_id"),
                "family": run.get("task_family"),
                "digest": task_digest,
                "computed_digest": computed_task_digest,
            },
            "baseline": {
                "baseline_id": run.get("baseline_id"),
                "baseline_version": run.get("baseline_version"),
                "digest": run.get("baseline_digest"),
                "deployment_digest": run.get(
                    "baseline_deployment_digest"
                ),
            },
            "task_instance": sealed_identity,
            "evaluation_protocol": {
                "frozen_fingerprint": component_fingerprints.get(
                    "evaluation_protocol"
                ),
                "task_result": copy.deepcopy(official.get("protocol")),
                "verified": official.get("protocol_verified"),
            },
            "checkpoint": checkpoint,
        },
        "bootstrap": {
            "statistic": "arithmetic_mean",
            "confidence": 0.95,
            "method": "percentile_case_resampling",
            "case_value": "mean_across_evaluated_inference_seeds",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "group_seed_derivation": (
                "uint64_be(sha256('<seed>:<group-label>')[:8])"
            ),
        },
        "count_semantics": {
            "expected": "jobs frozen in plan.json",
            "completed": "planned jobs with prediction status=complete",
            "evaluated": (
                "planned jobs with evaluation status=evaluated and a "
                "finite score"
            ),
            "failed": (
                "union of jobs with prediction status error/failed or "
                "evaluation status error/failed"
            ),
            "unavailable": (
                "evaluation status unavailable/unsupported, excluding jobs "
                "already counted as failed"
            ),
            "missing_values": (
                "missing records and absent scores remain explicit/null and "
                "are never imputed as zero"
            ),
            "std": "population standard deviation over available case means",
            "scene_partition": (
                "descriptive case-macro statistics after averaging available "
                "inference seeds within each Case; job_micro_mean separately "
                "reports the mean over evaluated jobs"
            ),
            "by_scene": (
                "descriptive case-macro statistics over available Cases in a "
                "scene; this is not the official macro mean over required "
                "partitions; job_micro_mean is also reported"
            ),
            "overall": (
                "descriptive case-macro summary plus an explicit evaluated-job "
                "micro mean; official Benchmark macro score remains in "
                "official_task_result"
            ),
        },
        "scene_partition": scene_partition,
        "by_scene": by_scene,
        "overall": overall,
        "jobs": job_records,
        "problem_job_ids": {
            "failed": overall["failed_job_ids"],
            "unavailable": overall["unavailable_job_ids"],
            "missing_prediction": overall[
                "missing_prediction_job_ids"
            ],
            "missing_evaluation": overall[
                "missing_evaluation_job_ids"
            ],
            "other_terminal": overall["other_terminal_job_ids"],
        },
        "unplanned_records": {
            "prediction_job_ids": unknown_predictions,
            "evaluation_job_ids": unknown_evaluations,
        },
        "official_task_result": official,
        "training_evidence": training_evidence,
        "training_acceptance": training_acceptance,
        "sources": sources,
        "integrity_issues": integrity_issues,
    }


def _format_number(value: Any) -> str:
    if not _finite_number(value):
        return "N/A"
    return f"{float(value):.6f}"


def _format_ci(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A"
    return (
        f"[{_format_number(value.get('low'))}, "
        f"{_format_number(value.get('high'))}]"
    )


def _format_boolean(value: Any) -> str:
    if value is None:
        return "N/A"
    return "yes" if value is True else "no"


def _markdown_cell(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _markdown_metric_row(
    label_cells: Sequence[str],
    value: dict[str, Any],
) -> str:
    cells = [
        *label_cells,
        str(value["expected"]),
        str(value["completed"]),
        str(value["evaluated"]),
        str(value["failed"]),
        str(value["unavailable"]),
        str(value["missing_prediction"]),
        str(value["missing_evaluation"]),
        _format_number(value["coverage"]),
        _format_number(value["job_micro_mean"]),
        _format_number(value["mean"]),
        _format_number(value["median"]),
        _format_number(value["std"]),
        _format_ci(value["ci95"]),
    ]
    return "| " + " | ".join(cells) + " |\n"


def render_markdown(summary: dict[str, Any]) -> str:
    provenance = summary["provenance"]
    checkpoint = provenance["checkpoint"]
    official = summary["official_task_result"]
    lines = [
        "# WAN2.2 quantity-embedding AtomicRun summary\n\n",
        "## Identity\n\n",
        "| Object | ID | Status / digest |\n",
        "|---|---|---|\n",
        f"| Run | `{provenance['run']['run_id']}` | "
        f"`{provenance['run']['status']}` / "
        f"`{provenance['run']['digest']}` |\n",
        f"| Task | `{provenance['task']['task_id']}` | "
        f"`{provenance['task']['digest']}` |\n",
        f"| Baseline | `{provenance['baseline']['baseline_id']}` | "
        f"`{provenance['baseline']['digest']}` |\n",
        f"| Checkpoint | `{checkpoint.get('path') or 'N/A'}` | "
        f"`{checkpoint.get('digest') or 'N/A'}` |\n",
        "\n",
        f"`run.json` available: "
        f"`{_format_boolean(provenance['run']['record_available'])}`. "
        f"Run digest semantics: "
        f"`{provenance['run']['digest_semantics']}`.\n\n",
        "Scores below use only successfully evaluated cases. Missing, failed, "
        "and unavailable jobs are never converted to zero. `N/A` means no "
        "score was available for that cell. The 95% interval is a fixed-seed "
        f"percentile bootstrap ({summary['bootstrap']['iterations']} "
        f"resamples, base seed `{summary['bootstrap']['seed']}`). A partial "
        "run remains reportable, but its strict official score is not "
        "publishable.\n\n",
        "## Descriptive scene × partition statistics\n\n",
        "The case-macro column first averages available inference seeds within "
        "each Case and then weights Cases equally. Job micro weights every "
        "evaluated inference job equally. Neither replaces the official strict "
        "aggregation below.\n\n",
        "| Scene | Partition | Expected | Completed | Evaluated | Failed | "
        "Unavailable | Missing prediction | Missing evaluation | Coverage | "
        "Job micro mean | Case macro mean | Median | Std | "
        "95% bootstrap CI |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in summary["scene_partition"]:
        lines.append(
            _markdown_metric_row(
                [f"`{row['scene_id']}`", f"`{row['partition']}`"],
                row,
            )
        )
    lines.extend([
        "\n## Descriptive by-scene statistics\n\n",
        "These rows report both evaluated-job micro and Case-macro summaries "
        "within each scene. They are not the "
        "Benchmark's official macro mean over required partitions.\n\n",
        "| Scene | Expected | Completed | Evaluated | Failed | Unavailable | "
        "Missing prediction | Missing evaluation | Coverage | Job micro mean | "
        "Case macro mean | Median | Std | 95% bootstrap CI |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ])
    for row in summary["by_scene"]:
        lines.append(
            _markdown_metric_row([f"`{row['scene_id']}`"], row)
        )
    lines.extend([
        "\n## Descriptive overall statistics\n\n",
        "| Scope | Expected | Completed | Evaluated | Failed | Unavailable | "
        "Missing prediction | Missing evaluation | Coverage | Job micro mean | "
        "Case macro mean | Median | Std | 95% bootstrap CI |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        _markdown_metric_row(["All planned jobs"], summary["overall"]),
        "\nThe overall row exposes both descriptive weighting policies. It does "
        "not replace the Benchmark's strict macro score.\n\n",
        "## Official Task result\n\n",
        f"- Protocol ID: "
        f"`{_markdown_cell((official.get('protocol') or {}).get('id'))}`\n",
        f"- Protocol fingerprint: "
        f"`{_markdown_cell((official.get('protocol') or {}).get('fingerprint'))}`\n",
        f"- Protocol verified against frozen run: "
        f"`{_format_boolean(official.get('protocol_verified'))}`\n",
        f"- Aggregation independently recomputed: "
        f"`{_format_boolean(official.get('aggregation_verified'))}`\n",
        f"- Status: `{official.get('status') or 'N/A'}`\n",
        f"- Expected jobs: `"
        f"{official.get('expected_jobs') if official.get('expected_jobs') is not None else 'N/A'}`\n",
        f"- Evaluated jobs: `"
        f"{official.get('evaluated_jobs') if official.get('evaluated_jobs') is not None else 'N/A'}`\n",
        f"- Coverage: `{_format_number(official.get('coverage'))}`\n",
        f"- Strict Task score: `{_format_number(official.get('score'))}`\n",
        f"- Observed mean score: "
        f"`{_format_number(official.get('observed_mean_score'))}`\n",
        f"- Aggregation: `{official.get('aggregation_policy') or 'N/A'}`\n",
        f"- Benchmark score publishable: "
        f"`{_format_boolean(summary['reporting_status'].get('benchmark_score_publishable'))}`\n",
        f"- Official integrity issues: `"
        f"{json.dumps(official.get('integrity_issues'), sort_keys=True)}`\n\n",
        "### Official partition breakdown\n\n",
        "| Scene / partition | Expected | Evaluated | Coverage | Strict score | "
        "Observed mean | Status counts |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
    ])
    for label, value in sorted(official.get("breakdown", {}).items()):
        lines.append(
            f"| `{_markdown_cell(label)}` | "
            f"{value.get('expected_jobs', 'N/A')} | "
            f"{value.get('evaluated_jobs', 'N/A')} | "
            f"{_format_number(value.get('coverage'))} | "
            f"{_format_number(value.get('score'))} | "
            f"{_format_number(value.get('observed_mean_score'))} | "
            f"`{json.dumps(value.get('status_counts', {}), sort_keys=True)}` |\n"
        )
    if not official.get("breakdown"):
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")
    lines.extend([
        "\nZero-job cells such as `free_fall/test_ood1` are shown as `N/A` "
        "in the descriptive grid above; the official evaluator does not "
        "invent an empty breakdown group.\n\n",
        "### Official by-scene macro result\n\n",
        "| Scene | Expected | Evaluated | Coverage | Strict score | "
        "Observed macro mean | Aggregation |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
    ])
    for scene_id, value in sorted(official.get("by_scene", {}).items()):
        lines.append(
            f"| `{_markdown_cell(scene_id)}` | "
            f"{value.get('expected_jobs', 'N/A')} | "
            f"{value.get('evaluated_jobs', 'N/A')} | "
            f"{_format_number(value.get('coverage'))} | "
            f"{_format_number(value.get('score'))} | "
            f"{_format_number(value.get('observed_mean_score'))} | "
            f"`{_markdown_cell(value.get('aggregation_policy'))}` |\n"
        )
    if not official.get("by_scene"):
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")

    problem = summary["problem_job_ids"]
    lines.extend([
        "\n## Problem job IDs\n\n",
        f"- Failed: `{json.dumps(problem['failed'])}`\n",
        f"- Unavailable: `{json.dumps(problem['unavailable'])}`\n",
        f"- Missing prediction: "
        f"`{json.dumps(problem['missing_prediction'])}`\n",
        f"- Missing evaluation: "
        f"`{json.dumps(problem['missing_evaluation'])}`\n",
        f"- Other terminal status: "
        f"`{json.dumps(problem['other_terminal'])}`\n\n",
        "## Per-job results\n\n",
        "| Scene | Partition | Case | Job | Seed | Prediction | Evaluation | "
        "Score | Reason code | Evaluator | Metrics | Artifacts | "
        "Reason / prediction error |\n",
        "|---|---|---|---|---:|---|---|---:|---|---|---|---|---|\n",
    ])
    for record in summary["jobs"]:
        detail = record.get("reason") or record.get("prediction_error")
        evaluator = record.get("evaluator")
        evaluator_id = (
            evaluator.get("id") if isinstance(evaluator, dict) else None
        )
        lines.append(
            f"| `{_markdown_cell(record['scene_id'])}` | "
            f"`{_markdown_cell(record['partition'])}` | "
            f"`{_markdown_cell(record['case_id'])}` | "
            f"`{_markdown_cell(record['job_id'])}` | "
            f"{record['seed']} | "
            f"`{_markdown_cell(record.get('prediction_status'))}` | "
            f"`{_markdown_cell(record.get('evaluation_status'))}` | "
            f"{_format_number(record.get('score'))} | "
            f"`{_markdown_cell(record.get('reason_code'))}` | "
            f"`{_markdown_cell(evaluator_id)}` | "
            f"`{_markdown_cell(json.dumps(record.get('metrics') or {}, sort_keys=True))}` | "
            f"`{_markdown_cell(json.dumps(record.get('artifacts') or {}, sort_keys=True))}` | "
            f"{_markdown_cell(detail)} |\n"
        )

    acceptance = summary["training_acceptance"]
    lines.extend([
        "\n## Training acceptance\n\n",
        "| Check | Passed |\n",
        "|---|---|\n",
        f"| Combined checkpoint inventory and digest | "
        f"{_format_boolean(acceptance.get('checkpoint_passed'))} |\n",
        f"| Frozen sampling plan | "
        f"{_format_boolean(acceptance.get('sampling_plan_passed'))} |\n",
        f"| Loss steps and artifacts | "
        f"{_format_boolean(acceptance.get('loss_passed'))} |\n",
        f"| Quantity gradients and frozen UMT5 | "
        f"{_format_boolean(acceptance.get('gradient_passed'))} |\n",
        f"| Training quantity-token Case coverage | "
        f"{_format_boolean(acceptance.get('quantity_token_audit_passed'))} |\n",
        f"| Overall | {_format_boolean(acceptance.get('passed'))} |\n\n",
        "The loss audit checks complete step coverage and finite values. "
        "Its TensorBoard scalar is the rank-0 local batch loss at each "
        "optimizer step, not an all-rank reduced loss.\n\n",
        "## Training evidence\n\n",
        "| Artifact | Available | SHA-256 | Selected audit fields |\n",
        "|---|---|---|---|\n",
    ])
    for name, reference in summary["training_evidence"].items():
        selected = reference.get("summary")
        selected_text = (
            "`" + json.dumps(selected, sort_keys=True) + "`"
            if selected is not None
            else "N/A"
        )
        lines.append(
            f"| `{name}` (`{reference['path']}`) | "
            f"{'yes' if reference['available'] else 'no'} | "
            f"`{reference['sha256'] or 'N/A'}` | {selected_text} |\n"
        )
    lines.extend([
        "\n## Integrity\n\n",
        f"- Checkpoint digest verified: "
        f"`{checkpoint.get('digest_verified')}`\n",
        f"- Full report integrity passed: "
        f"`{_format_boolean(summary['reporting_status']['integrity_passed'])}`\n",
        f"- All planned predictions complete: "
        f"`{_format_boolean(summary['reporting_status']['all_planned_predictions_complete'])}`\n",
        f"- Benchmark score publishable: "
        f"`{_format_boolean(summary['reporting_status']['benchmark_score_publishable'])}`\n",
    ])
    issues = summary["integrity_issues"]
    if issues:
        lines.append("- Issues:\n")
        for issue in issues:
            lines.append(
                "  - `" + json.dumps(issue, sort_keys=True) + "`\n"
            )
    else:
        lines.append("- Issues: none\n")
    return "".join(lines)


def write_summary(
    summary: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / OUTPUT_JSON
    markdown_path = directory / OUTPUT_MARKDOWN
    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a completed quantity-embedding AtomicRun without "
            "modifying the run."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize_run(args.run_dir)
    json_path, markdown_path = write_summary(summary, args.output_dir)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
