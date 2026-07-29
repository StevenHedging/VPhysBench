from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ..io import write_json, write_jsonl
from .contracts import (
    CaseEvaluationRequest,
    CaseEvaluationResult,
)
from .registry import SceneEvaluatorRegistry


def _failure_result(
    job: dict[str, Any],
    *,
    status: str,
    code: str,
    reason: str,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        job_id=job["job_id"],
        case_id=job["case_id"],
        scene_id=job["scene_id"],
        evaluator={
            "id": "task_evaluator_preflight",
            "version": "1.0",
            "scene_id": job["scene_id"],
        },
        status=status,
        score=None,
        reason_code=code,
        reason=reason,
    )


def _prediction_zero_result(
    job: dict[str, Any],
    *,
    code: str,
    reason: str,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        job_id=job["job_id"],
        case_id=job["case_id"],
        scene_id=job["scene_id"],
        evaluator={
            "id": "task_evaluator_prediction_output",
            "version": "1.1",
            "scene_id": job["scene_id"],
            "primary_score": "scene_subject_state_similarity",
        },
        status="evaluated",
        score=0.0,
        reason_code=code,
        reason=reason,
        metrics={
            "scene_subject_state_similarity": {
                "score": 0.0,
                "components": {
                    "physics_state": 0.0,
                    "subject": 0.0,
                },
                "degraded": True,
                "degradation_code": code,
                "degradation_reason": reason,
            }
        },
        quality={
            "degraded": True,
            "degradation_codes": [code],
            "temporal_coverage": 0.0,
        },
        provenance={
            "degradation": {
                "origin": "prediction",
                "policy": "conservative_zero_not_evaluator_failure",
            }
        },
    )


def aggregate_task_results(
    *,
    plan: dict[str, Any],
    case_results: list[dict[str, Any]],
    include_degraded_diagnostics: bool = False,
) -> dict[str, Any]:
    official = [
        item
        for item in case_results
        if item["evaluation_partition"] != "train_seen"
    ]
    statuses = Counter(item["status"] for item in official)
    evaluated = [item for item in official if item["status"] == "evaluated"]
    coverage = len(evaluated) / len(official) if official else 0.0

    partition_groups: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in official:
        partition_groups[
            (item["scene_id"], item["evaluation_partition"])
        ].append(item)
        scene_groups[item["scene_id"]].append(item)

    breakdown: dict[str, Any] = {}
    for (scene_id, partition), items in sorted(partition_groups.items()):
        scores = [
            float(item["score"])
            for item in items
            if item["status"] == "evaluated"
        ]
        item_coverage = len(scores) / len(items)
        breakdown[f"{scene_id}/{partition}"] = {
            "expected_jobs": len(items),
            "evaluated_jobs": len(scores),
            "coverage": item_coverage,
            "score": mean(scores) if item_coverage == 1.0 else None,
            "observed_mean_score": mean(scores) if scores else None,
            "status_counts": dict(
                sorted(Counter(item["status"] for item in items).items())
            ),
        }

    by_scene: dict[str, Any] = {}
    for scene_id in plan["scene_ids"]:
        items = scene_groups.get(scene_id, [])
        scores = [
            float(item["score"])
            for item in items
            if item["status"] == "evaluated"
        ]
        item_coverage = len(scores) / len(items) if items else 0.0
        if plan["family"] == "finetune_eval":
            partitions = sorted(
                {
                    item["evaluation_partition"]
                    for item in items
                    if item["evaluation_partition"] != "train_seen"
                }
            )
            partition_scores = [
                breakdown[f"{scene_id}/{partition}"]["score"]
                for partition in partitions
            ]
            strict_score = (
                mean(float(score) for score in partition_scores)
                if partitions and all(score is not None for score in partition_scores)
                else None
            )
            observed_partition_scores = [
                breakdown[f"{scene_id}/{partition}"]["observed_mean_score"]
                for partition in partitions
                if breakdown[f"{scene_id}/{partition}"][
                    "observed_mean_score"
                ]
                is not None
            ]
            observed = (
                mean(float(score) for score in observed_partition_scores)
                if observed_partition_scores
                else None
            )
            policy = "macro_mean_required_partitions"
        else:
            strict_score = mean(scores) if items and item_coverage == 1.0 else None
            observed = mean(scores) if scores else None
            policy = "mean_all_scene_cases_groups_are_diagnostics"
        by_scene[scene_id] = {
            "expected_jobs": len(items),
            "evaluated_jobs": len(scores),
            "coverage": item_coverage,
            "score": strict_score,
            "observed_mean_score": observed,
            "aggregation_policy": policy,
            "status_counts": dict(
                sorted(Counter(item["status"] for item in items).items())
            ),
        }

    scene_scores = [by_scene[scene_id]["score"] for scene_id in plan["scene_ids"]]
    observed_scene_scores = [
        by_scene[scene_id]["observed_mean_score"]
        for scene_id in plan["scene_ids"]
        if by_scene[scene_id]["observed_mean_score"] is not None
    ]
    task_score = (
        mean(float(score) for score in scene_scores)
        if scene_scores and all(score is not None for score in scene_scores)
        else None
    )
    result = {
        "status": "complete" if coverage == 1.0 else "partial",
        "expected_jobs": len(official),
        "evaluated_jobs": len(evaluated),
        "coverage": coverage,
        "status_counts": dict(sorted(statuses.items())),
        "score": task_score,
        "observed_mean_score": (
            mean(float(score) for score in observed_scene_scores)
            if observed_scene_scores
            else None
        ),
        "aggregation_policy": (
            "strict_complete_coverage_then_macro_mean_selected_scenes"
        ),
        "by_scene": by_scene,
        "breakdown": breakdown,
    }
    if include_degraded_diagnostics:
        degraded = [
            item
            for item in official
            if item["status"] == "evaluated"
            and item.get("quality", {}).get("degraded") is True
        ]
        result["robustness"] = {
            "degraded_evaluated_jobs": len(degraded),
            "degraded_evaluated_ratio": (
                len(degraded) / len(official) if official else 0.0
            ),
            "reference_or_media_unavailable_jobs": statuses.get(
                "unavailable", 0
            ),
            "evaluator_error_jobs": statuses.get("error", 0),
            "degradation_reason_counts": dict(
                sorted(
                    Counter(
                        item.get("reason_code") or "unspecified_degradation"
                        for item in degraded
                    ).items()
                )
            ),
        }
    return result


def evaluate_task(
    *,
    plan: dict[str, Any],
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    asset_root: str | Path,
    protocol: dict[str, Any],
    output_dir: str | Path,
    registry: SceneEvaluatorRegistry | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every frozen inference job and aggregate a strict Task score."""
    directory = Path(output_dir)
    case_root = directory / "cases"
    directory.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)
    by_case = {case["case_id"]: case for case in cases}
    predictions_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        job_id = prediction.get("job_id")
        if isinstance(job_id, str):
            predictions_by_job[job_id].append(prediction)
    planned_ids = {job["job_id"] for job in plan["jobs"]}
    integrity_issues = [
        {
            "code": "unknown_prediction_job",
            "job_id": job_id,
            "records": len(records),
        }
        for job_id, records in sorted(predictions_by_job.items())
        if job_id not in planned_ids
    ]
    active_registry = registry or SceneEvaluatorRegistry(protocol)
    robust_prediction_zeros = (
        protocol.get("robustness", {}).get(
            "prediction_record_failure_policy"
        )
        == "evaluated_zero"
    )
    results: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        artifact_dir = case_root / job["job_id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        case = by_case.get(job["case_id"])
        records = predictions_by_job.get(job["job_id"], [])
        if case is None:
            outcome = _failure_result(
                job,
                status="error",
                code="case_missing_from_frozen_dataset",
                reason=f"frozen case not found: {job['case_id']}",
            )
        elif len(records) > 1:
            outcome = _failure_result(
                job,
                status="error",
                code="duplicate_prediction_records",
                reason=f"job has {len(records)} prediction records",
            )
        elif (
            protocol["scenes"]
            .get(job["scene_id"], {"type": "unsupported"})
            .get("type")
            != "unsupported"
            and not records
        ):
            outcome = (
                _prediction_zero_result(
                    job,
                    code="prediction_record_missing",
                    reason="planned job has no prediction record",
                )
                if robust_prediction_zeros
                else _failure_result(
                    job,
                    status="unavailable",
                    code="prediction_record_missing",
                    reason="planned job has no prediction record",
                )
            )
        elif (
            protocol["scenes"]
            .get(job["scene_id"], {"type": "unsupported"})
            .get("type")
            != "unsupported"
            and records[0].get("status") != "complete"
        ):
            reason = f"prediction status is {records[0].get('status')!r}"
            outcome = (
                _prediction_zero_result(
                    job,
                    code="prediction_incomplete",
                    reason=reason,
                )
                if robust_prediction_zeros
                else _failure_result(
                    job,
                    status="unavailable",
                    code="prediction_incomplete",
                    reason=reason,
                )
            )
        else:
            request = CaseEvaluationRequest(
                job=job,
                case=case,
                case_catalog=by_case,
                prediction=records[0] if records else None,
                asset_root=Path(asset_root).resolve(),
                artifact_dir=artifact_dir,
                evaluator_config=protocol["scenes"].get(job["scene_id"], {}),
            )
            try:
                evaluator = active_registry.resolve(job["scene_id"])
                outcome = evaluator.evaluate(request)
            except Exception as exc:
                outcome = _failure_result(
                    job,
                    status="error",
                    code="unhandled_case_evaluator_error",
                    reason=f"{type(exc).__name__}: {exc}",
                )
        record = {
            **outcome.to_dict(),
            "evaluation_partition": job["evaluation_partition"],
            "seed": job["seed"],
        }
        results.append(record)
        write_json(artifact_dir / "result.json", record)

    aggregation = aggregate_task_results(
        plan=plan,
        case_results=results,
        include_degraded_diagnostics=bool(protocol.get("robustness")),
    )
    task_result = {
        "schema_version": "1.0",
        "task_id": plan["task_id"],
        "task_family": plan["family"],
        "protocol": {
            "id": protocol["protocol_id"],
            "fingerprint": protocol["fingerprint"],
            "path": protocol["path"],
        },
        "integrity_issues": integrity_issues,
        **aggregation,
    }
    manifest = {
        "schema_version": "1.0",
        "protocol": task_result["protocol"],
        "asset_root": str(Path(asset_root).resolve()),
        "planned_job_ids": [job["job_id"] for job in plan["jobs"]],
        "prediction_records": len(predictions),
        "registry": active_registry.describe(),
        "case_result_file": "case_results.jsonl",
        "task_result_file": "task_result.json",
    }
    write_jsonl(directory / "case_results.jsonl", results)
    write_json(directory / "task_result.json", task_result)
    write_json(directory / "manifest.json", manifest)
    return results, task_result
