from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..datasets import load_dataset
from ..io import write_json
from ..tasks import load_task, plan_atomic_task
from .protocols import load_evaluation_protocol
from .task_evaluator import evaluate_task


def _finite_unit_score(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_preflight_results(
    *,
    plan: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate total, finite evaluation for every planned official job."""

    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("plan.jobs must be a list")
    planned = {
        str(job["job_id"]): job
        for job in jobs
        if isinstance(job, Mapping) and isinstance(job.get("job_id"), str)
    }
    if len(planned) != len(jobs):
        raise ValueError("plan.jobs must contain unique string job IDs")

    observed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in case_results:
        job_id = result.get("job_id")
        if isinstance(job_id, str):
            observed[job_id].append(result)

    issues: list[dict[str, Any]] = []
    for job_id in sorted(set(observed) - set(planned)):
        issues.append(
            {
                "code": "unknown_result",
                "job_id": job_id,
                "record_count": len(observed[job_id]),
            }
        )
    for job_id, job in planned.items():
        records = observed.get(job_id, [])
        if not records:
            issues.append(
                {
                    "code": "missing_result",
                    "job_id": job_id,
                    "case_id": job.get("case_id"),
                }
            )
            continue
        if len(records) != 1:
            issues.append(
                {
                    "code": "duplicate_result",
                    "job_id": job_id,
                    "case_id": job.get("case_id"),
                    "record_count": len(records),
                }
            )
            continue
        result = records[0]
        status = result.get("status")
        if status != "evaluated":
            issues.append(
                {
                    "code": "case_not_evaluated",
                    "job_id": job_id,
                    "case_id": job.get("case_id"),
                    "status": status,
                    "reason_code": result.get("reason_code"),
                    "reason": result.get("reason"),
                }
            )
            continue
        if not _finite_unit_score(result.get("score")):
            issues.append(
                {
                    "code": "primary_score_invalid",
                    "job_id": job_id,
                    "case_id": job.get("case_id"),
                    "score": result.get("score"),
                }
            )
        metrics = result.get("metrics")
        csti = metrics.get("csti") if isinstance(metrics, Mapping) else None
        if not isinstance(csti, Mapping):
            issues.append(
                {
                    "code": "csti_metric_missing",
                    "job_id": job_id,
                    "case_id": job.get("case_id"),
                }
            )
            continue
        csti_status = csti.get("status")
        csti_score = csti.get("score")
        if csti_status == "evaluated":
            if not _finite_unit_score(csti_score):
                issues.append(
                    {
                        "code": "csti_score_invalid",
                        "job_id": job_id,
                        "case_id": job.get("case_id"),
                        "score": csti_score,
                    }
                )
        elif csti_status == "not_applicable":
            if csti_score is not None:
                issues.append(
                    {
                        "code": "csti_not_applicable_score_nonnull",
                        "job_id": job_id,
                        "case_id": job.get("case_id"),
                        "score": csti_score,
                    }
                )
        else:
            issues.append(
                {
                    "code": "csti_status_invalid",
                    "job_id": job_id,
                    "case_id": job.get("case_id"),
                    "status": csti_status,
                }
            )

    status_counts = Counter(
        str(records[0].get("status"))
        for job_id, records in observed.items()
        if job_id in planned and len(records) == 1
    )
    evaluated_jobs = sum(
        1
        for job_id, records in observed.items()
        if (
            job_id in planned
            and len(records) == 1
            and records[0].get("status") == "evaluated"
        )
    )
    fingerprints = sorted(
        {
            str(evaluator["fingerprint"])
            for result in case_results
            if isinstance(result.get("evaluator"), Mapping)
            for evaluator in [result["evaluator"]]
            if isinstance(evaluator.get("fingerprint"), str)
        }
    )
    return {
        "schema_version": "1.0",
        "status": "complete" if not issues else "failed",
        "task_id": plan.get("task_id"),
        "expected_jobs": len(planned),
        "evaluated_jobs": evaluated_jobs,
        "status_counts": dict(sorted(status_counts.items())),
        "evaluator_fingerprints": fingerprints,
        "issues": issues,
    }


def build_reference_predictions(
    *,
    plan: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    asset_root: str | Path,
) -> list[dict[str, Any]]:
    """Use each Case's canonical reference video as a known-legal prediction."""

    root = Path(asset_root).resolve(strict=True)
    by_case = {str(case["case_id"]): case for case in cases}
    predictions: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        case_id = str(job["case_id"])
        case = by_case[case_id]
        assets = case.get("assets")
        reference_value = (
            assets.get("reference_video")
            if isinstance(assets, Mapping)
            else None
        )
        if not isinstance(reference_value, str) or not reference_value:
            raise ValueError(f"Case {case_id!r} has no reference_video asset")
        path = (root / reference_value).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Case {case_id!r} reference_video escapes the asset root"
            ) from exc
        predictions.append(
            {
                "job_id": str(job["job_id"]),
                "case_id": case_id,
                "scene_id": str(job["scene_id"]),
                "seed": int(job["seed"]),
                "status": "complete",
                "video_path": str(path),
                "provenance": {
                    "policy": "canonical_reference_as_legal_prediction_v1"
                },
            }
        )
    return predictions


def run_reference_preflight(
    *,
    dataset_path: str | Path,
    task_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate all official jobs and write an authoritative preflight report."""

    dataset = load_dataset(dataset_path, check_assets=True)
    task = load_task(task_path)
    plan = plan_atomic_task(task, dataset).value
    protocol_id = str(task.value["evaluation"]["protocol"])
    protocol = load_evaluation_protocol(protocol_id)
    predictions = build_reference_predictions(
        plan=plan,
        cases=dataset.cases,
        asset_root=dataset.asset_root,
    )
    root = Path(output_dir).resolve()
    case_results, task_result = evaluate_task(
        plan=plan,
        cases=list(dataset.cases),
        predictions=predictions,
        asset_root=dataset.asset_root,
        protocol=protocol,
        output_dir=root / "evaluation",
        run_id="reference-evaluability-preflight",
        save_visualizations=False,
    )
    summary = validate_preflight_results(
        plan=plan,
        case_results=case_results,
    )
    summary.update(
        {
            "dataset_id": dataset.dataset_id,
            "dataset_digest": dataset.digest,
            "task_digest": task.digest,
            "protocol_id": protocol_id,
            "protocol_fingerprint": protocol["fingerprint"],
            "task_result_status": task_result.get("status"),
            "task_result_score": task_result.get("score"),
        }
    )
    write_json(root / "reference_preflight.json", summary)
    return summary


__all__ = [
    "build_reference_predictions",
    "run_reference_preflight",
    "validate_preflight_results",
]
