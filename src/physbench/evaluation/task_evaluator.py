from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean
from typing import Any

from ..io import write_json, write_jsonl
from .contracts import (
    CaseEvaluationRequest,
    CaseEvaluationResult,
)
from .common.csti import (
    CSTIConfig,
    not_applicable_csti_metric,
    zero_csti_metric,
)
from .common.entities import ReferenceCapability, materialize_entity_manifest
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
    case: dict[str, Any],
    csti_config: CSTIConfig | None,
    code: str,
    reason: str,
) -> CaseEvaluationResult:
    metrics = {
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
    }
    if csti_config is not None:
        manifest = materialize_entity_manifest(case)
        if manifest.reference_capability is ReferenceCapability.SAME_CASE_GT:
            metrics["csti"] = zero_csti_metric(
                expected_entities=tuple(
                    (entity.entity_id, entity.role_id)
                    for entity in manifest.entities
                ),
                config=csti_config,
                degradation_code=code,
                degradation_reason=reason,
            )
        else:
            metrics["csti"] = not_applicable_csti_metric(
                config=csti_config,
                reason_code="csti_requires_same_case_gt",
            )
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
        metrics=metrics,
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


def _diagnostic_group_result(
    items: list[dict[str, Any]],
    *,
    minimum_jobs: int,
) -> dict[str, Any]:
    scores = [
        float(item["score"])
        for item in items
        if item["status"] == "evaluated"
    ]
    expected = len(items)
    coverage = len(scores) / expected if expected else 0.0
    if expected == 0:
        status = "not_applicable"
    elif expected < minimum_jobs:
        status = "insufficient_samples"
    elif coverage == 1.0:
        status = "complete"
    else:
        status = "partial"
    return {
        "status": status,
        "minimum_jobs": minimum_jobs,
        "expected_jobs": expected,
        "evaluated_jobs": len(scores),
        "coverage": coverage,
        "score": (
            mean(scores)
            if expected >= minimum_jobs and coverage == 1.0
            else None
        ),
        "observed_mean_score": mean(scores) if scores else None,
        "status_counts": dict(
            sorted(Counter(item["status"] for item in items).items())
        ),
    }


@dataclass(frozen=True)
class DimensionCaseRecord:
    job_id: str
    scene_id: str
    evaluation_partition: str
    status: str
    score: float | None
    quality: dict[str, Any]
    reason_code: str | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scene_id": self.scene_id,
            "evaluation_partition": self.evaluation_partition,
            "status": self.status,
            "score": self.score,
            "quality": self.quality,
            "reason_code": self.reason_code,
        }


def _aggregate_dimension(
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
        if (
            plan["family"] == "finetune_eval"
            and "evaluation_annotations" in plan
        ):
            strict_score = (
                mean(scores) if items and item_coverage == 1.0 else None
            )
            observed = mean(scores) if scores else None
            policy = "mean_all_test_jobs_regimes_are_diagnostics"
        elif plan["family"] == "finetune_eval":
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
    evaluation_annotations = plan.get("evaluation_annotations")
    if isinstance(evaluation_annotations, dict):
        reporting = plan.get("reporting_policy", {})
        minimum_jobs = reporting.get("minimum_subgroup_jobs", 1)
        if (
            isinstance(minimum_jobs, bool)
            or not isinstance(minimum_jobs, int)
            or minimum_jobs < 1
        ):
            raise ValueError(
                "plan.reporting_policy.minimum_subgroup_jobs must be positive"
            )
        result_by_job = {item["job_id"]: item for item in official}
        unknown_annotations = set(evaluation_annotations) - set(result_by_job)
        if unknown_annotations:
            raise ValueError(
                "plan evaluation annotations reference jobs absent from results: "
                f"{sorted(unknown_annotations)}"
            )

        regime_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scene_regime_groups: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        factor_groups: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for job_id, annotation in evaluation_annotations.items():
            item = result_by_job[job_id]
            regime = annotation["generalization_regime"]
            regime_groups[regime].append(item)
            scene_regime_groups[(item["scene_id"], regime)].append(item)
            for factor in annotation["ood_factors"]:
                factor_groups[(factor["category"], factor["name"])].append(item)

        by_regime = {
            regime: _diagnostic_group_result(
                regime_groups.get(regime, []),
                minimum_jobs=minimum_jobs,
            )
            for regime in ("id", "ood", "mixed")
        }
        by_scene_regime = {
            f"{scene_id}/{regime}": _diagnostic_group_result(
                scene_regime_groups.get((scene_id, regime), []),
                minimum_jobs=minimum_jobs,
            )
            for scene_id in plan["scene_ids"]
            for regime in ("id", "ood", "mixed")
        }
        by_ood_factor = {
            f"{category}/{name}": {
                "category": category,
                "factor": name,
                **_diagnostic_group_result(
                    items,
                    minimum_jobs=minimum_jobs,
                ),
            }
            for (category, name), items in sorted(factor_groups.items())
        }
        result["generalization_breakdown"] = {
            "regimes_are_relative_to": "view_a.train",
            "minimum_subgroup_jobs": minimum_jobs,
            "by_regime": by_regime,
            "by_scene_regime": by_scene_regime,
            "by_ood_factor": by_ood_factor,
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
            "prediction_protocol_error_jobs": statuses.get(
                "protocol_error", 0
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


def aggregate_task_results(
    *,
    plan: dict[str, Any],
    case_results: list[dict[str, Any]],
    include_degraded_diagnostics: bool = False,
    general_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate expert scores and optional general metrics independently."""

    expert = _aggregate_dimension(
        plan=plan,
        case_results=case_results,
        include_degraded_diagnostics=include_degraded_diagnostics,
    )
    csti_config = (
        general_metrics.get("csti")
        if isinstance(general_metrics, Mapping)
        else None
    )
    if (
        not isinstance(csti_config, Mapping)
        or csti_config.get("enabled") is not True
    ):
        return expert

    csti_records = [
        _csti_dimension_record(item).to_mapping()
        for item in case_results
    ]
    csti = _aggregate_csti_dimension(plan=plan, records=csti_records)
    expert["dimensions"] = {
        "expert": {
            "score": expert["score"],
            "source": "top_level_score",
        },
        "csti": csti,
    }
    return expert


def _aggregate_csti_dimension(
    *,
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    applicable = [
        item for item in records
        if item["status"] != "not_applicable"
    ]
    not_applicable = [
        item
        for item in records
        if (
            item["evaluation_partition"] != "train_seen"
            and item["status"] == "not_applicable"
        )
    ]
    aggregate_plan = plan
    annotations = plan.get("evaluation_annotations")
    if isinstance(annotations, dict):
        applicable_ids = {item["job_id"] for item in applicable}
        aggregate_plan = {
            **plan,
            "evaluation_annotations": {
                job_id: annotation
                for job_id, annotation in annotations.items()
                if job_id in applicable_ids
            },
        }
    result = _aggregate_dimension(
        plan=aggregate_plan,
        case_results=applicable,
    )
    result["not_applicable_jobs"] = len(not_applicable)
    not_applicable_by_scene = Counter(
        item["scene_id"] for item in not_applicable
    )
    for scene_id in plan["scene_ids"]:
        scene = result["by_scene"][scene_id]
        scene["not_applicable_jobs"] = not_applicable_by_scene[scene_id]
        if scene["expected_jobs"] == 0 and not_applicable_by_scene[scene_id]:
            scene["status"] = "not_applicable"
        elif scene["expected_jobs"] and scene["coverage"] == 1.0:
            scene["status"] = "complete"
        else:
            scene["status"] = "partial"

    applicable_scenes = [
        result["by_scene"][scene_id]
        for scene_id in plan["scene_ids"]
        if result["by_scene"][scene_id]["status"] != "not_applicable"
    ]
    if not applicable_scenes:
        result["status"] = "not_applicable"
        result["score"] = None
        result["observed_mean_score"] = None
    else:
        strict_scores = [scene["score"] for scene in applicable_scenes]
        result["score"] = (
            mean(float(score) for score in strict_scores)
            if all(score is not None for score in strict_scores)
            else None
        )
        observed_scores = [
            scene["observed_mean_score"]
            for scene in applicable_scenes
            if scene["observed_mean_score"] is not None
        ]
        result["observed_mean_score"] = (
            mean(float(score) for score in observed_scores)
            if observed_scores
            else None
        )
        result["status"] = (
            "complete"
            if result["coverage"] == 1.0 and result["score"] is not None
            else "partial"
        )
    return result


def _csti_dimension_record(item: Mapping[str, Any]) -> DimensionCaseRecord:
    case_status = str(item["status"])
    score: float | None
    if case_status != "evaluated":
        status = case_status
        score = None
    else:
        metrics = item.get("metrics", {})
        metric = metrics.get("csti") if isinstance(metrics, Mapping) else None
        if not isinstance(metric, Mapping):
            status = "error"
            score = None
        else:
            status = str(metric.get("status"))
            raw_score = metric.get("score")
            if status == "evaluated":
                if (
                    isinstance(raw_score, bool)
                    or not isinstance(raw_score, (int, float))
                    or not math.isfinite(float(raw_score))
                    or not 0.0 <= float(raw_score) <= 1.0
                ):
                    raise ValueError(
                        f"CSTI score for job {item['job_id']!r} must be finite and in [0,1]"
                    )
                score = float(raw_score)
            else:
                if raw_score is not None:
                    raise ValueError(
                        f"non-evaluated CSTI record {item['job_id']!r} must have a null score"
                    )
                score = None
    return DimensionCaseRecord(
        job_id=str(item["job_id"]),
        scene_id=str(item["scene_id"]),
        evaluation_partition=str(item["evaluation_partition"]),
        status=status,
        score=score,
        quality=(
            dict(item.get("quality", {}))
            if isinstance(item.get("quality", {}), Mapping)
            else {}
        ),
        reason_code=(
            str(item["reason_code"])
            if item.get("reason_code") is not None
            else None
        ),
    )


def evaluate_task(
    *,
    plan: dict[str, Any],
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    asset_root: str | Path,
    protocol: dict[str, Any],
    output_dir: str | Path,
    registry: SceneEvaluatorRegistry | None = None,
    run_id: str | None = None,
    save_visualizations: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every frozen inference job and aggregate a strict Task score."""
    directory = Path(output_dir)
    effective_run_id = run_id or directory.resolve().parent.name
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
    csti_value = protocol.get("general_metrics", {}).get("csti")
    csti_config = (
        CSTIConfig.from_mapping(csti_value)
        if csti_value is not None
        else None
    )
    results: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        artifact_dir = case_root / job["job_id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        case = by_case.get(job["case_id"])
        records = predictions_by_job.get(job["job_id"], [])
        media_protocol_error: tuple[str, str] | None = None
        if (
            len(records) == 1
            and records[0].get("status") == "complete"
            and records[0].get("media_contract") is not None
        ):
            from ..baseline_runtime.media_contract import (
                MediaContractError,
                validate_prediction_video,
            )

            try:
                validate_prediction_video(
                    records[0].get("video_path"),
                    records[0]["media_contract"],
                )
            except MediaContractError as exc:
                media_protocol_error = (exc.code, str(exc))
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
        elif media_protocol_error is not None:
            outcome = _failure_result(
                job,
                status="protocol_error",
                code=media_protocol_error[0],
                reason=media_protocol_error[1],
            )
        elif (
            records
            and protocol["scenes"]
            .get(job["scene_id"], {"type": "unsupported"})
            .get("type")
            != "unsupported"
            and records[0].get("status") == "protocol_error"
        ):
            error = records[0].get("protocol_error", {})
            outcome = _failure_result(
                job,
                status="protocol_error",
                code=str(error.get("code", "prediction_protocol_error")),
                reason=str(
                    error.get(
                        "reason",
                        "prediction violates the sealed media contract",
                    )
                ),
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
                    case=case,
                    csti_config=csti_config,
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
                    case=case,
                    csti_config=csti_config,
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
                run_id=effective_run_id,
                save_visualizations=save_visualizations,
                visualization_root=directory / "visualizations",
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
        general_metrics=protocol.get("general_metrics"),
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
