#!/usr/bin/env python3
"""Create a reproducible, read-only summary of a completed AtomicRun."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


BOOTSTRAP_SEED = 20260728
BOOTSTRAP_ITERATIONS = 10_000
OUTPUT_JSON = "quantity_run_summary.json"
OUTPUT_MARKDOWN = "quantity_run_summary.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required AtomicRun file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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
        if field in record and record[field] != job[field]:
            raise ValueError(
                f"{label} identity mismatch for {job['job_id']}: "
                f"{field}={record[field]!r}, expected {job[field]!r}"
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
    missing_prediction = 0
    missing_evaluation = 0
    other_terminal = 0
    scores_by_case: dict[str, list[float]] = defaultdict(list)

    for job in jobs:
        job_id = job["job_id"]
        prediction = predictions.get(job_id)
        evaluation = evaluations.get(job_id)

        if prediction is None:
            missing_prediction += 1
        else:
            status = str(prediction.get("status", "unknown"))
            prediction_statuses[status] += 1
            if status == "complete":
                completed += 1
            elif status in {"error", "failed"}:
                failed_ids.add(job_id)

        if evaluation is None:
            missing_evaluation += 1
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
            ):
                raise ValueError(
                    f"evaluated job {job_id} has no finite numeric score"
                )
            evaluated += 1
            scores_by_case[job["case_id"]].append(float(score))
        elif status in {"error", "failed"}:
            failed_ids.add(job_id)
        elif status in {"unavailable", "unsupported"}:
            if job_id not in failed_ids:
                unavailable_ids.add(job_id)
        else:
            other_terminal += 1

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
        "score_unit": "case_mean_across_evaluated_inference_seeds",
        "prediction_status_counts": dict(sorted(prediction_statuses.items())),
        "evaluation_status_counts": dict(sorted(evaluation_statuses.items())),
        "reason_code_counts": dict(sorted(reason_codes.items())),
    }


def _checkpoint_provenance(
    run_dir: Path,
    integrity_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    relative_manifest = "artifacts/wan22/checkpoint.json"
    manifest_path = run_dir / relative_manifest
    reference = _artifact_reference(run_dir, relative_manifest)
    if not manifest_path.is_file():
        return {
            "digest": None,
            "path": None,
            "available": False,
            "digest_verified": None,
            "manifest": reference,
        }
    manifest = _load_json(manifest_path)
    checkpoint_value = manifest.get("checkpoint")
    checkpoint_path: Path | None = None
    if isinstance(checkpoint_value, str) and checkpoint_value:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
    available = checkpoint_path is not None and checkpoint_path.is_file()
    declared = manifest.get("checkpoint_sha256")
    if declared is not None and (
        not isinstance(declared, str) or len(declared) != 64
    ):
        raise ValueError("checkpoint manifest has an invalid checkpoint_sha256")
    actual = _file_sha256(checkpoint_path) if available else None
    digest = declared or actual
    verified = (
        actual == declared
        if actual is not None and declared is not None
        else (True if actual is not None else None)
    )
    if verified is False:
        integrity_issues.append({
            "code": "checkpoint_digest_mismatch",
            "declared_sha256": declared,
            "actual_sha256": actual,
        })
    if checkpoint_path is not None and not available:
        integrity_issues.append({
            "code": "checkpoint_file_missing",
            "path": _relative_to_run(checkpoint_path, run_dir),
        })
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
        "status": manifest.get("status"),
        "source": manifest.get("source"),
        "inventory": manifest.get("inventory"),
        "training_state": manifest.get("training_state"),
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
            return list(dict.fromkeys(selected))
    return sorted(
        {
            str(job["evaluation_partition"])
            for job in plan["jobs"]
        }
    )


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


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Read and summarize one completed AtomicRun without modifying it."""
    directory = Path(run_dir).resolve()
    run = _load_json(directory / "run.json")
    if run.get("status") != "complete":
        raise ValueError(
            "quantity run summary requires run.json status=complete; "
            f"got {run.get('status')!r}"
        )
    plan = _load_json(directory / "plan.json")
    task = _load_json(directory / "frozen" / "task.json")
    task_result = _load_json(directory / "evaluation" / "task_result.json")
    component_fingerprints = _load_json(
        directory / "component_fingerprints.json"
    )
    jobs = _validate_plan_jobs(plan)
    predictions = _unique_by_job_id(
        _load_jsonl(directory / "predictions.jsonl"),
        label="predictions",
    )
    evaluations = _unique_by_job_id(
        _load_jsonl(directory / "evaluation" / "case_results.jsonl"),
        label="case evaluations",
    )
    planned_ids = {job["job_id"] for job in jobs}
    unknown_predictions = sorted(set(predictions) - planned_ids)
    unknown_evaluations = sorted(set(evaluations) - planned_ids)
    if unknown_predictions:
        raise ValueError(
            f"predictions contain unknown jobs: {unknown_predictions}"
        )
    if unknown_evaluations:
        raise ValueError(
            f"case evaluations contain unknown jobs: {unknown_evaluations}"
        )
    by_job = {job["job_id"]: job for job in jobs}
    for job_id, prediction in predictions.items():
        _validate_record_identity(
            prediction,
            by_job[job_id],
            label="prediction",
        )
    for job_id, evaluation in evaluations.items():
        _validate_record_identity(
            evaluation,
            by_job[job_id],
            label="case evaluation",
        )

    integrity_issues: list[dict[str, Any]] = []
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

    loss_summary_keys = (
        "recorded_steps",
        "expected_total_steps",
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
    training_evidence = {
        "loss_summary": _artifact_reference(
            directory,
            "artifacts/wan22/loss_analysis/loss_summary.json",
            json_summary_keys=loss_summary_keys,
        ),
        "loss_curve_csv": _artifact_reference(
            directory,
            "artifacts/wan22/loss_analysis/loss_curve.csv",
        ),
        "loss_curve_png": _artifact_reference(
            directory,
            "artifacts/wan22/loss_analysis/loss_curve.png",
        ),
        "gradient_audit": _artifact_reference(
            directory,
            "artifacts/wan22/checkpoints/gradient_audit.json",
            json_summary_keys=gradient_summary_keys,
        ),
        "quantity_token_audit": _artifact_reference(
            directory,
            "artifacts/wan22/training_quantity_token_audit.jsonl",
        ),
    }
    sources = {
        "run": _artifact_reference(directory, "run.json"),
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
    }
    return {
        "schema_version": "1.0",
        "summary_type": "quantity_atomic_run_evaluation",
        "source_run_dir": str(directory),
        "provenance": {
            "run": {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "digest": _canonical_sha256(run),
                "digest_semantics": "canonical_run_json_sha256",
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
            "task_instance": {
                "task_instance_id": run.get("task_instance_id"),
                "digest": run.get("task_instance_digest"),
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
            "overall": (
                "micro summary over planned jobs; official Benchmark macro "
                "score remains in official_task_result"
            ),
        },
        "scene_partition": scene_partition,
        "by_scene": by_scene,
        "overall": overall,
        "official_task_result": {
            "status": task_result.get("status"),
            "coverage": task_result.get("coverage"),
            "score": task_result.get("score"),
            "observed_mean_score": task_result.get("observed_mean_score"),
            "aggregation_policy": task_result.get("aggregation_policy"),
        },
        "training_evidence": training_evidence,
        "sources": sources,
        "integrity_issues": integrity_issues,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def _format_ci(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A"
    return (
        f"[{_format_number(value.get('low'))}, "
        f"{_format_number(value.get('high'))}]"
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
        "| Object | ID | Digest |\n",
        "|---|---|---|\n",
        f"| Run | `{provenance['run']['run_id']}` | "
        f"`{provenance['run']['digest']}` |\n",
        f"| Task | `{provenance['task']['task_id']}` | "
        f"`{provenance['task']['digest']}` |\n",
        f"| Baseline | `{provenance['baseline']['baseline_id']}` | "
        f"`{provenance['baseline']['digest']}` |\n",
        f"| Checkpoint | `{checkpoint.get('path') or 'N/A'}` | "
        f"`{checkpoint.get('digest') or 'N/A'}` |\n",
        "\n",
        "Scores below use only successfully evaluated cases. Missing, failed, "
        "and unavailable jobs are never converted to zero. `N/A` means no "
        "score was available for that cell. The 95% interval is a fixed-seed "
        f"percentile bootstrap ({summary['bootstrap']['iterations']} "
        f"resamples, base seed `{summary['bootstrap']['seed']}`).\n\n",
        "## Scene × partition\n\n",
        "| Scene | Partition | Expected | Completed | Evaluated | Failed | "
        "Unavailable | Missing prediction | Missing evaluation | Coverage | "
        "Mean | Median | Std | 95% bootstrap CI |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in summary["scene_partition"]:
        lines.append(
            _markdown_metric_row(
                [f"`{row['scene_id']}`", f"`{row['partition']}`"],
                row,
            )
        )
    lines.extend([
        "\n## By scene\n\n",
        "| Scene | Expected | Completed | Evaluated | Failed | Unavailable | "
        "Missing prediction | Missing evaluation | Coverage | Mean | Median | "
        "Std | 95% bootstrap CI |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ])
    for row in summary["by_scene"]:
        lines.append(
            _markdown_metric_row([f"`{row['scene_id']}`"], row)
        )
    lines.extend([
        "\n## Overall\n\n",
        "| Scope | Expected | Completed | Evaluated | Failed | Unavailable | "
        "Missing prediction | Missing evaluation | Coverage | Mean | Median | "
        "Std | 95% bootstrap CI |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        _markdown_metric_row(["All planned jobs"], summary["overall"]),
        "\nThe overall row is a case-level micro summary for descriptive "
        "statistics. It does not replace the Benchmark's strict macro score.\n\n",
        "## Official Task result\n\n",
        f"- Status: `{official.get('status') or 'N/A'}`\n",
        f"- Coverage: `{_format_number(official.get('coverage'))}`\n",
        f"- Strict Task score: `{_format_number(official.get('score'))}`\n",
        f"- Observed mean score: "
        f"`{_format_number(official.get('observed_mean_score'))}`\n",
        f"- Aggregation: `{official.get('aggregation_policy') or 'N/A'}`\n\n",
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
