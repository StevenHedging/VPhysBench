from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .baselines import create_adapter
from .io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from .metrics import evaluate_cases
from .prompts import PromptRegistry
from .reporting import render_report
from .task_planner import plan_task
from .validation import errors, load_scene_configs, validate_cases


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _default_run_id(task_id: str, baseline_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _slug(f"{stamp}__{task_id}__{baseline_id}")


DEFAULT_PROMPTS = Path(__file__).resolve().parents[2] / "configs" / "prompts"


def run_benchmark(
    *,
    task_path: str | Path,
    baseline_path: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    scene_config_dir: str | Path,
    metric_config_path: str | Path,
    output_root: str | Path,
    execute: bool = False,
    run_id: str | None = None,
    stop_after_training: bool = False,
    train_preview_per_scene: int | None = None,
    train_preview_seed: int | None = None,
    prompt_config_dir: str | Path = DEFAULT_PROMPTS,
    train_prompt_profile: str | None = None,
    eval_prompt_profiles: list[str] | None = None,
) -> Path:
    task = load_json(task_path)
    baseline = load_json(baseline_path)
    cases = load_jsonl(manifest_path)
    split = load_json(split_path)
    scenes = load_scene_configs(scene_config_dir)
    metric_config = load_json(metric_config_path)
    issues = validate_cases(cases, manifest_path=manifest_path, scene_configs=scenes)
    if errors(issues):
        message = "; ".join(f"{item.case_id or '-'}:{item.code}:{item.message}" for item in errors(issues))
        raise ValueError(f"invalid manifest: {message}")
    plan = plan_task(
        task,
        cases,
        split,
        train_preview_per_scene=train_preview_per_scene,
        train_preview_seed=train_preview_seed,
        train_prompt_profile=train_prompt_profile,
        eval_prompt_profiles=eval_prompt_profiles,
    )
    prompt_registry = PromptRegistry(prompt_config_dir)
    prompt_registry.require([
        plan["prompt_profiles"]["train"],
        *plan["prompt_profiles"]["eval"],
    ])
    by_id = {case["case_id"]: case for case in cases}
    resolved_prompts = []
    train_profile = plan["prompt_profiles"]["train"]
    if train_profile:
        resolved_prompts.extend(
            prompt_registry.resolve(by_id[case_id], train_profile, role="train")
            for case_id in plan["train_case_ids"]
        )
    eval_prompt_map = {}
    for raw_job in plan["jobs"]:
        key = (raw_job["case_id"], raw_job["prompt_profile_id"])
        if key not in eval_prompt_map:
            eval_prompt_map[key] = prompt_registry.resolve(
                by_id[raw_job["case_id"]],
                raw_job["prompt_profile_id"],
                role="eval",
            )
    resolved_prompts.extend(eval_prompt_map[key] for key in sorted(eval_prompt_map))

    adapter = create_adapter(baseline, execute=execute)
    identifier = run_id or _default_run_id(task["task_id"], baseline["baseline_id"])
    run_dir = (Path(output_root) / identifier).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "jobs").mkdir()
    (run_dir / "predictions").mkdir()
    (run_dir / "artifacts").mkdir()

    write_json(run_dir / "data_context.json", {
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_dir": str(Path(manifest_path).resolve().parent),
    })
    write_json(run_dir / "frozen_task.json", task)
    write_json(run_dir / "frozen_baseline.json", baseline)
    write_json(run_dir / "frozen_split.json", split)
    write_json(run_dir / "frozen_metrics.json", metric_config)
    write_jsonl(run_dir / "frozen_cases.jsonl", cases)
    write_json(run_dir / "plan.json", plan)
    write_json(
        run_dir / "frozen_prompt_profiles.json",
        {
            **prompt_registry.snapshot([
                plan["prompt_profiles"]["train"],
                *plan["prompt_profiles"]["eval"],
            ]),
            "selection": plan["prompt_profiles"],
        },
    )

    write_jsonl(run_dir / "resolved_prompts.jsonl", resolved_prompts)

    training = adapter.prepare_training(plan["train_case_ids"], run_dir)
    write_json(run_dir / "training_job.json", training)
    training = adapter.train(training, run_dir / "training_job.json")
    write_json(run_dir / "training_stage.json", training)
    if execute and training.get("status") == "failed":
        raise RuntimeError("baseline training failed; inference was not started")
    predictions = []
    for raw_job in plan["jobs"]:
        prompt_record = eval_prompt_map[
            (raw_job["case_id"], raw_job["prompt_profile_id"])
        ]
        prepared = adapter.prepare_job(
            {**raw_job, "resolved_prompt": prompt_record},
            by_id[raw_job["case_id"]],
            run_dir,
        )
        job_path = run_dir / "jobs" / f"{raw_job['job_id']}.json"
        write_json(job_path, prepared)
        if stop_after_training:
            predictions.append({
                "job_id": prepared["job_id"],
                "case_id": prepared["case_id"],
                "baseline_id": baseline["baseline_id"],
                "evaluation_partition": prepared["evaluation_partition"],
                "prompt_profile_id": prepared["prompt_profile_id"],
                "evaluation_reference_video": prepared.get("evaluation_reference_video"),
                "visual_reference_video": prepared.get("visual_reference_video"),
                "status": "staged",
                "video_path": None,
                "manual_scores": {},
                "job_spec": str(job_path),
            })
        else:
            predictions.append(adapter.generate(prepared, job_path))
    write_jsonl(run_dir / "predictions.jsonl", predictions)

    case_metrics, summary = evaluate_cases(cases, predictions, scenes, metric_config)
    write_jsonl(run_dir / "case_metrics.jsonl", case_metrics)
    write_json(run_dir / "summary.json", summary)
    run = {
        "schema_version": "1.0",
        "benchmark_version": __version__,
        "run_id": identifier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "baseline_id": baseline["baseline_id"],
        "prompt_profiles": plan["prompt_profiles"],
        "manifest_source": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "execute_external_commands": execute,
        "status": "training_complete_inference_staged" if stop_after_training else (
        "complete_with_placeholders"
        if training.get("status") not in {"complete", "not_requested"}
        or any(item["status"] != "complete" for item in predictions)
        else "complete"),
    }
    write_json(run_dir / "run.json", run)
    (run_dir / "report.md").write_text(render_report(run, plan, summary), encoding="utf-8")
    return run_dir


def reevaluate_run(run_dir: str | Path, scene_config_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    cases = load_jsonl(directory / "frozen_cases.jsonl")
    predictions = load_jsonl(directory / "predictions.jsonl")
    scenes = load_scene_configs(scene_config_dir)
    metric_config = load_json(directory / "frozen_metrics.json")
    results, summary = evaluate_cases(cases, predictions, scenes, metric_config)
    write_jsonl(directory / "case_metrics.jsonl", results)
    write_json(directory / "summary.json", summary)
    run = load_json(directory / "run.json")
    plan = load_json(directory / "plan.json")
    (directory / "report.md").write_text(render_report(run, plan, summary), encoding="utf-8")
    return summary
