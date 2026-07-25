from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..baseline_api import load_baseline_bundle, load_baseline_plugin
from ..datasets import load_dataset_v2
from ..domain import BaselineTaskInstance, TaskSpec
from ..evaluation import evaluate_task, load_evaluation_protocol
from ..io import (
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)
from ..tasks import load_task_v2


def build_task_instance(
    *,
    dataset_path: str | Path,
    task_path: str | Path,
    baseline_path: str | Path,
    check_assets: bool = True,
) -> BaselineTaskInstance:
    """Public file-based facade for a Baseline-owned TaskBuilder."""
    dataset = load_dataset_v2(dataset_path, check_assets=check_assets)
    task = load_task_v2(task_path)
    baseline = load_baseline_bundle(baseline_path)
    plugin = load_baseline_plugin(baseline)
    instance = plugin.task_builder.build(dataset, task)
    instance.verify()
    return instance


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _run_id(task_id: str, baseline_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _slug(f"{stamp}__{task_id}__{baseline_id}")


def _state(run_dir: Path, stage: str, **extra: Any) -> None:
    current = (
        load_json(run_dir / "state.json")
        if (run_dir / "state.json").is_file()
        else {"schema_version": "2.0", "history": []}
    )
    current["stage"] = stage
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    current["history"].append({
        "stage": stage,
        "at": current["updated_at"],
        **extra,
    })
    write_json(run_dir / "state.json", current)


def _render_atomic_report(
    run: dict[str, Any], plan: dict[str, Any], summary: dict[str, Any]
) -> str:
    task_score = (
        "—" if summary["score"] is None else f"{summary['score']:.4f}"
    )
    lines = [
        f"# AtomicRun: {run['run_id']}\n\n",
        "```text\nTaskSpec → CanonicalTaskPlan → BaselineTaskInstance → AtomicRun\n```\n\n",
        f"- Dataset：`{run['dataset_id']}` / `{run['dataset_digest']}`\n",
        f"- Task：`{run['task_id']}` / `{run['task_family']}`\n",
        f"- Conditioning：`{run['conditioning']}`\n",
        f"- Baseline：`{run['baseline_id']}` / `{run['baseline_digest']}`\n",
        f"- Task instance：`{run['task_instance_id']}` / "
        f"`{run['task_instance_digest']}`\n",
        f"- Training seed：`{run['training_seed']}`\n",
        f"- Train cases：`{len(plan['train_case_ids'])}`\n",
        f"- Inference jobs：`{len(plan['jobs'])}`\n",
        f"- Status：`{run['status']}`\n\n",
        "## Evaluation\n\n",
        f"- Evaluation status：`{summary['status']}`\n",
        f"- Evaluation coverage：`{summary['coverage']:.4f}`\n",
        f"- Task score：`{task_score}`\n\n",
        "| Scene / partition | Expected | Evaluated | Score | Coverage |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name, item in summary["breakdown"].items():
        score = "—" if item["score"] is None else f"{item['score']:.4f}"
        lines.append(
            f"| `{name}` | {item['expected_jobs']} | {item['evaluated_jobs']} | "
            f"{score} | {item['coverage']:.4f} |\n"
        )
    lines.extend([
        "\n## Isolation\n\n",
        "- Dataset snapshot 不包含 prompt 或模型 input view。\n",
        "- 完整模型输入由 Baseline-owned DataAdapter 生成。\n",
        "- 物理注入是 DataAdapter 内的基线私有阶段，不预设为文本。\n",
        "- Trainer/Predictor 只消费已封印的 BaselineTaskInstance。\n",
        "- 本 AtomicRun 只包含一种 conditioning 和一份独立模型产物。\n",
    ])
    return "".join(lines)


def run_atomic(
    *,
    dataset_path: str | Path,
    task_path: str | Path,
    baseline_path: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
    execute: bool = False,
    stop_after_training: bool = False,
    check_assets: bool = True,
    scene_ids: list[str] | None = None,
    groups: list[str] | None = None,
    case_ids: list[str] | None = None,
) -> Path:
    dataset = load_dataset_v2(dataset_path, check_assets=check_assets)
    task = load_task_v2(task_path)
    protocol_id = task.value.get("evaluation", {}).get(
        "protocol", "scene_default_v1"
    )
    evaluation_protocol = load_evaluation_protocol(protocol_id)
    if scene_ids is not None or groups is not None or case_ids is not None:
        value = copy.deepcopy(task.value)
        if scene_ids is not None:
            value["selection"]["scene_ids"] = list(dict.fromkeys(scene_ids))
        if groups is not None:
            if task.family != "direct_eval":
                raise ValueError("--group is only valid for direct_eval")
            value["selection"]["groups"] = list(dict.fromkeys(groups))
        if case_ids is not None:
            if task.family != "direct_eval":
                raise ValueError("--case-id is only valid for direct_eval")
            value["selection"]["case_ids"] = list(dict.fromkeys(case_ids))
        task = TaskSpec(task.path, value, canonical_sha256(value))
    baseline = load_baseline_bundle(baseline_path)
    plugin = load_baseline_plugin(baseline)
    instance = plugin.task_builder.build(dataset, task)
    instance.verify()
    plan = instance.canonical_plan

    identifier = run_id or _run_id(task.task_id, baseline.baseline_id)
    run_dir = (Path(output_root) / identifier).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    for directory in (
        run_dir,
        run_dir / "frozen",
        run_dir / "task_instance",
        run_dir / "adaptations",
        run_dir / "training",
        run_dir / "jobs",
        run_dir / "predictions",
        run_dir / "evaluation",
        run_dir / "artifacts",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "frozen" / "dataset.json", dataset.descriptor)
    write_jsonl(run_dir / "frozen" / "cases.jsonl", dataset.cases)
    write_json(run_dir / "frozen" / "task.json", task.value)
    write_json(run_dir / "frozen" / "baseline.json", baseline.value)
    write_json(run_dir / "frozen" / "views.json", dataset.views)
    if dataset.asset_lock is not None:
        write_json(run_dir / "frozen" / "assets.lock.json", dataset.asset_lock)
    write_json(run_dir / "plan.json", plan.value)
    instance_value = instance.value
    write_json(
        run_dir / "task_instance" / "manifest.json",
        instance_value,
    )
    write_json(
        run_dir / "task_instance" / "canonical_plan.json",
        instance_value["canonical_plan"],
    )
    write_jsonl(
        run_dir / "task_instance" / "adaptations.jsonl",
        instance_value["adaptations"],
    )
    write_json(
        run_dir / "task_instance" / "training.json",
        instance_value["training"],
    )
    write_jsonl(
        run_dir / "task_instance" / "inference_jobs.jsonl",
        instance_value["inference"]["jobs"],
    )
    write_json(
        run_dir / "task_instance" / "execution_graph.json",
        instance_value["execution_graph"],
    )
    write_json(
        run_dir / "task_instance" / "cache_bindings.json",
        {"bindings": instance_value["cache_bindings"]},
    )
    write_json(
        run_dir / "task_instance" / "baseline_payload.json",
        instance_value["baseline_payload"],
    )
    write_json(run_dir / "component_fingerprints.json", {
        "dataset": dataset.digest,
        "task": task.digest,
        "baseline": baseline.digest,
        "task_builder": plugin.task_builder.fingerprint,
        "task_instance": instance.digest,
        "data_adapter": plugin.task_builder.data_adapter.fingerprint,
        "data_adapter_materialization": (
            plugin.task_builder.data_adapter.materialization_fingerprint
        ),
        "evaluation_protocol": evaluation_protocol["fingerprint"],
    })
    write_json(run_dir / "task_builder.json", plugin.task_builder.describe())
    write_json(
        run_dir / "data_adapter.json",
        plugin.task_builder.data_adapter.describe(),
    )
    _state(run_dir, "frozen")

    try:
        training, predictions = plugin.run_task(
            instance=instance,
            run_dir=run_dir,
            execute=execute,
            stop_after_training=stop_after_training,
        )
    except BaseException as exc:
        _state(run_dir, "failed", error=repr(exc))
        raise
    _state(run_dir, "training_complete_or_staged", status=training.get("status"))
    write_jsonl(run_dir / "predictions.jsonl", predictions)

    case_metrics, summary = evaluate_task(
        plan=plan.value,
        cases=list(dataset.cases),
        predictions=predictions,
        asset_root=dataset.asset_root,
        protocol=evaluation_protocol,
        output_dir=run_dir / "evaluation",
    )
    # Compatibility projections for consumers of the original v2 scaffold.
    write_jsonl(run_dir / "evaluation" / "case_metrics.jsonl", case_metrics)
    write_json(run_dir / "evaluation" / "summary.json", summary)
    status = (
        "training_complete_inference_staged"
        if stop_after_training
        else (
            "complete"
            if predictions and all(item.get("status") == "complete" for item in predictions)
            else ("planned" if not execute else "inference_incomplete")
        )
    )
    run = {
        "schema_version": "2.0",
        "benchmark_version": __version__,
        "run_id": identifier,
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest,
        "task_id": task.task_id,
        "task_family": task.family,
        "conditioning": task.conditioning,
        "baseline_id": baseline.baseline_id,
        "baseline_digest": baseline.digest,
        "task_instance_id": instance.instance_id,
        "task_instance_digest": instance.digest,
        "task_builder_fingerprint": plugin.task_builder.fingerprint,
        "training_seed": plan.value["training_seed"],
        "execute": execute,
        "status": status,
        "evaluation_status": summary["status"],
        "evaluation_coverage": summary["coverage"],
        "evaluation_score": summary["score"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "run.json", run)
    (run_dir / "report.md").write_text(
        _render_atomic_report(run, plan.value, summary), encoding="utf-8"
    )
    _state(run_dir, status)
    return run_dir


def reevaluate_atomic(run_dir: str | Path) -> dict[str, Any]:
    """Re-run Benchmark-owned evaluation for a frozen AtomicRun."""
    directory = Path(run_dir).resolve()
    plan = load_json(directory / "plan.json")
    task = load_json(directory / "frozen" / "task.json")
    cases = load_jsonl(directory / "frozen" / "cases.jsonl")
    predictions = load_jsonl(directory / "predictions.jsonl")
    instance = load_json(directory / "task_instance" / "manifest.json")
    protocol_id = task.get("evaluation", {}).get(
        "protocol", "scene_default_v1"
    )
    protocol = load_evaluation_protocol(protocol_id)
    case_results, summary = evaluate_task(
        plan=plan,
        cases=cases,
        predictions=predictions,
        asset_root=instance["source"]["asset_root"],
        protocol=protocol,
        output_dir=directory / "evaluation",
    )
    write_jsonl(directory / "evaluation" / "case_metrics.jsonl", case_results)
    write_json(directory / "evaluation" / "summary.json", summary)
    run_path = directory / "run.json"
    if run_path.is_file():
        run = load_json(run_path)
        run["evaluation_status"] = summary["status"]
        run["evaluation_coverage"] = summary["coverage"]
        run["evaluation_score"] = summary["score"]
        write_json(run_path, run)
        (directory / "report.md").write_text(
            _render_atomic_report(run, plan, summary), encoding="utf-8"
        )
    return summary


def _paired_plan_signature(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": plan["dataset_id"],
        "dataset_digest": plan["dataset_digest"],
        "scene_ids": plan["scene_ids"],
        "train_case_ids": plan["train_case_ids"],
        "training_seed": plan["training_seed"],
        "jobs": sorted(
            (
                job["case_id"],
                job["scene_id"],
                job["evaluation_partition"],
                job["seed"],
            )
            for job in plan["jobs"]
        ),
    }


def run_matrix(
    *,
    dataset_path: str | Path,
    task_paths: list[str | Path],
    baseline_path: str | Path,
    output_root: str | Path,
    matrix_id: str,
    execute: bool = False,
    stop_after_training: bool = False,
) -> list[Path]:
    if len(task_paths) < 2:
        raise ValueError("a task matrix requires at least two atomic tasks")
    dataset = load_dataset_v2(dataset_path, check_assets=True)
    tasks = [load_task_v2(path) for path in task_paths]
    baseline = load_baseline_bundle(baseline_path)
    plugin = load_baseline_plugin(baseline)
    instances = [plugin.task_builder.build(dataset, task) for task in tasks]
    plans = [instance.canonical_plan.value for instance in instances]
    first_signature = _paired_plan_signature(plans[0])
    if any(_paired_plan_signature(plan) != first_signature for plan in plans[1:]):
        raise ValueError(
            "paired task matrix must use identical data, splits, seeds, and evaluation cases"
        )
    conditioning = [task.conditioning for task in tasks]
    if len(conditioning) != len(set(conditioning)):
        raise ValueError("task matrix contains duplicate conditioning variants")
    output = Path(output_root).resolve()
    index_path = output / f"{matrix_id}.matrix.json"
    if index_path.exists():
        raise FileExistsError(f"matrix index already exists: {index_path}")
    index = {
        "schema_version": "2.0",
        "matrix_id": matrix_id,
        "elements": ["dataset", "task", "baseline"],
        "dataset": str(Path(dataset_path).resolve()),
        "baseline": str(Path(baseline_path).resolve()),
        "task_builder_fingerprint": plugin.task_builder.fingerprint,
        "tasks": [str(task.path) for task in tasks],
        "task_instances": [
            {
                "instance_id": instance.instance_id,
                "digest": instance.digest,
            }
            for instance in instances
        ],
        "conditioning": conditioning,
        "atomic_runs": [],
        "paired_plan_signature": first_signature,
        "status": "running",
    }
    write_json(index_path, index)
    run_dirs = []
    try:
        for task in tasks:
            run_dirs.append(run_atomic(
                dataset_path=dataset_path,
                task_path=task.path,
                baseline_path=baseline_path,
                output_root=output,
                run_id=f"{matrix_id}__{task.conditioning}",
                execute=execute,
                stop_after_training=stop_after_training,
            ))
            index["atomic_runs"] = [str(path) for path in run_dirs]
            write_json(index_path, index)
    except BaseException as exc:
        index["status"] = "failed"
        index["error"] = repr(exc)
        write_json(index_path, index)
        raise
    index["status"] = "complete"
    write_json(index_path, index)
    return run_dirs
