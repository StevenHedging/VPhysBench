from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..artifacts import ARTIFACT_POLICY, validate_prediction_records
from ..baseline_api import load_baseline_bundle, load_baseline_plugin
from ..datasets import load_dataset
from ..domain import TaskSpec
from ..evaluation import load_evaluation_protocol
from ..identifiers import require_safe_id
from ..io import (
    canonical_sha256,
    load_json,
    write_json,
    write_jsonl,
)
from ..tasks import load_task
from .task_compiler import build_task_instance, compile_task_instance


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
        f"- Baseline：`{run['baseline_id']}` / bundle "
        f"`{run['baseline_digest']}` / deployment "
        f"`{run['baseline_deployment_digest']}`\n",
        f"- Task instance：`{run['task_instance_id']}` / "
        f"`{run['task_instance_digest']}`\n",
        f"- Training seed：`{run['training_seed']}`\n",
        f"- Train cases：`{len(plan['train_case_ids'])}`\n",
        f"- Inference jobs：`{len(plan['jobs'])}`\n",
        f"- Visualization videos：`{run['save_visualizations']}`\n",
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
        "- 原始文本是 Dataset Case 的冻结属性。\n",
        "- 同一份可条件化 Case 由 Baseline-owned DataAdapter 转换。\n",
        "- 是否使用物理信息及其表示由 Baseline input policy 固定声明。\n",
        "- Trainer/Predictor 只消费已封印的 BaselineTaskInstance。\n",
        "- 本 AtomicRun 对应一个不可变 Baseline identity 和一份模型产物。\n",
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
    save_visualizations: bool = False,
) -> Path:
    dataset = load_dataset(dataset_path, check_assets=check_assets)
    task = load_task(task_path)
    protocol_id = task.value.get("evaluation", {}).get(
        "protocol", "scene_default_v1"
    )
    evaluation_protocol = load_evaluation_protocol(protocol_id)
    if scene_ids is not None or groups is not None or case_ids is not None:
        value = copy.deepcopy(task.value)
        if scene_ids is not None:
            value["selection"]["evaluation_scene_ids"] = list(
                dict.fromkeys(scene_ids)
            )
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
    instance = compile_task_instance(plugin, dataset, task)
    instance.verify()
    plan = instance.canonical_plan

    identifier = run_id or _run_id(task.task_id, baseline.baseline_id)
    require_safe_id(identifier, label="run_id")
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
        run_dir / "logs",
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
        "baseline_deployment": baseline.deployment_digest,
        "task_builder": plugin.task_builder.fingerprint,
        "task_instance": instance.digest,
        "data_adapter": plugin.task_builder.data_adapter.fingerprint,
        "data_adapter_materialization": (
            plugin.task_builder.data_adapter.materialization_fingerprint
        ),
        "evaluation_protocol": evaluation_protocol["fingerprint"],
    })
    write_json(run_dir / "artifact_policy.json", {
        "schema_version": "1.0",
        **ARTIFACT_POLICY,
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
        prediction_artifacts = validate_prediction_records(
            predictions,
            jobs=instance.value["inference"]["jobs"],
            baseline_id=baseline.baseline_id,
            run_dir=run_dir,
        )
    except BaseException as exc:
        _state(run_dir, "failed", error=repr(exc))
        raise
    _state(run_dir, "training_complete_or_staged", status=training.get("status"))
    write_json(
        run_dir / "artifacts" / "prediction_artifacts.json",
        prediction_artifacts,
    )
    write_jsonl(run_dir / "predictions.jsonl", predictions)

    # Scene evaluators depend on the optional scene-evaluation extra.  Keep
    # their import in the evaluation phase so planning and execution staging
    # remain usable in lightweight installations.
    from ..evaluation import evaluate_task

    case_metrics, summary = evaluate_task(
        plan=plan.value,
        cases=list(dataset.cases),
        predictions=predictions,
        asset_root=dataset.asset_root,
        protocol=evaluation_protocol,
        output_dir=run_dir / "evaluation",
        run_id=identifier,
        save_visualizations=save_visualizations,
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
        "baseline_id": baseline.baseline_id,
        "baseline_version": baseline.baseline_version,
        "baseline_digest": baseline.digest,
        "baseline_deployment_digest": baseline.deployment_digest,
        "task_instance_id": instance.instance_id,
        "task_instance_digest": instance.digest,
        "task_builder_fingerprint": plugin.task_builder.fingerprint,
        "input_policy": baseline.value["input_policy"],
        "training_seed": plan.value["training_seed"],
        "execute": execute,
        "status": status,
        "evaluation_status": summary["status"],
        "evaluation_coverage": summary["coverage"],
        "evaluation_score": summary["score"],
        "save_visualizations": bool(save_visualizations),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "run.json", run)
    (run_dir / "report.md").write_text(
        _render_atomic_report(run, plan.value, summary), encoding="utf-8"
    )
    _state(run_dir, status)
    return run_dir


def reevaluate_atomic(run_dir: str | Path) -> dict[str, Any]:
    """Fail closed instead of overwriting a schema-v2 AtomicRun.

    The symbol remains exported so older Python callers receive an actionable
    error.  Use ``reevaluate_atomic_variant`` for AtomicRun v2.  Historical
    schema-v1 directories use ``physbench.runner.reevaluate_run``.
    """
    raise RuntimeError(
        "in-place AtomicRun reevaluation is forbidden; use "
        "reevaluate_atomic_variant(run_dir, protocol_id=..., "
        "evaluation_id=...) or `physbench evaluate --protocol-id ... "
        "--evaluation-id ...`"
    )


def _paired_plan_signature(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": plan["dataset_id"],
        "dataset_digest": plan["dataset_digest"],
        "training_scene_ids": plan["training_scene_ids"],
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
    task_path: str | Path,
    baseline_paths: list[str | Path],
    output_root: str | Path,
    matrix_id: str,
    execute: bool = False,
    stop_after_training: bool = False,
    save_visualizations: bool = False,
) -> list[Path]:
    if len(baseline_paths) < 2:
        raise ValueError("a task matrix requires at least two Baselines")
    require_safe_id(matrix_id, label="matrix_id")
    dataset = load_dataset(dataset_path, check_assets=True)
    task = load_task(task_path)
    baselines = [load_baseline_bundle(path) for path in baseline_paths]
    baseline_ids = [baseline.baseline_id for baseline in baselines]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("task matrix contains duplicate Baseline identities")
    plugins = [load_baseline_plugin(baseline) for baseline in baselines]
    instances = [
        compile_task_instance(plugin, dataset, task)
        for plugin in plugins
    ]
    plans = [instance.canonical_plan.value for instance in instances]
    first_signature = _paired_plan_signature(plans[0])
    if any(_paired_plan_signature(plan) != first_signature for plan in plans[1:]):
        raise ValueError(
            "Baseline matrix must use identical data, splits, seeds, and "
            "evaluation cases"
        )
    output = Path(output_root).resolve()
    index_path = output / f"{matrix_id}.matrix.json"
    if index_path.exists():
        raise FileExistsError(f"matrix index already exists: {index_path}")
    index = {
        "schema_version": "2.0",
        "matrix_id": matrix_id,
        "elements": ["dataset", "task", "baselines"],
        "dataset": str(Path(dataset_path).resolve()),
        "task": str(task.path),
        "baselines": [
            {
                "descriptor": str(baseline.descriptor_path),
                "baseline_id": baseline.baseline_id,
                "digest": baseline.digest,
                "deployment_digest": baseline.deployment_digest,
                "input_policy": baseline.value["input_policy"],
                "task_builder_fingerprint": plugin.task_builder.fingerprint,
            }
            for baseline, plugin in zip(baselines, plugins, strict=True)
        ],
        "task_instances": [
            {
                "instance_id": instance.instance_id,
                "digest": instance.digest,
            }
            for instance in instances
        ],
        "atomic_runs": [],
        "paired_plan_signature": first_signature,
        "status": "running",
        "orchestration_status": "running",
        "atomic_run_statuses": {},
        "save_visualizations": bool(save_visualizations),
    }
    write_json(index_path, index)
    run_dirs = []
    try:
        for baseline, expected_instance in zip(
            baselines,
            instances,
            strict=True,
        ):
            run_dir = run_atomic(
                dataset_path=dataset_path,
                task_path=task.path,
                baseline_path=baseline.descriptor_path,
                output_root=output,
                run_id=f"{matrix_id}__{baseline.baseline_id}",
                execute=execute,
                stop_after_training=stop_after_training,
                save_visualizations=save_visualizations,
            )
            actual_run = load_json(run_dir / "run.json")
            if (
                actual_run.get("task_instance_digest")
                != expected_instance.digest
            ):
                raise RuntimeError(
                    "matrix preflight TaskInstance differs from the "
                    f"executed AtomicRun for {baseline.baseline_id}"
                )
            run_dirs.append(run_dir)
            index["atomic_runs"] = [str(path) for path in run_dirs]
            index["atomic_run_statuses"][baseline.baseline_id] = (
                actual_run["status"]
            )
            write_json(index_path, index)
    except BaseException as exc:
        index["status"] = "failed"
        index["orchestration_status"] = "failed"
        index["error"] = repr(exc)
        write_json(index_path, index)
        raise
    statuses = set(index["atomic_run_statuses"].values())
    index["orchestration_status"] = "complete"
    if statuses == {"complete"}:
        index["status"] = "complete"
    elif statuses == {"planned"}:
        index["status"] = "planned"
    elif statuses == {"training_complete_inference_staged"}:
        index["status"] = "training_complete_inference_staged"
    else:
        index["status"] = "incomplete"
    write_json(index_path, index)
    return run_dirs
