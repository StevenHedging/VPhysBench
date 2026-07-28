from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import import_prediction_video
from .baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from .io import load_json, load_jsonl, write_json
from .datasets import load_dataset_v2
from .orchestration import (
    build_task_instance,
    reevaluate_atomic,
    run_atomic,
    run_matrix,
)
from .prompts import PromptRegistry, SUPPORTED_PROMPT_PROFILES
from .runner import reevaluate_run, run_benchmark
from .splitters import build_view_a, build_view_b
from .task_planner import plan_task
from .validation import errors, load_scene_configs, validate_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = PROJECT_ROOT / "configs" / "scenes"
DEFAULT_METRICS = PROJECT_ROOT / "configs" / "metrics" / "default.json"
DEFAULT_PROMPTS = PROJECT_ROOT / "configs" / "prompts"


def _validate(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.manifest)
    scenes = load_scene_configs(args.scene_config_dir)
    issues = validate_cases(
        cases,
        manifest_path=args.manifest,
        scene_configs=scenes,
        check_assets=args.check_assets,
        asset_root=args.asset_root,
    )
    for issue in issues:
        print(json.dumps(issue.to_dict(), ensure_ascii=False))
    count = len(errors(issues))
    print(f"validated_cases={len(cases)} errors={count} warnings={len(issues)-count}")
    return 1 if count else 0


def _split(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.manifest)
    issues = validate_cases(cases, manifest_path=args.manifest)
    if errors(issues):
        raise ValueError("manifest must validate before splitting")
    split = build_view_a(cases) if args.view == "A" else build_view_b(cases, args.groups, args.seed)
    write_json(args.output, split)
    print(args.output)
    return 0


def _plan(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.manifest)
    plan = plan_task(
        load_json(args.task),
        cases,
        load_json(args.split),
        train_preview_per_scene=args.train_preview_per_scene,
        train_preview_seed=args.train_preview_seed,
        train_prompt_profile=args.train_prompt_profile,
        eval_prompt_profiles=args.eval_prompt_profile,
    )
    registry = PromptRegistry(args.prompt_config_dir)
    registry.require([
        plan["prompt_profiles"]["train"],
        *plan["prompt_profiles"]["eval"],
    ])
    by_id = {case["case_id"]: case for case in cases}
    train_profile = plan["prompt_profiles"]["train"]
    if train_profile:
        for case_id in plan["train_case_ids"]:
            registry.resolve(by_id[case_id], train_profile, role="train")
    for job in plan["jobs"]:
        registry.resolve(
            by_id[job["case_id"]], job["prompt_profile_id"], role="eval"
        )
    write_json(args.output, plan)
    print(f"train_cases={len(plan['train_case_ids'])} eval_jobs={len(plan['jobs'])} output={args.output}")
    return 0


def _run(args: argparse.Namespace) -> int:
    directory = run_benchmark(
        task_path=args.task,
        baseline_path=args.baseline,
        manifest_path=args.manifest,
        split_path=args.split,
        scene_config_dir=args.scene_config_dir,
        metric_config_path=args.metrics,
        output_root=args.output_root,
        execute=args.execute,
        run_id=args.run_id,
        stop_after_training=args.stop_after_training,
        train_preview_per_scene=args.train_preview_per_scene,
        train_preview_seed=args.train_preview_seed,
        prompt_config_dir=args.prompt_config_dir,
        train_prompt_profile=args.train_prompt_profile,
        eval_prompt_profiles=args.eval_prompt_profile,
    )
    print(directory)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    directory = Path(args.run_dir)
    summary = (
        reevaluate_atomic(directory)
        if (directory / "task_instance" / "manifest.json").is_file()
        else reevaluate_run(directory, args.scene_config_dir)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prediction_import(args: argparse.Namespace) -> int:
    record = import_prediction_video(
        source=args.source,
        run_dir=args.run_dir,
        baseline_id=args.baseline_id,
        case_id=args.case_id,
        job_id=args.job_id,
        seed=args.seed,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    fixture = PROJECT_ROOT / "examples" / "fixtures"
    run_id = "smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = run_benchmark(
        task_path=fixture / "task_view_a.json",
        baseline_path=PROJECT_ROOT / "configs" / "baselines" / "dummy_i2v.json",
        manifest_path=fixture / "cases.jsonl",
        split_path=fixture / "view_a.json",
        scene_config_dir=DEFAULT_SCENES,
        metric_config_path=DEFAULT_METRICS,
        output_root=args.output_root,
        run_id=run_id,
    )
    print(directory)
    return 0


def _validate_dataset_v2(args: argparse.Namespace) -> int:
    dataset = load_dataset_v2(
        args.dataset,
        check_assets=args.check_assets or args.check_asset_hashes,
        check_asset_hashes=args.check_asset_hashes,
    )
    print(
        f"dataset_id={dataset.dataset_id} cases={len(dataset.cases)} "
        f"scenes={len(dataset.scene_configs)} digest={dataset.digest}"
    )
    return 0


def _atomic_run(args: argparse.Namespace) -> int:
    directory = run_atomic(
        dataset_path=args.dataset,
        task_path=args.task,
        baseline_path=args.baseline,
        output_root=args.output_root,
        run_id=args.run_id,
        execute=args.execute,
        stop_after_training=args.stop_after_training,
        scene_ids=args.scene_id,
        groups=args.group,
        case_ids=args.case_id,
    )
    print(directory)
    return 0


def _task_build(args: argparse.Namespace) -> int:
    instance = build_task_instance(
        dataset_path=args.dataset,
        task_path=args.task,
        baseline_path=args.baseline,
        check_assets=not args.skip_asset_check,
    )
    write_json(args.output, instance.value)
    print(
        f"instance_id={instance.instance_id} digest={instance.digest} "
        f"output={args.output}"
    )
    return 0


def _matrix_run(args: argparse.Namespace) -> int:
    directories = run_matrix(
        dataset_path=args.dataset,
        task_paths=args.task,
        baseline_path=args.baseline,
        output_root=args.output_root,
        matrix_id=args.matrix_id,
        execute=args.execute,
        stop_after_training=args.stop_after_training,
    )
    for directory in directories:
        print(directory)
    return 0


def _baseline_list(args: argparse.Namespace) -> int:
    discovered = discover_baseline_bundles(args.root)
    records = []
    for baseline_id, path in sorted(discovered.items()):
        value = load_json(path)
        records.append({
            "baseline_id": baseline_id,
            "baseline_version": value.get("baseline_version"),
            "implementation_kind": value.get("implementation", {}).get("kind"),
            "descriptor_path": str(path),
        })
    print(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _baseline_inspect(args: argparse.Namespace) -> int:
    bundle = load_baseline_bundle(args.baseline, baselines_root=args.root)
    plugin = load_baseline_plugin(bundle)
    result = {
        "baseline_id": bundle.baseline_id,
        "baseline_version": bundle.baseline_version,
        "descriptor_path": str(bundle.descriptor_path),
        "root": str(bundle.root),
        "bundle_digest": bundle.digest,
        "deployment_digest": bundle.deployment_digest,
        "local_override_applied": (
            bundle.root / "baseline.local.json"
        ).is_file(),
        "implementation": bundle.value["implementation"],
        "capabilities": bundle.value["capabilities"],
        "supported_scenes": bundle.value.get("supported_scenes", "all"),
        "model": bundle.value.get("model", {}),
        "runtime": bundle.value.get("runtime", {}),
        "components": bundle.value.get("components", {}),
        "adapter_recipe": bundle.value.get("adapter"),
        "runner": bundle.value.get("runner"),
        "trainer": bundle.value.get("trainer"),
        "task_builder": plugin.task_builder.describe(),
        "data_adapter": plugin.task_builder.data_adapter.describe(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _baseline_validate(args: argparse.Namespace) -> int:
    bundle = load_baseline_bundle(args.baseline, baselines_root=args.root)
    plugin = load_baseline_plugin(bundle)
    result = {
        "status": "valid",
        "baseline_id": bundle.baseline_id,
        "baseline_version": bundle.baseline_version,
        "bundle_digest": bundle.digest,
        "deployment_digest": bundle.deployment_digest,
        "task_builder_fingerprint": plugin.task_builder.fingerprint,
        "data_adapter_fingerprint": plugin.task_builder.data_adapter.fingerprint,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _baseline_init(args: argparse.Namespace) -> int:
    from .baseline_runtime import create_baseline_scaffold

    directory = create_baseline_scaffold(
        name=args.name,
        backend=args.backend,
        root=args.root or (PROJECT_ROOT / "baselines"),
    )
    # A generated directory must satisfy the same Registry path as a real
    # integration before it is reported to the caller.
    load_baseline_plugin(load_baseline_bundle(directory))
    print(directory)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physbench")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a case manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--scene-config-dir", default=str(DEFAULT_SCENES))
    validate.add_argument("--check-assets", action="store_true")
    validate.add_argument("--asset-root")
    validate.set_defaults(func=_validate)

    split = sub.add_parser("split", help="build a frozen A or B view")
    split.add_argument("--view", choices=["A", "B"], required=True)
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--groups", type=int, default=5)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)

    plan = sub.add_parser("plan", help="materialize train and inference jobs")
    plan.add_argument("--task", required=True)
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--split", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--prompt-config-dir", default=str(DEFAULT_PROMPTS))
    plan.add_argument(
        "--train-prompt-profile", choices=SUPPORTED_PROMPT_PROFILES,
        help="prompt profile used to build finetuning metadata",
    )
    plan.add_argument(
        "--eval-prompt-profile", action="append", choices=SUPPORTED_PROMPT_PROFILES,
        help="repeat to evaluate the same model with multiple prompt profiles",
    )
    plan.add_argument(
        "--train-preview-per-scene", type=int,
        help="view A only: also schedule this many randomly selected seen training cases per scene",
    )
    plan.add_argument(
        "--train-preview-seed", type=int,
        help="seed used only for deterministic training-case preview sampling (default: 42)",
    )
    plan.set_defaults(func=_plan)

    run = sub.add_parser("run", help="run orchestration and placeholder evaluation")
    run.add_argument("--task", required=True)
    run.add_argument("--baseline", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--split", required=True)
    run.add_argument("--scene-config-dir", default=str(DEFAULT_SCENES))
    run.add_argument("--metrics", default=str(DEFAULT_METRICS))
    run.add_argument("--prompt-config-dir", default=str(DEFAULT_PROMPTS))
    run.add_argument(
        "--train-prompt-profile", choices=SUPPORTED_PROMPT_PROFILES,
        help="prompt profile used to build finetuning metadata",
    )
    run.add_argument(
        "--eval-prompt-profile", action="append", choices=SUPPORTED_PROMPT_PROFILES,
        help="repeat to evaluate the same adapter with multiple prompt profiles",
    )
    run.add_argument("--output-root", default="runs")
    run.add_argument("--run-id")
    run.add_argument("--execute", action="store_true", help="allow reviewed command adapters to execute")
    run.add_argument(
        "--stop-after-training", action="store_true",
        help="materialize inference jobs after training but leave generation staged",
    )
    run.add_argument(
        "--train-preview-per-scene", type=int,
        help="view A only: also infer this many randomly selected seen training cases per scene",
    )
    run.add_argument(
        "--train-preview-seed", type=int,
        help="seed used only for deterministic training-case preview sampling (default: 42)",
    )
    run.set_defaults(func=_run)

    evaluate = sub.add_parser("evaluate", help="re-evaluate a run after predictions or scores are added")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--scene-config-dir", default=str(DEFAULT_SCENES))
    evaluate.set_defaults(func=_evaluate)

    prediction_import = sub.add_parser(
        "prediction-import",
        help="copy an existing prediction into a run-owned artifact tree",
    )
    prediction_import.add_argument("--source", required=True)
    prediction_import.add_argument("--run-dir", required=True)
    prediction_import.add_argument("--baseline-id", required=True)
    prediction_import.add_argument("--case-id", required=True)
    prediction_import.add_argument("--job-id", required=True)
    prediction_import.add_argument("--seed", required=True, type=int)
    prediction_import.set_defaults(func=_prediction_import)

    smoke = sub.add_parser("smoke", help="run fixture end-to-end without models")
    smoke.add_argument("--output-root", default="runs/smoke")
    smoke.set_defaults(func=_smoke)

    validate_v2 = sub.add_parser(
        "validate-dataset", help="validate a prompt-free Dataset v2 bundle"
    )
    validate_v2.add_argument("--dataset", required=True)
    validate_v2.add_argument("--check-assets", action="store_true")
    validate_v2.add_argument(
        "--check-asset-hashes",
        action="store_true",
        help="also verify every referenced asset against assets.lock.json",
    )
    validate_v2.set_defaults(func=_validate_dataset_v2)

    atomic = sub.add_parser(
        "atomic-run", help="run Dataset × one atomic Task × Baseline"
    )
    atomic.add_argument("--dataset", required=True)
    atomic.add_argument("--task", required=True)
    atomic.add_argument("--baseline", required=True)
    atomic.add_argument("--output-root", default="runs_v2")
    atomic.add_argument("--run-id")
    atomic.add_argument("--scene-id", action="append")
    atomic.add_argument("--group", action="append")
    atomic.add_argument("--case-id", action="append")
    atomic.add_argument("--execute", action="store_true")
    atomic.add_argument("--stop-after-training", action="store_true")
    atomic.set_defaults(func=_atomic_run)

    task_build = sub.add_parser(
        "task-build",
        help="compile Dataset + Task into a sealed BaselineTaskInstance",
    )
    task_build.add_argument("--dataset", required=True)
    task_build.add_argument("--task", required=True)
    task_build.add_argument("--baseline", required=True)
    task_build.add_argument("--output", required=True)
    task_build.add_argument("--skip-asset-check", action="store_true")
    task_build.set_defaults(func=_task_build)

    matrix = sub.add_parser(
        "matrix-run",
        help="run paired atomic Tasks with identical Dataset/Baseline/splits",
    )
    matrix.add_argument("--dataset", required=True)
    matrix.add_argument("--task", action="append", required=True)
    matrix.add_argument("--baseline", required=True)
    matrix.add_argument("--output-root", default="runs_v2")
    matrix.add_argument("--matrix-id", required=True)
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--stop-after-training", action="store_true")
    matrix.set_defaults(func=_matrix_run)

    baseline = sub.add_parser(
        "baseline", help="discover and validate self-registering Baseline Bundles"
    )
    baseline_sub = baseline.add_subparsers(
        dest="baseline_command", required=True
    )
    baseline_list = baseline_sub.add_parser(
        "list", help="list discovered manifests without executing Baseline code"
    )
    baseline_list.add_argument("--root")
    baseline_list.set_defaults(func=_baseline_list)
    baseline_inspect = baseline_sub.add_parser(
        "inspect", help="inspect a Baseline Bundle and its declared components"
    )
    baseline_inspect.add_argument("baseline")
    baseline_inspect.add_argument("--root")
    baseline_inspect.set_defaults(func=_baseline_inspect)
    baseline_validate = baseline_sub.add_parser(
        "validate",
        help="validate a manifest, deployment, fingerprints and runtime",
    )
    baseline_validate.add_argument("baseline")
    baseline_validate.add_argument("--root")
    baseline_validate.set_defaults(func=_baseline_validate)
    baseline_init = baseline_sub.add_parser(
        "init", help="create a lightweight managed or submission Baseline"
    )
    baseline_init.add_argument("name")
    baseline_init.add_argument(
        "--backend",
        choices=["managed-i2v", "managed-v2v", "submission"],
        default="managed-i2v",
    )
    baseline_init.add_argument("--root")
    baseline_init.set_defaults(func=_baseline_init)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
