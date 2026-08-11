from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .artifacts import import_prediction_video
from .baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from .io import load_json, load_jsonl, write_json
from .datasets import load_dataset
from .dataset_hub import diagnose_project, diagnostics_succeeded, pull_dataset
from .runner import reevaluate_run
from .splitters import build_view_a, build_view_b
from .validation import errors, load_scene_configs, validate_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = PROJECT_ROOT / "configs" / "scenes"
ATOMIC_RUN_MARKERS = (
    "component_fingerprints.json",
    "frozen",
    "task_instance",
)


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


def _evaluate(args: argparse.Namespace) -> int:
    from .orchestration import reevaluate_atomic_variant

    directory = Path(args.run_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"run directory is not a directory: {directory}")
    run_path = directory / "run.json"
    if run_path.is_symlink():
        raise ValueError("run.json must not be a symlink")
    run = load_json(run_path) if run_path.is_file() else {}
    schema_version = run.get("schema_version")
    has_atomic_marker = any(
        os.path.lexists(directory / marker)
        for marker in ATOMIC_RUN_MARKERS
    )
    is_atomic = schema_version == "2.0" or has_atomic_marker
    if is_atomic:
        if not args.protocol_id or not args.evaluation_id:
            raise ValueError(
                "AtomicRun evaluation requires both --protocol-id and "
                "--evaluation-id; canonical evaluation is never overwritten"
            )
        result = reevaluate_atomic_variant(
            directory,
            protocol_id=args.protocol_id,
            evaluation_id=args.evaluation_id,
            save_visualizations=args.save_visualizations,
        )
    else:
        if schema_version != "1.0":
            raise ValueError(
                "evaluate requires either a schema_version=2.0 AtomicRun or "
                "a schema_version=1.0 legacy run"
            )
        if args.protocol_id or args.evaluation_id or args.save_visualizations:
            raise ValueError(
                "--protocol-id, --evaluation-id and --save-visualizations "
                "are only valid for AtomicRun directories"
            )
        result = reevaluate_run(directory, args.scene_config_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
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


def _validate_dataset(args: argparse.Namespace) -> int:
    dataset = load_dataset(
        args.dataset,
        check_assets=args.check_assets or args.check_asset_hashes,
        check_asset_hashes=args.check_asset_hashes,
    )
    print(
        f"dataset_id={dataset.dataset_id} cases={len(dataset.cases)} "
        f"scenes={len(dataset.scene_configs)} digest={dataset.digest}"
    )
    return 0


def _dataset_pull(args: argparse.Namespace) -> int:
    descriptor = pull_dataset(
        args.binding,
        local_dir=args.local_dir,
        check_assets=not args.skip_asset_check,
    )
    print(descriptor)
    return 0


def _doctor(args: argparse.Namespace) -> int:
    diagnostics = diagnose_project(args.project_root, level=args.level)
    for item in diagnostics:
        print(json.dumps({
            "name": item.name,
            "status": item.status,
            "detail": item.detail,
        }, ensure_ascii=False, sort_keys=True))
    return 0 if diagnostics_succeeded(diagnostics) else 1


def _atomic_run(args: argparse.Namespace) -> int:
    from .orchestration import run_atomic

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
        save_visualizations=args.save_visualizations,
    )
    print(directory)
    return 0


def _task_build(args: argparse.Namespace) -> int:
    from .orchestration import build_task_instance

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
    from .orchestration import run_matrix

    directories = run_matrix(
        dataset_path=args.dataset,
        task_path=args.task,
        baseline_paths=args.baseline,
        output_root=args.output_root,
        matrix_id=args.matrix_id,
        execute=args.execute,
        stop_after_training=args.stop_after_training,
        save_visualizations=args.save_visualizations,
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
        "input_policy": bundle.value["input_policy"],
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

    evaluate = sub.add_parser(
        "evaluate",
        help="create a coexisting AtomicRun evaluation variant",
    )
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--protocol-id")
    evaluate.add_argument("--evaluation-id")
    evaluate.add_argument(
        "--save-visualizations",
        action="store_true",
        help="save external per-Case visualization videos (default: off)",
    )
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

    validate_v2 = sub.add_parser(
        "validate-dataset", help="validate the current Dataset bundle"
    )
    validate_v2.add_argument("--dataset", required=True)
    validate_v2.add_argument("--check-assets", action="store_true")
    validate_v2.add_argument(
        "--check-asset-hashes",
        action="store_true",
        help="also verify every referenced asset against assets.lock.json",
    )
    validate_v2.set_defaults(func=_validate_dataset)

    dataset = sub.add_parser(
        "dataset", help="download the immutable bound Dataset release"
    )
    dataset_sub = dataset.add_subparsers(
        dest="dataset_command", required=True
    )
    dataset_pull = dataset_sub.add_parser(
        "pull", help="download and validate the bound Hugging Face Dataset"
    )
    dataset_pull.add_argument(
        "--binding",
        default=str(PROJECT_ROOT / "datasets" / "huggingface.json"),
    )
    dataset_pull.add_argument(
        "--local-dir",
        default=str(PROJECT_ROOT / "datasets"),
    )
    dataset_pull.add_argument(
        "--skip-asset-check",
        action="store_true",
        help=(
            "prefetch and fully validate assets in staging without publishing "
            "them as active"
        ),
    )
    dataset_pull.set_defaults(func=_dataset_pull)

    doctor = sub.add_parser(
        "doctor", help="diagnose metadata or full evaluation readiness"
    )
    doctor.add_argument(
        "--level",
        choices=["metadata", "evaluation"],
        default="metadata",
    )
    doctor.add_argument("--project-root", default=str(PROJECT_ROOT))
    doctor.set_defaults(func=_doctor)

    atomic = sub.add_parser(
        "atomic-run", help="run Dataset × one atomic Task × Baseline"
    )
    atomic.add_argument("--dataset", required=True)
    atomic.add_argument("--task", required=True)
    atomic.add_argument("--baseline", required=True)
    atomic.add_argument("--output-root", default="run")
    atomic.add_argument("--run-id")
    atomic.add_argument("--scene-id", action="append")
    atomic.add_argument("--group", action="append")
    atomic.add_argument("--case-id", action="append")
    atomic.add_argument("--execute", action="store_true")
    atomic.add_argument("--stop-after-training", action="store_true")
    atomic.add_argument(
        "--save-visualizations",
        action="store_true",
        help="save external per-Case visualization videos (default: off)",
    )
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
        help="run one Task across multiple Baseline identities",
    )
    matrix.add_argument("--dataset", required=True)
    matrix.add_argument("--task", required=True)
    matrix.add_argument("--baseline", action="append", required=True)
    matrix.add_argument("--output-root", default="run")
    matrix.add_argument("--matrix-id", required=True)
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--stop-after-training", action="store_true")
    matrix.add_argument(
        "--save-visualizations",
        action="store_true",
        help="save external per-Case visualization videos (default: off)",
    )
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
