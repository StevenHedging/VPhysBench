#!/usr/bin/env python3
"""Run targeted collision-v5 prediction and GT self-consistency audits.

This standalone audit intentionally does not write process videos; those
belong to an AtomicRun evaluation and are enabled through the main CLI.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.reference import resolve_physics_reference
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected collision videos with scene_default_v5. "
            "Predictions use CASE_ID=VIDEO entries; --self-check also scores "
            "the immutable reference as a detector/evaluator sanity check."
        )
    )
    parser.add_argument("--dataset", type=Path, default=LATEST_DATASET)
    parser.add_argument("--protocol", default="scene_default_v5")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="CASE_ID=VIDEO",
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prediction_mapping(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        case_id, separator, path_value = value.partition("=")
        if not separator or not case_id or not path_value:
            raise ValueError(
                "--prediction must use the exact form CASE_ID=VIDEO"
            )
        if case_id in output:
            raise ValueError(f"duplicate prediction for case {case_id}")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[case_id] = path
    return output


def _job_id(case_id: str, variant: str) -> str:
    normalized = "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in case_id
    )
    return f"collision_v5_audit__{normalized}__{variant}"


def _reference_video(
    *,
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
) -> Path:
    """Resolve same-case or physics-parent GT through the evaluator contract."""
    request = CaseEvaluationRequest(
        job={
            "job_id": _job_id(case["case_id"], "reference_resolution"),
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
        },
        case=case,
        case_catalog=catalog,
        prediction=None,
        asset_root=asset_root,
        artifact_dir=Path("."),
        evaluator_config={},
    )
    return resolve_physics_reference(request)[0]


def _failure_record(
    *,
    case_id: str,
    variant: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "scene_id": "collision_1d",
        "status": "error",
        "score": None,
        "reason_code": getattr(exc, "code", "audit_execution_failed"),
        "reason": str(exc),
        "audit_variant": variant,
        "audit_error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _selected_case_ids(
    case_ids: list[str],
    predictions: dict[str, Path],
    *,
    self_check: bool,
) -> list[str]:
    selected = list(dict.fromkeys([*case_ids, *predictions]))
    if not selected:
        raise ValueError("select at least one --case-id or --prediction")
    unscored = [
        case_id
        for case_id in case_ids
        if not self_check and case_id not in predictions
    ]
    if unscored:
        raise ValueError(
            "--case-id entries without predictions require --self-check: "
            + ", ".join(sorted(set(unscored)))
        )
    return selected


def _collision_v5_config(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    try:
        config = protocol["scenes"]["collision_1d"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "audit protocol has no collision_1d scene"
        ) from exc
    if config.get("type") != "collision_1d_state_v5":
        raise ValueError(
            "audit_collision_evaluator_v5 requires "
            "collision_1d_state_v5, got "
            f"{config.get('type')!r}"
        )
    return config


def _digest_files(root: Path, paths: list[Path]) -> dict[str, Any]:
    root = root.resolve()
    unique = sorted({path.resolve() for path in paths})
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in unique:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"source snapshot path escapes project root: {path}"
            ) from exc
        payload = path.read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        relative_text = relative.as_posix()
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
        digest.update(b"\0")
        records.append(
            {
                "path": relative_text,
                "sha256": file_digest,
                "size_bytes": len(payload),
            }
        )
    return {
        "algorithm": "sha256(path_nul_file_sha256_nul)",
        "digest": digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _implementation_snapshot(protocol_path: Path) -> dict[str, Any]:
    source_paths = sorted(
        (PROJECT_ROOT / "src" / "physbench").rglob("*.py")
    )
    source_paths.extend(
        [
            Path(__file__).resolve(),
            protocol_path.resolve(),
            (
                PROJECT_ROOT
                / "schemas"
                / "v2"
                / "evaluation_protocol.schema.json"
            ).resolve(),
        ]
    )
    snapshot = _digest_files(PROJECT_ROOT, source_paths)
    git: dict[str, Any] = {
        "head": None,
        "dirty": None,
        "status_entries": None,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        git = {
            "head": head,
            "dirty": bool(status),
            "status_entries": len(status),
        }
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"source_tree": snapshot, "git": git}


def _evaluate(
    evaluator: Any,
    *,
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
    video_path: Path,
    output: Path,
    variant: str,
    evaluator_config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    job_id = _job_id(case["case_id"], variant)
    artifact_dir = output / "cases" / job_id
    request = CaseEvaluationRequest(
        job={
            "job_id": job_id,
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
        },
        case=case,
        case_catalog=catalog,
        prediction={
            "job_id": job_id,
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "status": "complete",
            "video_path": str(video_path),
        },
        asset_root=asset_root,
        artifact_dir=artifact_dir,
        evaluator_config=evaluator_config,
        run_id=run_id,
        save_visualizations=False,
    )
    result = evaluator.evaluate(request).to_dict()
    result["audit_variant"] = variant
    write_json(artifact_dir / "result.json", result)
    return result


def main() -> None:
    args = _arguments()
    predictions = _prediction_mapping(args.prediction)
    dataset = load_dataset(args.dataset)
    catalog = {case["case_id"]: case for case in dataset.cases}
    selected = _selected_case_ids(
        args.case_id,
        predictions,
        self_check=args.self_check,
    )
    for case_id in selected:
        if case_id not in catalog:
            raise KeyError(f"case is absent from the frozen dataset: {case_id}")
        if catalog[case_id]["scene_id"] != "collision_1d":
            raise ValueError(f"case is not collision_1d: {case_id}")

    protocol = copy.deepcopy(load_evaluation_protocol(args.protocol))
    implementation = _implementation_snapshot(Path(protocol["path"]))
    collision_config = _collision_v5_config(protocol)
    collision_config["sam2"]["device"] = args.device
    evaluator = SceneEvaluatorRegistry(protocol).resolve("collision_1d")
    output = args.output.expanduser().resolve()
    run_id = args.run_id or output.name
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for case_id in selected:
        case = catalog[case_id]
        variants: list[tuple[str, Path]] = []
        if args.self_check:
            try:
                variants.append(
                    (
                        "gt_self",
                        _reference_video(
                            case=case,
                            catalog=catalog,
                            asset_root=dataset.asset_root,
                        ),
                    )
                )
            except Exception as exc:
                result = _failure_record(
                    case_id=case_id,
                    variant="gt_self",
                    exc=exc,
                )
                records.append(result)
                write_json(
                    output
                    / "cases"
                    / _job_id(case_id, "gt_self")
                    / "result.json",
                    result,
                )
                print(
                    f"[{len(records)}] {case_id}/gt_self: "
                    f"error score=None",
                    flush=True,
                )
        if case_id in predictions:
            variants.append(("prediction", predictions[case_id]))
        for variant, video_path in variants:
            try:
                result = _evaluate(
                    evaluator,
                    case=case,
                    catalog=catalog,
                    asset_root=dataset.asset_root,
                    video_path=video_path,
                    output=output,
                    variant=variant,
                    evaluator_config=collision_config,
                    run_id=run_id,
                )
            except Exception as exc:
                result = _failure_record(
                    case_id=case_id,
                    variant=variant,
                    exc=exc,
                )
                write_json(
                    output
                    / "cases"
                    / _job_id(case_id, variant)
                    / "result.json",
                    result,
                )
            records.append(result)
            print(
                f"[{len(records)}] {case_id}/{variant}: "
                f"{result['status']} score={result['score']}",
                flush=True,
            )

    implementation_end = _implementation_snapshot(Path(protocol["path"]))
    implementation["stable_during_run"] = (
        implementation["source_tree"]["digest"]
        == implementation_end["source_tree"]["digest"]
    )
    implementation["end_source_tree_digest"] = implementation_end[
        "source_tree"
    ]["digest"]
    implementation["git_at_end"] = implementation_end["git"]
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "digest": dataset.digest,
            "descriptor": str(Path(args.dataset).resolve()),
        },
        "protocol": {
            "id": protocol["protocol_id"],
            "fingerprint": protocol["fingerprint"],
            "runtime_overrides": {
                "sam2.device": args.device,
                "save_visualizations": False,
            },
        },
        "implementation": implementation,
        "evaluator": evaluator.describe(),
        "records": records,
        "summary": {
            "records": len(records),
            "evaluated": sum(
                value["status"] == "evaluated" for value in records
            ),
            "errors": sum(value["status"] == "error" for value in records),
            "unavailable": sum(
                value["status"] == "unavailable" for value in records
            ),
        },
    }
    write_json(output / "audit_report.json", report)
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
