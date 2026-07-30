#!/usr/bin/env python3
"""Run targeted scene-default-v6 prediction and reference audits.

This entry point covers the four non-collision ``open_world_v2`` evaluators.
Collision remains frozen on evaluator 2.2 and has its own v5 audit script.
Large overlays are written through each evaluator's configured external
visualization root; the local audit directory contains scores, curves, and
the machine-readable open-world audit.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.reference import resolve_physics_reference
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.io import write_json


SUPPORTED_SCENES = {
    "pendulum",
    "free_fall",
    "inclined_plane_slide",
    "uniform_circular_motion",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected non-collision cases with scene_default_v6. "
            "Predictions use CASE_ID=VIDEO entries; --self-check also scores "
            "the resolved immutable reference."
        )
    )
    parser.add_argument("--dataset", type=Path, default=LATEST_DATASET)
    parser.add_argument("--protocol", default="scene_default_v6")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="CASE_ID=VIDEO",
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--device", default="auto")
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


def _job_id(case_id: str, variant: str) -> str:
    normalized = "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in case_id
    )
    return f"open_world_v6_audit__{normalized}__{variant}"


def _reference_resolution(
    *,
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
) -> tuple[Path, str, str | None]:
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
    return resolve_physics_reference(request)


def _failure_record(
    *,
    case: dict[str, Any],
    variant: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
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


def _scene_config(
    protocol: dict[str, Any],
    *,
    scene_id: str,
    device: str,
) -> dict[str, Any]:
    if scene_id not in SUPPORTED_SCENES:
        raise ValueError(
            f"scene {scene_id!r} is not a non-collision v6 audit scene"
        )
    config = protocol["scenes"][scene_id]
    if config.get("observer_protocol") != "open_world_v2":
        raise ValueError(
            f"{scene_id} is not configured for open_world_v2"
        )
    if not str(config.get("type", "")).endswith("_v6"):
        raise ValueError(f"{scene_id} is not routed to a v6 evaluator")
    if isinstance(config.get("sam2"), dict):
        config["sam2"]["device"] = device
    return config


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
    reference_mode: str | None = None,
    reference_parent_id: str | None = None,
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
    )
    result = evaluator.evaluate(request).to_dict()
    result["audit_variant"] = variant
    if reference_mode is not None:
        result["audit_reference"] = {
            "mode": reference_mode,
            "parent_case_id": reference_parent_id,
        }
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
        scene_id = catalog[case_id]["scene_id"]
        if scene_id not in SUPPORTED_SCENES:
            raise ValueError(
                f"case {case_id} belongs to unsupported scene {scene_id}"
            )

    protocol = copy.deepcopy(load_evaluation_protocol(args.protocol))
    registry = SceneEvaluatorRegistry(protocol)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    evaluators: dict[str, dict[str, Any]] = {}

    for case_id in selected:
        case = catalog[case_id]
        scene_id = case["scene_id"]
        config = _scene_config(
            protocol,
            scene_id=scene_id,
            device=args.device,
        )
        evaluator = registry.resolve(scene_id)
        evaluators[scene_id] = evaluator.describe()
        variants: list[
            tuple[str, Path, str | None, str | None]
        ] = []
        if args.self_check:
            try:
                reference, mode, parent_id = _reference_resolution(
                    case=case,
                    catalog=catalog,
                    asset_root=dataset.asset_root,
                )
                variants.append(
                    ("gt_self", reference, mode, parent_id)
                )
            except Exception as exc:
                result = _failure_record(
                    case=case,
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
                    "error score=None",
                    flush=True,
                )
        if case_id in predictions:
            variants.append(
                ("prediction", predictions[case_id], None, None)
            )
        for variant, video_path, mode, parent_id in variants:
            try:
                result = _evaluate(
                    evaluator,
                    case=case,
                    catalog=catalog,
                    asset_root=dataset.asset_root,
                    video_path=video_path,
                    output=output,
                    variant=variant,
                    evaluator_config=config,
                    reference_mode=mode,
                    reference_parent_id=parent_id,
                )
            except Exception as exc:
                result = _failure_record(
                    case=case,
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
            },
        },
        "evaluators": evaluators,
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
