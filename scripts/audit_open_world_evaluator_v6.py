#!/usr/bin/env python3
"""Run targeted open-world-v2 prediction and reference audits.

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
import hashlib
import math
from pathlib import Path
from statistics import mean, median
import subprocess
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected non-collision open-world cases. "
            "Predictions use CASE_ID=VIDEO entries; --self-check scores a "
            "same-case reference as GT-self, while OOD physics parents are "
            "reported separately as capability probes."
        )
    )
    parser.add_argument("--dataset", type=Path, default=LATEST_DATASET)
    parser.add_argument("--protocol", default="scene_default_v6")
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        choices=sorted(SUPPORTED_SCENES),
        help=(
            "select every frozen Dataset case in a non-collision scene; "
            "repeat for multiple scenes"
        ),
    )
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


def _case_ids_for_scenes(
    cases: list[dict[str, Any]],
    scene_ids: list[str],
) -> list[str]:
    selected = set(scene_ids)
    return [
        case["case_id"]
        for case in cases
        if case["scene_id"] in selected
    ]


def _job_id(protocol_id: str, case_id: str, variant: str) -> str:
    normalized_protocol = "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in protocol_id
    )
    normalized_case = "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in case_id
    )
    return (
        f"open_world_v2_audit__{normalized_protocol}__"
        f"{normalized_case}__{variant}"
    )


def _self_check_variant(
    *,
    reference_mode: str | None = None,
    case: dict[str, Any] | None = None,
) -> str:
    """Name a reference-as-prediction probe without overstating its meaning.

    A same-case immutable reference is a genuine GT-self check.  For an OOD
    case, however, ``resolve_physics_reference`` returns a physics-identical
    parent whose pixels, background, apparatus, and initial gauge may differ
    from the child.  Scoring that parent as the child prediction is useful as
    a capability/isolation probe, but it is not GT-self and must never be
    pooled into GT-self calibration statistics.
    """

    if reference_mode == "same_case_reference":
        return "gt_self"
    if reference_mode == "parent_physics_reference":
        return "physics_parent_as_prediction"
    if case is not None:
        if (
            case.get("has_real_reference_video", False)
            and case.get("assets", {}).get("physics_reference_video")
        ):
            return "gt_self"
        if case.get("provenance", {}).get("parent_case_id"):
            return "physics_parent_as_prediction"
    return "reference_resolution_failure"


def _reference_resolution(
    *,
    protocol_id: str,
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
) -> tuple[Path, str, str | None]:
    request = CaseEvaluationRequest(
        job={
            "job_id": _job_id(
                protocol_id,
                case["case_id"],
                "reference_resolution",
            ),
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


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    finite_scores = [
        float(record["score"])
        for record in records
        if record.get("status") == "evaluated"
        and isinstance(record.get("score"), (int, float))
        and not isinstance(record.get("score"), bool)
        and math.isfinite(float(record["score"]))
    ]
    return {
        "records": len(records),
        "evaluated": sum(
            record.get("status") == "evaluated" for record in records
        ),
        "errors": sum(
            record.get("status") == "error" for record in records
        ),
        "unavailable": sum(
            record.get("status") == "unavailable" for record in records
        ),
        "finite_scores": len(finite_scores),
        "score_min": min(finite_scores) if finite_scores else None,
        "score_mean": mean(finite_scores) if finite_scores else None,
        "score_median": median(finite_scores) if finite_scores else None,
        "score_max": max(finite_scores) if finite_scores else None,
    }


def _digest_files(
    root: Path,
    paths: list[Path],
) -> dict[str, Any]:
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
        # The content digest above is authoritative even in an exported
        # source tree without Git metadata.
        pass
    return {
        "source_tree": snapshot,
        "git": git,
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
    if not str(config.get("type", "")).endswith(("_v6", "_v7")):
        raise ValueError(
            f"{scene_id} is not routed to a supported open-world evaluator"
        )
    if isinstance(config.get("sam2"), dict):
        config["sam2"]["device"] = device
    return config


def _evaluate(
    evaluator: Any,
    *,
    protocol_id: str,
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
    job_id = _job_id(protocol_id, case["case_id"], variant)
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
    scene_case_ids = _case_ids_for_scenes(dataset.cases, args.scene)
    selected = _selected_case_ids(
        [*args.case_id, *scene_case_ids],
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
    implementation = _implementation_snapshot(Path(protocol["path"]))
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
            self_check_variant = _self_check_variant(case=case)
            try:
                reference, mode, parent_id = _reference_resolution(
                    protocol_id=protocol["protocol_id"],
                    case=case,
                    catalog=catalog,
                    asset_root=dataset.asset_root,
                )
                self_check_variant = _self_check_variant(
                    reference_mode=mode,
                    case=case,
                )
                variants.append(
                    (self_check_variant, reference, mode, parent_id)
                )
            except Exception as exc:
                result = _failure_record(
                    case=case,
                    variant=self_check_variant,
                    exc=exc,
                )
                records.append(result)
                write_json(
                    output
                    / "cases"
                    / _job_id(
                        protocol["protocol_id"],
                        case_id,
                        self_check_variant,
                    )
                    / "result.json",
                    result,
                )
                print(
                    f"[{len(records)}] {case_id}/{self_check_variant}: "
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
                    protocol_id=protocol["protocol_id"],
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
                    / _job_id(
                        protocol["protocol_id"], case_id, variant
                    )
                    / "result.json",
                    result,
                )
            records.append(result)
            print(
                f"[{len(records)}] {case_id}/{variant}: "
                f"{result['status']} score={result['score']}",
                flush=True,
            )

    variants = sorted(
        {str(record["audit_variant"]) for record in records}
    )
    by_variant = {
        variant: _record_summary(
            [
                record
                for record in records
                if record["audit_variant"] == variant
            ]
        )
        for variant in variants
    }
    scenes = sorted({str(record["scene_id"]) for record in records})
    by_scene = {
        scene_id: _record_summary(
            [
                record
                for record in records
                if record["scene_id"] == scene_id
            ]
        )
        for scene_id in scenes
    }
    by_scene_variant = {
        f"{scene_id}/{variant}": _record_summary(
            [
                record
                for record in records
                if record["scene_id"] == scene_id
                and record["audit_variant"] == variant
            ]
        )
        for scene_id in scenes
        for variant in variants
        if any(
            record["scene_id"] == scene_id
            and record["audit_variant"] == variant
            for record in records
        )
    }
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
            },
        },
        "implementation": implementation,
        "evaluators": evaluators,
        "records": records,
        "summary": {
            **_record_summary(records),
            "by_variant": by_variant,
            "by_scene": by_scene,
            "by_scene_variant": by_scene_variant,
        },
        "self_check_semantics": {
            "gt_self": (
                "same-case immutable reference scored as its own prediction"
            ),
            "physics_parent_as_prediction": (
                "physics-identical parent pixels scored as an OOD child "
                "prediction; capability/isolation probe, not GT-self"
            ),
        },
    }
    write_json(output / "audit_report.json", report)
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
