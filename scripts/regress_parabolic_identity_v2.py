#!/usr/bin/env python3
"""Run frozen false-high and positive regressions for parabolic identity v2."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.io import load_json, load_jsonl, sha256_file, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIR_RUN = Path(
    "/root/Steven/VPhysBench/run/wan22_pair_text_cross_attention_2184_v2"
)
PAIR_PREDICTION_ROOT = (
    PAIR_RUN
    / "predictions/wan22_ti2v_5b_lora_r32_pair_text_cross_attention_v1"
)
REGRESSION_CASES: Mapping[str, Mapping[str, object]] = {
    "parabolic_img_0592": {
        "prediction_path": str(
            PAIR_PREDICTION_ROOT
            / "seven_scene_symbol_value_finetune_eval_v13__"
            "parabolic_img_0592__seed000042.mp4"
        ),
        "old_expert_score": 0.5656459893608466,
        "maximum_score": 0.25,
    },
    "parabolic_img_0623": {
        "prediction_path": str(
            PAIR_PREDICTION_ROOT
            / "seven_scene_symbol_value_finetune_eval_v13__"
            "parabolic_img_0623__seed000042.mp4"
        ),
        "old_expert_score": 0.5270753640342365,
        "maximum_score": 0.25,
    },
}

PHYSICS_RUN = Path(
    "/root/Steven/VPhysBench/run/wan22_physics_text_lora_2184_v1"
)
PHYSICS_PREDICTION_ROOT = (
    PHYSICS_RUN / "predictions/wan22_ti2v_5b_lora_r32_physics_text_v1"
)
REVIEWED_GOOD_CONTROLS: Mapping[str, Mapping[str, object]] = {
    "parabolic_img_0577": {
        "prediction_path": str(
            PHYSICS_PREDICTION_ROOT
            / "seven_scene_symbol_value_finetune_eval_v13__"
            "parabolic_img_0577__seed000042.mp4"
        ),
        "old_expert_score": 0.8473541946432471,
        "minimum_expert_score": 0.35,
    }
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=LATEST_DATASET)
    parser.add_argument("--protocol", default="scene_default_v13")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=(
            PROJECT_ROOT
            / "docs/evidence/parabolic_identity_v2_regression.json"
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse only results whose evaluator identity and both video "
            "hashes match exactly."
        ),
    )
    return parser.parse_args()


def preflight(
    *, dataset_path: Path = LATEST_DATASET
) -> tuple[Any, dict[str, dict[str, Any]]]:
    dataset = load_dataset(dataset_path, check_assets=True)
    catalog = {case["case_id"]: case for case in dataset.cases}
    required = {*REGRESSION_CASES, *REVIEWED_GOOD_CONTROLS}
    missing = sorted(required - set(catalog))
    if missing:
        raise KeyError(f"regression Cases are absent from Dataset: {missing}")
    for case_id in sorted(required):
        manifest_value = catalog[case_id].get("assets", {}).get(
            "first_frame_mask_manifest"
        )
        if not isinstance(manifest_value, str):
            raise ValueError(f"{case_id} has no frozen subject manifest")
        manifest_path = dataset.asset_root / manifest_value
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
    for spec in (*REGRESSION_CASES.values(), *REVIEWED_GOOD_CONTROLS.values()):
        prediction = Path(str(spec["prediction_path"]))
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
    return dataset, catalog


def _load_reusable_result(
    *,
    artifact_dir: Path,
    job_id: str,
    case_id: str,
    variant: str,
    evaluator_fingerprint: str,
    prediction_path: Path,
    reference_path: Path,
) -> dict[str, Any] | None:
    result_path = artifact_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = load_json(result_path)
        prediction_hash = sha256_file(prediction_path)
        reference_hash = sha256_file(reference_path)
    except (OSError, ValueError):
        return None
    evaluator = result.get("evaluator")
    provenance = result.get("provenance")
    if not isinstance(evaluator, Mapping) or not isinstance(
        provenance, Mapping
    ):
        return None
    expected = {
        "job_id": job_id,
        "case_id": case_id,
        "scene_id": "parabolic_motion",
        "status": "evaluated",
        "regression_variant": variant,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return None
    if evaluator.get("fingerprint") != evaluator_fingerprint:
        return None
    if provenance.get("prediction_video_sha256") != prediction_hash:
        return None
    if provenance.get("reference_video_sha256") != reference_hash:
        return None
    return result


def _evaluate(
    evaluator: Any,
    *,
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
    prediction_path: Path,
    output: Path,
    variant: str,
    evaluator_config: dict[str, Any],
    prediction_record: Mapping[str, Any] | None = None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    job_id = f"parabolic_identity_v2__{case['case_id']}__{variant}"
    artifact_dir = output / "cases" / job_id
    reference_path = asset_root / case["assets"]["reference_video"]
    if reuse_existing:
        reusable = _load_reusable_result(
            artifact_dir=artifact_dir,
            job_id=job_id,
            case_id=str(case["case_id"]),
            variant=variant,
            evaluator_fingerprint=str(evaluator.describe()["fingerprint"]),
            prediction_path=prediction_path,
            reference_path=reference_path,
        )
        if reusable is not None:
            return reusable
    prediction = dict(prediction_record or {})
    prediction.update(
        {
            "job_id": job_id,
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "status": "complete",
            "video_path": str(prediction_path),
        }
    )
    request = CaseEvaluationRequest(
        job={
            "job_id": job_id,
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
        },
        case=case,
        case_catalog=catalog,
        prediction=prediction,
        asset_root=asset_root,
        artifact_dir=artifact_dir,
        evaluator_config=evaluator_config,
        run_id=output.name,
        save_visualizations=False,
    )
    result = evaluator.evaluate(request).to_dict()
    result["regression_variant"] = variant
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "result.json", result)
    return result


def _score_record(
    result: Mapping[str, Any], *, prediction_path: Path
) -> dict[str, Any]:
    csti = result.get("metrics", {}).get("csti", {})
    binding = (
        result.get("metrics", {})
        .get("physical_subject_similarity", {})
        .get("components", {})
        .get("frame_zero_binding")
    )
    observation = result.get("provenance", {}).get("observation", {})
    prediction_observation = (
        observation.get("prediction", {})
        if isinstance(observation, Mapping)
        else {}
    )
    return {
        "status": result.get("status"),
        "reason_code": result.get("reason_code"),
        "expert_score": result.get("score"),
        "csti_score": csti.get("score"),
        "csti_status": csti.get("status"),
        "frame_zero_binding": binding,
        "prediction_identity_state": prediction_observation.get(
            "identity_state"
        ),
        "identity_termination_frame": prediction_observation.get(
            "identity_termination_frame"
        ),
        "identity_termination_reason": prediction_observation.get(
            "identity_termination_reason"
        ),
        "degradation_codes": result.get("quality", {}).get(
            "degradation_codes", []
        ),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "result_path": str(
            Path(result.get("artifacts", {}).get("open_world_audit", ""))
            .parent
            / "result.json"
        ),
        "open_world_audit": result.get("artifacts", {}).get(
            "open_world_audit"
        ),
        "entity_position_curve": result.get("artifacts", {}).get(
            "entity_position_curve"
        ),
    }


def _numeric(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{key} is unavailable: {value!r}")
    return float(value)


def _prediction_records(run_root: Path) -> dict[str, dict[str, Any]]:
    path = run_root / "predictions.jsonl"
    records = load_jsonl(path)
    output = {str(record["case_id"]): record for record in records}
    if len(output) != len(records):
        raise ValueError(f"duplicate Case prediction records in {path}")
    return output


def main() -> None:
    args = _arguments()
    dataset, catalog = preflight(dataset_path=args.dataset)
    protocol = copy.deepcopy(load_evaluation_protocol(args.protocol))
    config = protocol["scenes"]["parabolic_motion"]
    evaluator = SceneEvaluatorRegistry(protocol).resolve(
        "parabolic_motion"
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_records = _prediction_records(PAIR_RUN)
    physics_records = _prediction_records(PHYSICS_RUN)
    failures: list[str] = []
    negatives: dict[str, dict[str, Any]] = {}
    exact_controls: dict[str, dict[str, Any]] = {}
    good_controls: dict[str, dict[str, Any]] = {}

    for case_id, spec in REGRESSION_CASES.items():
        case = catalog[case_id]
        prediction_path = Path(str(spec["prediction_path"]))
        result = _evaluate(
            evaluator,
            case=case,
            catalog=catalog,
            asset_root=dataset.asset_root,
            prediction_path=prediction_path,
            output=output,
            variant="false_high_negative",
            evaluator_config=config,
            prediction_record=pair_records[case_id],
            reuse_existing=args.reuse_existing,
        )
        record = _score_record(result, prediction_path=prediction_path)
        record.update(
            {
                "old_expert_score": spec["old_expert_score"],
                "maximum_score": spec["maximum_score"],
            }
        )
        negatives[case_id] = record
        try:
            maximum = float(spec["maximum_score"])
            if record["status"] != "evaluated":
                raise AssertionError(f"status={record['status']!r}")
            if _numeric(record, "expert_score") > maximum:
                raise AssertionError("expert score exceeds maximum")
            if _numeric(record, "csti_score") > maximum:
                raise AssertionError("CSTI score exceeds maximum")
        except AssertionError as exc:
            failures.append(f"negative {case_id}: {exc}")

        reference_path = dataset.asset_root / case["assets"]["reference_video"]
        exact_result = _evaluate(
            evaluator,
            case=case,
            catalog=catalog,
            asset_root=dataset.asset_root,
            prediction_path=reference_path,
            output=output,
            variant="exact_gt_positive",
            evaluator_config=config,
            reuse_existing=args.reuse_existing,
        )
        exact = _score_record(exact_result, prediction_path=reference_path)
        exact["minimum_score"] = 0.95
        exact_controls[case_id] = exact
        try:
            if exact["status"] != "evaluated":
                raise AssertionError(f"status={exact['status']!r}")
            if _numeric(exact, "expert_score") < 0.95:
                raise AssertionError("expert score is below exact-GT minimum")
            if _numeric(exact, "csti_score") < 0.95:
                raise AssertionError("CSTI score is below exact-GT minimum")
        except AssertionError as exc:
            failures.append(f"exact positive {case_id}: {exc}")

    for case_id, spec in REVIEWED_GOOD_CONTROLS.items():
        case = catalog[case_id]
        prediction_path = Path(str(spec["prediction_path"]))
        result = _evaluate(
            evaluator,
            case=case,
            catalog=catalog,
            asset_root=dataset.asset_root,
            prediction_path=prediction_path,
            output=output,
            variant="reviewed_good_positive",
            evaluator_config=config,
            prediction_record=physics_records[case_id],
            reuse_existing=args.reuse_existing,
        )
        record = _score_record(result, prediction_path=prediction_path)
        record.update(
            {
                "old_expert_score": spec["old_expert_score"],
                "minimum_expert_score": spec["minimum_expert_score"],
            }
        )
        good_controls[case_id] = record
        try:
            minimum = float(spec["minimum_expert_score"])
            if record["status"] != "evaluated":
                raise AssertionError(f"status={record['status']!r}")
            if _numeric(record, "expert_score") < minimum:
                raise AssertionError("expert score is below reviewed minimum")
            binding = record.get("frame_zero_binding")
            if not isinstance(binding, Mapping) or not binding.get("accepted"):
                raise AssertionError("reviewed frame-zero identity is unconfirmed")
        except AssertionError as exc:
            failures.append(f"reviewed positive {case_id}: {exc}")

    evidence = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["fingerprint"],
        "evaluator": evaluator.describe(),
        "dataset_id": dataset.dataset_id,
        "dataset_release": dataset.descriptor["release"],
        "output": str(output),
        "negative_controls": negatives,
        "exact_gt_positive_controls": exact_controls,
        "reviewed_good_positive_controls": good_controls,
        "passed": not failures,
        "failures": failures,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.evidence, evidence)
    write_json(output / "regression_summary.json", evidence)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
