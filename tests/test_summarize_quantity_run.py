from __future__ import annotations

import hashlib
import json
import statistics
import struct
import tempfile
import unittest
from pathlib import Path

from physbench.artifacts import prediction_artifact_manifest
from physbench.baselines.wan22_quantity import Wan22QuantityLoraAdapter
from physbench.baselines.wan22_quantity_model import (
    QUANTITY_CHECKPOINT_PREFIX,
    QUANTITY_ENCODER_STATE_KEYS,
    WAN22_TI2V_5B_LORA_RANK,
    expected_wan22_ti2v_5b_lora_targets,
)
from physbench.domain import BaselineTaskInstance
from physbench.evaluation.task_evaluator import aggregate_task_results
from scripts.summarize_quantity_run import (
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    _canonical_sha256,
    _file_sha256,
    main,
    render_markdown,
    summarize_run,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class QuantityRunSummaryTests(unittest.TestCase):
    BASELINE_ID = "wan22_quantity_fixture"
    DATASET_ID = "fixture_dataset"
    TASK_ID = "quantity_fixture_task"
    PROTOCOL_ID = "scene_default_v1"
    PROTOCOL_FINGERPRINT = _sha("fixture-protocol")
    REGISTRY_ID = "fixture_quantity_registry"
    REGISTRY_FINGERPRINT = _sha("fixture-registry")

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, values: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(value, sort_keys=True) + "\n"
                for value in values
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _quantity(scene_id: str) -> dict:
        is_velocity = scene_id == "collision_1d"
        return {
            "name": (
                "initial_velocity" if is_velocity else "characteristic_length"
            ),
            "canonical_si_unit": "m/s" if is_velocity else "m",
            "dimension": (
                [1, 0, -1, 0, 0, 0, 0]
                if is_velocity
                else [1, 0, 0, 0, 0, 0, 0]
            ),
            "quantity_type": "velocity" if is_velocity else "length",
            "quantity_type_id": 5 if is_velocity else 1,
            "sentinel": "<extra_id_0>",
            "si_value": 1.0,
            "raw_unit": "m/s" if is_velocity else "m",
            "raw_value": 1.0,
            "rendered_quantity": (
                "1.0 m/s" if is_velocity else "1.0 m"
            ),
            "source_role": "primary",
        }

    def _native_inputs(self, case_id: str, scene_id: str) -> dict:
        unit = "m/s" if scene_id == "collision_1d" else "m"
        return {
            "text": {
                "prompt": f"{case_id}: value <extra_id_0>.",
                "audited_prompt": f"{case_id}: value 1.0 {unit}.",
            },
            "physics": {
                "registry_id": self.REGISTRY_ID,
                "registry_fingerprint": self.REGISTRY_FINGERPRINT,
                "representation": "quantity_token_embedding_v1",
                "quantities": [self._quantity(scene_id)],
            },
            "vision": {"first_frame_asset": f"{case_id}.png"},
        }

    @staticmethod
    def _audit_quantity(quantity: dict) -> dict:
        return {
            **quantity,
            "sentinel_token_id": 256299,
            "token_span": [4, 5],
        }

    def _training_audit(
        self,
        case_id: str,
        scene_id: str,
        native_inputs: dict,
    ) -> dict:
        return {
            "schema_version": "1.0",
            "case_id": case_id,
            "scene_id": scene_id,
            "prompt": native_inputs["text"]["prompt"],
            "audited_prompt": native_inputs["text"]["audited_prompt"],
            "quantity_registry_id": self.REGISTRY_ID,
            "quantity_registry_fingerprint": (
                self.REGISTRY_FINGERPRINT
            ),
            "quantities": [
                self._audit_quantity(
                    native_inputs["physics"]["quantities"][0]
                )
            ],
        }

    def _inference_audit(self, job: dict) -> dict:
        native_inputs = job["native_inputs"]
        return {
            "schema_version": "1.0",
            "case_id": job["case_id"],
            "job_id": job["job_id"],
            "registry_id": self.REGISTRY_ID,
            "registry_fingerprint": self.REGISTRY_FINGERPRINT,
            "prompt": native_inputs["text"]["prompt"],
            "audited_prompt": native_inputs["text"]["audited_prompt"],
            "quantities": [
                self._audit_quantity(
                    native_inputs["physics"]["quantities"][0]
                )
            ],
        }

    @staticmethod
    def _write_real_safetensors(
        path: Path,
        *,
        bad_topology: bool = False,
        non_finite: bool = False,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        entries: dict[str, dict] = {}
        offset = 0
        targets = sorted(expected_wan22_ti2v_5b_lora_targets())
        if bad_topology:
            targets[0] = f"{targets[0]}_unexpected"
        for target in targets:
            for side in ("A", "B"):
                key = (
                    f"diffusion_model.{target}."
                    f"lora_{side}.default.weight"
                )
                shape = (
                    [WAN22_TI2V_5B_LORA_RANK, 1]
                    if side == "A"
                    else [1, WAN22_TI2V_5B_LORA_RANK]
                )
                tensor_bytes = (
                    shape[0] * shape[1] * struct.calcsize("<f")
                )
                entries[key] = {
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + tensor_bytes],
                }
                offset += tensor_bytes
        for state_key in sorted(QUANTITY_ENCODER_STATE_KEYS):
            key = f"{QUANTITY_CHECKPOINT_PREFIX}{state_key}"
            entries[key] = {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [offset, offset + 4],
            }
            offset += 4
        header = json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = bytearray(offset)
        if non_finite:
            struct.pack_into("<f", payload, 0, float("inf"))
        path.write_bytes(
            struct.pack("<Q", len(header))
            + header
            + payload
        )

    def _case_result(
        self,
        job: dict,
        score: float,
    ) -> dict:
        return {
            **{
                key: job[key]
                for key in (
                    "job_id",
                    "case_id",
                    "scene_id",
                    "evaluation_partition",
                    "seed",
                )
            },
            "evaluator": {
                "id": f"{job['scene_id']}_case_evaluator",
                "version": "1.0",
                "scene_id": job["scene_id"],
            },
            "status": "evaluated",
            "score": score,
            "reason_code": None,
            "reason": None,
            "metrics": {"subject_iou_mean": score},
            "quality": {"valid_frames": 5},
            "artifacts": {
                "iou_curve": (
                    f"evaluation/cases/{job['job_id']}/iou_curve.png"
                )
            },
            "provenance": {
                "protocol_fingerprint": self.PROTOCOL_FINGERPRINT
            },
        }

    def _seal_instance(
        self,
        *,
        run_dir: Path,
        plan: dict,
        task_digest: str,
        dataset_digest: str,
        baseline_digest: str,
        deployment_digest: str,
        task_builder_fingerprint: str,
        data_adapter_fingerprint: str,
        materialization_fingerprint: str,
        cases: list[dict],
        baseline_version: str,
    ) -> BaselineTaskInstance:
        train_ids = plan["train_case_ids"]
        eval_ids = list(dict.fromkeys(
            job["case_id"] for job in plan["jobs"]
        ))
        by_case = {case["case_id"]: case for case in cases}
        adaptations = []
        by_key = {}
        for role, case_ids in (("train", train_ids), ("eval", eval_ids)):
            for case_id in case_ids:
                case = by_case[case_id]
                adaptation = {
                    "adaptation_id": f"{case_id}__{role}",
                    "case_id": case_id,
                    "scene_id": case["scene_id"],
                    "role": role,
                    "native_inputs": self._native_inputs(
                        case_id,
                        case["scene_id"],
                    ),
                }
                adaptations.append(adaptation)
                by_key[(case_id, role)] = adaptation
        inference_jobs = [
            {
                **job,
                "adaptation_id": f"{job['case_id']}__eval",
                "model_ref": "artifact://train/model",
                "native_inputs": by_key[
                    (job["case_id"], "eval")
                ]["native_inputs"],
            }
            for job in plan["jobs"]
        ]
        operations = [
            {
                "operation_id": "train",
                "kind": "train",
                "depends_on": [],
            },
            {
                "operation_id": "infer",
                "kind": "infer",
                "depends_on": ["train"],
                "job_ids": [
                    job["job_id"] for job in inference_jobs
                ],
            },
            {
                "operation_id": "evaluate",
                "kind": "evaluate",
                "depends_on": ["infer"],
                "job_ids": [
                    job["job_id"] for job in inference_jobs
                ],
            },
        ]
        return BaselineTaskInstance.seal({
            "schema_version": "3.0",
            "instance_id": (
                f"{self.TASK_ID}__{self.BASELINE_ID}__fixture"
            ),
            "identity": {
                "dataset": {
                    "dataset_id": self.DATASET_ID,
                    "digest": dataset_digest,
                },
                "task": {
                    "task_id": self.TASK_ID,
                    "digest": task_digest,
                },
                "baseline": {
                    "baseline_id": self.BASELINE_ID,
                    "baseline_version": baseline_version,
                    "digest": baseline_digest,
                    "deployment_digest": deployment_digest,
                },
                "task_builder": {
                    "type": "fixture_builder_v1",
                    "fingerprint": task_builder_fingerprint,
                },
                "data_adapter": {
                    "fingerprint": data_adapter_fingerprint,
                    "materialization_fingerprint": (
                        materialization_fingerprint
                    ),
                },
                "canonical_plan_digest": _canonical_sha256(plan),
            },
            "semantics": {
                "family": "finetune_eval",
                "scene_ids": plan["scene_ids"],
            },
            "canonical_plan": plan,
            "source": {
                "asset_root": str(run_dir / "fixture_assets"),
                "cases": cases,
            },
            "adaptations": adaptations,
            "training": {
                "operation_id": "train",
                "case_ids": train_ids,
                "adaptation_ids": [
                    f"{case_id}__train" for case_id in train_ids
                ],
                "seed": 42,
            },
            "inference": {
                "predictor": {"type": "fixture_predictor"},
                "jobs": inference_jobs,
            },
            "execution_graph": {"operations": operations},
            "cache_bindings": [],
            "baseline_payload": {},
        })

    def _fixture_run(
        self,
        root: Path,
        *,
        inference_seeds: tuple[int, ...] = (42,),
        include_ood2: bool = False,
        baseline_version: str = "1.0.0",
    ) -> Path:
        run_dir = root / "atomic_run"
        train_cases = [
            {
                "case_id": "pendulum_train",
                "scene_id": "pendulum",
            },
            {
                "case_id": "free_fall_train",
                "scene_id": "free_fall",
            },
        ]
        eval_cases = [
            {
                "case_id": "pendulum_id_a",
                "scene_id": "pendulum",
                "partition": "test_id",
                "score": 0.2,
            },
            {
                "case_id": "pendulum_id_b",
                "scene_id": "pendulum",
                "partition": "test_id",
                "score": 0.4,
            },
            {
                "case_id": "pendulum_ood",
                "scene_id": "pendulum",
                "partition": "test_ood1",
                "score": 0.6,
            },
            {
                "case_id": "free_fall_id",
                "scene_id": "free_fall",
                "partition": "test_id",
                "score": 0.8,
            },
        ]
        if include_ood2:
            eval_cases.append({
                "case_id": "collision_ood2",
                "scene_id": "collision_1d",
                "partition": "test_ood2",
                "score": 0.5,
            })
        cases = [
            *train_cases,
            *[
                {
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                }
                for case in eval_cases
            ],
        ]
        scene_ids = ["pendulum", "free_fall"]
        if include_ood2:
            scene_ids.append("collision_1d")
        task = {
            "schema_version": "3.0",
            "task_id": self.TASK_ID,
            "family": "finetune_eval",
            "dataset_id": self.DATASET_ID,
            "dataset_view": "view_a",
            "selection": {
                "scene_ids": ["pendulum", "free_fall"],
                "eval_partitions": ["test_id", "test_ood1"],
            },
            "ood2": {
                "enabled": include_ood2,
                "heldout_scenes": (
                    ["collision_1d"] if include_ood2 else []
                ),
            },
            "seeds": {
                "training": [42],
                "inference": list(inference_seeds),
            },
            "evaluation": {"protocol": self.PROTOCOL_ID},
        }
        jobs = []
        for case in eval_cases:
            for seed in inference_seeds:
                jobs.append({
                    "job_id": (
                        f"{self.TASK_ID}__{case['case_id']}"
                        f"__seed{seed:06d}"
                    ),
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                    "evaluation_partition": case["partition"],
                    "seed": seed,
                })
        plan = {
            "schema_version": "3.0",
            "task_id": self.TASK_ID,
            "family": "finetune_eval",
            "dataset_id": self.DATASET_ID,
            "dataset_digest": _sha("fixture-dataset"),
            "scene_ids": scene_ids,
            "train_case_ids": [
                case["case_id"] for case in train_cases
            ],
            "training_seed": 42,
            "jobs": jobs,
        }
        task_digest = _canonical_sha256(task)
        baseline_digest = _sha("fixture-baseline")
        deployment_digest = _sha("fixture-deployment")
        task_builder_fingerprint = _sha("fixture-builder")
        data_adapter_fingerprint = _sha("fixture-adapter")
        materialization_fingerprint = _sha("fixture-materialization")
        instance = self._seal_instance(
            run_dir=run_dir,
            plan=plan,
            task_digest=task_digest,
            dataset_digest=plan["dataset_digest"],
            baseline_digest=baseline_digest,
            deployment_digest=deployment_digest,
            task_builder_fingerprint=task_builder_fingerprint,
            data_adapter_fingerprint=data_adapter_fingerprint,
            materialization_fingerprint=materialization_fingerprint,
            cases=cases,
            baseline_version=baseline_version,
        )
        instance_value = instance.value
        components = {
            "dataset": plan["dataset_digest"],
            "task": task_digest,
            "baseline": baseline_digest,
            "baseline_deployment": deployment_digest,
            "task_builder": task_builder_fingerprint,
            "task_instance": instance.digest,
            "data_adapter": data_adapter_fingerprint,
            "data_adapter_materialization": (
                materialization_fingerprint
            ),
            "evaluation_protocol": self.PROTOCOL_FINGERPRINT,
        }

        self._write_json(run_dir / "plan.json", plan)
        self._write_json(run_dir / "frozen" / "task.json", task)
        self._write_json(
            run_dir / "frozen" / "dataset.json",
            {"dataset_id": self.DATASET_ID},
        )
        self._write_json(
            run_dir / "frozen" / "baseline.json",
            {
                "baseline_id": self.BASELINE_ID,
                "baseline_version": baseline_version,
                "trainer": {
                    "config": {"save_optimizer_state": True}
                },
            },
        )
        self._write_jsonl(
            run_dir / "frozen" / "cases.jsonl",
            cases,
        )
        self._write_json(
            run_dir / "component_fingerprints.json",
            components,
        )
        self._write_json(
            run_dir / "task_instance" / "manifest.json",
            instance_value,
        )
        self._write_json(
            run_dir / "task_instance" / "canonical_plan.json",
            plan,
        )

        inference_jobs = {
            job["job_id"]: job
            for job in instance_value["inference"]["jobs"]
        }
        predictions = []
        for job in jobs:
            video = (
                run_dir / "predictions" / self.BASELINE_ID
                / f"{job['job_id']}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video:{job['job_id']}".encode("utf-8"))
            predictions.append({
                **job,
                "baseline_id": self.BASELINE_ID,
                "status": "complete",
                "video_path": str(video),
                "video_sha256": _file_sha256(video),
            })
            audit_path = (
                run_dir / "artifacts" / "wan22"
                / "inference_quantity_token_audits"
                / f"{job['job_id']}.json"
            )
            self._write_json(
                audit_path,
                self._inference_audit(inference_jobs[job["job_id"]]),
            )
        self._write_jsonl(run_dir / "predictions.jsonl", predictions)
        self._write_json(
            run_dir / "artifacts" / "prediction_artifacts.json",
            prediction_artifact_manifest(predictions, run_dir),
        )

        score_by_case = {
            case["case_id"]: case["score"] for case in eval_cases
        }
        case_results = [
            self._case_result(job, score_by_case[job["case_id"]])
            for job in jobs
        ]
        self._write_jsonl(
            run_dir / "evaluation" / "case_results.jsonl",
            case_results,
        )
        for record in case_results:
            self._write_json(
                run_dir / "evaluation" / "cases"
                / record["job_id"] / "result.json",
                record,
            )
        aggregation = aggregate_task_results(
            plan=plan,
            case_results=case_results,
        )
        task_result = {
            "schema_version": "1.0",
            "task_id": self.TASK_ID,
            "task_family": "finetune_eval",
            "protocol": {
                "id": self.PROTOCOL_ID,
                "fingerprint": self.PROTOCOL_FINGERPRINT,
                "path": "configs/evaluation/scene_default_v1.json",
            },
            "integrity_issues": [],
            **aggregation,
        }
        self._write_json(
            run_dir / "evaluation" / "task_result.json",
            task_result,
        )
        run = {
            "schema_version": "2.0",
            "run_id": "quantity_fixture_run",
            "status": "complete",
            "dataset_id": self.DATASET_ID,
            "dataset_digest": plan["dataset_digest"],
            "task_id": self.TASK_ID,
            "task_family": "finetune_eval",
            "baseline_id": self.BASELINE_ID,
            "baseline_version": baseline_version,
            "baseline_digest": baseline_digest,
            "baseline_deployment_digest": deployment_digest,
            "task_instance_id": instance.instance_id,
            "task_instance_digest": instance.digest,
            "evaluation_status": aggregation["status"],
            "evaluation_coverage": aggregation["coverage"],
            "evaluation_score": aggregation["score"],
        }
        self._write_json(run_dir / "run.json", run)
        self._write_json(
            run_dir / "state.json",
            {
                "schema_version": "2.0",
                "stage": "complete",
                "updated_at": "2026-07-28T00:00:00+00:00",
                "history": [{"stage": "complete"}],
            },
        )

        sampling = {
            "unique_case_count": len(train_cases),
            "metadata_row_count": 4,
            "dataset_repeat": 1,
            "num_epochs": 1,
            "world_size": 1,
            "expected_optimizer_steps_per_epoch": 4,
            "expected_total_optimizer_steps": 4,
        }
        self._write_json(
            run_dir / "artifacts" / "wan22"
            / "training_sampling_plan.json",
            sampling,
        )
        losses = [0.4, 0.3, 0.2, 0.1]
        loss_root = (
            run_dir / "artifacts" / "wan22" / "loss_analysis"
        )
        loss_root.mkdir(parents=True, exist_ok=True)
        (loss_root / "loss_curve.csv").write_text(
            "step,loss\n"
            + "".join(
                f"{step},{loss}\n"
                for step, loss in enumerate(losses, 1)
            ),
            encoding="utf-8",
        )
        (loss_root / "loss_curve.png").write_bytes(
            b"\x89PNG\r\n\x1a\nfixture"
        )
        loss_mean = statistics.fmean(losses)
        self._write_json(
            loss_root / "loss_summary.json",
            {
                "recorded_steps": 4,
                "first_step": 1,
                "last_step": 4,
                "expected_total_steps": 4,
                "finite_fraction": 1.0,
                "mean": loss_mean,
                "median": statistics.median(losses),
                "first_100_mean": loss_mean,
                "last_100_mean": loss_mean,
                "last_over_first_100_mean": 1.0,
            },
        )
        self._write_json(
            run_dir / "training" / "loss_export.json",
            {"status": "complete", "return_code": 0},
        )

        checkpoint = (
            run_dir / "artifacts" / "wan22" / "checkpoints"
            / "step-4.safetensors"
        )
        self._write_real_safetensors(checkpoint)
        strict_inventory = Wan22QuantityLoraAdapter._checkpoint_inventory(
            checkpoint
        )
        legacy_inventory_fields = (
            "tensor_count",
            "parameter_count",
            "lora_tensor_count",
            "quantity_encoder_tensor_count",
            "dtype_tensor_counts",
        )
        inventory = (
            strict_inventory
            if baseline_version == "1.0.1"
            else {
                field: strict_inventory[field]
                for field in legacy_inventory_fields
            }
        )
        state_root = checkpoint.parent / "training_state_latest"
        state_root.mkdir(parents=True, exist_ok=True)
        optimizer = state_root / "optimizer_scheduler.pt"
        optimizer.write_bytes(b"optimizer fixture")
        (state_root / "rng_rank_00.pt").write_bytes(b"rng fixture")
        self._write_json(
            state_root / "state.json",
            {
                "schema_version": "1.0",
                "status": "complete",
                "global_step": 4,
                "epoch_id": 0,
                "model_checkpoint": checkpoint.name,
                "world_size": 1,
            },
        )
        self._write_json(
            run_dir / "artifacts" / "wan22" / "checkpoint.json",
            {
                "schema_version": "2.0",
                "status": "complete",
                "source": "task1_finetune",
                "baseline_id": self.BASELINE_ID,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "inventory": inventory,
                "training_state": {
                    "directory": str(state_root),
                    "optimizer_scheduler": str(optimizer),
                    "optimizer_scheduler_sha256": (
                        _file_sha256(optimizer)
                    ),
                    "state_manifest": str(state_root / "state.json"),
                },
            },
        )
        self._write_json(
            checkpoint.parent / "gradient_audit.json",
            {
                "sample_count": 4,
                "positive_quantity_gradient_count": 4,
                "min_positive_quantity_gradient_l2": 0.01,
                "max_quantity_gradient_l2": 0.04,
                "text_encoder_gradient_tensor_count_max": 0,
                "samples": [
                    {
                        "step": step,
                        "quantity_gradient_l2": step / 100,
                        "quantity_gradient_tensor_count": 19,
                        "text_encoder_gradient_tensor_count": 0,
                    }
                    for step in range(1, 5)
                ],
            },
        )
        training_adaptations = {
            adaptation["case_id"]: adaptation
            for adaptation in instance_value["adaptations"]
            if adaptation["role"] == "train"
        }
        self._write_jsonl(
            run_dir / "artifacts" / "wan22"
            / "training_quantity_token_audit.jsonl",
            [
                self._training_audit(
                    case["case_id"],
                    case["scene_id"],
                    training_adaptations[
                        case["case_id"]
                    ]["native_inputs"],
                )
                for case in train_cases
            ],
        )
        return run_dir

    @staticmethod
    def _row(
        summary: dict,
        scene_id: str,
        partition: str,
    ) -> dict:
        return next(
            row
            for row in summary["scene_partition"]
            if row["scene_id"] == scene_id
            and row["partition"] == partition
        )

    def _rewrite_official_from_cases(self, run_dir: Path) -> None:
        plan = self._read_json(run_dir / "plan.json")
        records = self._read_jsonl(
            run_dir / "evaluation" / "case_results.jsonl"
        )
        aggregation = aggregate_task_results(
            plan=plan,
            case_results=records,
        )
        result = self._read_json(
            run_dir / "evaluation" / "task_result.json"
        )
        for key in (
            "status",
            "expected_jobs",
            "evaluated_jobs",
            "coverage",
            "status_counts",
            "score",
            "observed_mean_score",
            "aggregation_policy",
            "by_scene",
            "breakdown",
        ):
            result[key] = aggregation[key]
        self._write_json(
            run_dir / "evaluation" / "task_result.json",
            result,
        )
        run = self._read_json(run_dir / "run.json")
        run["evaluation_status"] = aggregation["status"]
        run["evaluation_coverage"] = aggregation["coverage"]
        run["evaluation_score"] = aggregation["score"]
        self._write_json(run_dir / "run.json", run)

    def test_complete_golden_run_is_publishable_and_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            first = summarize_run(run_dir)
            second = summarize_run(run_dir)

            self.assertEqual([], first["integrity_issues"])
            self.assertTrue(
                first["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )
            self.assertTrue(
                first["official_task_result"]["aggregation_verified"]
            )
            self.assertTrue(
                first["official_task_result"]["protocol_verified"]
            )
            self.assertAlmostEqual(
                0.625,
                first["official_task_result"]["score"],
            )
            self.assertTrue(first["training_acceptance"]["passed"])
            self.assertEqual(
                619,
                first["provenance"]["checkpoint"]["inventory"][
                    "tensor_count"
                ],
            )
            checkpoint = first["provenance"]["checkpoint"]
            self.assertEqual(
                "wan22_quantity_checkpoint_inventory_v1_legacy",
                checkpoint["inventory_profile"]["profile_id"],
            )
            self.assertTrue(
                checkpoint["inventory_profile"]["verified"]
            )
            self.assertTrue(
                checkpoint["inventory_declaration"]["verified"]
            )
            self.assertEqual(
                {
                    "finite_payload_verified",
                    "lora_pair_count",
                    "lora_rank",
                    "lora_target_topology",
                    "safetensors_layout_verified",
                },
                set(
                    checkpoint["inventory_declaration"][
                        "derived_not_declared"
                    ]
                ),
            )
            self.assertTrue(
                first["provenance"]["checkpoint"]["training_state"][
                    "passed"
                ]
            )
            self.assertEqual(
                first["scene_partition"],
                second["scene_partition"],
            )
            pendulum_job = next(
                row
                for row in first["jobs"]
                if row["scene_id"] == "pendulum"
            )
            self.assertIn("subject_iou_mean", pendulum_job["metrics"])
            self.assertIn("iou_curve", pendulum_job["artifacts"])
            markdown = render_markdown(first)
            self.assertIn(self.PROTOCOL_FINGERPRINT, markdown)
            self.assertIn("Benchmark score publishable: `yes`", markdown)

    def test_cli_writes_json_and_markdown_without_mutating_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._fixture_run(root)
            output = root / "report"
            run_before = (run_dir / "run.json").read_bytes()
            self.assertEqual(
                0,
                main([
                    "--run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(output),
                ]),
            )
            payload = self._read_json(output / OUTPUT_JSON)
            markdown = (output / OUTPUT_MARKDOWN).read_text(
                encoding="utf-8"
            )
            self.assertTrue(
                payload["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )
            self.assertIn("Case macro mean", markdown)
            self.assertEqual(
                run_before,
                (run_dir / "run.json").read_bytes(),
            )

    def test_protocol_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            path = run_dir / "evaluation" / "task_result.json"
            result = self._read_json(path)
            result["protocol"]["fingerprint"] = _sha("revised-protocol")
            self._write_json(path, result)

            summary = summarize_run(run_dir)
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )
            self.assertFalse(
                summary["official_task_result"]["protocol_verified"]
            )
            self.assertIn(
                "official_evaluation_protocol_mismatch",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )

    def test_forged_official_macro_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            result_path = (
                run_dir / "evaluation" / "task_result.json"
            )
            result = self._read_json(result_path)
            result["score"] = 0.99
            result["by_scene"]["pendulum"]["score"] = 0.99
            self._write_json(result_path, result)
            run_path = run_dir / "run.json"
            run = self._read_json(run_path)
            run["evaluation_score"] = 0.99
            self._write_json(run_path, run)

            summary = summarize_run(run_dir)
            self.assertFalse(
                summary["official_task_result"][
                    "aggregation_verified"
                ]
            )
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )
            mismatches = [
                issue
                for issue in summary["integrity_issues"]
                if issue["code"] == "official_aggregation_mismatch"
            ]
            self.assertTrue(mismatches)

    def test_invalid_prediction_status_and_case_score_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            predictions_path = run_dir / "predictions.jsonl"
            predictions = self._read_jsonl(predictions_path)
            predictions[0]["status"] = "mystery"
            self._write_jsonl(predictions_path, predictions)
            results_path = (
                run_dir / "evaluation" / "case_results.jsonl"
            )
            results = self._read_jsonl(results_path)
            results[0]["score"] = 1.5
            self._write_jsonl(results_path, results)

            summary = summarize_run(run_dir)
            codes = {
                issue["code"] for issue in summary["integrity_issues"]
            }
            self.assertIn("prediction_status_invalid", codes)
            self.assertIn("case_evaluation_score_invalid", codes)
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )

    def test_evaluated_case_with_failed_prediction_cannot_publish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            predictions_path = run_dir / "predictions.jsonl"
            predictions = self._read_jsonl(predictions_path)
            failed_job_id = predictions[0]["job_id"]
            predictions[0]["status"] = "failed"
            self._write_jsonl(predictions_path, predictions)
            self._write_json(
                run_dir / "artifacts" / "prediction_artifacts.json",
                prediction_artifact_manifest(predictions, run_dir),
            )
            (
                run_dir / "artifacts" / "wan22"
                / "inference_quantity_token_audits"
                / f"{failed_job_id}.json"
            ).unlink()

            summary = summarize_run(run_dir)
            codes = {
                issue["code"] for issue in summary["integrity_issues"]
            }
            self.assertIn(
                "complete_run_has_incomplete_predictions",
                codes,
            )
            self.assertIn(
                "evaluated_case_without_complete_prediction",
                codes,
            )
            self.assertFalse(
                summary["reporting_status"][
                    "all_planned_predictions_complete"
                ]
            )
            self.assertFalse(
                summary["official_task_result"][
                    "aggregation_verified"
                ]
            )
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )

    def test_sealed_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            run_path = run_dir / "run.json"
            run = self._read_json(run_path)
            run["dataset_id"] = "mixed_dataset"
            self._write_json(run_path, run)

            summary = summarize_run(run_dir)
            self.assertIn(
                "run_sealed_identity_mismatch",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )

    def test_training_and_inference_token_audits_are_semantic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            training_path = (
                run_dir / "artifacts" / "wan22"
                / "training_quantity_token_audit.jsonl"
            )
            records = self._read_jsonl(training_path)
            records[0]["quantities"][0]["si_value"] = 99.0
            self._write_jsonl(training_path, records)
            inference_root = (
                run_dir / "artifacts" / "wan22"
                / "inference_quantity_token_audits"
            )
            inference_paths = sorted(inference_root.glob("*.json"))
            inference_paths[0].unlink()
            inference_record = self._read_json(inference_paths[1])
            inference_record["registry_fingerprint"] = _sha(
                "wrong-registry"
            )
            self._write_json(inference_paths[1], inference_record)

            summary = summarize_run(run_dir)
            codes = {
                issue["code"] for issue in summary["integrity_issues"]
            }
            self.assertIn("quantity_token_audit_value_mismatch", codes)
            self.assertIn(
                "inference_quantity_token_audit_coverage_mismatch",
                codes,
            )
            self.assertIn(
                "quantity_token_audit_identity_mismatch",
                codes,
            )
            self.assertFalse(
                summary["training_acceptance"][
                    "quantity_token_audit_passed"
                ]
            )

    def test_checkpoint_is_read_from_real_safetensors_header(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            manifest_path = (
                run_dir / "artifacts" / "wan22" / "checkpoint.json"
            )
            manifest = self._read_json(manifest_path)
            checkpoint = Path(manifest["checkpoint"])
            checkpoint.write_bytes(b"not a safetensors checkpoint")
            manifest["checkpoint_sha256"] = _file_sha256(checkpoint)
            manifest["checkpoint_size"] = checkpoint.stat().st_size
            self._write_json(manifest_path, manifest)

            summary = summarize_run(run_dir)
            self.assertIn(
                "checkpoint_inventory_read_failed",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )

    def test_hardened_checkpoint_inventory_profile_is_publishable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(
                Path(temporary),
                baseline_version="1.0.1",
            )

            summary = summarize_run(run_dir)
            checkpoint = summary["provenance"]["checkpoint"]
            self.assertEqual([], summary["integrity_issues"])
            self.assertTrue(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )
            self.assertEqual(
                "wan22_quantity_checkpoint_inventory_v1_hardened",
                checkpoint["inventory_profile"]["profile_id"],
            )
            self.assertTrue(
                checkpoint["inventory_profile"]["verified"]
            )
            self.assertEqual(
                [],
                checkpoint["inventory_declaration"][
                    "derived_not_declared"
                ],
            )
            self.assertTrue(
                checkpoint["inventory_declaration"]["verified"]
            )

    def test_legacy_checkpoint_declared_value_tamper_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            manifest_path = (
                run_dir / "artifacts" / "wan22" / "checkpoint.json"
            )
            manifest = self._read_json(manifest_path)
            manifest["inventory"]["tensor_count"] = 1
            self._write_json(manifest_path, manifest)

            summary = summarize_run(run_dir)
            checkpoint = summary["provenance"]["checkpoint"]
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )
            self.assertFalse(
                checkpoint["inventory_declaration"]["verified"]
            )
            self.assertIn(
                "checkpoint_declared_inventory_mismatch",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )

    def test_hardened_checkpoint_missing_field_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(
                Path(temporary),
                baseline_version="1.0.1",
            )
            manifest_path = (
                run_dir / "artifacts" / "wan22" / "checkpoint.json"
            )
            manifest = self._read_json(manifest_path)
            manifest["inventory"].pop("finite_payload_verified")
            self._write_json(manifest_path, manifest)

            summary = summarize_run(run_dir)
            checkpoint = summary["provenance"]["checkpoint"]
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )
            self.assertFalse(
                checkpoint["inventory_declaration"]["verified"]
            )
            self.assertIn(
                "checkpoint_inventory_declaration_missing_field",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )

    def test_checkpoint_inventory_profile_identity_mismatch_fails_closed(
        self,
    ) -> None:
        for source in ("run", "frozen_baseline", "sealed_task_instance"):
            with self.subTest(source=source):
                with tempfile.TemporaryDirectory() as temporary:
                    run_dir = self._fixture_run(Path(temporary))
                    if source == "run":
                        path = run_dir / "run.json"
                        document = self._read_json(path)
                        document["baseline_version"] = "1.0.1"
                    elif source == "frozen_baseline":
                        path = run_dir / "frozen" / "baseline.json"
                        document = self._read_json(path)
                        document["baseline_version"] = "1.0.1"
                    else:
                        path = (
                            run_dir / "task_instance" / "manifest.json"
                        )
                        document = self._read_json(path)
                        document["identity"]["baseline"][
                            "baseline_version"
                        ] = "1.0.1"
                    self._write_json(path, document)

                    summary = summarize_run(run_dir)
                    checkpoint = summary["provenance"]["checkpoint"]
                    self.assertFalse(
                        summary["training_acceptance"][
                            "checkpoint_passed"
                        ]
                    )
                    self.assertFalse(
                        checkpoint["inventory_profile"]["verified"]
                    )
                    self.assertIn(
                        (
                            "checkpoint_inventory_profile_"
                            "identity_mismatch"
                        ),
                        {
                            issue["code"]
                            for issue in summary["integrity_issues"]
                        },
                    )

    def test_unknown_checkpoint_inventory_profile_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(
                Path(temporary),
                baseline_version="9.9.9",
            )

            summary = summarize_run(run_dir)
            checkpoint = summary["provenance"]["checkpoint"]
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )
            self.assertFalse(
                checkpoint["inventory_profile"]["verified"]
            )
            self.assertIn(
                "checkpoint_inventory_profile_unsupported",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )

    def test_bad_lora_topology_fails_checkpoint_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            manifest_path = (
                run_dir / "artifacts" / "wan22" / "checkpoint.json"
            )
            manifest = self._read_json(manifest_path)
            checkpoint = Path(manifest["checkpoint"])
            self._write_real_safetensors(
                checkpoint,
                bad_topology=True,
            )
            manifest["checkpoint_sha256"] = _file_sha256(checkpoint)
            manifest["checkpoint_size"] = checkpoint.stat().st_size
            self._write_json(manifest_path, manifest)

            summary = summarize_run(run_dir)
            self.assertIn(
                "checkpoint_inventory_read_failed",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )

    def test_non_finite_checkpoint_payload_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            manifest_path = (
                run_dir / "artifacts" / "wan22" / "checkpoint.json"
            )
            manifest = self._read_json(manifest_path)
            checkpoint = Path(manifest["checkpoint"])
            self._write_real_safetensors(
                checkpoint,
                non_finite=True,
            )
            manifest["checkpoint_sha256"] = _file_sha256(checkpoint)
            manifest["checkpoint_size"] = checkpoint.stat().st_size
            self._write_json(manifest_path, manifest)

            summary = summarize_run(run_dir)
            self.assertIn(
                "checkpoint_inventory_read_failed",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )

    def test_empty_rng_sidecar_fails_checkpoint_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            (
                run_dir / "artifacts" / "wan22" / "checkpoints"
                / "training_state_latest" / "rng_rank_00.pt"
            ).write_bytes(b"")

            summary = summarize_run(run_dir)
            self.assertIn(
                "checkpoint_rng_state_empty",
                {
                    issue["code"]
                    for issue in summary["integrity_issues"]
                },
            )
            self.assertFalse(
                summary["training_acceptance"]["checkpoint_passed"]
            )
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )

    def test_loss_csv_must_cover_every_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            csv_path = (
                run_dir / "artifacts" / "wan22" / "loss_analysis"
                / "loss_curve.csv"
            )
            csv_path.write_text(
                "step,loss\n1,0.4\n",
                encoding="utf-8",
            )

            summary = summarize_run(run_dir)
            codes = {
                issue["code"] for issue in summary["integrity_issues"]
            }
            self.assertIn("loss_curve_step_coverage_mismatch", codes)
            self.assertIn("loss_summary_curve_mismatch", codes)
            self.assertFalse(
                summary["training_acceptance"]["loss_passed"]
            )

    def test_ood2_partition_is_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(
                Path(temporary),
                include_ood2=True,
            )
            summary = summarize_run(run_dir)
            row = self._row(
                summary,
                "collision_1d",
                "test_ood2",
            )
            self.assertEqual(1, row["expected"])
            self.assertEqual(1, row["evaluated"])
            self.assertTrue(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )

    def test_multiseed_case_macro_and_job_micro_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(
                Path(temporary),
                inference_seeds=(42, 43),
            )
            path = run_dir / "evaluation" / "case_results.jsonl"
            records = self._read_jsonl(path)
            for record in records:
                if record["case_id"] == "pendulum_id_a":
                    record["score"] = (
                        0.0 if record["seed"] == 42 else 1.0
                    )
                    record["metrics"]["subject_iou_mean"] = record["score"]
                if (
                    record["case_id"] == "pendulum_id_b"
                    and record["seed"] == 42
                ):
                    record["score"] = 1.0
                    record["metrics"]["subject_iou_mean"] = 1.0
                if (
                    record["case_id"] == "pendulum_id_b"
                    and record["seed"] == 43
                ):
                    record.update({
                        "status": "unavailable",
                        "score": None,
                        "reason_code": "fixture_unavailable",
                        "reason": "fixture unavailable",
                    })
            self._write_jsonl(path, records)
            self._rewrite_official_from_cases(run_dir)

            summary = summarize_run(run_dir)
            row = self._row(summary, "pendulum", "test_id")
            self.assertAlmostEqual(2 / 3, row["job_micro_mean"])
            self.assertAlmostEqual(0.75, row["mean"])
            self.assertEqual(
                "case_macro_after_within_case_inference_seed_mean",
                row["aggregation_policy"],
            )
            self.assertEqual([], summary["integrity_issues"])
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )

    def test_malformed_nested_official_result_reports_instead_of_crashing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            path = run_dir / "evaluation" / "task_result.json"
            result = self._read_json(path)
            result["by_scene"] = {"pendulum": "not-an-object"}
            self._write_json(path, result)

            summary = summarize_run(run_dir)
            markdown = render_markdown(summary)
            self.assertIn("official_by_scene_entry_invalid", {
                issue["code"] for issue in summary["integrity_issues"]
            })
            self.assertIn("Benchmark score publishable: `no`", markdown)

    def test_failed_run_with_missing_results_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            (run_dir / "run.json").unlink()
            self._write_json(
                run_dir / "state.json",
                {
                    "schema_version": "2.0",
                    "stage": "failed",
                    "updated_at": "2026-07-28T00:00:00+00:00",
                    "history": [
                        {
                            "stage": "failed",
                            "error": "fixture training failure",
                        }
                    ],
                },
            )
            (run_dir / "predictions.jsonl").unlink()
            (
                run_dir / "evaluation" / "case_results.jsonl"
            ).unlink()
            (run_dir / "evaluation" / "task_result.json").unlink()

            summary = summarize_run(run_dir)
            self.assertTrue(summary["reporting_status"]["partial_report"])
            self.assertFalse(
                summary["reporting_status"][
                    "benchmark_score_publishable"
                ]
            )
            self.assertEqual(
                len(self._read_json(run_dir / "plan.json")["jobs"]),
                summary["overall"]["missing_prediction"],
            )
            self.assertFalse(
                summary["provenance"]["run"]["record_available"]
            )

    def test_planned_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            run_path = run_dir / "run.json"
            run = self._read_json(run_path)
            run["status"] = "planned"
            self._write_json(run_path, run)
            with self.assertRaisesRegex(ValueError, "reportable"):
                summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
