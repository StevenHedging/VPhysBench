from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any

from physbench.baselines.wan22_lora import Wan22LoraAdapter
from physbench.domain import BaselineBundle, BaselineTaskInstance
from physbench.io import load_jsonl, write_json, write_jsonl


class Wan22ExecutionEngine:
    """Compatibility execution layer behind the managed WAN driver."""

    def __init__(
        self,
        bundle: BaselineBundle,
        task_builder: Any,
    ):
        self.bundle = bundle
        self.task_builder = task_builder

    def _components(self) -> dict[str, Any]:
        value = self.bundle.value
        if "components" in value:
            return value["components"]
        components = {"predictor": value["runner"]}
        if "trainer" in value:
            components["trainer"] = value["trainer"]
        return components

    def _adapter_class(self):
        return Wan22LoraAdapter

    def _worker_script(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "wan22_generate_batch.py"
        )

    def _finalize_prepared_job(
        self,
        prepared: dict[str, Any],
        raw_job: dict[str, Any],
    ) -> None:
        # Standard WAN Baselines consume physics only through rendered text.
        # No parallel structured side channel reaches their model boundary.
        prepared["model_input"].pop("physical_parameters", None)

    def _legacy_config(
        self, instance_value: dict[str, Any]
    ) -> dict[str, Any]:
        value = self.bundle.value
        components = self._components()
        frozen_checkpoint = value.get("model", {}).get("frozen_lora_checkpoint")
        is_training = instance_value["semantics"]["family"] == "finetune_eval"
        trainer_config = copy.deepcopy(
            components.get("trainer", {}).get("config", {})
        )
        if is_training:
            trainer_config["seed"] = int(instance_value["training"]["seed"])
        cache_binding = instance_value["cache_bindings"][0]
        return {
            "schema_version": "1.0",
            "baseline_id": value["baseline_id"],
            "adapter": "wan22_lora",
            "input_view": "auto_i2v_from_case_or_video",
            "capabilities": value["capabilities"],
            "supported_scenes": value.get("supported_scenes", "all"),
            "require_image_condition": True,
            "require_lora_checkpoint": bool(is_training or frozen_checkpoint),
            "initial_lora_checkpoint": None if is_training else frozen_checkpoint,
            "runtime": value["runtime"],
            "media_adapter": self.task_builder.data_adapter.media_config,
            "lora": trainer_config,
            "generation": instance_value["inference"]["predictor"]["config"],
            "shared_media_cache": {
                "enabled": True,
                "root": cache_binding["root"],
            },
        }

    @staticmethod
    def _legacy_case(
        case: dict[str, Any], *, asset_root: Path, train_case_ids: set[str]
    ) -> dict[str, Any]:
        assets = dict(case["assets"])
        supervised = case.get("supervised_targets", {}).get("video")
        if isinstance(supervised, dict):
            assets[supervised["asset_key"]] = supervised["asset"]
        split = (
            "train"
            if case["case_id"] in train_case_ids
            else ("test_ood1" if case["ood"]["level"] == "ood1" else "test_id")
        )
        input_views: dict[str, dict[str, Any]] = {}
        if assets.get("first_frame"):
            input_views["i2v"] = {"first_frame": assets["first_frame"]}
        else:
            input_views["t2v"] = {}
        return {
            "schema_version": "1.0",
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "view_a_split": split,
            "physical_parameters": case.get("physics", {}),
            "appearance": case["appearance"],
            "temporal": case["temporal"],
            "alignment": case.get("alignment"),
            "assets": assets,
            "has_real_reference_video": case.get(
                "has_real_reference_video", False
            ),
            "ood": case["ood"],
            "provenance": case.get(
                "provenance",
                {
                    "source_kind": "managed_runtime_projection",
                    "parent_case_id": None,
                },
            ),
            "text": {
                "description": case["text"]["prompt"],
                "prompt": case["text"]["prompt"],
            },
            "input_views": input_views,
            "_dataset_asset_root": str(asset_root),
        }

    def _parallel_generate(self, run_dir: Path, planned_jobs: int) -> list[dict[str, Any]]:
        runtime = self.bundle.value["runtime"]
        available_gpus = [
            value.strip()
            for value in str(runtime.get("cuda_visible_devices", "0")).split(",")
            if value.strip()
        ]
        gpus = available_gpus[: max(1, min(len(available_gpus), planned_jobs))]
        worker_root = run_dir / "artifacts" / "wan22" / "inference_workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        worker_script = self._worker_script()
        processes = []
        for index, gpu in enumerate(gpus):
            result = worker_root / f"worker_{index:02d}.jsonl"
            log = (worker_root / f"worker_{index:02d}.log").open(
                "w", encoding="utf-8"
            )
            command = [
                str(runtime["python"]),
                str(worker_script),
                "--run-dir",
                str(run_dir),
                "--worker-index",
                str(index),
                "--worker-count",
                str(len(gpus)),
                "--result",
                str(result),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            processes.append((index, gpu, process, log, command))
        return_codes = {}
        for index, gpu, process, log, _ in processes:
            return_codes[f"worker_{index:02d}_gpu_{gpu}"] = process.wait()
            log.close()
        records = []
        for path in sorted(worker_root.glob("worker_*.jsonl")):
            records.extend(load_jsonl(path))
        records.sort(key=lambda item: item["job_id"])
        write_json(worker_root / "summary.json", {
            "gpu_assignments": gpus,
            "worker_return_codes": return_codes,
            "planned_jobs": planned_jobs,
            "recorded_jobs": len(records),
            "completed_jobs": sum(item["status"] == "complete" for item in records),
            "failed_jobs": sum(item["status"] != "complete" for item in records),
            "persistent_model_per_worker": True,
        })
        return records

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        instance.verify()
        instance_value = instance.value
        identity = instance_value["identity"]
        if identity["baseline"]["baseline_id"] != self.bundle.baseline_id:
            raise ValueError("task instance targets a different baseline")
        if identity["baseline"]["digest"] != self.bundle.digest:
            raise ValueError("task instance baseline snapshot digest mismatch")
        if (
            identity["baseline"].get("deployment_digest")
            != self.bundle.deployment_digest
        ):
            raise ValueError("task instance baseline deployment digest mismatch")
        if (
            identity["task_builder"]["fingerprint"]
            != self.task_builder.fingerprint
        ):
            raise ValueError("task instance TaskBuilder fingerprint mismatch")

        plan = instance.canonical_plan
        source_cases = instance_value["source"]["cases"]
        asset_root = Path(instance_value["source"]["asset_root"])
        train_ids = set(plan.train_case_ids)
        legacy_cases = [
            self._legacy_case(
                case, asset_root=asset_root, train_case_ids=train_ids
            )
            for case in source_cases
        ]
        write_jsonl(run_dir / "frozen_cases.jsonl", legacy_cases)
        write_json(run_dir / "data_context.json", {
            "manifest_path": str(run_dir / "task_instance" / "manifest.json"),
            "manifest_dir": str(asset_root),
        })

        adaptation_records = instance_value["adaptations"]
        adaptations_by_id = {
            item["adaptation_id"]: item for item in adaptation_records
        }
        write_jsonl(
            run_dir / "adaptations" / "case_adaptations.jsonl",
            adaptation_records,
        )

        legacy_config = self._legacy_config(instance_value)
        cache_binding = instance_value["cache_bindings"][0]
        write_json(run_dir / "data_adapter_cache_binding.json", {
            "schema_version": "2.0",
            "policy": cache_binding["policy"],
            "dataset_digest": cache_binding["dataset_digest"],
            "data_adapter_fingerprint": (
                identity["data_adapter"]["fingerprint"]
            ),
            "materialization_fingerprint": cache_binding[
                "materialization_fingerprint"
            ],
            "cache_root": cache_binding["root"],
            "rebuildable": True,
            "source_assets_mutated": False,
        })
        legacy_adapter = self._adapter_class()(
            legacy_config,
            execute=execute,
            media_adapter=self.task_builder.data_adapter.media,
        )
        training = legacy_adapter.prepare_training(plan.train_case_ids, run_dir)
        write_json(run_dir / "training" / "training_job.json", training)
        training = legacy_adapter.train(
            training, run_dir / "training_job.compat.json"
        )
        write_json(run_dir / "training" / "training_stage.json", training)
        if execute and training.get("status") == "failed":
            raise RuntimeError("WAN training failed; inference was not started")
        if (
            execute
            and instance_value["semantics"]["family"] == "finetune_eval"
            and training.get("status") == "complete"
        ):
            loss_command = [
                str(self.bundle.value["runtime"]["python"]),
                str(Path(__file__).resolve().parents[3] / "scripts" / "plot_wan22_loss.py"),
                "--run-dir",
                str(run_dir),
            ]
            completed = subprocess.run(loss_command, check=False)
            write_json(run_dir / "training" / "loss_export.json", {
                "command": loss_command,
                "return_code": completed.returncode,
                "status": "complete" if completed.returncode == 0 else "failed",
            })

        legacy_by_id = {case["case_id"]: case for case in legacy_cases}
        predictions = []
        for raw_job in instance_value["inference"]["jobs"]:
            adaptation = adaptations_by_id[raw_job["adaptation_id"]]
            compatibility_job = {
                **raw_job,
                "text_transform_id": adaptation["text_transform_id"],
                "adaptation": adaptation,
            }
            prepared = legacy_adapter.prepare_job(
                compatibility_job,
                legacy_by_id[raw_job["case_id"]],
                run_dir,
            )
            expected_prompt = raw_job["native_inputs"]["text"]["prompt"]
            if prepared["model_input"]["prompt"] != expected_prompt:
                raise AssertionError(
                    f"WAN compatibility renderer changed native input for "
                    f"{raw_job['job_id']}"
                )
            self._finalize_prepared_job(prepared, raw_job)
            job_path = run_dir / "jobs" / f"{raw_job['job_id']}.json"
            write_json(job_path, prepared)
            if stop_after_training:
                predictions.append({
                    "job_id": prepared["job_id"],
                    "case_id": prepared["case_id"],
                    "baseline_id": self.bundle.baseline_id,
                    "evaluation_partition": prepared["evaluation_partition"],
                    "status": "staged",
                    "video_path": None,
                    "manual_scores": {},
                    "job_spec": str(job_path),
                    "seed": int(prepared["seed"]),
                })
            elif not execute:
                prediction = legacy_adapter.generate(prepared, job_path)
                predictions.append(prediction)
        if execute and not stop_after_training:
            predictions = self._parallel_generate(
                run_dir, len(instance_value["inference"]["jobs"])
            )
        return training, predictions
