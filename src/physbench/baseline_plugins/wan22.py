from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import BaselinePlugin, DataAdapter, TaskBuilder
from ..baselines.wan22_lora import Wan22LoraAdapter
from ..baselines.wan22_media import Wan22MediaAdapter
from ..domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from ..io import canonical_sha256, load_jsonl, write_json, write_jsonl
from ..prompts import PromptRegistry


class Wan22DataAdapter(DataAdapter):
    """WAN-owned, five-stage Case-to-model adaptation pipeline."""

    REQUIRED_STAGES = {"spatial", "temporal", "paradigm", "text", "physics"}

    def __init__(self, config: dict[str, Any], bundle_root: Path):
        self.config = config
        missing = self.REQUIRED_STAGES - set(config)
        if missing:
            raise ValueError(f"WAN data adapter missing stages: {sorted(missing)}")
        profile_dir = bundle_root / config["text"]["profiles_dir"]
        self.registry = PromptRegistry(profile_dir)
        if set(self.registry.profiles) != {"generic", "physics"}:
            raise ValueError(
                "WAN data-adapter profiles must define exactly generic and physics"
            )
        self.media = Wan22MediaAdapter(self.media_config)

    @property
    def stage_fingerprints(self) -> dict[str, str]:
        profiles = self.registry.snapshot(["generic", "physics"])["profiles"]
        return {
            "spatial": canonical_sha256(self.config["spatial"]),
            "temporal": canonical_sha256(self.config["temporal"]),
            "paradigm": canonical_sha256(self.config["paradigm"]),
            "text": canonical_sha256({
                "config": self.config["text"],
                "generic_profile": profiles["generic"],
            }),
            "physics": canonical_sha256({
                "config": self.config["physics"],
                "physics_profile": profiles["physics"],
            }),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "wan22_case_data_v2",
            "stages": self.stage_fingerprints,
        })

    @property
    def materialization_fingerprint(self) -> str:
        fingerprints = self.stage_fingerprints
        return canonical_sha256({
            name: fingerprints[name]
            for name in ("spatial", "temporal", "paradigm")
        })

    @property
    def media_config(self) -> dict[str, Any]:
        spatial = self.config["spatial"]
        temporal = self.config["temporal"]
        return {
            "width": spatial["width"],
            "height": spatial["height"],
            "spatial_policy": spatial["policy"],
            "pad_color": spatial["pad_color"],
            "aspect_ratio_buckets": spatial["aspect_ratio_buckets"],
            "fps": temporal["fps"],
            "max_frames": temporal["max_frames"],
            "min_frames": temporal["min_frames"],
            "temporal_policy": temporal["policy"],
            "cache_policy": self.config.get("cache", {}).get(
                "policy", "run_private"
            ),
        }

    def _spatial_profile(self, scene_id: str) -> dict[str, Any]:
        buckets = self.config["spatial"]["aspect_ratio_buckets"]
        if not buckets.get("enabled", False):
            return {
                "name": "default",
                "width": self.config["spatial"]["width"],
                "height": self.config["spatial"]["height"],
            }
        for name, profile in buckets["buckets"].items():
            if scene_id in profile.get("scene_ids", []):
                return {
                    "name": name,
                    "width": profile["width"],
                    "height": profile["height"],
                }
        raise ValueError(f"no WAN spatial profile configured for scene {scene_id}")

    def describe(self) -> dict[str, Any]:
        return {
            "type": "wan22_case_data_v2",
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "stage_fingerprints": self.stage_fingerprints,
            "stages": [
                "spatial",
                "temporal",
                "paradigm",
                "text",
                "physics",
            ],
            "ownership": "baseline",
            "source_assets_mutated": False,
            "native_inputs_are_opaque_to_benchmark": True,
            "cache_policy": self.config.get("cache", {}).get(
                "policy", "run_private"
            ),
            "config": self.config,
        }

    def adapt_case(
        self, case: dict[str, Any], conditioning: str, *, role: str
    ) -> dict[str, Any]:
        if conditioning not in {"generic", "physics"}:
            raise ValueError(f"unsupported WAN conditioning {conditioning}")
        # Text adaptation never receives structured physics.  The physics stage is
        # the only stage allowed to observe case["physics"].
        text_case = {
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "physical_parameters": {},
        }
        generic_record = self.registry.resolve(text_case, "generic", role=role)
        if generic_record["used_parameters"]:
            raise AssertionError("WAN text adaptation leaked physical parameters")

        if conditioning == "physics":
            physics_case = {
                "case_id": case["case_id"],
                "scene_id": case["scene_id"],
                "physical_parameters": case["physics"],
            }
            record = self.registry.resolve(physics_case, "physics", role=role)
        else:
            record = generic_record

        assets = case["assets"]
        first_frame = assets.get("first_frame")
        first_frame_source = (
            "assets.first_frame"
            if first_frame
            else "assets.physics_reference_video:frame0"
        )
        profile = self._spatial_profile(case["scene_id"])
        physics_strategy = (
            self.config["physics"]["strategy"]
            if conditioning == "physics"
            else "disabled"
        )
        native_inputs = {
            "vision": {
                "paradigm": self.config["paradigm"]["mode"],
                "first_frame_source": first_frame_source,
            },
            # This field is WAN-native.  Another baseline may expose tokens,
            # tensors, control streams, or any other opaque native payload.
            "text": {"prompt": record["prompt"]},
        }

        record["conditioning"] = conditioning
        record["schema_version"] = "2.0"
        record["data_adapter_fingerprint"] = self.fingerprint
        record["materialization_fingerprint"] = self.materialization_fingerprint
        record["stages"] = {
            "spatial": {
                "type": self.config["spatial"]["type"],
                "policy": self.config["spatial"]["policy"],
                "target_profile": profile,
                "materialization": "deferred_to_baseline_runtime",
            },
            "temporal": {
                "type": self.config["temporal"]["type"],
                "policy": self.config["temporal"]["policy"],
                "target_fps": self.config["temporal"]["fps"],
                "max_frames": self.config["temporal"]["max_frames"],
                "valid_frame_rule": "4n+1",
                "materialization": "deferred_to_baseline_runtime",
            },
            "paradigm": {
                "type": self.config["paradigm"]["type"],
                "mode": self.config["paradigm"]["mode"],
                "source": first_frame_source,
            },
            "text": {
                "type": self.config["text"]["type"],
                "generic_description": generic_record["prompt"],
                "contains_detailed_physics": False,
            },
            "physics": {
                "type": self.config["physics"]["type"],
                "enabled": conditioning == "physics",
                "strategy": physics_strategy,
                "native_target": (
                    self.config["physics"].get("native_target")
                    if conditioning == "physics"
                    else None
                ),
                "used_parameters": record["used_parameters"],
            },
        }
        record["native_inputs"] = native_inputs
        if conditioning == "generic" and record["used_parameters"]:
            raise AssertionError("generic conditioning leaked physical parameters")
        return record


class Wan22TaskBuilder(TaskBuilder):
    """Compile a canonical Benchmark plan into a sealed WAN-native task."""

    TYPE = "wan22_task_builder_v1"

    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        components = bundle.value["components"]
        builder_config = components["task_builder"]["config"]
        self.data_adapter = Wan22DataAdapter(
            builder_config["data_adapter"], bundle.root
        )

    @property
    def fingerprint(self) -> str:
        components = self.bundle.value["components"]
        return canonical_sha256({
            "type": self.TYPE,
            "data_adapter": self.data_adapter.fingerprint,
            "trainer": components["trainer"],
            "predictor": components["predictor"],
            "model": self.bundle.value.get("model", {}),
        })

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "fingerprint": self.fingerprint,
            "ownership": "baseline",
            "build_is_side_effect_free": True,
            "canonical_plan_owner": "benchmark",
            "data_adapter": self.data_adapter.describe(),
            "output": "BaselineTaskInstance",
        }

    def _validate_compatibility(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        plan: AtomicPlan,
    ) -> None:
        value = self.bundle.value
        supported_families = set(value["capabilities"]["task_families"])
        if task.family not in supported_families:
            raise ValueError(f"baseline does not support task family {task.family}")
        supported_conditioning = set(value["capabilities"]["conditioning"])
        if task.conditioning not in supported_conditioning:
            raise ValueError(f"baseline does not support {task.conditioning}")
        supported_scenes = value.get("supported_scenes", "all")
        if supported_scenes != "all":
            requested_scenes = set(plan.value["scene_ids"]) | {
                job["scene_id"] for job in plan.jobs
            }
            unknown = requested_scenes - set(supported_scenes)
            if unknown:
                raise ValueError(
                    f"baseline does not support scenes {sorted(unknown)}"
                )
        if task.family == "direct_eval":
            checkpoint = value.get("model", {}).get("frozen_lora_checkpoint")
            if checkpoint and not Path(checkpoint).is_file():
                raise FileNotFoundError(
                    f"frozen LoRA checkpoint not found: {checkpoint}"
                )

    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance:
        self._validate_compatibility(dataset, task, canonical_plan)
        by_id = {case["case_id"]: case for case in dataset.cases}
        train_ids = list(canonical_plan.train_case_ids)
        eval_case_ids = sorted({job["case_id"] for job in canonical_plan.jobs})
        selected_ids = sorted(set(train_ids) | set(eval_case_ids))

        adaptations: list[dict[str, Any]] = []
        adaptation_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for role, case_ids in (("train", train_ids), ("eval", eval_case_ids)):
            for case_id in case_ids:
                adaptation = self.data_adapter.adapt_case(
                    by_id[case_id], task.conditioning, role=role
                )
                adaptation_id = (
                    f"{case_id}::{role}::{task.conditioning}"
                )
                adaptation["adaptation_id"] = adaptation_id
                adaptations.append(adaptation)
                adaptation_by_key[(case_id, role)] = adaptation
        adaptations.sort(key=lambda item: item["adaptation_id"])

        model_ref = (
            "artifact://train/model"
            if task.family == "finetune_eval"
            else "baseline://frozen_model"
        )
        inference_jobs = []
        for job in canonical_plan.jobs:
            adaptation = adaptation_by_key[(job["case_id"], "eval")]
            inference_jobs.append({
                **job,
                "adaptation_id": adaptation["adaptation_id"],
                "model_ref": model_ref,
                "native_inputs": adaptation["native_inputs"],
            })
        inference_jobs.sort(key=lambda item: item["job_id"])

        components = self.bundle.value["components"]
        training = None
        operations = []
        if task.family == "finetune_eval":
            training = {
                "operation_id": "train",
                "case_ids": train_ids,
                "adaptation_ids": [
                    adaptation_by_key[(case_id, "train")]["adaptation_id"]
                    for case_id in train_ids
                ],
                "seed": canonical_plan.value["training_seed"],
                "trainer": components["trainer"],
                "outputs": {"model": "artifact://train/model"},
            }
            operations.append({
                "operation_id": "train",
                "kind": "train",
                "depends_on": [],
                "outputs": ["artifact://train/model"],
            })
        operations.extend([
            {
                "operation_id": "infer",
                "kind": "infer",
                "depends_on": ["train"] if training else [],
                "model_ref": model_ref,
                "job_ids": [job["job_id"] for job in inference_jobs],
            },
            {
                "operation_id": "evaluate",
                "kind": "evaluate",
                "depends_on": ["infer"],
                "job_ids": [job["job_id"] for job in inference_jobs],
            },
        ])

        cache_root = (
            Path(__file__).resolve().parents[3]
            / "cache"
            / "baselines"
            / self.bundle.baseline_id
            / self.data_adapter.materialization_fingerprint
            / dataset.digest
        )
        plan_digest = canonical_sha256(canonical_plan.value)
        instance_id = (
            f"{task.task_id}__{self.bundle.baseline_id}"
            f"__{self.fingerprint[:12]}"
        )
        return BaselineTaskInstance.seal({
            "schema_version": "2.1",
            "instance_id": instance_id,
            "identity": {
                "dataset": {
                    "dataset_id": dataset.dataset_id,
                    "digest": dataset.digest,
                },
                "task": {
                    "task_id": task.task_id,
                    "digest": task.digest,
                },
                "baseline": {
                    "baseline_id": self.bundle.baseline_id,
                    "digest": self.bundle.digest,
                },
                "task_builder": {
                    "type": self.TYPE,
                    "fingerprint": self.fingerprint,
                },
                "data_adapter": {
                    "fingerprint": self.data_adapter.fingerprint,
                    "materialization_fingerprint": (
                        self.data_adapter.materialization_fingerprint
                    ),
                },
                "canonical_plan_digest": plan_digest,
            },
            "semantics": {
                "family": task.family,
                "conditioning": task.conditioning,
                "scene_ids": canonical_plan.value["scene_ids"],
            },
            "canonical_plan": canonical_plan.value,
            "source": {
                "asset_root": str(dataset.asset_root),
                "cases": [by_id[case_id] for case_id in selected_ids],
            },
            "adaptations": adaptations,
            "training": training,
            "inference": {
                "predictor": components["predictor"],
                "jobs": inference_jobs,
            },
            "execution_graph": {"operations": operations},
            "cache_bindings": [{
                "kind": "media_derivatives",
                "policy": "content_addressed_shared_immutable",
                "root": str(cache_root),
                "dataset_digest": dataset.digest,
                "materialization_fingerprint": (
                    self.data_adapter.materialization_fingerprint
                ),
            }],
            "baseline_payload": {
                "type": "wan22_task_v1",
                "model": self.bundle.value.get("model", {}),
                "runtime": self.bundle.value["runtime"],
                "media_adapter": self.data_adapter.media_config,
            },
        })


class Wan22BaselinePlugin(BaselinePlugin):
    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self.task_builder = Wan22TaskBuilder(bundle)

    def _legacy_config(
        self, instance_value: dict[str, Any]
    ) -> dict[str, Any]:
        value = self.bundle.value
        components = value["components"]
        frozen_checkpoint = value.get("model", {}).get("frozen_lora_checkpoint")
        is_training = instance_value["semantics"]["family"] == "finetune_eval"
        trainer_config = copy.deepcopy(components["trainer"]["config"])
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
            "physical_parameters": case["physics"],
            "appearance": case["appearance"],
            "temporal": case["temporal"],
            "alignment": case.get("alignment"),
            "assets": assets,
            "has_real_reference_video": case["has_real_reference_video"],
            "ood": case["ood"],
            "provenance": case["provenance"],
            "text": {"description": f"{case['scene_id']} physical video case"},
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
        worker_script = (
            Path(__file__).resolve().parents[3] / "scripts" / "wan22_generate_batch.py"
        )
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
        # Private compatibility projection consumed by the existing WAN trainer.
        # It is not a Benchmark-level ConditionAdapter contract.
        write_jsonl(run_dir / "resolved_prompts.jsonl", adaptation_records)

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
        legacy_adapter = Wan22LoraAdapter(
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
        conditioning = instance_value["semantics"]["conditioning"]
        for raw_job in instance_value["inference"]["jobs"]:
            adaptation = adaptations_by_id[raw_job["adaptation_id"]]
            compatibility_job = {
                **raw_job,
                "prompt_profile_id": conditioning,
                "resolved_prompt": adaptation,
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
            # ModelInput is the exact consumable boundary. Structured physics stays
            # in the frozen Dataset and condition audit, never as a side channel.
            prepared["model_input"].pop("physical_parameters", None)
            job_path = run_dir / "jobs" / f"{raw_job['job_id']}.json"
            write_json(job_path, prepared)
            if stop_after_training:
                predictions.append({
                    "job_id": prepared["job_id"],
                    "case_id": prepared["case_id"],
                    "baseline_id": self.bundle.baseline_id,
                    "conditioning": conditioning,
                    "prompt_profile_id": conditioning,
                    "evaluation_partition": prepared["evaluation_partition"],
                    "status": "staged",
                    "video_path": None,
                    "manual_scores": {},
                    "job_spec": str(job_path),
                })
            elif not execute:
                prediction = legacy_adapter.generate(prepared, job_path)
                prediction["conditioning"] = conditioning
                predictions.append(prediction)
        if execute and not stop_after_training:
            predictions = self._parallel_generate(
                run_dir, len(instance_value["inference"]["jobs"])
            )
            for prediction in predictions:
                prediction["conditioning"] = conditioning
        return training, predictions
