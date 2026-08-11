from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .wan22_lora import Wan22LoraAdapter
from .wan22_st_tube_iou_masks import (
    DEFAULT_SAM2_MODEL_ID,
    materialize_subject_mask_tube,
    release_mask_segmenter,
    resolve_training_mask_manifest,
)
from ..evaluation.common.masks.sam2 import Sam2VideoSegmenter
from ..io import load_json, load_jsonl, sha256_file, write_json, write_jsonl


class Wan22STTubeIoULoraAdapter(Wan22LoraAdapter):
    """WAN LoRA orchestration with training-only subject-tube supervision."""

    def _training_parallelism(
        self,
        *,
        metadata_row_count: int | None = None,
    ) -> dict[str, int]:
        visible_devices = [
            item.strip()
            for item in str(self.runtime.get("cuda_visible_devices", "0")).split(",")
            if item.strip()
        ]
        world_size = len(visible_devices)
        if world_size < 1:
            raise ValueError("ST tube-IoU training requires at least one GPU")
        lora = self.config["lora"]
        micro_batch_size = int(lora["micro_batch_size"])
        global_batch_size = int(lora["global_batch_size"])
        distributed_micro_batch = world_size * micro_batch_size
        if micro_batch_size < 1 or global_batch_size < 1:
            raise ValueError("training batch sizes must be positive")
        if global_batch_size % distributed_micro_batch:
            raise ValueError(
                "global_batch_size must be divisible by world_size times "
                "micro_batch_size"
            )
        accumulation = global_batch_size // distributed_micro_batch
        contract = {
            "world_size": world_size,
            "micro_batch_size": micro_batch_size,
            "global_batch_size": global_batch_size,
            "gradient_accumulation_steps": accumulation,
        }
        if metadata_row_count is None:
            return contract
        repeated_rows = metadata_row_count * int(lora["dataset_repeat"])
        if repeated_rows % world_size:
            raise ValueError("repeated training rows must divide the world size")
        micro_steps = repeated_rows // world_size
        if micro_steps % accumulation:
            raise ValueError("micro steps per epoch must divide accumulation")
        optimizer_steps = micro_steps // accumulation
        contract.update({
            "micro_steps_per_epoch": micro_steps,
            "optimizer_steps_per_epoch": optimizer_steps,
            "total_optimizer_steps": optimizer_steps * int(lora["num_epochs"]),
        })
        return contract

    def prepare_training(
        self,
        train_case_ids: list[str],
        run_dir: Path,
    ) -> dict[str, Any]:
        prepared = super().prepare_training(train_case_ids, run_dir)
        prepared.update({
            "adapter": "wan22_st_tube_iou",
            "st_tube_iou": self.config["st_tube_iou"],
            "checkpoint_format": "dit_lora_plus_separate_st_occupancy_head_v1",
            "mask_supervision_scope": "training_only",
        })
        return prepared

    def _training_metadata_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / "metadata.jsonl"

    def _training_metadata_row(
        self,
        *,
        case: dict[str, Any],
        adaptation: dict[str, Any],
        video: str,
        subject_mask: str | None = None,
    ) -> dict[str, Any]:
        if subject_mask is None:
            subject_mask = f"masks/{case['case_id']}.npz"
        return {
            "video": video,
            "subject_mask": subject_mask,
            "prompt": adaptation["native_inputs"]["text"]["prompt"],
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "text_transform_id": adaptation["text_transform_id"],
        }

    def _write_training_metadata(
        self,
        path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        write_jsonl(path, rows)

    def _balance_training_rows(
        self,
        rows: list[dict[str, Any]],
        artifact_root: Path,
    ) -> list[dict[str, Any]]:
        policy = self.config["lora"].get("scene_balancing")
        if policy != "oversample_each_scene_to_largest_world_aligned":
            raise ValueError(f"unsupported ST tube-IoU balancing policy: {policy}")
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scene[row["scene_id"]].append(row)
        world_size = self._training_parallelism()["world_size"]
        scene_count = len(by_scene)
        alignment = world_size // math.gcd(world_size, scene_count)
        largest = max((len(items) for items in by_scene.values()), default=0)
        target = ((largest + alignment - 1) // alignment) * alignment
        balanced: list[dict[str, Any]] = []
        for scene_id in sorted(by_scene):
            items = sorted(by_scene[scene_id], key=lambda row: row["case_id"])
            balanced.extend(dict(items[index % len(items)]) for index in range(target))
        parallelism = self._training_parallelism(metadata_row_count=len(balanced))
        lora = self.config["lora"]
        write_json(artifact_root / "training_sampling_plan.json", {
            "policy": policy,
            "unique_case_count": len(rows),
            "metadata_row_count": len(balanced),
            "input_scene_counts": dict(sorted(Counter(row["scene_id"] for row in rows).items())),
            "balanced_scene_counts": dict(sorted(Counter(row["scene_id"] for row in balanced).items())),
            "per_scene_target": target,
            "world_alignment": alignment,
            "dataset_repeat": int(lora["dataset_repeat"]),
            "num_epochs": int(lora["num_epochs"]),
            **parallelism,
            "expected_optimizer_steps_per_epoch": parallelism["optimizer_steps_per_epoch"],
            "expected_total_optimizer_steps": parallelism["total_optimizer_steps"],
            "shuffle": True,
            "sampler_seed": int(lora["seed"]),
            "sampler_generator": "torch.Generator",
            "cross_rank_duplicate_policy": "forbidden_after_balancing",
        })
        return balanced

    @staticmethod
    def _accelerate_config_path(run_dir: Path) -> Path:
        return run_dir / "artifacts" / "wan22" / "accelerate_config.yaml"

    @staticmethod
    def _accelerate_config_audit_path(run_dir: Path) -> Path:
        return run_dir / "artifacts" / "wan22" / "accelerate_config.audit.json"

    def _configured_accelerate_config(self) -> Path:
        configured = self.runtime.get("accelerate_config")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("runtime.accelerate_config is required")
        path = Path(configured)
        if not path.is_absolute():
            path = Path(self.runtime["project_root"]) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Accelerate config not found: {path}")
        return path

    def _freeze_accelerate_config(self, run_dir: Path) -> dict[str, Any]:
        source = self._configured_accelerate_config()
        frozen = self._accelerate_config_path(run_dir).resolve()
        audit_path = self._accelerate_config_audit_path(run_dir)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.is_file() and audit_path.is_file():
            value = load_json(audit_path)
            if value.get("frozen_sha256") != sha256_file(frozen):
                raise ValueError("run-local Accelerate config hash mismatch")
            return value
        if frozen.exists() or audit_path.exists():
            raise RuntimeError("incomplete run-local Accelerate config lock")
        source_hash = sha256_file(source)
        shutil.copyfile(source, frozen)
        if sha256_file(frozen) != source_hash:
            raise RuntimeError("Accelerate config changed while being frozen")
        audit = {
            "schema_version": "1.0",
            "source_path": str(source),
            "source_sha256": source_hash,
            "frozen_path": str(frozen),
            "frozen_sha256": source_hash,
            "size": frozen.stat().st_size,
        }
        write_json(audit_path, audit)
        return audit

    def _training_command(self, runtime_root: Path) -> list[str]:
        del runtime_root
        return [
            "bash",
            str(self.project_root / "scripts" / "train_wan22_st_tube_iou.sh"),
        ]

    def _training_environment(
        self,
        run_dir: Path,
        dataset_dir: Path,
        metadata: Path,
    ) -> dict[str, str]:
        environment = super()._training_environment(run_dir, dataset_dir, metadata)
        parallelism = self._training_parallelism()
        accelerate = self._freeze_accelerate_config(run_dir)
        environment.update({
            "WAN_PROJECT_ROOT": str(self.runtime["project_root"]),
            "WAN_PYTHON": str(self.runtime["python"]),
            "BENCHMARK_ROOT": str(self.project_root),
            "ST_TUBE_IOU_CONFIG_JSON": json.dumps(
                self.config["st_tube_iou"], sort_keys=True, separators=(",", ":")
            ),
            "ST_TUBE_IOU_METRICS_PATH": str(
                run_dir / "artifacts" / "wan22" / "training_st_tube_iou_metrics.jsonl"
            ),
            "SAVE_OPTIMIZER_STATE": (
                "1" if self.config["lora"].get("save_optimizer_state") else "0"
            ),
            "GRAD_ACCUM": str(parallelism["gradient_accumulation_steps"]),
            "ACCELERATE_CONFIG": accelerate["frozen_path"],
            "ACCELERATE_CONFIG_SHA256": accelerate["frozen_sha256"],
            "ACCELERATE_CONFIG_AUDIT_PATH": str(
                self._accelerate_config_audit_path(run_dir)
            ),
        })
        return environment

    def _generation_script(self) -> Path:
        return self.project_root / "scripts" / "wan22_generate.py"

    def train(self, prepared_training: dict[str, Any], job_path: Path) -> dict[str, Any]:
        run_dir = job_path.parent
        artifact_root = run_dir / "artifacts" / "wan22"
        artifact_root.mkdir(parents=True, exist_ok=True)
        train_ids = prepared_training["case_ids"]
        if not train_ids:
            return super().train(prepared_training, job_path)

        cases = {case["case_id"]: case for case in load_jsonl(run_dir / "frozen_cases.jsonl")}
        adaptations = {
            item["case_id"]: item
            for item in load_jsonl(run_dir / "adaptations" / "case_adaptations.jsonl")
            if item.get("role") == "train"
        }
        missing = sorted(set(train_ids) - set(adaptations))
        if missing:
            raise ValueError(f"training adaptations missing cases: {missing}")
        dataset_root = Path(self._context(run_dir)["manifest_dir"]).resolve()
        dataset_dir = artifact_root / "dataset"
        metadata_path = self._training_metadata_path(dataset_dir)
        media_audits: list[dict[str, Any]] = []
        tube_audits: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        segmenter = None
        if self.execute:
            segmenter = Sam2VideoSegmenter({
                "model_id": self.config["st_tube_iou"].get(
                    "mask_segmenter_model_id", DEFAULT_SAM2_MODEL_ID
                ),
                "device": "cuda",
            })
        for case_id in train_ids:
            case = cases[case_id]
            self._check_scene(case["scene_id"])
            source_value = case.get("assets", {}).get("reference_video")
            if not source_value:
                raise ValueError(f"training case {case_id} has no video target")
            source = self._source(source_value, run_dir)
            output_video = self._media_output(
                dataset_dir / "videos" / f"{case_id}.mp4",
                category="training_videos",
                filename=f"{case_id}.mp4",
            )
            media_record = self.media.normalize_video(
                source,
                output_video,
                materialize=self.execute,
                speed_factor=self._speed_factor(case),
                scene_id=case["scene_id"],
            )
            media_record.update(case_id=case_id, role="train")
            media_audits.append(media_record)
            output_mask = dataset_dir / "masks" / f"{case_id}.npz"
            if self.execute:
                manifest = resolve_training_mask_manifest(
                    source,
                    dataset_root=dataset_root,
                    case_id=case_id,
                )
                tube_audits.append(materialize_subject_mask_tube(
                    normalized_video=output_video,
                    mask_manifest=manifest,
                    dataset_root=dataset_root,
                    output=output_mask,
                    case_id=case_id,
                    segmenter=segmenter,
                ))
            rows.append(self._training_metadata_row(
                case=case,
                adaptation=adaptations[case_id],
                video=(
                    output_video.relative_to(dataset_dir).as_posix()
                    if output_video.is_relative_to(dataset_dir)
                    else str(output_video)
                ),
                subject_mask=output_mask.relative_to(dataset_dir).as_posix(),
            ))
        release_mask_segmenter(segmenter)
        segmenter = None
        rows = self._balance_training_rows(rows, artifact_root)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_training_metadata(metadata_path, rows)
        write_jsonl(artifact_root / "training_media_audit.jsonl", media_audits)
        write_jsonl(artifact_root / "training_mask_tube_audit.jsonl", tube_audits)

        runtime_root = Path(self.runtime["project_root"])
        command = self._training_command(runtime_root)
        environment = self._training_environment(run_dir, dataset_dir, metadata_path)
        spec = {
            **prepared_training,
            "status": "planned",
            "command": command,
            "environment_overrides": environment,
            "metadata": str(metadata_path),
            "media_audit": str(artifact_root / "training_media_audit.jsonl"),
            "mask_tube_audit": str(artifact_root / "training_mask_tube_audit.jsonl"),
        }
        write_json(artifact_root / "training_spec.json", spec)
        if not self.execute:
            self._write_checkpoint(run_dir, "planned", None, "task1_finetune")
            return {
                **spec,
                "status": "planned",
                "checkpoint_manifest": str(self._checkpoint_manifest(run_dir)),
            }

        env = os.environ.copy()
        env.update(environment)
        completed = subprocess.run(command, cwd=runtime_root, env=env, check=False)
        if completed.returncode != 0:
            return {**spec, "status": "failed", "return_code": completed.returncode}
        checkpoints = sorted(
            (
                path for path in (artifact_root / "checkpoints").glob("*.safetensors")
                if not path.name.endswith(".st-head.safetensors")
            ),
            key=self._checkpoint_order,
        )
        if not checkpoints:
            return {**spec, "status": "failed", "error": "trainer produced no LoRA checkpoint"}
        checkpoint = checkpoints[-1]
        head = checkpoint.with_name(f"{checkpoint.stem}.st-head.safetensors")
        if not head.is_file():
            return {**spec, "status": "failed", "error": "trainer produced no paired ST head"}
        manifest = {
            "status": "complete",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "auxiliary_head": str(head),
            "auxiliary_head_sha256": sha256_file(head),
            "auxiliary_head_inference_required": False,
            "source": "task1_finetune",
            "baseline_id": self.baseline_id,
        }
        write_json(self._checkpoint_manifest(run_dir), manifest)
        return {
            **spec,
            "status": "complete",
            "return_code": 0,
            "checkpoint": str(checkpoint),
            "auxiliary_head": str(head),
            "checkpoint_manifest": str(self._checkpoint_manifest(run_dir)),
        }


__all__ = ["Wan22STTubeIoULoraAdapter"]
