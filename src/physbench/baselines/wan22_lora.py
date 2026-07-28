from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .base import BaselineAdapter
from .wan22_media import Wan22MediaAdapter
from ..io import load_json, load_jsonl, write_json, write_jsonl


class Wan22LoraAdapter(BaselineAdapter):
    """WAN2.2-TI2V-5B + LoRA adapter for both benchmark task modes."""

    def __init__(
        self,
        config: dict[str, Any],
        execute: bool = False,
        media_adapter: Wan22MediaAdapter | None = None,
    ):
        super().__init__(config, execute=execute)
        self.runtime = config.get("runtime", {})
        self.generation = config.get("generation", {})
        self.media = media_adapter or Wan22MediaAdapter(config["media_adapter"])
        shared_cache = config.get("shared_media_cache", {})
        self.shared_media_cache_root = (
            Path(shared_cache["root"]).resolve()
            if shared_cache.get("enabled") and shared_cache.get("root")
            else None
        )
        self.project_root = Path(__file__).resolve().parents[3]

    def _check_scene(self, scene_id: str) -> None:
        supported = self.config.get("supported_scenes", "all")
        if supported != "all" and scene_id not in supported:
            raise ValueError(
                f"WAN baseline {self.baseline_id} is not declared valid for scene {scene_id}; "
                f"supported={supported}"
            )

    def _prompt(
        self,
        case: dict[str, Any],
        adaptation: dict[str, Any] | None = None,
    ) -> str:
        if adaptation is None:
            raise ValueError(
                f"case {case.get('case_id')} has no managed adaptation record"
            )
        if adaptation.get("case_id") != case.get("case_id"):
            raise ValueError(
                f"adaptation case {adaptation.get('case_id')} does not match "
                f"{case.get('case_id')}"
            )
        prompt = adaptation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"case {case.get('case_id')} has an invalid adapted prompt"
            )
        return prompt.strip()

    @staticmethod
    def _context(run_dir: Path) -> dict[str, Any]:
        return load_json(run_dir / "data_context.json")

    def _source(self, value: str, run_dir: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(self._context(run_dir)["manifest_dir"]) / path

    def _checkpoint_manifest(self, run_dir: Path) -> Path:
        return run_dir / "artifacts" / "wan22" / "checkpoint.json"

    def _media_output(
        self, default: Path, *, category: str, filename: str
    ) -> Path:
        if self.shared_media_cache_root is None:
            return default
        return self.shared_media_cache_root / category / filename

    @staticmethod
    def _speed_factor(case: dict[str, Any]) -> float:
        temporal = case.get("temporal", {})
        value = float(temporal.get("encoded_to_physical_speed", 1.0))
        if value <= 0:
            raise ValueError(
                f"case {case.get('case_id')} has invalid encoded_to_physical_speed={value}"
            )
        return value

    def _write_checkpoint(self, run_dir: Path, status: str, path: str | None, source: str) -> None:
        write_json(self._checkpoint_manifest(run_dir), {
            "status": status,
            "checkpoint": path,
            "source": source,
            "baseline_id": self.baseline_id,
        })

    def prepare_training(self, train_case_ids: list[str], run_dir: Path) -> dict[str, Any]:
        value = super().prepare_training(train_case_ids, run_dir)
        value.update({
            "adapter": "wan22_lora",
            "media_policy": self.config["media_adapter"],
            "lora": self.config.get("lora", {}),
            "wan_artifact_dir": str(run_dir / "artifacts" / "wan22"),
        })
        return value

    def _training_environment(self, run_dir: Path, dataset_dir: Path, metadata: Path) -> dict[str, str]:
        lora = self.config.get("lora", {})
        runtime_root = Path(self.runtime["project_root"])
        environment = {
            "DATA_DIR": str(dataset_dir),
            "METADATA_PATH": str(metadata),
            "MODEL_BASE": str(Path(self.runtime.get("model_base", runtime_root / "models"))),
            "OUTPUT_DIR": str(run_dir / "artifacts" / "wan22" / "checkpoints"),
            "RUN_NAME": run_dir.name,
            "NUM_FRAMES": str(self.media.max_frames),
            "DATASET_REPEAT": str(lora.get("dataset_repeat", 10)),
            "NUM_EPOCHS": str(lora.get("num_epochs", 5)),
            "LEARNING_RATE": str(lora.get("learning_rate", 1e-4)),
            "WEIGHT_DECAY": str(lora.get("weight_decay", 0.01)),
            "LORA_RANK": str(lora.get("rank", 32)),
            "LORA_TARGET_MODULES": str(lora.get("target_modules", "q,k,v,o,ffn.0,ffn.2")),
            "GRAD_ACCUM": str(lora.get("gradient_accumulation", 1)),
            "NUM_WORKERS": str(lora.get("num_workers", 4)),
            "TRAIN_SEED": str(lora.get("seed", 42)),
        }
        if self.media.dynamic_resolution:
            environment["DYNAMIC_RESOLUTION"] = "1"
            environment["MAX_PIXELS"] = str(self.media.max_pixels)
        else:
            environment["HEIGHT"] = str(self.media.height)
            environment["WIDTH"] = str(self.media.width)
        if lora.get("save_steps") is not None:
            environment["SAVE_STEPS"] = str(lora["save_steps"])
        initial = self.config.get("initial_lora_checkpoint")
        if initial:
            environment["LORA_CHECKPOINT"] = str(initial)
        visible = self.runtime.get("cuda_visible_devices")
        if visible is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(visible)
        accelerate_config = self.runtime.get("accelerate_config")
        if accelerate_config is not None:
            environment["ACCELERATE_CONFIG"] = str(accelerate_config)
        return environment

    def _balance_training_rows(
        self, rows: list[dict[str, str]], artifact_root: Path
    ) -> list[dict[str, str]]:
        policy = self.config.get("lora", {}).get("scene_balancing", "none")
        input_counts = Counter(row["scene_id"] for row in rows)
        if policy == "none":
            balanced = list(rows)
        elif policy == "oversample_each_scene_to_largest":
            by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_scene[row["scene_id"]].append(row)
            target = max((len(items) for items in by_scene.values()), default=0)
            balanced = []
            for scene_id in sorted(by_scene):
                items = sorted(by_scene[scene_id], key=lambda row: row["case_id"])
                balanced.extend(dict(items[index % len(items)]) for index in range(target))
        else:
            raise ValueError(f"unsupported lora.scene_balancing policy: {policy}")
        output_counts = Counter(row["scene_id"] for row in balanced)
        dataset_repeat = int(self.config.get("lora", {}).get("dataset_repeat", 10))
        num_epochs = int(self.config.get("lora", {}).get("num_epochs", 5))
        world_size = len(str(self.runtime.get("cuda_visible_devices", "0")).split(","))
        write_json(artifact_root / "training_sampling_plan.json", {
            "policy": policy,
            "unique_case_count": len(rows),
            "metadata_row_count": len(balanced),
            "input_scene_counts": dict(sorted(input_counts.items())),
            "balanced_scene_counts": dict(sorted(output_counts.items())),
            "dataset_repeat": dataset_repeat,
            "num_epochs": num_epochs,
            "world_size": world_size,
            "expected_optimizer_steps_per_epoch": (
                len(balanced) * dataset_repeat + world_size - 1
            ) // world_size,
            "expected_total_optimizer_steps": (
                ((len(balanced) * dataset_repeat + world_size - 1) // world_size)
                * num_epochs
            ),
        })
        return balanced

    def train(self, prepared_training: dict[str, Any], job_path: Path) -> dict[str, Any]:
        run_dir = job_path.parent
        artifact_root = run_dir / "artifacts" / "wan22"
        artifact_root.mkdir(parents=True, exist_ok=True)
        train_ids = prepared_training["case_ids"]
        if not train_ids:
            checkpoint = self.config.get("initial_lora_checkpoint")
            if not checkpoint and self.config.get("require_lora_checkpoint", True):
                raise ValueError("zero-training WAN2.2+LoRA task requires initial_lora_checkpoint")
            if self.execute and checkpoint and not Path(checkpoint).is_file():
                raise FileNotFoundError(f"frozen LoRA checkpoint not found: {checkpoint}")
            self._write_checkpoint(run_dir, "frozen" if checkpoint else "base_model", checkpoint, "baseline_config")
            return {
                **prepared_training,
                "status": "not_requested",
                "checkpoint_manifest": str(self._checkpoint_manifest(run_dir)),
                "checkpoint": checkpoint,
                "note": "Task 2: no task-specific training or finetuning was performed.",
            }

        cases = {case["case_id"]: case for case in load_jsonl(run_dir / "frozen_cases.jsonl")}
        adaptations = {
            item["case_id"]: item
            for item in load_jsonl(
                run_dir / "adaptations" / "case_adaptations.jsonl"
            )
            if item.get("role") == "train"
        }
        missing_adaptations = sorted(set(train_ids) - set(adaptations))
        if missing_adaptations:
            raise ValueError(
                "training adaptations missing cases: "
                f"{missing_adaptations}"
            )
        dataset_dir = artifact_root / "dataset"
        metadata_path = dataset_dir / "metadata.csv"
        audit_records = []
        metadata_rows = []
        for case_id in train_ids:
            case = cases[case_id]
            self._check_scene(case["scene_id"])
            source_value = case.get("assets", {}).get("reference_video") or case.get("assets", {}).get("physics_reference_video")
            if not source_value:
                raise ValueError(f"training case {case_id} has no video asset")
            source = self._source(source_value, run_dir)
            output = self._media_output(
                dataset_dir / "videos" / f"{case_id}.mp4",
                category="training_videos",
                filename=f"{case_id}.mp4",
            )
            record = self.media.normalize_video(
                source, output, materialize=self.execute,
                speed_factor=self._speed_factor(case),
                scene_id=case["scene_id"],
            )
            record.update(case_id=case_id, role="train")
            audit_records.append(record)
            metadata_rows.append({
                "video": (
                    output.relative_to(dataset_dir).as_posix()
                    if output.is_relative_to(dataset_dir)
                    else str(output)
                ),
                "prompt": self._prompt(case, adaptations[case_id]),
                "case_id": case_id,
                "scene_id": case["scene_id"],
                "text_transform_id": adaptations[case_id][
                    "text_transform_id"
                ],
            })
        metadata_rows = self._balance_training_rows(metadata_rows, artifact_root)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "video",
                    "prompt",
                    "case_id",
                    "scene_id",
                    "text_transform_id",
                ],
            )
            writer.writeheader()
            writer.writerows(metadata_rows)
        write_jsonl(artifact_root / "training_media_audit.jsonl", audit_records)

        runtime_root = Path(self.runtime["project_root"])
        command = ["bash", str(runtime_root / "scripts" / "training" / "train_lora.sh")]
        environment = self._training_environment(run_dir, dataset_dir, metadata_path)
        spec = {
            **prepared_training,
            "status": "planned",
            "command": command,
            "environment_overrides": environment,
            "metadata": str(metadata_path),
            "media_audit": str(artifact_root / "training_media_audit.jsonl"),
        }
        spec_path = artifact_root / "training_spec.json"
        write_json(spec_path, spec)
        if not self.execute:
            self._write_checkpoint(run_dir, "planned", None, "task1_finetune")
            return {
                **spec,
                "status": "planned",
                "checkpoint_manifest": str(self._checkpoint_manifest(run_dir)),
                "note": "WAN media derivatives and training command are planned; execution is disabled.",
            }

        env = os.environ.copy()
        env.update(environment)
        completed = subprocess.run(command, cwd=runtime_root, env=env, check=False)
        if completed.returncode != 0:
            return {**spec, "status": "failed", "return_code": completed.returncode}
        checkpoints = sorted(
            (run_dir / "artifacts" / "wan22" / "checkpoints").glob("*.safetensors"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not checkpoints:
            return {**spec, "status": "failed", "return_code": 0, "error": "trainer produced no LoRA checkpoint"}
        checkpoint = str(checkpoints[-1])
        self._write_checkpoint(run_dir, "complete", checkpoint, "task1_finetune")
        return {
            **spec,
            "status": "complete",
            "return_code": 0,
            "checkpoint": checkpoint,
            "checkpoint_manifest": str(self._checkpoint_manifest(run_dir)),
        }

    def _configured_checkpoint(self, run_dir: Path) -> str | None:
        manifest = self._checkpoint_manifest(run_dir)
        if manifest.is_file():
            return load_json(manifest).get("checkpoint")
        return self.config.get("initial_lora_checkpoint")

    def prepare_job(self, job: dict[str, Any], case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        self._check_scene(case["scene_id"])
        assets = case.get("assets", {})
        views = case.get("input_views", {})
        first_value = views.get("i2v", {}).get("first_frame") or assets.get("first_frame")
        physics_reference_value = assets.get("physics_reference_video")
        visual_reference_value = assets.get("reference_video") if case.get("has_real_reference_video") else None

        first_source_is_video = False
        if not first_value and physics_reference_value:
            first_value = physics_reference_value
            first_source_is_video = True
        if not first_value and self.config.get("require_image_condition", True):
            raise ValueError(f"case {case['case_id']} cannot provide or derive a WAN first frame")

        cache = run_dir / "artifacts" / "wan22"
        first_record = None
        normalized_first = None
        if first_value:
            first_source = self._source(first_value, run_dir)
            normalized_first = self._media_output(
                cache / "first_frames" / f"{case['case_id']}.png",
                category="first_frames",
                filename=f"{case['case_id']}.png",
            )
            first_record = self.media.normalize_first_frame(
                first_source, normalized_first,
                source_is_video=first_source_is_video, materialize=self.execute,
                scene_id=case["scene_id"],
            )

        reference_record = None
        normalized_reference = None
        if physics_reference_value:
            source = self._source(physics_reference_value, run_dir)
            normalized_reference = self._media_output(
                cache / "evaluation_references" / f"{case['case_id']}__physics.mp4",
                category="evaluation_references",
                filename=f"{case['case_id']}__physics.mp4",
            )
            reference_record = self.media.normalize_video(
                source, normalized_reference, materialize=self.execute,
                speed_factor=self._speed_factor(case),
                scene_id=case["scene_id"],
            )

        visual_record = None
        normalized_visual = None
        if visual_reference_value:
            if visual_reference_value == physics_reference_value:
                normalized_visual = normalized_reference
                visual_record = reference_record
            else:
                source = self._source(visual_reference_value, run_dir)
                normalized_visual = self._media_output(
                    cache / "evaluation_references" / f"{case['case_id']}__visual.mp4",
                    category="evaluation_references",
                    filename=f"{case['case_id']}__visual.mp4",
                )
                visual_record = self.media.normalize_video(
                    source, normalized_visual, materialize=self.execute,
                    speed_factor=self._speed_factor(case),
                    scene_id=case["scene_id"],
                )

        num_frames = (
            reference_record["generation_target_frames"]
            if reference_record
            else self.media.max_frames
        )
        output = (
            run_dir
            / "predictions"
            / self.baseline_id
            / f"{job['job_id']}.mp4"
        )
        checkpoint = self._configured_checkpoint(run_dir)
        media_profile = self.media.profile(case["scene_id"])
        return {
            **job,
            "baseline_id": self.baseline_id,
            "input_view": "i2v" if normalized_first else "t2v",
            "model_input": {
                "prompt": self._prompt(case, job.get("adaptation")),
                "first_frame": str(normalized_first) if normalized_first else None,
                "physical_parameters": case["physical_parameters"],
            },
            "output_video": str(output),
            "checkpoint": checkpoint,
            "checkpoint_manifest": str(self._checkpoint_manifest(run_dir)),
            "evaluation_reference_video": str(normalized_reference) if normalized_reference else None,
            "visual_reference_video": str(normalized_visual) if normalized_visual else None,
            "media_adaptation": {
                "first_frame": first_record,
                "physics_reference": reference_record,
                "visual_reference": visual_record,
            },
            "wan22": {
                "runtime": self.runtime,
                "generation": {**self.generation, "height": media_profile["height"], "width": media_profile["width"],
                               "fps": self.media.fps, "num_frames": num_frames},
            },
        }

    def generate(self, prepared_job: dict[str, Any], job_path: Path) -> dict[str, Any]:
        script = self.project_root / "scripts" / "wan22_generate.py"
        python = str(self.runtime.get("python", "python"))
        command = [python, str(script), "--job", str(job_path)]
        common = {
            "job_id": prepared_job["job_id"],
            "case_id": prepared_job["case_id"],
            "baseline_id": self.baseline_id,
            "evaluation_partition": prepared_job["evaluation_partition"],
            "manual_scores": {},
            "command": command,
            "seed": int(prepared_job["seed"]),
        }
        if not self.execute:
            return {
                **common,
                "status": "planned",
                "video_path": None,
                "note": "WAN generation is planned; execution is disabled.",
            }
        checkpoint = prepared_job.get("checkpoint")
        if self.config.get("require_lora_checkpoint", True) and not checkpoint:
            raise FileNotFoundError("WAN LoRA checkpoint is unavailable after training")
        if checkpoint and not Path(checkpoint).is_file():
            raise FileNotFoundError(f"WAN LoRA checkpoint not found: {checkpoint}")
        env = os.environ.copy()
        visible = self.runtime.get("cuda_visible_devices")
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(visible)
        if self.runtime.get("model_base"):
            env["MODEL_BASE"] = str(self.runtime["model_base"])
        completed = subprocess.run(
            command,
            cwd=Path(self.runtime["project_root"]),
            env=env,
            check=False,
        )
        output = Path(prepared_job["output_video"])
        status = "complete" if completed.returncode == 0 and output.is_file() else "failed"
        return {
            **common,
            "status": status,
            "video_path": str(output) if output.is_file() else None,
            "return_code": completed.returncode,
            "probe": self.media.probe(output) if output.is_file() else None,
        }
