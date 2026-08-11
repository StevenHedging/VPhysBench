from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .wan22_lora import Wan22LoraAdapter
from .wan22_quantity import Wan22QuantityLoraAdapter
from .wan22_quantity_model import (
    WAN22_TI2V_5B_LORA_PAIR_COUNT,
    WAN22_TI2V_5B_LORA_RANK,
    WAN22_TI2V_5B_LORA_TENSOR_COUNT,
    expected_wan22_ti2v_5b_lora_targets,
)
from .wan22_symbol_value_model import (
    SYMBOL_VALUE_CHECKPOINT_PREFIX,
    SymbolValueConditioner,
    _conditioner_state_dict,
    _validate_conditioner_state_dict,
    symbol_value_pipeline_shared_fingerprint,
    verify_symbol_value_checkpoint_manifest,
)
from ..io import load_json, sha256_file, write_json, write_jsonl


_LORA_KEY = re.compile(
    r"^(?P<target>.+)\.lora_(?P<side>A|B)(?P<adapter>\.default)?\.weight$"
)


class Wan22SymbolValueLoraAdapter(Wan22QuantityLoraAdapter):
    """WAN media orchestration for symbol/value text cross-attention."""

    def _training_parallelism(
        self,
        *,
        metadata_row_count: int | None = None,
    ) -> dict[str, int]:
        visible_devices = [
            item.strip()
            for item in str(
                self.runtime.get("cuda_visible_devices", "0")
            ).split(",")
            if item.strip()
        ]
        world_size = len(visible_devices)
        if world_size < 1:
            raise ValueError("symbol-value training requires at least one GPU")
        lora = self.config["lora"]
        micro_batch_size = int(lora["micro_batch_size"])
        global_batch_size = int(lora["global_batch_size"])
        if micro_batch_size < 1 or global_batch_size < 1:
            raise ValueError("training batch sizes must be positive")
        distributed_micro_batch = world_size * micro_batch_size
        if global_batch_size % distributed_micro_batch:
            raise ValueError(
                "global_batch_size must be divisible by world_size times "
                "micro_batch_size: "
                f"{global_batch_size} vs {world_size}*{micro_batch_size}"
            )
        gradient_accumulation_steps = (
            global_batch_size // distributed_micro_batch
        )
        contract = {
            "world_size": world_size,
            "micro_batch_size": micro_batch_size,
            "global_batch_size": global_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        if metadata_row_count is None:
            return contract
        dataset_repeat = int(lora["dataset_repeat"])
        num_epochs = int(lora["num_epochs"])
        repeated_rows = metadata_row_count * dataset_repeat
        if repeated_rows % world_size:
            raise ValueError(
                "repeated training rows must divide the distributed world"
            )
        micro_steps_per_epoch = repeated_rows // world_size
        if micro_steps_per_epoch % gradient_accumulation_steps:
            raise ValueError(
                "micro steps per epoch must divide gradient accumulation"
            )
        optimizer_steps_per_epoch = (
            micro_steps_per_epoch // gradient_accumulation_steps
        )
        contract.update({
            "micro_steps_per_epoch": micro_steps_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": optimizer_steps_per_epoch * num_epochs,
        })
        return contract

    def prepare_training(
        self,
        train_case_ids: list[str],
        run_dir: Path,
    ) -> dict[str, Any]:
        prepared = Wan22LoraAdapter.prepare_training(self, train_case_ids, run_dir)
        prepared.update({
            "adapter": "wan22_symbol_value_cross_attention",
            "symbol_value_conditioner": self.config["symbol_value_conditioner"],
            "checkpoint_format": (
                "combined_dit_lora_and_symbol_value_conditioner_safetensors_v1"
            ),
        })
        return prepared

    def _training_metadata_row(
        self,
        *,
        case: dict[str, Any],
        adaptation: dict[str, Any],
        video: str,
    ) -> dict[str, Any]:
        native = adaptation["native_inputs"]
        physics = native["physics"]
        return {
            "video": video,
            "prompt": native["text"]["prompt"],
            "quantities": physics["quantities"],
            "quantity_registry_id": physics["registry_id"],
            "quantity_registry_fingerprint": physics["registry_fingerprint"],
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
        policy = self.config.get("lora", {}).get("scene_balancing")
        if policy != "oversample_each_scene_to_largest_world_aligned":
            raise ValueError(f"unsupported symbol-value balancing policy: {policy}")
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
        if len(balanced) % world_size:
            raise AssertionError("world-aligned balancing did not divide world size")
        lora = self.config["lora"]
        parallelism = self._training_parallelism(
            metadata_row_count=len(balanced),
        )
        plan = {
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
            "expected_optimizer_steps_per_epoch": parallelism[
                "optimizer_steps_per_epoch"
            ],
            "expected_total_optimizer_steps": parallelism[
                "total_optimizer_steps"
            ],
            "shuffle": True,
            "sampler_seed": int(lora["seed"]),
            "sampler_generator": "torch.Generator",
            "cross_rank_duplicate_policy": "forbidden_after_balancing",
        }
        write_json(artifact_root / "training_sampling_plan.json", plan)
        return balanced

    def _training_command(self, runtime_root: Path) -> list[str]:
        del runtime_root
        return ["bash", str(self.project_root / "scripts" / "train_wan22_symbol_value.sh")]

    def _training_environment(
        self,
        run_dir: Path,
        dataset_dir: Path,
        metadata: Path,
    ) -> dict[str, str]:
        environment = Wan22LoraAdapter._training_environment(
            self, run_dir, dataset_dir, metadata
        )
        parallelism = self._training_parallelism()
        environment.update({
            "WAN_PROJECT_ROOT": str(self.runtime["project_root"]),
            "WAN_PYTHON": str(self.runtime["python"]),
            "BENCHMARK_ROOT": str(self.project_root),
            "SYMBOL_VALUE_CONDITIONER_CONFIG_JSON": json.dumps(
                self.config["symbol_value_conditioner"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "SYMBOL_VALUE_TOKEN_AUDIT_PATH": str(
                run_dir / "artifacts" / "wan22" / "training_symbol_value_token_audit.jsonl"
            ),
            "SAVE_OPTIMIZER_STATE": (
                "1" if self.config.get("lora", {}).get("save_optimizer_state") else "0"
            ),
            "GRAD_ACCUM": str(
                parallelism["gradient_accumulation_steps"]
            ),
        })
        accelerate = self._freeze_accelerate_config(run_dir)
        environment.update({
            "ACCELERATE_CONFIG": accelerate["frozen_path"],
            "ACCELERATE_CONFIG_SHA256": accelerate["frozen_sha256"],
            "ACCELERATE_CONFIG_AUDIT_PATH": str(
                self._accelerate_config_audit_path(run_dir)
            ),
        })
        return environment

    def _generation_script(self) -> Path:
        return self.project_root / "scripts" / "wan22_symbol_value_generate.py"

    def _checkpoint_inventory(self, path: Path) -> dict[str, Any]:
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        conditioner_state = _conditioner_state_dict(state)
        conditioner = SymbolValueConditioner(self.config["symbol_value_conditioner"])
        _validate_conditioner_state_dict(conditioner, conditioner_state)
        conditioner_keys = {
            key for key in state
            if key.startswith(SYMBOL_VALUE_CHECKPOINT_PREFIX)
            or key.startswith("symbol_value_conditioner.")
        }
        lora = {key: value for key, value in state.items() if key not in conditioner_keys}
        if len(lora) != WAN22_TI2V_5B_LORA_TENSOR_COUNT:
            raise ValueError(
                f"combined checkpoint requires 600 LoRA tensors, got {len(lora)}"
            )
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in lora.items():
            match = _LORA_KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"unsupported combined checkpoint tensor: {key}")
            target = match.group("target").removeprefix("diffusion_model.")
            pairs.setdefault(target, {})[match.group("side")] = tensor
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"LoRA tensor must be finite floating point: {key}")
        if len(pairs) != WAN22_TI2V_5B_LORA_PAIR_COUNT:
            raise ValueError(f"combined checkpoint requires 300 LoRA pairs, got {len(pairs)}")
        if set(pairs) != expected_wan22_ti2v_5b_lora_targets():
            raise ValueError("combined checkpoint LoRA target topology mismatch")
        for target, pair in pairs.items():
            if set(pair) != {"A", "B"}:
                raise ValueError(f"incomplete LoRA pair: {target}")
            if pair["A"].ndim != 2 or pair["B"].ndim != 2:
                raise ValueError(f"LoRA tensors must be matrices: {target}")
            if pair["A"].shape[0] != WAN22_TI2V_5B_LORA_RANK or pair["B"].shape[1] != WAN22_TI2V_5B_LORA_RANK:
                raise ValueError(f"LoRA rank mismatch: {target}")
        return {
            "tensor_count": len(state),
            "lora_tensor_count": len(lora),
            "lora_pair_count": len(pairs),
            "lora_rank": WAN22_TI2V_5B_LORA_RANK,
            "symbol_value_conditioner_tensor_count": len(conditioner_state),
            "symbol_value_conditioner_state_keys": sorted(conditioner_state),
            "finite_payload_verified": True,
            "topology_verified": True,
        }

    def train(
        self,
        prepared_training: dict[str, Any],
        job_path: Path,
    ) -> dict[str, Any]:
        run_dir = job_path.parent
        if self.execute and prepared_training["case_ids"]:
            self._write_model_asset_manifest(run_dir)
        result = Wan22LoraAdapter.train(self, prepared_training, job_path)
        checkpoint_value = result.get("checkpoint")
        if self.execute and result.get("status") == "complete" and checkpoint_value:
            checkpoint = Path(checkpoint_value).resolve()
            inventory = self._checkpoint_inventory(checkpoint)
            state_root = checkpoint.parent / "training_state_latest"
            state_path = state_root / "optimizer_scheduler.pt"
            manifest = {
                "schema_version": "2.0",
                "status": "complete",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "source": "task1_finetune",
                "baseline_id": self.baseline_id,
                "inventory": inventory,
                "symbol_value_conditioner_config": self.config[
                    "symbol_value_conditioner"
                ],
                "base_model_assets": str(self._model_asset_manifest(run_dir)),
                "training_state": (
                    {
                        "directory": str(state_root),
                        "optimizer_scheduler": str(state_path),
                        "optimizer_scheduler_sha256": sha256_file(state_path),
                        "state_manifest": str(state_root / "state.json"),
                    }
                    if state_path.is_file() else None
                ),
            }
            write_json(self._checkpoint_manifest(run_dir), manifest)
            result["checkpoint_sha256"] = manifest["checkpoint_sha256"]
            result["checkpoint_inventory"] = inventory
        return result

    def prepare_job(
        self,
        job: dict[str, Any],
        case: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        prepared = Wan22LoraAdapter.prepare_job(self, job, case, run_dir)
        native = job["adaptation"]["native_inputs"]
        prepared["model_input"].pop("physical_parameters", None)
        prepared["model_input"]["quantity_payload"] = native["physics"]
        prepared["symbol_value_token_audit"] = str(
            run_dir / "artifacts" / "wan22" / "inference_symbol_value_token_audits"
            / f"{job['job_id']}.json"
        )
        prepared["wan22"]["symbol_value_conditioner"] = self.config[
            "symbol_value_conditioner"
        ]
        return prepared

    def generate(
        self,
        prepared_job: dict[str, Any],
        job_path: Path,
    ) -> dict[str, Any]:
        if self.execute:
            checkpoint = prepared_job.get("checkpoint")
            manifest = prepared_job.get("checkpoint_manifest")
            if not checkpoint or not manifest:
                raise FileNotFoundError(
                    "symbol-value generation requires checkpoint and manifest"
                )
            verify_symbol_value_checkpoint_manifest(checkpoint, manifest)
        result = Wan22LoraAdapter.generate(self, prepared_job, job_path)
        result["scene_id"] = prepared_job["scene_id"]
        if self.execute and result.get("status") == "complete":
            audit_path = Path(prepared_job["symbol_value_token_audit"])
            audit = load_json(audit_path)
            verification = audit.get("checkpoint_verification")
            if not isinstance(verification, dict) or verification.get("verified") is not True:
                raise ValueError("symbol-value audit lacks checkpoint verification")
            expected = symbol_value_pipeline_shared_fingerprint(prepared_job)
            if audit.get("pipeline_shared_config_fingerprint") != expected:
                raise ValueError("symbol-value shared pipeline configuration drifted")
            if audit.get("load_boundary_verified") is not True:
                raise ValueError("symbol-value load boundary was not verified")
            result.update({
                "symbol_value_token_audit": str(audit_path),
                "quantity_count": len(
                    prepared_job["model_input"]["quantity_payload"]["quantities"]
                ),
                "checkpoint_sha256": verification["checkpoint_sha256"],
                "checkpoint_size": verification["checkpoint_size"],
                "checkpoint_manifest": verification["manifest"],
                "load_boundary_verified": True,
                "checkpoint_load_mode": audit["checkpoint_load_mode"],
                "pipeline_shared_config_fingerprint": expected,
            })
        return result


__all__ = ["Wan22SymbolValueLoraAdapter"]
