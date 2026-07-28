from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from .wan22_lora import Wan22LoraAdapter
from .wan22_quantity_model import (
    QUANTITY_ENCODER_STATE_KEYS,
    QUANTITY_ENCODER_TENSOR_COUNT,
    WAN22_TI2V_5B_LORA_PAIR_COUNT,
    WAN22_TI2V_5B_LORA_RANK,
    WAN22_TI2V_5B_LORA_TENSOR_COUNT,
    expected_wan22_ti2v_5b_lora_targets,
    verify_quantity_checkpoint_manifest,
)
from ..io import load_json, sha256_file, write_json, write_jsonl


_INVENTORY_LORA_KEY = re.compile(
    r"^(?P<target>.+)\.lora_(?P<side>A|B)"
    r"(?P<adapter>\.default)?\.weight$"
)


class Wan22QuantityLoraAdapter(Wan22LoraAdapter):
    """WAN media orchestration for the quantity-embedding model."""

    def prepare_training(
        self,
        train_case_ids: list[str],
        run_dir: Path,
    ) -> dict[str, Any]:
        prepared = super().prepare_training(train_case_ids, run_dir)
        prepared.update({
            "adapter": "wan22_quantity_embedding",
            "quantity_encoder": self.config["quantity_encoder"],
            "checkpoint_format": (
                "combined_dit_lora_and_quantity_encoder_safetensors_v1"
            ),
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
    ) -> dict[str, Any]:
        native = adaptation["native_inputs"]
        physics = native["physics"]
        return {
            "video": video,
            "prompt": native["text"]["prompt"],
            "audited_prompt": native["text"]["audited_prompt"],
            "quantities": physics["quantities"],
            "quantity_registry_id": physics["registry_id"],
            "quantity_registry_fingerprint": physics[
                "registry_fingerprint"
            ],
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
        balanced = super()._balance_training_rows(rows, artifact_root)
        plan_path = artifact_root / "training_sampling_plan.json"
        plan = load_json(plan_path)
        sampler_seed = int(self.config.get("lora", {}).get("seed", 42))
        plan.update({
            "shuffle": True,
            "sampler_seed": sampler_seed,
            "sampler_seed_source": (
                "trainer.config.seed -> TRAIN_SEED -> --sampler_seed"
            ),
            "sampler_generator": "torch.Generator",
            "sampler_binding": "DataLoader(generator=...)",
            "process_seed_policy": "sampler_seed + distributed_rank",
        })
        write_json(plan_path, plan)
        return balanced

    def _training_command(self, runtime_root: Path) -> list[str]:
        return [
            "bash",
            str(self.project_root / "scripts" / "train_wan22_quantity.sh"),
        ]

    def _training_environment(
        self,
        run_dir: Path,
        dataset_dir: Path,
        metadata: Path,
    ) -> dict[str, str]:
        environment = super()._training_environment(
            run_dir,
            dataset_dir,
            metadata,
        )
        environment.update({
            "WAN_PROJECT_ROOT": str(self.runtime["project_root"]),
            "WAN_PYTHON": str(self.runtime["python"]),
            "BENCHMARK_ROOT": str(self.project_root),
            "QUANTITY_ENCODER_CONFIG_JSON": json.dumps(
                self.config["quantity_encoder"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "QUANTITY_TOKEN_AUDIT_PATH": str(
                run_dir
                / "artifacts"
                / "wan22"
                / "training_quantity_token_audit.jsonl"
            ),
            "SAVE_OPTIMIZER_STATE": (
                "1"
                if self.config.get("lora", {}).get(
                    "save_optimizer_state",
                    False,
                )
                else "0"
            ),
        })
        if (
            self.runtime.get("accelerate_config") is not None
            or self.execute
        ):
            accelerate_config = self._freeze_accelerate_config(run_dir)
            environment.update({
                "ACCELERATE_CONFIG": accelerate_config["frozen_path"],
                "ACCELERATE_CONFIG_SHA256": accelerate_config[
                    "frozen_sha256"
                ],
                "ACCELERATE_CONFIG_AUDIT_PATH": str(
                    self._accelerate_config_audit_path(run_dir)
                ),
            })
        return environment

    def _generation_script(self) -> Path:
        return self.project_root / "scripts" / "wan22_quantity_generate.py"

    def _model_asset_manifest(self, run_dir: Path) -> Path:
        return (
            run_dir
            / "artifacts"
            / "wan22"
            / "base_model_assets.json"
        )

    @staticmethod
    def _accelerate_config_path(run_dir: Path) -> Path:
        return (
            run_dir
            / "artifacts"
            / "wan22"
            / "accelerate_config.yaml"
        )

    @staticmethod
    def _accelerate_config_audit_path(run_dir: Path) -> Path:
        return (
            run_dir
            / "artifacts"
            / "wan22"
            / "accelerate_config.audit.json"
        )

    def _configured_accelerate_config(
        self,
        *,
        require_exists: bool = True,
    ) -> Path:
        configured = self.runtime.get("accelerate_config")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(
                "runtime.accelerate_config is required for WAN quantity "
                "training"
            )
        path = Path(configured)
        if not path.is_absolute():
            path = Path(self.runtime["project_root"]) / path
        path = path.resolve()
        if require_exists and not path.is_file():
            raise FileNotFoundError(
                f"Accelerate config not found: {path}"
            )
        return path

    def _freeze_accelerate_config(
        self,
        run_dir: Path,
    ) -> dict[str, Any]:
        """Copy and seal the launcher config before any training process."""
        frozen = self._accelerate_config_path(run_dir).resolve()
        audit_path = self._accelerate_config_audit_path(run_dir)
        frozen.parent.mkdir(parents=True, exist_ok=True)

        if frozen.exists() or audit_path.exists():
            source = self._configured_accelerate_config(
                require_exists=False
            )
            if not frozen.is_file() or not audit_path.is_file():
                raise RuntimeError(
                    "incomplete run-local Accelerate config lock: "
                    f"config={frozen.is_file()}, "
                    f"audit={audit_path.is_file()}"
                )
            audit = load_json(audit_path)
            if audit.get("schema_version") != "1.0":
                raise ValueError(
                    "unsupported Accelerate config audit schema"
                )
            expected = {
                "source_path": str(source),
                "frozen_path": str(frozen),
            }
            for field, value in expected.items():
                if audit.get(field) != value:
                    raise ValueError(
                        "run-local Accelerate config audit does not match "
                        f"{field}: expected={value!r}, "
                        f"recorded={audit.get(field)!r}"
                    )
            frozen_sha256 = sha256_file(frozen)
            if audit.get("frozen_sha256") != frozen_sha256:
                raise ValueError(
                    "run-local Accelerate config differs from its recorded "
                    "SHA-256"
                )
            if audit.get("source_sha256") != frozen_sha256:
                raise ValueError(
                    "Accelerate config audit was not sealed from the "
                    "recorded source bytes"
                )
            if audit.get("size") != frozen.stat().st_size:
                raise ValueError(
                    "run-local Accelerate config size differs from its "
                    "audit"
                )
            return audit

        source = self._configured_accelerate_config()
        source_sha256 = sha256_file(source)
        shutil.copyfile(source, frozen)
        frozen_sha256 = sha256_file(frozen)
        if source_sha256 != frozen_sha256:
            frozen.unlink(missing_ok=True)
            raise RuntimeError(
                "Accelerate config changed while it was being frozen"
            )
        audit = {
            "schema_version": "1.0",
            "source_path": str(source),
            "source_sha256": source_sha256,
            "frozen_path": str(frozen),
            "frozen_sha256": frozen_sha256,
            "size": frozen.stat().st_size,
        }
        write_json(audit_path, audit)
        return audit

    def _verified_diffsynth_commit(self, runtime_root: Path) -> str:
        expected_commit = self.runtime.get("diffsynth_commit")
        if (
            not isinstance(expected_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None
        ):
            raise ValueError(
                "runtime.diffsynth_commit must pin an exact lowercase "
                "40-character Git commit"
            )
        repository = runtime_root / "vendor" / "DiffSynth-Studio"
        try:
            actual_commit = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repository),
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "cannot fingerprint the DiffSynth dependency"
            ) from exc
        if re.fullmatch(r"[0-9a-f]{40}", actual_commit) is None:
            raise RuntimeError(
                "DiffSynth dependency returned an invalid Git commit"
            )
        if actual_commit != expected_commit:
            raise ValueError(
                "DiffSynth deployment commit differs from the Baseline "
                f"manifest: expected={expected_commit}, "
                f"actual={actual_commit}"
            )
        return actual_commit

    def _write_model_asset_manifest(self, run_dir: Path) -> None:
        accelerate_config = self._freeze_accelerate_config(run_dir)
        runtime_root = Path(self.runtime["project_root"]).resolve()
        model_base = Path(self.runtime["model_base"])
        if not model_base.is_absolute():
            model_base = runtime_root / model_base
        model_dir = (
            model_base
            / "Wan-AI"
            / "Wan2.2-TI2V-5B"
        )
        paths = [
            model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            *sorted(
                model_dir.glob(
                    "diffusion_pytorch_model-*-of-*.safetensors"
                )
            ),
            model_dir / "Wan2.2_VAE.pth",
            *sorted(
                path
                for path in (
                    model_dir / "google" / "umt5-xxl"
                ).iterdir()
                if path.is_file()
            ),
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"WAN base model assets are incomplete: {missing}"
            )
        diffsynth_commit = self._verified_diffsynth_commit(runtime_root)
        write_json(self._model_asset_manifest(run_dir), {
            "schema_version": "1.0",
            "base_model": "WAN2.2-TI2V-5B",
            "diffsynth_commit": diffsynth_commit,
            "accelerate_config": accelerate_config,
            "files": [
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        })

    @staticmethod
    def _checkpoint_inventory(path: Path) -> dict[str, Any]:
        tensor_count = 0
        parameter_count = 0
        quantity_keys: set[str] = set()
        quantity_prefixes: set[str] = set()
        lora_pairs: dict[tuple[str, str], dict[str, list[int]]] = {}
        dtypes: dict[str, int] = {}
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                raise ValueError("invalid safetensors checkpoint header")
            header_size = struct.unpack("<Q", header_size_raw)[0]
            header = json.loads(
                handle.read(header_size).decode("utf-8")
            )
            for key, tensor in header.items():
                if key == "__metadata__":
                    continue
                shape = tensor["shape"]
                count = 1
                for size in shape:
                    count *= int(size)
                tensor_count += 1
                parameter_count += count
                dtype = str(tensor["dtype"])
                if dtype not in {"BF16", "F16", "F32", "F64"}:
                    raise ValueError(
                        "combined checkpoint tensor must be floating point: "
                        f"{key} has dtype {dtype}"
                    )
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                if key.startswith("pipe.quantity_encoder."):
                    quantity_prefixes.add("pipe.quantity_encoder.")
                    quantity_keys.add(
                        key.removeprefix("pipe.quantity_encoder.")
                    )
                    continue
                if key.startswith("quantity_encoder."):
                    quantity_prefixes.add("quantity_encoder.")
                    quantity_keys.add(
                        key.removeprefix("quantity_encoder.")
                    )
                    continue
                match = _INVENTORY_LORA_KEY.fullmatch(key)
                if match is None:
                    raise ValueError(
                        "combined checkpoint contains an unsupported tensor: "
                        f"{key}"
                    )
                pair_id = (
                    match.group("target"),
                    match.group("adapter") or "",
                )
                side = match.group("side")
                pair = lora_pairs.setdefault(pair_id, {})
                if side in pair:
                    raise ValueError(
                        "combined checkpoint contains duplicate LoRA "
                        f"{side} tensors for {pair_id[0]}"
                    )
                pair[side] = [int(size) for size in shape]
        lora_tensors = sum(len(pair) for pair in lora_pairs.values())
        if tensor_count != (
            WAN22_TI2V_5B_LORA_TENSOR_COUNT
            + QUANTITY_ENCODER_TENSOR_COUNT
        ):
            raise ValueError(
                "combined checkpoint must contain exactly 619 tensors: "
                f"got {tensor_count}"
            )
        if (
            len(quantity_keys) != QUANTITY_ENCODER_TENSOR_COUNT
            or quantity_keys != QUANTITY_ENCODER_STATE_KEYS
            or len(quantity_prefixes) != 1
        ):
            raise ValueError(
                "combined checkpoint must contain the exact 19-tensor "
                "QuantityEncoder topology"
            )
        if lora_tensors != WAN22_TI2V_5B_LORA_TENSOR_COUNT:
            raise ValueError(
                "combined checkpoint must contain exactly "
                f"{WAN22_TI2V_5B_LORA_TENSOR_COUNT} LoRA tensors, "
                f"got {lora_tensors}"
            )
        if len(lora_pairs) != WAN22_TI2V_5B_LORA_PAIR_COUNT:
            raise ValueError(
                "combined checkpoint must contain exactly "
                f"{WAN22_TI2V_5B_LORA_PAIR_COUNT} LoRA A/B pairs, "
                f"got {len(lora_pairs)}"
            )
        normalized_targets: set[str] = set()
        for (raw_target, _adapter), pair in lora_pairs.items():
            if set(pair) != {"A", "B"}:
                raise ValueError(
                    "combined checkpoint contains an incomplete LoRA pair: "
                    f"{raw_target}"
                )
            target = (
                raw_target.removeprefix("diffusion_model.")
                if raw_target.startswith("diffusion_model.")
                else raw_target
            )
            if target in normalized_targets:
                raise ValueError(
                    "combined checkpoint maps duplicate LoRA pairs to "
                    f"{target}"
                )
            normalized_targets.add(target)
            shape_a = pair["A"]
            shape_b = pair["B"]
            if (
                len(shape_a) != 2
                or len(shape_b) != 2
                or shape_a[0] != WAN22_TI2V_5B_LORA_RANK
                or shape_b[1] != WAN22_TI2V_5B_LORA_RANK
            ):
                raise ValueError(
                    "combined checkpoint LoRA rank must be exactly "
                    f"{WAN22_TI2V_5B_LORA_RANK}: {raw_target}"
                )
        if normalized_targets != expected_wan22_ti2v_5b_lora_targets():
            expected = expected_wan22_ti2v_5b_lora_targets()
            raise ValueError(
                "combined checkpoint LoRA target topology mismatch: "
                f"missing={sorted(expected - normalized_targets)[:10]}, "
                f"unexpected={sorted(normalized_targets - expected)[:10]}"
            )
        return {
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "lora_tensor_count": lora_tensors,
            "lora_pair_count": len(lora_pairs),
            "lora_rank": WAN22_TI2V_5B_LORA_RANK,
            "lora_target_topology": "wan22_ti2v_5b_30x10_v1",
            "quantity_encoder_tensor_count": len(quantity_keys),
            "dtype_tensor_counts": dict(sorted(dtypes.items())),
        }

    def train(
        self,
        prepared_training: dict[str, Any],
        job_path: Path,
    ) -> dict[str, Any]:
        run_dir = job_path.parent
        if self.execute and prepared_training["case_ids"]:
            self._write_model_asset_manifest(run_dir)
        result = super().train(prepared_training, job_path)
        checkpoint_value = result.get("checkpoint")
        if (
            self.execute
            and result.get("status") in {"complete", "not_requested"}
            and checkpoint_value
        ):
            checkpoint = Path(checkpoint_value)
            inventory = self._checkpoint_inventory(checkpoint)
            state_root = checkpoint.parent / "training_state_latest"
            state_path = state_root / "optimizer_scheduler.pt"
            manifest = {
                "schema_version": "2.0",
                "status": result["status"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "source": (
                    "task1_finetune"
                    if prepared_training["case_ids"]
                    else "baseline_config"
                ),
                "baseline_id": self.baseline_id,
                "inventory": inventory,
                "quantity_encoder_config": self.config[
                    "quantity_encoder"
                ],
                "base_model_assets": str(
                    self._model_asset_manifest(run_dir)
                ),
                "training_state": (
                    {
                        "directory": str(state_root),
                        "optimizer_scheduler": str(state_path),
                        "optimizer_scheduler_sha256": sha256_file(
                            state_path
                        ),
                        "state_manifest": str(state_root / "state.json"),
                    }
                    if state_path.is_file()
                    else None
                ),
            }
            write_json(self._checkpoint_manifest(run_dir), manifest)
            result["checkpoint_sha256"] = manifest[
                "checkpoint_sha256"
            ]
            result["checkpoint_inventory"] = inventory
        return result

    def prepare_job(
        self,
        job: dict[str, Any],
        case: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        prepared = super().prepare_job(job, case, run_dir)
        adaptation = job["adaptation"]
        native = adaptation["native_inputs"]
        prepared["model_input"].pop("physical_parameters", None)
        prepared["model_input"].update({
            "audited_prompt": native["text"]["audited_prompt"],
            "quantity_payload": native["physics"],
        })
        prepared["quantity_token_audit"] = str(
            run_dir
            / "artifacts"
            / "wan22"
            / "inference_quantity_token_audits"
            / f"{job['job_id']}.json"
        )
        prepared["wan22"]["quantity_encoder"] = self.config[
            "quantity_encoder"
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
            if not checkpoint:
                raise FileNotFoundError(
                    "quantity-embedding generation requires a checkpoint"
                )
            if not manifest:
                raise FileNotFoundError(
                    "quantity-embedding generation requires a checkpoint "
                    "manifest"
                )
            verify_quantity_checkpoint_manifest(checkpoint, manifest)
        return super().generate(prepared_job, job_path)


__all__ = ["Wan22QuantityLoraAdapter"]
