#!/usr/bin/env python3
"""Train WAN2.2 DiT LoRA jointly with the SI-aware symbol-value conditioner."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadVideo,
    ToAbsolutePath,
)
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    save_training_args,
)
from examples.wanvideo.model_training.train import (
    WanTrainingModule,
    wan_parser,
)
from physbench.baselines.wan22_symbol_value_model import (
    NUMERIC_FEATURE_NAMES,
    SymbolValueConditioner,
    install_symbol_value_prompt_unit,
    load_symbol_value_conditioner_checkpoint,
    pool_symbol_embeddings,
)
from physbench.baselines.wan22_quantity_model import distributed_sampling_contract
from physbench.io import write_json, write_jsonl


def seed_process() -> int:
    seed = int(os.environ.get("TRAIN_SEED", "42"))
    if seed < 0:
        raise ValueError("TRAIN_SEED must be non-negative")
    rank = int(os.environ.get("RANK", "0"))
    worker_seed = seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    return worker_seed


def seeded_training_generator(seed: int) -> torch.Generator:
    """Build the generator explicitly owned by the shuffled DataLoader."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sampler seed must be a non-negative integer")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class SymbolValueWanTrainingModule(WanTrainingModule):
    def __init__(
        self,
        *args,
        symbol_value_conditioner_config: dict,
        conditioner_checkpoint: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conditioner = SymbolValueConditioner(symbol_value_conditioner_config).to(
            device=self.pipe.device,
            dtype=torch.float32,
        )
        install_symbol_value_prompt_unit(self.pipe, conditioner)
        if conditioner_checkpoint:
            load_symbol_value_conditioner_checkpoint(
                self.pipe.symbol_value_conditioner,
                conditioner_checkpoint,
                required=False,
            )
        self.pipe.symbol_value_conditioner.train()
        self.pipe.symbol_value_conditioner.requires_grad_(True)

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = (
            super().get_pipeline_inputs(data)
        )
        quantities = data.get("quantities")
        if not isinstance(quantities, list) or not quantities:
            raise ValueError("training row requires non-empty quantities")
        inputs_posi["quantities"] = quantities
        inputs_nega["negative_quantities"] = None
        return inputs_shared, inputs_posi, inputs_nega


class RecoverableSymbolValueModelLogger(ModelLogger):
    """Weights plus a diagnostic optimizer/scheduler/RNG snapshot."""

    def __init__(self, *args, save_optimizer_state: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_optimizer_state = save_optimizer_state
        self.optimizer = None
        self.scheduler = None
        self.sampler_generator = None
        self.sampling_contract = None
        self.args = None
        self.epoch_id = -1
        self.gradient_samples: list[dict] = []

    def bind_state(
        self,
        optimizer,
        scheduler,
        args,
        *,
        sampler_generator: torch.Generator,
        sampling_contract: dict,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler_generator = sampler_generator
        self.sampling_contract = dict(sampling_contract)
        self.args = args

    @staticmethod
    def assert_loss_finite(
        accelerator: accelerate.Accelerator,
        loss: torch.Tensor,
    ) -> None:
        local_finite = bool(torch.isfinite(loss.detach()).all().item())
        flag = torch.tensor(
            1 if local_finite else 0,
            dtype=torch.int64,
            device=accelerator.device,
        )
        finite_processes = accelerator.reduce(flag, reduction="sum")
        if int(finite_processes.item()) != accelerator.num_processes:
            raise FloatingPointError(
                "non-finite training loss detected before backward: "
                f"finite_processes={int(finite_processes.item())}/"
                f"{accelerator.num_processes}"
            )

    def assert_conditioner_gradients_finite(
        self,
        accelerator: accelerate.Accelerator,
        model: torch.nn.Module,
    ) -> tuple[float, int]:
        unwrapped = accelerator.unwrap_model(model)
        quantity_squared_tensor = torch.zeros(
            (),
            dtype=torch.float32,
            device=accelerator.device,
        )
        local_finite_tensor = torch.ones(
            (),
            dtype=torch.bool,
            device=accelerator.device,
        )
        quantity_tensors = 0
        for parameter in unwrapped.pipe.symbol_value_conditioner.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                local_finite_tensor.logical_and_(
                    torch.isfinite(gradient).all().to(accelerator.device)
                )
                quantity_squared_tensor.add_(
                    gradient.float().pow(2).sum().to(accelerator.device)
                )
                quantity_tensors += 1
        quantity_squared = float(quantity_squared_tensor.item())
        local_finite = (
            bool(local_finite_tensor.item())
            and quantity_tensors > 0
            and math.isfinite(quantity_squared)
        )
        flag = torch.tensor(
            1 if local_finite else 0,
            dtype=torch.int64,
            device=accelerator.device,
        )
        finite_processes = accelerator.reduce(flag, reduction="sum")
        if int(finite_processes.item()) != accelerator.num_processes:
            raise FloatingPointError(
                "non-finite or missing SymbolValueConditioner gradients detected "
                "before optimizer.step: "
                f"finite_processes={int(finite_processes.item())}/"
                f"{accelerator.num_processes}"
            )
        return math.sqrt(quantity_squared), quantity_tensors

    def record_gradients(
        self,
        accelerator: accelerate.Accelerator,
        model: torch.nn.Module,
        *,
        conditioner_gradient_l2: float,
        conditioner_gradient_tensor_count: int,
    ) -> None:
        if not math.isfinite(conditioner_gradient_l2):
            raise FloatingPointError(
                "SymbolValueConditioner gradient norm is non-finite"
            )
        unwrapped = accelerator.unwrap_model(model)
        text_gradients = sum(
            parameter.grad is not None
            for parameter in unwrapped.pipe.text_encoder.parameters()
        )
        self.gradient_samples.append({
            "step": self.num_steps + 1,
            "conditioner_gradient_l2": conditioner_gradient_l2,
            "conditioner_gradient_tensor_count": (
                conditioner_gradient_tensor_count
            ),
            "text_encoder_gradient_tensor_count": text_gradients,
        })

    @staticmethod
    def _atomic_torch_save(value, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def _save_recovery_state(
        self,
        accelerator: accelerate.Accelerator,
        model_checkpoint_name: str,
    ) -> None:
        if not self.save_optimizer_state:
            return
        if (
            self.optimizer is None
            or self.scheduler is None
            or self.sampler_generator is None
            or self.sampling_contract is None
        ):
            raise RuntimeError("training state logger is not bound")
        state_root = Path(self.output_path) / "training_state_latest"
        state_root.mkdir(parents=True, exist_ok=True)
        rng_path = state_root / (
            f"rng_rank_{accelerator.process_index:02d}.pt"
        )
        self._atomic_torch_save({
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None
            ),
            "sampler_generator": self.sampler_generator.get_state(),
            "sampler_generator_initial_seed": (
                self.sampler_generator.initial_seed()
            ),
            "process_index": accelerator.process_index,
            "global_step": self.num_steps,
            "epoch_id": self.epoch_id,
        }, rng_path)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_path = state_root / "optimizer_scheduler.pt"
            self._atomic_torch_save({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.num_steps,
                "epoch_id": self.epoch_id,
                "model_checkpoint": model_checkpoint_name,
                "world_size": accelerator.num_processes,
                "gradient_accumulation_steps": (
                    accelerator.gradient_accumulation_steps
                ),
            }, state_path)
            write_json(state_root / "state.json", {
                "schema_version": "1.0",
                "status": "complete",
                "global_step": self.num_steps,
                "epoch_id": self.epoch_id,
                "model_checkpoint": model_checkpoint_name,
                "optimizer_scheduler": str(state_path),
                "rng_state_pattern": "rng_rank_XX.pt",
                "world_size": accelerator.num_processes,
                "sampling_contract": self.sampling_contract,
                "sampler_generator_state_captured": True,
                "dataloader_iterator_state_captured": False,
                "recovery_capability": "diagnostic_snapshot_only",
                "resume_semantics": (
                    "optimizer, scheduler, process RNG, and sampler generator "
                    "states are captured; the current DataLoader iterator "
                    "position/permutation and automatic CLI resume are not "
                    "implemented, so this is not an exact-resume checkpoint"
                ),
            })
        accelerator.wait_for_everyone()

    def save_model(self, accelerator, model, file_name):
        super().save_model(accelerator, model, file_name)
        self._save_recovery_state(accelerator, file_name)

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.epoch_id = int(epoch_id)
        super().on_epoch_end(accelerator, model, epoch_id)

    def on_training_end(self, accelerator, model, save_steps=None):
        super().on_training_end(accelerator, model, save_steps)
        if accelerator.is_main_process:
            positive = [
                item["conditioner_gradient_l2"]
                for item in self.gradient_samples
                if item["conditioner_gradient_l2"] > 0
            ]
            write_json(
                Path(self.output_path) / "gradient_audit.json",
                {
                    "schema_version": "1.0",
                    "sample_count": len(self.gradient_samples),
                    "positive_conditioner_gradient_count": len(positive),
                    "min_positive_conditioner_gradient_l2": (
                        min(positive) if positive else None
                    ),
                    "max_conditioner_gradient_l2": (
                        max(
                            (
                                item["conditioner_gradient_l2"]
                                for item in self.gradient_samples
                            ),
                            default=None,
                        )
                    ),
                    "text_encoder_gradient_tensor_count_max": max(
                        (
                            item["text_encoder_gradient_tensor_count"]
                            for item in self.gradient_samples
                        ),
                        default=0,
                    ),
                    "samples": self.gradient_samples,
                },
            )


def audit_training_tokens(
    model: SymbolValueWanTrainingModule,
    dataset: UnifiedDataset,
    output: Path,
) -> None:
    by_case = {}
    for row in dataset.data:
        by_case.setdefault(row["case_id"], row)
    records = []
    for case_id in sorted(by_case):
        row = by_case[case_id]
        _, tokens = pool_symbol_embeddings(
            model.pipe.tokenizer,
            model.pipe.text_encoder.token_embedding,
            row["quantities"],
        )
        records.append({
            "schema_version": "1.0",
            "case_id": case_id,
            "scene_id": row["scene_id"],
            "prompt": row["prompt"],
            "quantity_registry_id": row["quantity_registry_id"],
            "quantity_registry_fingerprint": row[
                "quantity_registry_fingerprint"
            ],
            "quantities": tokens,
        })
    write_jsonl(output, records)


def launch_symbol_value_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: SymbolValueWanTrainingModule,
    model_logger: RecoverableSymbolValueModelLogger,
    args,
) -> None:
    sampling_contract = distributed_sampling_contract(
        len(dataset),
        accelerator.num_processes,
    )
    if accelerator.is_main_process:
        save_training_args(args)
    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    sampler_generator = seeded_training_generator(args.sampler_seed)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        generator=sampler_generator,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
    )
    unprepared_sampler = type(dataloader.sampler).__name__
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    prepared_batch_sampler = type(dataloader.batch_sampler).__name__
    if accelerator.is_main_process:
        write_json(
            Path(args.output_path) / "training_sampling_runtime.json",
            {
                "schema_version": "1.0",
                "shuffle": True,
                "sampler_seed": int(args.sampler_seed),
                "sampler_generator": "torch.Generator",
                "generator_initial_seed": int(
                    sampler_generator.initial_seed()
                ),
                "generator_binding": "DataLoader(generator=...)",
                "sampler_before_accelerator_prepare": unprepared_sampler,
                "batch_sampler_after_accelerator_prepare": (
                    prepared_batch_sampler
                ),
                "dataloader_after_accelerator_prepare": (
                    type(dataloader).__name__
                ),
                **sampling_contract,
                "sampler_generator_state_checkpointed": bool(
                    args.save_optimizer_state
                ),
                "process_seed_policy": (
                    "TRAIN_SEED + distributed rank; sampler generator uses "
                    "the common TRAIN_SEED"
                ),
            },
        )
    model_logger.bind_state(
        optimizer,
        scheduler,
        args,
        sampler_generator=sampler_generator,
        sampling_contract=sampling_contract,
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(args.num_epochs):
        model_logger.epoch_id = epoch_id
        for data in dataloader:
            with accelerator.accumulate(model):
                loss = model(data)
                model_logger.assert_loss_finite(accelerator, loss)
                accelerator.backward(loss)
                conditioner_gradient_l2, conditioner_gradient_tensors = (
                    model_logger.assert_conditioner_gradients_finite(
                        accelerator,
                        model,
                    )
                )
                if accelerator.sync_gradients:
                    model_logger.record_gradients(
                        accelerator,
                        model,
                        conditioner_gradient_l2=conditioner_gradient_l2,
                        conditioner_gradient_tensor_count=(
                            conditioner_gradient_tensors
                        ),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(
                    accelerator,
                    model,
                    args.save_steps,
                    loss=loss,
                )
        if args.save_steps is None:
            model_logger.on_epoch_end(
                accelerator,
                model,
                epoch_id,
            )
    model_logger.on_training_end(
        accelerator,
        model,
        args.save_steps,
    )


def main() -> int:
    worker_seed = seed_process()
    parser = wan_parser()
    parser.add_argument(
        "--symbol_value_conditioner_config_json",
        required=True,
    )
    parser.add_argument(
        "--symbol_value_token_audit_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        required=True,
        help="Seed bound to the shuffled DataLoader torch.Generator.",
    )
    args = parser.parse_args()
    if args.task != "sft":
        raise ValueError("symbol-value Baseline currently supports task=sft only")
    train_seed = int(os.environ.get("TRAIN_SEED", "42"))
    if args.sampler_seed != train_seed:
        raise ValueError(
            "--sampler_seed must equal the sealed TRAIN_SEED: "
            f"{args.sampler_seed} != {train_seed}"
        )
    if args.lora_rank != 32:
        raise ValueError(
            "WAN2.2-TI2V-5B symbol-value Baseline requires lora_rank=32"
        )
    if args.lora_target_modules != "q,k,v,o,ffn.0,ffn.2":
        raise ValueError(
            "WAN2.2-TI2V-5B symbol-value Baseline requires the frozen LoRA "
            "target selector q,k,v,o,ffn.0,ffn.2"
        )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            # WAN's I2V first-frame path rounds spatial inputs to multiples
            # of 32.  Keeping the video path on the same grid prevents the
            # video and first-frame VAE latents from differing by one cell
            # when dynamic resolution is used.
            height_division_factor=32,
            width_division_factor=32,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": (
                ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(
                    args.num_frames,
                    4,
                    1,
                    frame_processor=ImageCropAndResize(
                        512,
                        512,
                        None,
                        16,
                        16,
                    ),
                )
            ),
            "input_audio": (
                ToAbsolutePath(args.dataset_base_path)
                >> LoadAudio(sr=16000)
            ),
            "wantodance_music_path": ToAbsolutePath(
                args.dataset_base_path
            ),
        },
    )
    conditioner_config = json.loads(args.symbol_value_conditioner_config_json)
    model = SymbolValueWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=(
            args.use_gradient_checkpointing_offload
        ),
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if (
                args.initialize_model_on_cpu
                or args.enable_model_cpu_offload
            )
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        symbol_value_conditioner_config=conditioner_config,
        conditioner_checkpoint=args.lora_checkpoint,
    )
    if accelerator.is_main_process:
        args.symbol_value_token_audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        audit_training_tokens(
            model,
            dataset,
            args.symbol_value_token_audit_path,
        )
        write_json(
            Path(args.output_path) / "symbol_value_conditioner_spec.json",
            {
                "schema_version": "1.0",
                "worker_seed_rank_0": worker_seed,
                "sampler_seed": args.sampler_seed,
                "numeric_features": list(NUMERIC_FEATURE_NAMES),
                "config": conditioner_config,
                "injection_stage": (
                    "post_frozen_umt5_pre_dit_cross_attention"
                ),
            },
        )
    accelerator.wait_for_everyone()
    model_logger = RecoverableSymbolValueModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        save_optimizer_state=args.save_optimizer_state,
    )
    launch_symbol_value_training(
        accelerator,
        dataset,
        model,
        model_logger,
        args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
