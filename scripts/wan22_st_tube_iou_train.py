#!/usr/bin/env python3
"""Train WAN2.2 DiT LoRA with a whole-video latent tube-IoU auxiliary loss."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import accelerate
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
from examples.wanvideo.model_training.train import WanTrainingModule, wan_parser
from safetensors.torch import load_file, save_file

from physbench.baselines.wan22_st_tube_iou_model import (
    LatentOccupancyHead,
    SpatioTemporalTubeIoULoss,
    align_mask_tube,
    flowmatch_clean_estimate,
    noise_weight,
)
from physbench.io import write_json


def effective_st_lambda(
    lambda_st: float,
    optimizer_step: int,
    warmup_steps: int,
) -> float:
    if lambda_st < 0:
        raise ValueError("lambda_st must be non-negative")
    if optimizer_step < 0 or warmup_steps < 0:
        raise ValueError("optimizer step and warmup must be non-negative")
    if warmup_steps == 0:
        return float(lambda_st)
    return float(lambda_st) * min(1.0, (optimizer_step + 1) / warmup_steps)


def st_head_checkpoint_path(lora_checkpoint: Path) -> Path:
    return lora_checkpoint.with_name(f"{lora_checkpoint.stem}.st-head.safetensors")


def lora_only_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    remove_prefix: str | None,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if ".lora_A" not in key and ".lora_B" not in key:
            continue
        name = key
        if remove_prefix is not None and name.startswith(remove_prefix):
            name = name[len(remove_prefix) :]
        output[name] = value
    if not output:
        raise ValueError("no DiT LoRA tensors were found for export")
    return output


def _scheduler_batch_values(
    scheduler: Any,
    timestep_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = timestep_ids.detach().to(device="cpu", dtype=torch.long)
    timesteps = scheduler.timesteps[ids].to(device=device, dtype=dtype)
    sigmas = scheduler.sigmas[ids].to(device=device, dtype=dtype)
    weights = scheduler.linear_timesteps_weights[ids].to(
        device=device,
        dtype=torch.float32,
    )
    return timesteps, sigmas, weights


def compute_flowmatch_st_objective(
    *,
    pipe: Any,
    inputs: dict[str, Any],
    subject_mask: torch.Tensor,
    occupancy_head: nn.Module,
    config: dict[str, Any],
    optimizer_step: int,
    timestep_ids: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate one random-timestep WAN Flow Matching and tube-IoU step."""

    shared = dict(inputs)
    clean_latents = shared["input_latents"]
    if clean_latents.ndim != 5:
        raise ValueError("WAN input_latents must use BCTHW layout")
    batch = clean_latents.shape[0]
    minimum = int(
        float(shared.get("min_timestep_boundary", 0.0))
        * len(pipe.scheduler.timesteps)
    )
    maximum = int(
        float(shared.get("max_timestep_boundary", 1.0))
        * len(pipe.scheduler.timesteps)
    )
    if timestep_ids is None:
        timestep_ids = torch.randint(minimum, maximum, (batch,))
    if timestep_ids.shape != (batch,):
        raise ValueError("one Flow Matching timestep id is required per sample")
    timesteps, sigmas, training_weights = _scheduler_batch_values(
        pipe.scheduler,
        timestep_ids,
        device=clean_latents.device,
        dtype=pipe.torch_dtype,
    )
    sigma_view = sigmas.reshape(batch, 1, 1, 1, 1)
    if noise is None:
        noise = torch.randn_like(clean_latents) * float(shared.get("noise_scale", 1.0))
    if noise.shape != clean_latents.shape:
        raise ValueError("Flow Matching noise shape differs from clean latents")
    noisy_latents = (1.0 - sigma_view) * clean_latents + sigma_view * noise
    training_target = noise - clean_latents
    first_frame = shared.get("first_frame_latents")
    if first_frame is not None:
        noisy_latents = torch.cat([first_frame, noisy_latents[:, :, 1:]], dim=2)
    shared["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    predicted_velocity = pipe.model_fn(
        **models,
        **shared,
        timestep=timesteps,
    )
    base_prediction = predicted_velocity
    base_target = training_target
    if first_frame is not None:
        base_prediction = base_prediction[:, :, 1:]
        base_target = base_target[:, :, 1:]
    per_sample_mse = F.mse_loss(
        base_prediction.float(),
        base_target.float(),
        reduction="none",
    ).flatten(start_dim=1).mean(dim=1)
    base_loss = (per_sample_mse * training_weights).mean()

    enabled = bool(config["enable_st_iou_loss"])
    base_scale = base_loss.new_tensor(float(config.get("base_loss_scale", 1.0)))
    if not enabled:
        zero = base_loss.new_zeros(())
        zero_batch = base_loss.new_zeros((batch,))
        return {
            "total_loss": base_scale * base_loss,
            "base_loss": base_loss,
            "st_loss": zero,
            "st_loss_unweighted": zero,
            "st_iou": zero,
            "st_iou_per_sample": zero_batch,
            "noise_sigma": sigmas.float(),
            "noise_weight": zero_batch,
            "lambda_effective": zero,
            "gt_foreground_fraction": zero,
            "pred_foreground_fraction": zero,
        }

    estimate = flowmatch_clean_estimate(
        noisy_latents,
        predicted_velocity,
        sigmas,
        first_frame_latents=first_frame,
    )
    prediction = occupancy_head(estimate).sigmoid().squeeze(1)
    target = align_mask_tube(
        subject_mask.to(device=prediction.device),
        latent_shape=tuple(prediction.shape),
    )
    weights = noise_weight(
        sigmas,
        mode=str(config["st_loss_weighting"]),
        threshold=float(config["st_noise_threshold"]),
    )
    tube_loss = SpatioTemporalTubeIoULoss(
        eps=float(config["st_iou_eps"])
    )(prediction, target, sample_weight=weights)
    lambda_effective = effective_st_lambda(
        float(config["lambda_st"]),
        optimizer_step,
        int(config["st_loss_warmup_steps"]),
    )
    lambda_tensor = base_loss.new_tensor(lambda_effective)
    total_loss = base_scale * base_loss + lambda_tensor * tube_loss.loss
    return {
        "total_loss": total_loss,
        "base_loss": base_loss,
        "st_loss": tube_loss.loss,
        "st_loss_unweighted": tube_loss.per_sample_loss.mean(),
        "st_iou": tube_loss.iou.mean(),
        "st_iou_per_sample": tube_loss.iou,
        "noise_sigma": sigmas.float(),
        "noise_weight": weights.float(),
        "lambda_effective": lambda_tensor,
        "gt_foreground_fraction": tube_loss.target_foreground_fraction.mean(),
        "pred_foreground_fraction": (
            tube_loss.prediction_foreground_fraction.mean()
        ),
    }


class LoadSubjectMaskTube:
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

    def __call__(self, value: str) -> torch.Tensor:
        path = Path(value)
        if not path.is_absolute():
            path = self.base_path / path
        with np.load(path, allow_pickle=False) as payload:
            if "masks" not in payload:
                raise ValueError(f"subject-mask tube has no masks array: {path}")
            tube = payload["masks"]
        if tube.ndim != 3 or tube.dtype != np.uint8:
            raise ValueError(f"subject-mask tube must be uint8 THW: {path}")
        if not set(np.unique(tube).tolist()).issubset({0, 1}):
            raise ValueError(f"subject-mask tube must be binary: {path}")
        return torch.from_numpy(tube.copy())


class STTubeIoUWanTrainingModule(WanTrainingModule):
    def __init__(self, *args, st_config: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.st_config = dict(st_config)
        self.occupancy_head = LatentOccupancyHead(
            latent_channels=int(st_config["latent_channels"]),
            hidden_channels=int(st_config["hidden_channels"]),
        ).to(device=self.pipe.device, dtype=torch.float32)
        self.occupancy_head.train().requires_grad_(True)
        self.optimizer_step = 0
        self.last_metrics: dict[str, torch.Tensor] = {}

    def forward(self, data, inputs=None):
        subject_mask = data.get("subject_mask")
        if not isinstance(subject_mask, torch.Tensor):
            raise ValueError("training row requires a subject_mask tensor")
        if subject_mask.ndim == 3:
            subject_mask = subject_mask.unsqueeze(0)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs,
            self.pipe.device,
            self.pipe.torch_dtype,
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, _negative = inputs
        metrics = compute_flowmatch_st_objective(
            pipe=self.pipe,
            inputs={**shared, **positive},
            subject_mask=subject_mask,
            occupancy_head=self.occupancy_head,
            config=self.st_config,
            optimizer_step=self.optimizer_step,
        )
        self.last_metrics = metrics
        return metrics

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return lora_only_state_dict(state_dict, remove_prefix=remove_prefix)


class STTubeIoUModelLogger(ModelLogger):
    def __init__(
        self,
        *args,
        st_config: dict[str, Any],
        metrics_path: Path,
        save_optimizer_state: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.st_config = dict(st_config)
        self.metrics_path = metrics_path
        self.save_optimizer_state = save_optimizer_state
        self.optimizer = None
        self.scheduler = None
        self.epoch_id = -1
        self.micro_step_in_epoch = 0
        self.gradient_samples: list[dict[str, Any]] = []

    def bind_state(self, optimizer, scheduler) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def _atomic_torch_save(value: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def _save_training_state(
        self,
        accelerator: accelerate.Accelerator,
        checkpoint_name: str,
    ) -> None:
        if not self.save_optimizer_state:
            return
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("ST training-state logger is not bound")
        root = Path(self.output_path) / "training_state_latest"
        root.mkdir(parents=True, exist_ok=True)
        rank_path = root / f"rng_rank_{accelerator.process_index:02d}.pt"
        self._atomic_torch_save({
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "process_index": accelerator.process_index,
            "global_step": self.num_steps,
            "epoch_id": self.epoch_id,
            "micro_step_in_epoch": self.micro_step_in_epoch,
        }, rank_path)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_path = root / "optimizer_scheduler.pt"
            self._atomic_torch_save({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.num_steps,
                "epoch_id": self.epoch_id,
                "micro_step_in_epoch": self.micro_step_in_epoch,
                "model_checkpoint": checkpoint_name,
                "world_size": accelerator.num_processes,
                "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
            }, state_path)
            write_json(root / "state.json", {
                "schema_version": "1.0",
                "status": "complete",
                "global_step": self.num_steps,
                "epoch_id": self.epoch_id,
                "micro_step_in_epoch": self.micro_step_in_epoch,
                "model_checkpoint": checkpoint_name,
                "optimizer_scheduler": str(state_path),
                "rng_state_pattern": "rng_rank_XX.pt",
                "world_size": accelerator.num_processes,
                "resume_semantics": (
                    "weights, optimizer, scheduler, process RNG, epoch and local "
                    "micro-step are captured; exact mid-epoch DataLoader iterator "
                    "restoration is not provided"
                ),
            })
        accelerator.wait_for_everyone()

    def save_model(self, accelerator, model, file_name):
        super().save_model(accelerator, model, file_name)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            state = {
                key: value.detach().float().cpu().contiguous()
                for key, value in unwrapped.occupancy_head.state_dict().items()
            }
            head_path = st_head_checkpoint_path(Path(self.output_path) / file_name)
            save_file(
                state,
                str(head_path),
                metadata={
                    "schema_version": "1.0",
                    "representation": "learned_latent_space_subject_occupancy_proxy",
                    "inference_required": "false",
                    "config_json": json.dumps(self.st_config, sort_keys=True),
                },
            )
        self._save_training_state(accelerator, file_name)

    def append_metrics(self, record: dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


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


def _gradient_audit(
    model: STTubeIoUWanTrainingModule,
    *,
    require_head: bool,
) -> dict[str, Any]:
    lora_squared = 0.0
    lora_count = 0
    head_squared = 0.0
    head_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite gradient: {name}")
        squared = float(gradient.square().sum())
        if ".lora_A" in name or ".lora_B" in name:
            lora_squared += squared
            lora_count += 1
        elif name.startswith("occupancy_head."):
            head_squared += squared
            head_count += 1
    if lora_count == 0:
        raise RuntimeError("missing LoRA gradients")
    if require_head and head_count == 0:
        raise RuntimeError("missing occupancy-head gradients")
    return {
        "lora_gradient_l2": math.sqrt(lora_squared),
        "lora_gradient_tensor_count": lora_count,
        "occupancy_head_gradient_l2": math.sqrt(head_squared),
        "occupancy_head_gradient_tensor_count": head_count,
    }


def _gather_scalar(
    accelerator: accelerate.Accelerator,
    value: torch.Tensor,
) -> float:
    gathered = accelerator.gather_for_metrics(value.detach().float().reshape(1))
    return float(gathered.mean().cpu())


def launch_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: STTubeIoUWanTrainingModule,
    logger: STTubeIoUModelLogger,
    args: Any,
) -> None:
    if accelerator.is_main_process:
        save_training_args(args)
    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    sampler_generator = torch.Generator().manual_seed(args.sampler_seed)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        generator=sampler_generator,
        collate_fn=lambda rows: rows[0],
        num_workers=args.dataset_num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    logger.bind_state(optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    step_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    for epoch_id in range(args.num_epochs):
        logger.epoch_id = epoch_id
        logger.micro_step_in_epoch = 0
        for data in dataloader:
            logger.micro_step_in_epoch += 1
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.optimizer_step = logger.num_steps
            with accelerator.accumulate(model):
                metrics = model(data)
                loss = metrics["total_loss"]
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise FloatingPointError("non-finite total training loss")
                accelerator.backward(loss)
                gradient = _gradient_audit(
                    unwrapped,
                    require_head=bool(unwrapped.st_config["enable_st_iou_loss"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if not accelerator.sync_gradients:
                    continue
                elapsed = time.perf_counter() - step_started
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(accelerator.device))
                    if torch.cuda.is_available()
                    else 0
                )
                logger.on_step_end(
                    accelerator,
                    model,
                    args.save_steps,
                    loss=loss,
                )
                record = {
                    "schema_version": "1.0",
                    "optimizer_step": logger.num_steps,
                    "epoch_id": epoch_id,
                    "base_loss": _gather_scalar(accelerator, metrics["base_loss"]),
                    "st_loss": _gather_scalar(accelerator, metrics["st_loss"]),
                    "st_loss_unweighted": _gather_scalar(
                        accelerator, metrics["st_loss_unweighted"]
                    ),
                    "st_iou": _gather_scalar(accelerator, metrics["st_iou"]),
                    "noise_sigma": _gather_scalar(
                        accelerator, metrics["noise_sigma"].mean()
                    ),
                    "noise_weight": _gather_scalar(
                        accelerator, metrics["noise_weight"].mean()
                    ),
                    "lambda_effective": _gather_scalar(
                        accelerator, metrics["lambda_effective"]
                    ),
                    "total_loss": _gather_scalar(accelerator, metrics["total_loss"]),
                    "gt_foreground_fraction": _gather_scalar(
                        accelerator, metrics["gt_foreground_fraction"]
                    ),
                    "pred_foreground_fraction": _gather_scalar(
                        accelerator, metrics["pred_foreground_fraction"]
                    ),
                    "step_time_seconds": elapsed,
                    "cuda_peak_memory_bytes": peak_memory,
                    **gradient,
                }
                if accelerator.is_main_process:
                    logger.append_metrics(record)
                step_started = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
        if args.save_steps is None:
            logger.on_epoch_end(accelerator, model, epoch_id)
    logger.on_training_end(accelerator, model, args.save_steps)


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "enable_st_iou_loss",
        "lambda_st",
        "st_iou_eps",
        "st_loss_weighting",
        "st_noise_threshold",
        "st_loss_warmup_steps",
        "latent_channels",
        "hidden_channels",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"ST tube-IoU config is missing fields: {missing}")
    if config["st_loss_weighting"] not in {"none", "linear_clean", "threshold"}:
        raise ValueError("unsupported st_loss_weighting")


def main() -> int:
    worker_seed = seed_process()
    parser = wan_parser()
    parser.add_argument("--st_tube_iou_config_json", required=True)
    parser.add_argument("--st_tube_iou_metrics_path", type=Path, required=True)
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--sampler_seed", type=int, required=True)
    args = parser.parse_args()
    if args.task != "sft":
        raise ValueError("ST tube-IoU baseline supports task=sft only")
    if args.lora_rank != 32:
        raise ValueError("WAN ST tube-IoU baseline requires lora_rank=32")
    if args.lora_target_modules != "q,k,v,o,ffn.0,ffn.2":
        raise ValueError("WAN ST tube-IoU baseline requires the sealed LoRA targets")
    config = json.loads(args.st_tube_iou_config_json)
    _validate_config(config)
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
            height_division_factor=32,
            width_division_factor=32,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "subject_mask": LoadSubjectMaskTube(args.dataset_base_path),
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )
    model = STTubeIoUWanTrainingModule(
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
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if args.initialize_model_on_cpu or args.enable_model_cpu_offload
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        st_config=config,
    )
    if args.lora_checkpoint:
        head_path = st_head_checkpoint_path(Path(args.lora_checkpoint))
        if not head_path.is_file():
            raise FileNotFoundError(f"paired ST head checkpoint is missing: {head_path}")
        model.occupancy_head.load_state_dict(load_file(str(head_path), device="cpu"))
    if accelerator.is_main_process:
        write_json(Path(args.output_path) / "st_tube_iou_spec.json", {
            "schema_version": "1.0",
            "worker_seed_rank_0": worker_seed,
            "sampler_seed": args.sampler_seed,
            "flow_parameterization": {
                "noisy": "x_s=(1-sigma)*x0+sigma*epsilon",
                "velocity": "epsilon-x0",
                "clean_estimate": "x0_hat=x_s-sigma*v_hat",
            },
            "occupancy_source": "learned_latent_space_subject_occupancy_proxy",
            "config": config,
        })
    logger = STTubeIoUModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        st_config=config,
        metrics_path=args.st_tube_iou_metrics_path,
        save_optimizer_state=args.save_optimizer_state,
    )
    launch_training(accelerator, dataset, model, logger, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
