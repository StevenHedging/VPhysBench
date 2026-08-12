#!/usr/bin/env python3
"""Train WAN2.2 LoRA with Tube-IoU and subject-motion auxiliaries."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import accelerate
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
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    save_training_args,
)
from examples.wanvideo.model_training.train import wan_parser
from safetensors.torch import load_file

from physbench.baselines.wan22_st_tube_iou_model import (
    SpatioTemporalTubeIoULoss,
    flowmatch_clean_estimate,
    noise_weight,
)
from physbench.baselines.wan22_subject_motion_model import (
    align_subject_support_tube,
    masked_subject_flow_loss,
    subject_temporal_difference_loss,
)
from physbench.io import write_json
from scripts.wan22_st_tube_iou_train import (
    LoadSubjectMaskTube,
    STTubeIoUModelLogger,
    STTubeIoUWanTrainingModule,
    _gather_scalar,
    _gradient_audit,
    _scheduler_batch_values,
    seed_process,
    st_head_checkpoint_path,
)


def auxiliary_warmup_ratio(optimizer_step: int, warmup_steps: int) -> float:
    """Linear auxiliary ramp keyed by completed synchronized optimizer steps."""

    if optimizer_step < 0 or warmup_steps < 0:
        raise ValueError("optimizer step and auxiliary warmup must be non-negative")
    if warmup_steps == 0:
        return 1.0
    return min(1.0, (optimizer_step + 1) / warmup_steps)


def _nonnegative_coefficient(config: dict[str, Any], key: str) -> float:
    value = float(config[key])
    if not torch.isfinite(torch.tensor(value)) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return value


def compute_subject_motion_objective(
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
    """Run one Wan Flow-Matching forward and all subject-motion losses."""

    shared = dict(inputs)
    clean_latents = shared["input_latents"]
    if not isinstance(clean_latents, torch.Tensor) or clean_latents.ndim != 5:
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
    if not bool(torch.isfinite(sigmas).all()) or bool((sigmas < 0).any()) or bool(
        (sigmas > 1).any()
    ):
        raise ValueError("Wan sigma values must be finite and lie in [0, 1]")

    sigma_view = sigmas.reshape(batch, 1, 1, 1, 1)
    if noise is None:
        noise = torch.randn_like(clean_latents) * float(
            shared.get("noise_scale", 1.0)
        )
    if noise.shape != clean_latents.shape:
        raise ValueError("Flow Matching noise shape differs from clean latents")
    noisy_latents = (1.0 - sigma_view) * clean_latents + sigma_view * noise
    training_target = noise - clean_latents
    first_frame = shared.get("first_frame_latents")
    if first_frame is not None:
        if (
            not isinstance(first_frame, torch.Tensor)
            or first_frame.shape != clean_latents[:, :, :1].shape
        ):
            raise ValueError("first_frame_latents must match one BCTHW latent slice")
        noisy_latents = torch.cat([first_frame, noisy_latents[:, :, 1:]], dim=2)
    shared["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    predicted_velocity = pipe.model_fn(
        **models,
        **shared,
        timestep=timesteps,
    )
    if predicted_velocity.shape != clean_latents.shape:
        raise ValueError("Wan predicted velocity shape differs from clean latents")

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
    base_scale = base_loss.new_tensor(float(config.get("base_loss_scale", 1.0)))

    lambda_st = _nonnegative_coefficient(config, "lambda_st")
    lambda_subject = _nonnegative_coefficient(config, "lambda_subject_flow")
    lambda_motion = _nonnegative_coefficient(config, "lambda_motion_delta")
    st_enabled = bool(config["enable_st_iou_loss"])
    if lambda_st > 0.0 and not st_enabled:
        raise ValueError("positive lambda_st requires enable_st_iou_loss=true")
    any_auxiliary = st_enabled or lambda_subject > 0.0 or lambda_motion > 0.0
    if not any_auxiliary:
        zero = base_loss.new_zeros(())
        zero_batch = base_loss.new_zeros((batch,))
        return {
            "total_loss": base_scale * base_loss,
            "base_loss": base_loss,
            "st_loss": zero,
            "st_loss_unweighted": zero,
            "st_iou": zero,
            "st_iou_per_sample": zero_batch,
            "subject_flow_loss": zero,
            "motion_delta_loss": zero,
            "subject_flow_contribution": zero,
            "motion_delta_contribution": zero,
            "st_contribution": zero,
            "noise_sigma": sigmas.float(),
            "noise_weight": zero_batch,
            "motion_noise_weight": zero_batch,
            "aux_warmup_ratio": zero,
            "lambda_st_effective": zero,
            "lambda_effective": zero,
            "lambda_subject_effective": zero,
            "lambda_motion_effective": zero,
            "gt_foreground_fraction": zero,
            "pred_foreground_fraction": zero,
            "subject_support_fraction": zero,
            "motion_support_fraction": zero,
        }

    aligned_mask = align_subject_support_tube(
        subject_mask.to(device=clean_latents.device),
        latent_shape=(
            clean_latents.shape[0],
            clean_latents.shape[2],
            clean_latents.shape[3],
            clean_latents.shape[4],
        ),
    )
    velocity_mask = (
        torch.maximum(aligned_mask[:, 1:], aligned_mask[:, :-1])
        if first_frame is not None
        else aligned_mask
    )
    subject_result = masked_subject_flow_loss(
        base_prediction,
        base_target,
        velocity_mask,
        training_weights,
        eps=float(config["subject_flow_eps"]),
    )

    estimate = flowmatch_clean_estimate(
        noisy_latents,
        predicted_velocity,
        sigmas,
        first_frame_latents=first_frame,
    )
    motion_weights = (1.0 - sigmas.float()).clamp(min=0.0, max=1.0)
    motion_result = subject_temporal_difference_loss(
        estimate,
        clean_latents,
        aligned_mask,
        motion_weights,
        eps=float(config["motion_delta_eps"]),
    )

    zero = base_loss.new_zeros(())
    zero_batch = base_loss.new_zeros((batch,))
    st_loss = zero
    st_loss_unweighted = zero
    st_iou = zero
    st_iou_per_sample = zero_batch
    gt_foreground_fraction = aligned_mask.float().mean()
    pred_foreground_fraction = zero
    st_weights = zero_batch
    if st_enabled:
        occupancy_prediction = occupancy_head(estimate).sigmoid().squeeze(1)
        if occupancy_prediction.shape != aligned_mask.shape:
            raise ValueError("occupancy head output differs from aligned subject mask")
        st_weights = noise_weight(
            sigmas,
            mode=str(config["st_loss_weighting"]),
            threshold=float(config["st_noise_threshold"]),
        )
        tube_result = SpatioTemporalTubeIoULoss(
            eps=float(config["st_iou_eps"])
        )(occupancy_prediction, aligned_mask, sample_weight=st_weights)
        st_loss = tube_result.loss
        st_loss_unweighted = tube_result.per_sample_loss.mean()
        st_iou = tube_result.iou.mean()
        st_iou_per_sample = tube_result.iou
        gt_foreground_fraction = tube_result.target_foreground_fraction.mean()
        pred_foreground_fraction = (
            tube_result.prediction_foreground_fraction.mean()
        )

    warmup_ratio = auxiliary_warmup_ratio(
        optimizer_step,
        int(config["aux_warmup_steps"]),
    )
    ratio_tensor = base_loss.new_tensor(warmup_ratio)
    lambda_st_tensor = base_loss.new_tensor(lambda_st * warmup_ratio)
    lambda_subject_tensor = base_loss.new_tensor(lambda_subject * warmup_ratio)
    lambda_motion_tensor = base_loss.new_tensor(lambda_motion * warmup_ratio)
    st_contribution = lambda_st_tensor * st_loss
    subject_contribution = lambda_subject_tensor * subject_result.loss
    motion_contribution = lambda_motion_tensor * motion_result.loss
    total_loss = (
        base_scale * base_loss
        + st_contribution
        + subject_contribution
        + motion_contribution
    )
    return {
        "total_loss": total_loss,
        "base_loss": base_loss,
        "st_loss": st_loss,
        "st_loss_unweighted": st_loss_unweighted,
        "st_iou": st_iou,
        "st_iou_per_sample": st_iou_per_sample,
        "subject_flow_loss": subject_result.loss,
        "motion_delta_loss": motion_result.loss,
        "st_contribution": st_contribution,
        "subject_flow_contribution": subject_contribution,
        "motion_delta_contribution": motion_contribution,
        "noise_sigma": sigmas.float(),
        "noise_weight": st_weights.float(),
        "motion_noise_weight": motion_weights,
        "aux_warmup_ratio": ratio_tensor,
        "lambda_st_effective": lambda_st_tensor,
        "lambda_effective": lambda_st_tensor,
        "lambda_subject_effective": lambda_subject_tensor,
        "lambda_motion_effective": lambda_motion_tensor,
        "gt_foreground_fraction": gt_foreground_fraction,
        "pred_foreground_fraction": pred_foreground_fraction,
        "subject_support_fraction": subject_result.support_fraction.mean(),
        "motion_support_fraction": motion_result.support_fraction.mean(),
    }


class SubjectMotionWanTrainingModule(STTubeIoUWanTrainingModule):
    """Tube-IoU module with direct subject velocity and motion supervision."""

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
        metrics = compute_subject_motion_objective(
            pipe=self.pipe,
            inputs={**shared, **positive},
            subject_mask=subject_mask,
            occupancy_head=self.occupancy_head,
            config=self.st_config,
            optimizer_step=self.optimizer_step,
        )
        self.last_metrics = metrics
        return metrics


def launch_subject_motion_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: SubjectMotionWanTrainingModule,
    logger: STTubeIoUModelLogger,
    args: Any,
) -> None:
    """Run the audited training loop with synchronized expanded metrics."""

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
                    require_head=(
                        bool(unwrapped.st_config["enable_st_iou_loss"])
                        and float(unwrapped.st_config["lambda_st"]) > 0.0
                    ),
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
                scalar_keys = (
                    "base_loss",
                    "st_loss",
                    "st_loss_unweighted",
                    "st_iou",
                    "subject_flow_loss",
                    "motion_delta_loss",
                    "st_contribution",
                    "subject_flow_contribution",
                    "motion_delta_contribution",
                    "aux_warmup_ratio",
                    "lambda_st_effective",
                    "lambda_subject_effective",
                    "lambda_motion_effective",
                    "total_loss",
                    "gt_foreground_fraction",
                    "pred_foreground_fraction",
                    "subject_support_fraction",
                    "motion_support_fraction",
                )
                record = {
                    "schema_version": "1.0",
                    "optimizer_step": logger.num_steps,
                    "epoch_id": epoch_id,
                    **{
                        key: _gather_scalar(accelerator, metrics[key])
                        for key in scalar_keys
                    },
                    "noise_sigma": _gather_scalar(
                        accelerator, metrics["noise_sigma"].mean()
                    ),
                    "noise_weight": _gather_scalar(
                        accelerator, metrics["noise_weight"].mean()
                    ),
                    "motion_noise_weight": _gather_scalar(
                        accelerator, metrics["motion_noise_weight"].mean()
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
        "lambda_subject_flow",
        "lambda_motion_delta",
        "st_iou_eps",
        "subject_flow_eps",
        "motion_delta_eps",
        "st_loss_weighting",
        "st_noise_threshold",
        "aux_warmup_steps",
        "latent_channels",
        "hidden_channels",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"subject-motion config is missing fields: {missing}")
    for key in ("lambda_st", "lambda_subject_flow", "lambda_motion_delta"):
        _nonnegative_coefficient(config, key)
    if float(config["lambda_st"]) > 0.0 and not bool(
        config["enable_st_iou_loss"]
    ):
        raise ValueError("positive lambda_st requires enable_st_iou_loss=true")
    if config["st_loss_weighting"] not in {"none", "linear_clean", "threshold"}:
        raise ValueError("unsupported st_loss_weighting")
    auxiliary_warmup_ratio(0, int(config["aux_warmup_steps"]))
    for key in ("st_iou_eps", "subject_flow_eps", "motion_delta_eps"):
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")


def main() -> int:
    worker_seed = seed_process()
    parser = wan_parser()
    parser.add_argument("--subject_motion_config_json", required=True)
    parser.add_argument("--subject_motion_metrics_path", type=Path, required=True)
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--sampler_seed", type=int, required=True)
    args = parser.parse_args()
    if args.task != "sft":
        raise ValueError("subject-motion baseline supports task=sft only")
    if args.lora_rank != 32:
        raise ValueError("WAN subject-motion baseline requires lora_rank=32")
    if args.lora_target_modules != "q,k,v,o,ffn.0,ffn.2":
        raise ValueError("WAN subject-motion baseline requires sealed LoRA targets")
    config = json.loads(args.subject_motion_config_json)
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
            "input_audio": ToAbsolutePath(args.dataset_base_path)
            >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )
    model = SubjectMotionWanTrainingModule(
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
            raise FileNotFoundError(
                f"paired subject-motion head checkpoint is missing: {head_path}"
            )
        model.occupancy_head.load_state_dict(load_file(str(head_path), device="cpu"))
    if accelerator.is_main_process:
        write_json(
            Path(args.output_path) / "subject_motion_spec.json",
            {
                "schema_version": "1.0",
                "worker_seed_rank_0": worker_seed,
                "sampler_seed": args.sampler_seed,
                "flow_parameterization": {
                    "noisy": "x_s=(1-sigma)*x0+sigma*epsilon",
                    "velocity": "epsilon-x0",
                    "clean_estimate": "x0_hat=x_s-sigma*v_hat",
                },
                "occupancy_source": (
                    "learned_latent_space_subject_occupancy_proxy"
                ),
                "config": config,
            },
        )
    logger = STTubeIoUModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        st_config=config,
        metrics_path=args.subject_motion_metrics_path,
        save_optimizer_state=args.save_optimizer_state,
    )
    launch_subject_motion_training(accelerator, dataset, model, logger, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
