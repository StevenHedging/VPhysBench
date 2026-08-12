#!/usr/bin/env python3
"""Train WAN2.2 LoRA with Tube-IoU and centroid-trajectory auxiliaries."""

from __future__ import annotations

import time
from typing import Any

import accelerate
import torch
from torch.utils.data import DataLoader

from diffsynth.core import UnifiedDataset
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    save_training_args,
)

from physbench.baselines.wan22_subject_motion_model import (
    align_subject_support_tube,
)
from physbench.baselines.wan22_subject_trajectory_model import (
    subject_centroid_trajectory_loss,
)
from scripts import wan22_subject_motion_train as subject_motion


_BASE_COMPUTE_OBJECTIVE = subject_motion.compute_subject_motion_objective
_BASE_VALIDATE_CONFIG = subject_motion._validate_config
_BASE_TRAINING_MODULE = subject_motion.SubjectMotionWanTrainingModule


def compute_subject_trajectory_objective(
    *,
    pipe: Any,
    inputs: dict[str, Any],
    subject_mask: torch.Tensor,
    occupancy_head: torch.nn.Module,
    config: dict[str, Any],
    optimizer_step: int,
    timestep_ids: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Add normalized centroid position/velocity to the audited base objective."""

    captured: dict[str, torch.Tensor] = {}

    def capture_occupancy(
        module: torch.nn.Module,
        module_inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, module_inputs
        captured["logits"] = output

    handle = occupancy_head.register_forward_hook(capture_occupancy)
    try:
        metrics = _BASE_COMPUTE_OBJECTIVE(
            pipe=pipe,
            inputs=inputs,
            subject_mask=subject_mask,
            occupancy_head=occupancy_head,
            config=config,
            optimizer_step=optimizer_step,
            timestep_ids=timestep_ids,
            noise=noise,
        )
    finally:
        handle.remove()

    lambda_position = subject_motion._nonnegative_coefficient(
        config, "lambda_centroid_position"
    )
    lambda_velocity = subject_motion._nonnegative_coefficient(
        config, "lambda_centroid_velocity"
    )
    if lambda_position == 0.0 and lambda_velocity == 0.0:
        zero = metrics["base_loss"].new_zeros(())
        metrics.update({
            "centroid_position_loss": zero,
            "centroid_velocity_loss": zero,
            "centroid_position_contribution": zero,
            "centroid_velocity_contribution": zero,
            "lambda_centroid_position_effective": zero,
            "lambda_centroid_velocity_effective": zero,
            "trajectory_valid_frame_fraction": zero,
            "trajectory_valid_velocity_fraction": zero,
        })
        return metrics
    if "logits" not in captured:
        raise RuntimeError(
            "positive centroid loss requires the latent occupancy head"
        )

    logits = captured["logits"]
    if logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError("trajectory occupancy logits must use B1THW layout")
    probability = logits.sigmoid().squeeze(1)
    target = align_subject_support_tube(
        subject_mask.to(device=probability.device),
        latent_shape=probability.shape,
    )
    trajectory = subject_centroid_trajectory_loss(
        probability,
        target,
        sample_weight=metrics["noise_weight"],
        eps=float(config["trajectory_eps"]),
        smooth_l1_beta=float(config["trajectory_smooth_l1_beta"]),
    )
    warmup = metrics["aux_warmup_ratio"]
    effective_position = warmup * metrics["base_loss"].new_tensor(
        lambda_position
    )
    effective_velocity = warmup * metrics["base_loss"].new_tensor(
        lambda_velocity
    )
    position_contribution = effective_position * trajectory.position_loss
    velocity_contribution = effective_velocity * trajectory.velocity_loss
    metrics["total_loss"] = (
        metrics["total_loss"]
        + position_contribution
        + velocity_contribution
    )
    metrics.update({
        "centroid_position_loss": trajectory.position_loss,
        "centroid_velocity_loss": trajectory.velocity_loss,
        "centroid_position_contribution": position_contribution,
        "centroid_velocity_contribution": velocity_contribution,
        "lambda_centroid_position_effective": effective_position,
        "lambda_centroid_velocity_effective": effective_velocity,
        "trajectory_valid_frame_fraction": trajectory.valid_frame_fraction,
        "trajectory_valid_velocity_fraction": (
            trajectory.valid_velocity_fraction
        ),
    })
    return metrics


class SubjectTrajectoryWanTrainingModule(_BASE_TRAINING_MODULE):
    """Tube-IoU WAN module with direct normalized trajectory supervision."""

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
        metrics = compute_subject_trajectory_objective(
            pipe=self.pipe,
            inputs={**shared, **positive},
            subject_mask=subject_mask,
            occupancy_head=self.occupancy_head,
            config=self.st_config,
            optimizer_step=self.optimizer_step,
        )
        self.last_metrics = metrics
        return metrics


def launch_subject_trajectory_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: SubjectTrajectoryWanTrainingModule,
    logger: subject_motion.STTubeIoUModelLogger,
    args: Any,
) -> None:
    """Run the synchronized loop and retain trajectory-specific metrics."""

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
                gradient = subject_motion._gradient_audit(
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
                    "centroid_position_loss",
                    "centroid_velocity_loss",
                    "centroid_position_contribution",
                    "centroid_velocity_contribution",
                    "aux_warmup_ratio",
                    "lambda_st_effective",
                    "lambda_subject_effective",
                    "lambda_motion_effective",
                    "lambda_centroid_position_effective",
                    "lambda_centroid_velocity_effective",
                    "total_loss",
                    "gt_foreground_fraction",
                    "pred_foreground_fraction",
                    "subject_support_fraction",
                    "motion_support_fraction",
                    "trajectory_valid_frame_fraction",
                    "trajectory_valid_velocity_fraction",
                )
                record = {
                    "schema_version": "1.0",
                    "optimizer_step": logger.num_steps,
                    "epoch_id": epoch_id,
                    **{
                        key: subject_motion._gather_scalar(
                            accelerator, metrics[key]
                        )
                        for key in scalar_keys
                    },
                    "noise_sigma": subject_motion._gather_scalar(
                        accelerator, metrics["noise_sigma"].mean()
                    ),
                    "noise_weight": subject_motion._gather_scalar(
                        accelerator, metrics["noise_weight"].mean()
                    ),
                    "motion_noise_weight": subject_motion._gather_scalar(
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


def _validate_trajectory_config(config: dict[str, Any]) -> None:
    _BASE_VALIDATE_CONFIG(config)
    required = {
        "lambda_centroid_position",
        "lambda_centroid_velocity",
        "trajectory_eps",
        "trajectory_smooth_l1_beta",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"subject-trajectory config is missing fields: {missing}")
    for key in ("lambda_centroid_position", "lambda_centroid_velocity"):
        subject_motion._nonnegative_coefficient(config, key)
    if (
        float(config["lambda_centroid_position"]) > 0.0
        or float(config["lambda_centroid_velocity"]) > 0.0
    ) and not bool(config["enable_st_iou_loss"]):
        raise ValueError("positive centroid losses require the occupancy head")
    for key in ("trajectory_eps", "trajectory_smooth_l1_beta"):
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")


def main() -> int:
    subject_motion.SubjectMotionWanTrainingModule = (
        SubjectTrajectoryWanTrainingModule
    )
    subject_motion.launch_subject_motion_training = (
        launch_subject_trajectory_training
    )
    subject_motion._validate_config = _validate_trajectory_config
    return subject_motion.main()


if __name__ == "__main__":
    raise SystemExit(main())
