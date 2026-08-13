#!/usr/bin/env python3
"""Train WAN2.2 with fixed-probe tube, motion, geometry, and trust losses."""

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

from physbench.baselines.wan22_subject_geometry_trust_model import (
    subject_spatial_geometry_loss,
)
from physbench.baselines.wan22_subject_motion_model import (
    align_subject_support_tube,
)
from scripts import wan22_subject_anchor_trust_train as anchor_trust
from scripts import wan22_subject_motion_train as subject_motion


_BASE_VALIDATE_CONFIG = subject_motion._validate_config


def compute_subject_geometry_trust_objective(
    *,
    pipe: Any,
    inputs: dict[str, Any],
    subject_mask: torch.Tensor,
    occupancy_head: torch.nn.Module,
    lora_reference: dict[str, torch.Tensor],
    config: dict[str, Any],
    optimizer_step: int,
    timestep_ids: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Add foreground mass and covariance to anchor/trust supervision."""

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
        metrics = anchor_trust.compute_subject_anchor_trust_objective(
            pipe=pipe,
            inputs=inputs,
            subject_mask=subject_mask,
            occupancy_head=occupancy_head,
            lora_reference=lora_reference,
            config=config,
            optimizer_step=optimizer_step,
            timestep_ids=timestep_ids,
            noise=noise,
        )
    finally:
        handle.remove()

    lambda_mass = subject_motion._nonnegative_coefficient(
        config, "lambda_subject_mass"
    )
    lambda_covariance = subject_motion._nonnegative_coefficient(
        config, "lambda_subject_covariance"
    )
    zero = metrics["base_loss"].new_zeros(())
    mass_loss = zero
    covariance_loss = zero
    valid_fraction = zero
    if lambda_mass > 0.0 or lambda_covariance > 0.0:
        logits = captured.get("logits")
        if logits is None:
            raise RuntimeError("subject geometry requires the occupancy probe")
        if logits.ndim != 5 or logits.shape[1] != 1:
            raise ValueError("geometry occupancy logits must use B1THW layout")
        probability = logits.sigmoid().squeeze(1)
        target = align_subject_support_tube(
            subject_mask.to(device=probability.device),
            latent_shape=probability.shape,
        )
        geometry = subject_spatial_geometry_loss(
            probability,
            target,
            sample_weight=metrics["noise_weight"],
            eps=float(config["geometry_eps"]),
            smooth_l1_beta=float(config["geometry_smooth_l1_beta"]),
        )
        mass_loss = geometry.mass_loss
        covariance_loss = geometry.covariance_loss
        valid_fraction = geometry.valid_frame_fraction

    warmup = metrics["aux_warmup_ratio"]
    effective_mass = warmup * metrics["base_loss"].new_tensor(lambda_mass)
    effective_covariance = warmup * metrics["base_loss"].new_tensor(
        lambda_covariance
    )
    mass_contribution = effective_mass * mass_loss
    covariance_contribution = effective_covariance * covariance_loss
    metrics["total_loss"] = (
        metrics["total_loss"] + mass_contribution + covariance_contribution
    )
    metrics.update({
        "subject_mass_loss": mass_loss,
        "subject_covariance_loss": covariance_loss,
        "subject_mass_contribution": mass_contribution,
        "subject_covariance_contribution": covariance_contribution,
        "lambda_subject_mass_effective": effective_mass,
        "lambda_subject_covariance_effective": effective_covariance,
        "geometry_valid_frame_fraction": valid_fraction,
    })
    return metrics


class SubjectGeometryTrustWanTrainingModule(
    anchor_trust.SubjectAnchorTrustWanTrainingModule
):
    """Use a frozen occupancy probe for absolute and relative geometry."""

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
        metrics = compute_subject_geometry_trust_objective(
            pipe=self.pipe,
            inputs={**shared, **positive},
            subject_mask=subject_mask,
            occupancy_head=self.occupancy_head,
            lora_reference=self.lora_reference,
            config=self.st_config,
            optimizer_step=self.optimizer_step,
        )
        self.last_metrics = metrics
        return metrics


def launch_subject_geometry_trust_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: SubjectGeometryTrustWanTrainingModule,
    logger: subject_motion.STTubeIoUModelLogger,
    args: Any,
) -> None:
    """Run synchronized training and audit every structural contribution."""

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
                    require_head=False,
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
                    "st_contribution",
                    "anchored_displacement_loss",
                    "anchored_displacement_contribution",
                    "lora_relative_drift",
                    "lora_trust_region_contribution",
                    "subject_mass_loss",
                    "subject_covariance_loss",
                    "subject_mass_contribution",
                    "subject_covariance_contribution",
                    "lambda_subject_mass_effective",
                    "lambda_subject_covariance_effective",
                    "geometry_valid_frame_fraction",
                    "aux_warmup_ratio",
                    "total_loss",
                    "gt_foreground_fraction",
                    "pred_foreground_fraction",
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


def _validate_geometry_trust_config(config: dict[str, Any]) -> None:
    _BASE_VALIDATE_CONFIG(config)
    required = {
        "freeze_occupancy_head",
        "lambda_anchored_displacement",
        "lambda_lora_trust_region",
        "lambda_subject_mass",
        "lambda_subject_covariance",
        "trajectory_eps",
        "trajectory_smooth_l1_beta",
        "lora_trust_region_eps",
        "geometry_eps",
        "geometry_smooth_l1_beta",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"subject-geometry config is missing fields: {missing}")
    if not bool(config["enable_st_iou_loss"]):
        raise ValueError("subject geometry requires the occupancy probe")
    if not bool(config["freeze_occupancy_head"]):
        raise ValueError("subject geometry requires freeze_occupancy_head=true")
    for key in (
        "lambda_anchored_displacement",
        "lambda_lora_trust_region",
        "lambda_subject_mass",
        "lambda_subject_covariance",
    ):
        subject_motion._nonnegative_coefficient(config, key)
    for key in (
        "trajectory_eps",
        "trajectory_smooth_l1_beta",
        "lora_trust_region_eps",
        "geometry_eps",
        "geometry_smooth_l1_beta",
    ):
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")


def main() -> int:
    subject_motion.SubjectMotionWanTrainingModule = (
        SubjectGeometryTrustWanTrainingModule
    )
    subject_motion.launch_subject_motion_training = (
        launch_subject_geometry_trust_training
    )
    subject_motion._validate_config = _validate_geometry_trust_config
    return subject_motion.main()


if __name__ == "__main__":
    raise SystemExit(main())
