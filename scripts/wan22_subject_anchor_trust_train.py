#!/usr/bin/env python3
"""Train WAN2.2 with a fixed motion probe and LoRA trust region."""

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

from physbench.baselines.wan22_subject_anchor_trust_model import (
    anchored_centroid_displacement_loss,
    relative_lora_drift_loss,
    snapshot_lora_parameters,
)
from physbench.baselines.wan22_subject_motion_model import (
    align_subject_support_tube,
)
from scripts import wan22_subject_motion_train as subject_motion


_BASE_VALIDATE_CONFIG = subject_motion._validate_config


def compute_subject_anchor_trust_objective(
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
    """Add anchor-relative motion and parameter trust to Flow Matching."""

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
        metrics = subject_motion.compute_subject_motion_objective(
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

    lambda_anchor = subject_motion._nonnegative_coefficient(
        config, "lambda_anchored_displacement"
    )
    lambda_trust = subject_motion._nonnegative_coefficient(
        config, "lambda_lora_trust_region"
    )
    if lambda_anchor > 0.0 and "logits" not in captured:
        raise RuntimeError("anchored displacement requires the occupancy probe")
    zero = metrics["base_loss"].new_zeros(())
    anchor_loss = zero
    valid_fraction = zero
    if lambda_anchor > 0.0:
        logits = captured["logits"]
        if logits.ndim != 5 or logits.shape[1] != 1:
            raise ValueError("anchor occupancy logits must use B1THW layout")
        probability = logits.sigmoid().squeeze(1)
        target = align_subject_support_tube(
            subject_mask.to(device=probability.device),
            latent_shape=probability.shape,
        )
        anchored = anchored_centroid_displacement_loss(
            probability,
            target,
            sample_weight=metrics["noise_weight"],
            eps=float(config["trajectory_eps"]),
            smooth_l1_beta=float(config["trajectory_smooth_l1_beta"]),
        )
        anchor_loss = anchored.loss
        valid_fraction = anchored.valid_frame_fraction

    drift = relative_lora_drift_loss(
        pipe.dit.named_parameters(),
        lora_reference,
        eps=float(config["lora_trust_region_eps"]),
    )
    warmup = metrics["aux_warmup_ratio"]
    effective_anchor = warmup * metrics["base_loss"].new_tensor(lambda_anchor)
    effective_trust = warmup * metrics["base_loss"].new_tensor(lambda_trust)
    anchor_contribution = effective_anchor * anchor_loss
    trust_contribution = effective_trust * drift
    metrics["total_loss"] = (
        metrics["total_loss"] + anchor_contribution + trust_contribution
    )
    metrics.update({
        "anchored_displacement_loss": anchor_loss,
        "anchored_displacement_contribution": anchor_contribution,
        "lambda_anchored_displacement_effective": effective_anchor,
        "anchored_valid_frame_fraction": valid_fraction,
        "lora_relative_drift": drift,
        "lora_trust_region_contribution": trust_contribution,
        "lambda_lora_trust_region_effective": effective_trust,
    })
    return metrics


class SubjectAnchorTrustWanTrainingModule(
    subject_motion.SubjectMotionWanTrainingModule
):
    """Treat the paired occupancy head as a fixed differentiable probe."""

    def __init__(self, *args, st_config: dict[str, Any], **kwargs):
        super().__init__(*args, st_config=st_config, **kwargs)
        if not bool(st_config["freeze_occupancy_head"]):
            raise ValueError("subject-anchor-trust requires a frozen occupancy head")
        self.occupancy_head.eval().requires_grad_(False)
        self.lora_reference = snapshot_lora_parameters(
            self.pipe.dit.named_parameters()
        )

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
        metrics = compute_subject_anchor_trust_objective(
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


def launch_subject_anchor_trust_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: SubjectAnchorTrustWanTrainingModule,
    logger: subject_motion.STTubeIoUModelLogger,
    args: Any,
) -> None:
    """Run the synchronized loop and audit anchor/trust losses."""

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
                    "subject_flow_loss",
                    "motion_delta_loss",
                    "st_contribution",
                    "subject_flow_contribution",
                    "motion_delta_contribution",
                    "anchored_displacement_loss",
                    "anchored_displacement_contribution",
                    "lambda_anchored_displacement_effective",
                    "anchored_valid_frame_fraction",
                    "lora_relative_drift",
                    "lora_trust_region_contribution",
                    "lambda_lora_trust_region_effective",
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


def _validate_anchor_trust_config(config: dict[str, Any]) -> None:
    _BASE_VALIDATE_CONFIG(config)
    required = {
        "freeze_occupancy_head",
        "lambda_anchored_displacement",
        "lambda_lora_trust_region",
        "trajectory_eps",
        "trajectory_smooth_l1_beta",
        "lora_trust_region_eps",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"subject-anchor-trust config is missing fields: {missing}")
    if not bool(config["enable_st_iou_loss"]):
        raise ValueError("anchored displacement requires the occupancy probe")
    if float(config["lambda_st"]) != 0.0:
        raise ValueError("fixed-probe recipe requires lambda_st=0")
    if not bool(config["freeze_occupancy_head"]):
        raise ValueError("fixed-probe recipe requires freeze_occupancy_head=true")
    for key in (
        "lambda_anchored_displacement",
        "lambda_lora_trust_region",
    ):
        subject_motion._nonnegative_coefficient(config, key)
    for key in (
        "trajectory_eps",
        "trajectory_smooth_l1_beta",
        "lora_trust_region_eps",
    ):
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")


def main() -> int:
    subject_motion.SubjectMotionWanTrainingModule = (
        SubjectAnchorTrustWanTrainingModule
    )
    subject_motion.launch_subject_motion_training = (
        launch_subject_anchor_trust_training
    )
    subject_motion._validate_config = _validate_anchor_trust_config
    return subject_motion.main()


if __name__ == "__main__":
    raise SystemExit(main())
