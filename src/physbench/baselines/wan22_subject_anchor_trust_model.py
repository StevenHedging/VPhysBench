from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from .wan22_subject_trajectory_model import soft_mask_centroid_trajectory


@dataclass(frozen=True)
class AnchoredDisplacementLossResult:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    valid_frame_fraction: torch.Tensor


def anchored_centroid_displacement_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    smooth_l1_beta: float = 0.05,
) -> AnchoredDisplacementLossResult:
    """Match motion relative to each tube's first valid subject position."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("anchored trajectory tensors must share BTHW shape")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("anchored trajectory sample weights must have shape [B]")
    if eps <= 0 or smooth_l1_beta <= 0:
        raise ValueError("anchored trajectory epsilon and beta must be positive")
    weights = sample_weight.to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("anchored trajectory weights must be finite and non-negative")

    predicted = soft_mask_centroid_trajectory(prediction, eps=eps)
    expected = soft_mask_centroid_trajectory(target, eps=eps)
    valid = expected.mass > float(eps)
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("anchored trajectory target contains an empty sample")
    first_index = valid.to(torch.int64).argmax(dim=1)
    gather_index = first_index.reshape(-1, 1, 1).expand(-1, 1, 2)
    predicted_anchor = predicted.centroids.gather(1, gather_index)
    expected_anchor = expected.centroids.gather(1, gather_index)
    predicted_displacement = predicted.centroids - predicted_anchor
    expected_displacement = expected.centroids - expected_anchor
    frame_error = F.smooth_l1_loss(
        predicted_displacement,
        expected_displacement,
        reduction="none",
        beta=float(smooth_l1_beta),
    ).mean(dim=-1)
    valid_f = valid.float()
    per_sample = (frame_error * valid_f).sum(dim=1) / valid_f.sum(
        dim=1
    ).clamp_min(1.0)
    return AnchoredDisplacementLossResult(
        loss=(per_sample * weights).mean(),
        per_sample_loss=per_sample,
        valid_frame_fraction=valid_f.mean(),
    )


def _is_lora_parameter(name: str) -> bool:
    return ".lora_A" in name or ".lora_B" in name or name.startswith(
        ("lora_A", "lora_B")
    )


def snapshot_lora_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    """Capture an immutable float32 trust-region center for LoRA tensors."""

    reference = {
        name: parameter.detach().float().clone()
        for name, parameter in named_parameters
        if parameter.requires_grad and _is_lora_parameter(name)
    }
    if not reference:
        raise ValueError("cannot snapshot an empty LoRA parameter set")
    return reference


def relative_lora_drift_loss(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    reference: dict[str, torch.Tensor],
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return squared LoRA displacement normalized by parent squared norm."""

    if eps <= 0:
        raise ValueError("LoRA trust-region epsilon must be positive")
    current = {
        name: parameter
        for name, parameter in named_parameters
        if parameter.requires_grad and _is_lora_parameter(name)
    }
    if not current or set(current) != set(reference):
        raise ValueError("LoRA trust-region parameter identity changed")
    first = next(iter(current.values()))
    numerator = torch.zeros((), device=first.device, dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    for name, parameter in current.items():
        center = reference[name]
        if center.shape != parameter.shape:
            raise ValueError(f"LoRA trust-region shape changed for {name}")
        center = center.to(device=parameter.device, dtype=torch.float32)
        if not bool(torch.isfinite(center).all()):
            raise ValueError(f"LoRA trust-region reference is non-finite for {name}")
        difference = parameter.float() - center
        numerator = numerator + difference.square().sum()
        denominator = denominator + center.square().sum()
    return numerator / denominator.clamp_min(float(eps))


__all__ = [
    "AnchoredDisplacementLossResult",
    "anchored_centroid_displacement_loss",
    "relative_lora_drift_loss",
    "snapshot_lora_parameters",
]
