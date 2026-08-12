from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .wan22_st_tube_iou_model import expected_wan_vae_latent_frames


@dataclass(frozen=True)
class MaskedLossResult:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    support_fraction: torch.Tensor


def align_subject_support_tube(
    mask: torch.Tensor,
    *,
    latent_shape: Sequence[int],
) -> torch.Tensor:
    """Conservatively pool a pixel tube onto the WAN latent grid.

    Max pooling preserves every occupied source region, including subjects
    smaller than one latent cell that nearest-neighbor sampling can miss.
    """

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError("subject mask must use THW or BTHW layout")
    if len(latent_shape) != 4:
        raise ValueError("latent mask shape must use BTHW layout")
    batch, latent_frames, latent_height, latent_width = map(int, latent_shape)
    if min(batch, latent_frames, latent_height, latent_width) < 1:
        raise ValueError("latent mask dimensions must be positive")
    if mask.shape[0] != batch:
        raise ValueError(
            f"mask batch {mask.shape[0]} does not match latent batch {batch}"
        )
    expected_frames = expected_wan_vae_latent_frames(int(mask.shape[1]))
    if expected_frames != latent_frames:
        raise ValueError(
            "subject-mask temporal compression does not match Wan VAE: "
            f"{mask.shape[1]} pixel frames imply {expected_frames} latent "
            f"frames, got {latent_frames}"
        )
    mask_f = mask.float()
    if not bool(torch.isfinite(mask_f).all()):
        raise ValueError("subject mask contains non-finite values")
    if bool((mask_f < 0).any()) or bool((mask_f > 1).any()):
        raise ValueError("subject mask must be in [0, 1]")
    source_support = mask_f.flatten(start_dim=1).sum(dim=1)
    if bool((source_support <= 0).any()):
        raise ValueError("subject mask contains an empty sample")

    aligned = F.adaptive_max_pool3d(
        mask_f.unsqueeze(1),
        output_size=(latent_frames, latent_height, latent_width),
    ).squeeze(1)
    aligned_support = aligned.flatten(start_dim=1).sum(dim=1)
    if bool((aligned_support <= 0).any()):
        raise AssertionError("area-preserving subject alignment lost support")
    return aligned


def masked_subject_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    subject_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    eps: float = 1e-6,
) -> MaskedLossResult:
    """Area-normalized velocity MSE inside one subject tube per sample."""

    if eps <= 0:
        raise ValueError("subject Flow epsilon must be positive")
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError("subject Flow tensors must use BCTHW layout")
    if prediction.shape != target.shape:
        raise ValueError("subject Flow prediction and target shapes differ")
    expected_mask_shape = (
        prediction.shape[0],
        prediction.shape[2],
        prediction.shape[3],
        prediction.shape[4],
    )
    if subject_mask.shape != expected_mask_shape:
        raise ValueError("subject Flow mask must match the prediction BTHW shape")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("subject Flow sample weights must have shape [B]")

    prediction_f = prediction.float()
    target_f = target.float()
    mask_f = subject_mask.to(device=prediction.device, dtype=torch.float32)
    weights_f = sample_weight.to(device=prediction.device, dtype=torch.float32)
    for name, value in (
        ("prediction", prediction_f),
        ("target", target_f),
        ("mask", mask_f),
        ("sample weights", weights_f),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"subject Flow {name} contains non-finite values")
    if bool((mask_f < 0).any()) or bool((mask_f > 1).any()):
        raise ValueError("subject Flow mask must be in [0, 1]")

    error = (prediction_f - target_f).square().mean(dim=1)
    support = mask_f.flatten(start_dim=1).sum(dim=1)
    if bool((support <= 0).any()):
        raise ValueError("subject Flow received an empty aligned subject mask")
    numerator = (error * mask_f).flatten(start_dim=1).sum(dim=1)
    per_sample = numerator / (support + float(eps))
    voxel_count = mask_f[0].numel()
    return MaskedLossResult(
        loss=(per_sample * weights_f).mean(),
        per_sample_loss=per_sample,
        support_fraction=support / voxel_count,
    )


def subject_temporal_difference_loss(
    clean_estimate: torch.Tensor,
    clean_target: torch.Tensor,
    subject_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    eps: float = 1e-6,
) -> MaskedLossResult:
    """Match consecutive clean-latent changes on subject endpoint unions."""

    if eps <= 0:
        raise ValueError("subject temporal-difference epsilon must be positive")
    if clean_estimate.ndim != 5 or clean_target.ndim != 5:
        raise ValueError("subject temporal-difference tensors must use BCTHW layout")
    if clean_estimate.shape != clean_target.shape:
        raise ValueError("clean estimate and target shapes differ")
    if clean_estimate.shape[2] < 2:
        raise ValueError("subject motion loss requires at least two latent frames")
    expected_mask_shape = (
        clean_estimate.shape[0],
        clean_estimate.shape[2],
        clean_estimate.shape[3],
        clean_estimate.shape[4],
    )
    if subject_mask.shape != expected_mask_shape:
        raise ValueError("subject motion mask must match the clean latent BTHW shape")
    if sample_weight.shape != (clean_estimate.shape[0],):
        raise ValueError("subject motion sample weights must have shape [B]")

    estimate_f = clean_estimate.float()
    target_f = clean_target.float()
    mask_f = subject_mask.to(device=clean_estimate.device, dtype=torch.float32)
    weights_f = sample_weight.to(device=clean_estimate.device, dtype=torch.float32)
    for name, value in (
        ("clean estimate", estimate_f),
        ("clean target", target_f),
        ("mask", mask_f),
        ("sample weights", weights_f),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"subject temporal-difference {name} contains non-finite values"
            )
    if bool((mask_f < 0).any()) or bool((mask_f > 1).any()):
        raise ValueError("subject motion mask must be in [0, 1]")

    estimate_delta = estimate_f[:, :, 1:] - estimate_f[:, :, :-1]
    target_delta = target_f[:, :, 1:] - target_f[:, :, :-1]
    error = ((estimate_delta - target_delta).square() + float(eps)).sqrt()
    error = error.mean(dim=1)
    support_mask = torch.maximum(mask_f[:, 1:], mask_f[:, :-1])
    support = support_mask.flatten(start_dim=1).sum(dim=1)
    if bool((support <= 0).any()):
        raise ValueError("subject motion received an empty endpoint-union mask")
    numerator = (error * support_mask).flatten(start_dim=1).sum(dim=1)
    per_sample = numerator / (support + float(eps))
    voxel_count = support_mask[0].numel()
    return MaskedLossResult(
        loss=(per_sample * weights_f).mean(),
        per_sample_loss=per_sample,
        support_fraction=support / voxel_count,
    )


__all__ = [
    "align_subject_support_tube",
    "MaskedLossResult",
    "masked_subject_flow_loss",
    "subject_temporal_difference_loss",
]
