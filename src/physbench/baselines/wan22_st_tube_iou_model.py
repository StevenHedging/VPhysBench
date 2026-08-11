from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TubeIoUResult:
    loss: torch.Tensor
    iou: torch.Tensor
    per_sample_loss: torch.Tensor
    intersection: torch.Tensor
    union: torch.Tensor
    prediction_foreground_fraction: torch.Tensor
    target_foreground_fraction: torch.Tensor


class SpatioTemporalTubeIoULoss(nn.Module):
    """Soft IoU over one complete ``T*H*W`` occupancy tube per sample."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        if eps <= 0:
            raise ValueError("tube IoU eps must be positive")
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> TubeIoUResult:
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("tube IoU expects prediction and target in BTHW layout")
        if prediction.shape != target.shape:
            raise ValueError(
                "tube IoU prediction and target shapes differ: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        prediction_f = prediction.float()
        target_f = target.float()
        if not bool(torch.isfinite(prediction_f).all()):
            raise ValueError("tube IoU prediction contains non-finite values")
        if not bool(torch.isfinite(target_f).all()):
            raise ValueError("tube IoU target contains non-finite values")
        if bool((prediction_f < 0).any()) or bool((prediction_f > 1).any()):
            raise ValueError("tube IoU prediction must be in [0, 1]")
        if bool((target_f < 0).any()) or bool((target_f > 1).any()):
            raise ValueError("tube IoU target must be in [0, 1]")

        prediction_flat = prediction_f.flatten(start_dim=1)
        target_flat = target_f.flatten(start_dim=1)
        intersection = (prediction_flat * target_flat).sum(dim=1)
        prediction_sum = prediction_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        union = prediction_sum + target_sum - intersection
        raw_iou = (intersection + self.eps) / (union + self.eps)
        iou = torch.where(union == 0, torch.ones_like(raw_iou), raw_iou)
        per_sample_loss = 1.0 - iou

        if sample_weight is None:
            weighted_loss = per_sample_loss
        else:
            weights = sample_weight.to(
                device=per_sample_loss.device,
                dtype=per_sample_loss.dtype,
            )
            if weights.ndim == 0:
                weights = weights.expand_as(per_sample_loss)
            if weights.shape != per_sample_loss.shape:
                raise ValueError(
                    "tube IoU sample weights must have shape [B]: "
                    f"got {tuple(weights.shape)} for batch {prediction.shape[0]}"
                )
            weighted_loss = per_sample_loss * weights

        voxel_count = prediction_flat.shape[1]
        return TubeIoUResult(
            loss=weighted_loss.mean(),
            iou=iou,
            per_sample_loss=per_sample_loss,
            intersection=intersection,
            union=union,
            prediction_foreground_fraction=prediction_sum / voxel_count,
            target_foreground_fraction=target_sum / voxel_count,
        )


def _batch_sigma(
    sigma: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError(
            f"flow sigma must be scalar or [B], got {tuple(value.shape)}"
        )
    return value.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def flowmatch_clean_estimate(
    noisy_latents: torch.Tensor,
    predicted_velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    first_frame_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert Wan flow velocity once: ``x0_hat = x_s - sigma * v_hat``."""

    if noisy_latents.ndim != 5 or predicted_velocity.ndim != 5:
        raise ValueError("Wan latent tensors must use BCTHW layout")
    sigma_view = _batch_sigma(sigma, noisy_latents)
    if predicted_velocity.shape == noisy_latents.shape:
        estimate = noisy_latents - sigma_view * predicted_velocity
        if first_frame_latents is not None:
            if first_frame_latents.shape != noisy_latents[:, :, :1].shape:
                raise ValueError("first-frame latent shape does not match Wan latent")
            estimate = torch.cat([first_frame_latents, estimate[:, :, 1:]], dim=2)
        return estimate

    expected_tail = noisy_latents[:, :, 1:].shape
    if first_frame_latents is None or predicted_velocity.shape != expected_tail:
        raise ValueError(
            "predicted velocity must match all latents, or the TI2V tail when "
            "first_frame_latents is supplied"
        )
    if first_frame_latents.shape != noisy_latents[:, :, :1].shape:
        raise ValueError("first-frame latent shape does not match Wan latent")
    tail = noisy_latents[:, :, 1:] - sigma_view * predicted_velocity
    return torch.cat([first_frame_latents, tail], dim=2)


def expected_wan_vae_latent_frames(pixel_frames: int) -> int:
    if pixel_frames < 1 or pixel_frames % 4 != 1:
        raise ValueError("Wan pixel frame count must satisfy 4n+1")
    return (pixel_frames - 1) // 4 + 1


def align_mask_tube(
    mask: torch.Tensor,
    *,
    latent_shape: Sequence[int],
) -> torch.Tensor:
    """Nearest-neighbor align a pixel mask tube to an actual Wan latent tube."""

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError("subject mask must use THW or BTHW layout")
    if len(latent_shape) != 4:
        raise ValueError("latent mask shape must use BTHW layout")
    batch, latent_frames, latent_height, latent_width = map(int, latent_shape)
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
    aligned = F.interpolate(
        mask.float().unsqueeze(1),
        size=(latent_frames, latent_height, latent_width),
        mode="nearest",
    )
    return aligned.squeeze(1)


def noise_weight(
    sigma: torch.Tensor,
    *,
    mode: str,
    threshold: float,
) -> torch.Tensor:
    sigma_f = sigma.float()
    if mode == "none":
        return torch.ones_like(sigma_f)
    if mode == "linear_clean":
        return (1.0 - sigma_f).clamp(0.0, 1.0)
    if mode == "threshold":
        if threshold < 0 or threshold > 1:
            raise ValueError("st_noise_threshold must be in [0, 1]")
        return (sigma_f <= threshold).to(dtype=sigma_f.dtype)
    raise ValueError(f"unsupported st_loss_weighting: {mode}")


class LatentOccupancyHead(nn.Module):
    """Learned latent-space proxy for physical-subject occupancy."""

    def __init__(self, latent_channels: int = 48, hidden_channels: int = 32):
        super().__init__()
        if latent_channels < 1 or hidden_channels < 1:
            raise ValueError("occupancy-head channel counts must be positive")
        self.features = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden_channels, 1, 1)

    def forward(self, clean_latents: torch.Tensor) -> torch.Tensor:
        if clean_latents.ndim != 5:
            raise ValueError("latent occupancy head expects BCTHW input")
        return self.output(self.features(clean_latents.float()))


__all__ = [
    "LatentOccupancyHead",
    "SpatioTemporalTubeIoULoss",
    "TubeIoUResult",
    "align_mask_tube",
    "expected_wan_vae_latent_frames",
    "flowmatch_clean_estimate",
    "noise_weight",
]
