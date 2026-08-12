from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SoftCentroidTrajectory:
    centroids: torch.Tensor
    mass: torch.Tensor


@dataclass(frozen=True)
class CentroidTrajectoryLossResult:
    position_loss: torch.Tensor
    velocity_loss: torch.Tensor
    per_sample_position_loss: torch.Tensor
    per_sample_velocity_loss: torch.Tensor
    valid_frame_fraction: torch.Tensor
    valid_velocity_fraction: torch.Tensor


def soft_mask_centroid_trajectory(
    probability: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> SoftCentroidTrajectory:
    """Return differentiable BTx(y,x) centroids in the [-1, 1] canvas."""

    if eps <= 0:
        raise ValueError("soft-centroid epsilon must be positive")
    if probability.ndim != 4:
        raise ValueError("soft centroid expects a BTHW probability tube")
    value = probability.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("soft-centroid probability contains non-finite values")
    if bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("soft-centroid probability must be in [0, 1]")

    height, width = value.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=value.device, dtype=value.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=value.device, dtype=value.dtype)
    mass = value.sum(dim=(-2, -1))
    denominator = mass + float(eps)
    centroid_y = (value * y.reshape(1, 1, height, 1)).sum(
        dim=(-2, -1)
    ) / denominator
    centroid_x = (value * x.reshape(1, 1, 1, width)).sum(
        dim=(-2, -1)
    ) / denominator
    return SoftCentroidTrajectory(
        centroids=torch.stack((centroid_y, centroid_x), dim=-1),
        mass=mass,
    )


def subject_centroid_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    smooth_l1_beta: float = 0.05,
) -> CentroidTrajectoryLossResult:
    """Match subject position and one-step velocity independent of canvas size."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("trajectory prediction and target must share BTHW shape")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("trajectory sample weights must have shape [B]")
    if eps <= 0 or smooth_l1_beta <= 0:
        raise ValueError("trajectory epsilon and Smooth-L1 beta must be positive")
    weights = sample_weight.to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("trajectory sample weights must be finite and non-negative")

    predicted = soft_mask_centroid_trajectory(prediction, eps=eps)
    expected = soft_mask_centroid_trajectory(target, eps=eps)
    valid_frame = expected.mass > float(eps)
    frame_error = F.smooth_l1_loss(
        predicted.centroids,
        expected.centroids,
        reduction="none",
        beta=float(smooth_l1_beta),
    ).mean(dim=-1)
    valid_f = valid_frame.float()
    per_sample_position = (frame_error * valid_f).sum(dim=1) / valid_f.sum(
        dim=1
    ).clamp_min(1.0)

    predicted_velocity = predicted.centroids[:, 1:] - predicted.centroids[:, :-1]
    expected_velocity = expected.centroids[:, 1:] - expected.centroids[:, :-1]
    valid_velocity = valid_frame[:, 1:] & valid_frame[:, :-1]
    velocity_error = F.smooth_l1_loss(
        predicted_velocity,
        expected_velocity,
        reduction="none",
        beta=float(smooth_l1_beta),
    ).mean(dim=-1)
    valid_velocity_f = valid_velocity.float()
    per_sample_velocity = (
        (velocity_error * valid_velocity_f).sum(dim=1)
        / valid_velocity_f.sum(dim=1).clamp_min(1.0)
    )

    return CentroidTrajectoryLossResult(
        position_loss=(per_sample_position * weights).mean(),
        velocity_loss=(per_sample_velocity * weights).mean(),
        per_sample_position_loss=per_sample_position,
        per_sample_velocity_loss=per_sample_velocity,
        valid_frame_fraction=valid_f.mean(),
        valid_velocity_fraction=(
            valid_velocity_f.mean()
            if valid_velocity_f.numel()
            else prediction.new_zeros((), dtype=torch.float32)
        ),
    )


__all__ = [
    "CentroidTrajectoryLossResult",
    "SoftCentroidTrajectory",
    "soft_mask_centroid_trajectory",
    "subject_centroid_trajectory_loss",
]
