from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .wan22_subject_trajectory_model import soft_mask_centroid_trajectory


@dataclass(frozen=True)
class SubjectSpatialGeometryLossResult:
    mass_loss: torch.Tensor
    covariance_loss: torch.Tensor
    per_sample_mass_loss: torch.Tensor
    per_sample_covariance_loss: torch.Tensor
    valid_frame_fraction: torch.Tensor


@dataclass(frozen=True)
class SubjectFramewiseIoULossResult:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    per_frame_iou: torch.Tensor
    target_valid_frame_fraction: torch.Tensor


def subject_framewise_iou_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> SubjectFramewiseIoULossResult:
    """Average soft subject IoU over frames instead of occupied voxels."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("framewise IoU tensors must share BTHW shape")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("framewise IoU sample weights must have shape [B]")
    if eps <= 0:
        raise ValueError("framewise IoU epsilon must be positive")
    prediction_f = prediction.float()
    target_f = target.float()
    weights = sample_weight.to(device=prediction.device, dtype=torch.float32)
    for name, value in (
        ("prediction", prediction_f),
        ("target", target_f),
        ("weights", weights),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"framewise IoU {name} contains non-finite values")
    if bool((prediction_f < 0).any()) or bool((prediction_f > 1).any()):
        raise ValueError("framewise IoU prediction must be in [0, 1]")
    if bool((target_f < 0).any()) or bool((target_f > 1).any()):
        raise ValueError("framewise IoU target must be in [0, 1]")
    if bool((weights < 0).any()):
        raise ValueError("framewise IoU weights must be non-negative")

    intersection = (prediction_f * target_f).sum(dim=(-2, -1))
    prediction_mass = prediction_f.sum(dim=(-2, -1))
    target_mass = target_f.sum(dim=(-2, -1))
    if bool((target_mass.sum(dim=1) <= 0).any()):
        raise ValueError("framewise IoU target contains an empty sample")
    union = prediction_mass + target_mass - intersection
    raw_iou = (intersection + float(eps)) / (union + float(eps))
    per_frame_iou = torch.where(union == 0, torch.ones_like(raw_iou), raw_iou)
    per_sample = (1.0 - per_frame_iou).mean(dim=1)
    return SubjectFramewiseIoULossResult(
        loss=(per_sample * weights).mean(),
        per_sample_loss=per_sample,
        per_frame_iou=per_frame_iou,
        target_valid_frame_fraction=(target_mass > float(eps)).float().mean(),
    )


def _soft_spatial_geometry(
    probability: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return frame mass fraction, centroid, and normalized 2-D covariance."""

    trajectory = soft_mask_centroid_trajectory(probability, eps=eps)
    value = probability.float()
    height, width = value.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=value.device, dtype=value.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=value.device, dtype=value.dtype)
    dy = y.reshape(1, 1, height, 1) - trajectory.centroids[..., 0].reshape(
        *value.shape[:2], 1, 1
    )
    dx = x.reshape(1, 1, 1, width) - trajectory.centroids[..., 1].reshape(
        *value.shape[:2], 1, 1
    )
    denominator = trajectory.mass + float(eps)
    covariance = torch.stack(
        (
            (value * dy.square()).sum(dim=(-2, -1)) / denominator,
            (value * dx.square()).sum(dim=(-2, -1)) / denominator,
            (value * dy * dx).sum(dim=(-2, -1)) / denominator,
        ),
        dim=-1,
    )
    mass_fraction = trajectory.mass / float(height * width)
    return mass_fraction, trajectory.centroids, covariance


def subject_spatial_geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    smooth_l1_beta: float = 0.05,
) -> SubjectSpatialGeometryLossResult:
    """Match subject mass and second-order spatial extent in every frame.

    The covariance term is translation invariant and therefore distinguishes a
    separated object pair from a collapsed pair even when both share a centroid.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("subject geometry tensors must share BTHW shape")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("subject geometry sample weights must have shape [B]")
    if eps <= 0 or smooth_l1_beta <= 0:
        raise ValueError("subject geometry epsilon and beta must be positive")
    weights = sample_weight.to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("subject geometry weights must be finite and non-negative")

    predicted_mass, _predicted_centroid, predicted_covariance = (
        _soft_spatial_geometry(prediction, eps=eps)
    )
    expected_mass, _expected_centroid, expected_covariance = (
        _soft_spatial_geometry(target, eps=eps)
    )
    valid = expected_mass > float(eps)
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("subject geometry target contains an empty sample")
    mass_error = F.smooth_l1_loss(
        (predicted_mass + float(eps)).log(),
        (expected_mass + float(eps)).log(),
        reduction="none",
        beta=float(smooth_l1_beta),
    )
    covariance_error = F.smooth_l1_loss(
        predicted_covariance,
        expected_covariance,
        reduction="none",
        beta=float(smooth_l1_beta),
    ).mean(dim=-1)
    valid_f = valid.float()
    denominator = valid_f.sum(dim=1).clamp_min(1.0)
    per_sample_mass = (mass_error * valid_f).sum(dim=1) / denominator
    per_sample_covariance = (
        (covariance_error * valid_f).sum(dim=1) / denominator
    )
    return SubjectSpatialGeometryLossResult(
        mass_loss=(per_sample_mass * weights).mean(),
        covariance_loss=(per_sample_covariance * weights).mean(),
        per_sample_mass_loss=per_sample_mass,
        per_sample_covariance_loss=per_sample_covariance,
        valid_frame_fraction=valid_f.mean(),
    )


__all__ = [
    "SubjectFramewiseIoULossResult",
    "SubjectSpatialGeometryLossResult",
    "subject_framewise_iou_loss",
    "subject_spatial_geometry_loss",
]
