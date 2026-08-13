from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import torch

from physbench.baseline_api import discover_baseline_bundles, load_baseline_bundle
from physbench.baselines.wan22_subject_geometry_trust_model import (
    subject_framewise_iou_loss,
    subject_spatial_geometry_loss,
)
from physbench.baselines.wan22_st_tube_iou_model import LatentOccupancyHead
from physbench.baselines.wan22_subject_anchor_trust_model import (
    snapshot_lora_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "wan22_ti2v_5b_lora_r32_subject_geometry_trust_v1"
BASELINE_PATH = ROOT / "baselines/wan22_subject_geometry_trust/baseline.json"
FIXED_TUBE_ID = "wan22_ti2v_5b_lora_r32_fixed_probe_tube_anchor_trust_v1"


class SubjectSpatialGeometryLossTests(unittest.TestCase):
    @staticmethod
    def _two_object_tube(distance: int) -> torch.Tensor:
        tube = torch.zeros((1, 3, 5, 9))
        center = 4
        for frame in range(3):
            tube[0, frame, 2, center - distance] = 1
            tube[0, frame, 2, center + distance] = 1
        return tube

    def test_identical_geometry_has_zero_loss(self) -> None:
        target = self._two_object_tube(distance=3)

        result = subject_spatial_geometry_loss(
            target,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.mass_loss, torch.tensor(0.0))
        torch.testing.assert_close(result.covariance_loss, torch.tensor(0.0))

    def test_same_centroid_but_collapsed_pair_is_penalized(self) -> None:
        target = self._two_object_tube(distance=3)
        collapsed = self._two_object_tube(distance=1)

        result = subject_spatial_geometry_loss(
            collapsed,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.mass_loss, torch.tensor(0.0))
        self.assertGreater(float(result.covariance_loss), 0.0)

    def test_missing_object_is_penalized_by_mass_and_covariance(self) -> None:
        target = self._two_object_tube(distance=3)
        prediction = target.clone()
        prediction[:, :, :, 7] = 0

        result = subject_spatial_geometry_loss(
            prediction,
            target,
            sample_weight=torch.ones(1),
        )

        self.assertGreater(float(result.mass_loss), 0.0)
        self.assertGreater(float(result.covariance_loss), 0.0)

    def test_geometry_loss_backpropagates_to_soft_prediction(self) -> None:
        logits = torch.randn((1, 3, 5, 9), requires_grad=True)

        result = subject_spatial_geometry_loss(
            logits.sigmoid(),
            self._two_object_tube(distance=3),
            sample_weight=torch.ones(1),
        )
        (result.mass_loss + result.covariance_loss).backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


class SubjectFramewiseIoULossTests(unittest.TestCase):
    def test_identical_tube_has_zero_loss(self) -> None:
        target = SubjectSpatialGeometryLossTests._two_object_tube(distance=3)

        result = subject_framewise_iou_loss(
            target,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.loss, torch.tensor(0.0))

    def test_small_and_large_frames_receive_equal_temporal_weight(self) -> None:
        target = torch.zeros((1, 2, 8, 8))
        target[:, 0, 0, 0] = 1
        target[:, 1] = 1
        prediction = target.clone()
        prediction[:, 0] = 0

        result = subject_framewise_iou_loss(
            prediction,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.loss, torch.tensor(0.5))

    def test_framewise_iou_backpropagates_to_soft_prediction(self) -> None:
        logits = torch.randn((1, 3, 5, 9), requires_grad=True)

        result = subject_framewise_iou_loss(
            logits.sigmoid(),
            SubjectSpatialGeometryLossTests._two_object_tube(distance=3),
            sample_weight=torch.ones(1),
        )
        result.loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5])
        self.linear_timesteps_weights = torch.ones(2)


class _FakeDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor(0.25))


class _FakePipe:
    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()
        self.dit = _FakeDiT()
        self.in_iteration_models = ["dit"]
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")

    def model_fn(self, *, dit, latents, timestep, **kwargs):
        del timestep, kwargs
        return latents * dit.lora_A


class SubjectGeometryTrustObjectiveTests(unittest.TestCase):
    def test_all_fixed_probe_terms_backpropagate_only_to_lora(self) -> None:
        module = importlib.import_module(
            "scripts.wan22_subject_geometry_trust_train"
        )
        pipe = _FakePipe()
        reference = snapshot_lora_parameters(pipe.dit.named_parameters())
        pipe.dit.lora_A.data.add_(0.1)
        head = LatentOccupancyHead(latent_channels=1, hidden_channels=2)
        head.requires_grad_(False)
        mask = torch.zeros((1, 9, 3, 5))
        for frame in range(9):
            mask[0, frame, 1, min(4, frame // 2)] = 1
            mask[0, frame, 1, max(0, 4 - frame // 2)] = 1
        config = {
            "enable_st_iou_loss": True,
            "freeze_occupancy_head": True,
            "lambda_st": 0.02,
            "lambda_framewise_iou": 0.01,
            "lambda_subject_flow": 0.0,
            "lambda_motion_delta": 0.0,
            "lambda_anchored_displacement": 0.02,
            "lambda_subject_mass": 0.01,
            "lambda_subject_covariance": 0.02,
            "lambda_lora_trust_region": 0.25,
            "st_iou_eps": 1e-6,
            "subject_flow_eps": 1e-6,
            "motion_delta_eps": 1e-6,
            "trajectory_eps": 1e-6,
            "trajectory_smooth_l1_beta": 0.05,
            "geometry_eps": 1e-6,
            "geometry_smooth_l1_beta": 0.05,
            "lora_trust_region_eps": 1e-12,
            "st_loss_weighting": "linear_clean",
            "st_noise_threshold": 0.5,
            "aux_warmup_steps": 0,
            "latent_channels": 1,
            "hidden_channels": 2,
        }

        metrics = module.compute_subject_geometry_trust_objective(
            pipe=pipe,
            inputs={"input_latents": torch.randn((1, 1, 3, 3, 5))},
            subject_mask=mask,
            occupancy_head=head,
            lora_reference=reference,
            config=config,
            optimizer_step=0,
            timestep_ids=torch.tensor([1]),
            noise=torch.zeros((1, 1, 3, 3, 5)),
        )

        self.assertGreater(float(metrics["st_contribution"].detach()), 0.0)
        self.assertGreater(
            float(metrics["subject_framewise_iou_contribution"].detach()), 0.0
        )
        self.assertGreater(
            float(metrics["anchored_displacement_contribution"].detach()), 0.0
        )
        self.assertGreater(
            float(metrics["subject_mass_contribution"].detach()), 0.0
        )
        self.assertGreater(
            float(metrics["subject_covariance_contribution"].detach()), 0.0
        )
        metrics["total_loss"].backward()
        self.assertIsNotNone(pipe.dit.lora_A.grad)
        self.assertGreater(float(pipe.dit.lora_A.grad.abs()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))


class SubjectGeometryTrustRegistrationTests(unittest.TestCase):
    def test_fixed_probe_validator_accepts_positive_tube_weight(self) -> None:
        module = importlib.import_module(
            "scripts.wan22_subject_geometry_trust_train"
        )
        bundle = load_baseline_bundle(FIXED_TUBE_ID)

        module._validate_geometry_trust_config(
            bundle.value["trainer"]["config"]
        )

    def test_geometry_bundle_registers_second_order_recipe(self) -> None:
        self.assertEqual(BASELINE_PATH, discover_baseline_bundles()[BASELINE_ID])
        bundle = load_baseline_bundle(BASELINE_PATH)
        config = bundle.value["trainer"]["config"]

        self.assertEqual(0.000025, config["learning_rate"])
        self.assertEqual(0.02, config["lambda_st"])
        self.assertEqual(0.01, config["lambda_framewise_iou"])
        self.assertEqual(0.02, config["lambda_anchored_displacement"])
        self.assertEqual(0.01, config["lambda_subject_mass"])
        self.assertEqual(0.10, config["lambda_subject_covariance"])
        self.assertEqual(1.0, config["lambda_lora_trust_region"])
        self.assertTrue(config["freeze_occupancy_head"])


if __name__ == "__main__":
    unittest.main()
