from __future__ import annotations

import unittest
from pathlib import Path
import importlib

import torch

from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task
from physbench.baselines.wan22_subject_trajectory_model import (
    soft_mask_centroid_trajectory,
    subject_centroid_trajectory_loss,
)
from physbench.baselines.wan22_st_tube_iou_model import LatentOccupancyHead


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "wan22_ti2v_5b_lora_r32_subject_trajectory_v1"
BASELINE_ROOT = ROOT / "baselines" / "wan22_subject_trajectory"
BASELINE_PATH = BASELINE_ROOT / "baseline.json"
SPLIT_TASK = ROOT / "tasks/experiments/six_scene_train_five_scene_eval_v14.json"


class SoftMaskCentroidTrajectoryTests(unittest.TestCase):
    def test_centroids_use_canvas_normalized_coordinates(self) -> None:
        mask = torch.zeros((1, 2, 3, 5))
        mask[0, 0, 0, 0] = 1
        mask[0, 1, 2, 4] = 1

        result = soft_mask_centroid_trajectory(mask)

        torch.testing.assert_close(
            result.centroids,
            torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]]),
        )
        torch.testing.assert_close(result.mass, torch.ones((1, 2)))

    def test_soft_centroid_is_differentiable(self) -> None:
        probability = torch.rand((1, 3, 4, 5), requires_grad=True)

        result = soft_mask_centroid_trajectory(probability)
        result.centroids.square().sum().backward()

        self.assertIsNotNone(probability.grad)
        self.assertGreater(float(probability.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(probability.grad).all()))


class SubjectCentroidTrajectoryLossTests(unittest.TestCase):
    @staticmethod
    def _moving_target() -> torch.Tensor:
        target = torch.zeros((1, 3, 3, 5))
        target[0, 0, 1, 0] = 1
        target[0, 1, 1, 2] = 1
        target[0, 2, 1, 4] = 1
        return target

    def test_identical_tubes_have_zero_position_and_velocity_loss(self) -> None:
        target = self._moving_target()

        result = subject_centroid_trajectory_loss(
            target,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.position_loss, torch.tensor(0.0))
        torch.testing.assert_close(result.velocity_loss, torch.tensor(0.0))
        torch.testing.assert_close(result.valid_frame_fraction, torch.tensor(1.0))

    def test_stationary_prediction_is_penalized_for_missing_motion(self) -> None:
        target = self._moving_target()
        prediction = torch.zeros_like(target)
        prediction[0, :, 1, 0] = 1

        result = subject_centroid_trajectory_loss(
            prediction,
            target,
            sample_weight=torch.ones(1),
        )

        self.assertGreater(float(result.position_loss), 0.0)
        self.assertGreater(float(result.velocity_loss), 0.0)

    def test_empty_target_frames_do_not_change_the_loss(self) -> None:
        target = self._moving_target()
        target[:, 1] = 0
        prediction = target.clone()
        prediction[:, 1] = 1

        result = subject_centroid_trajectory_loss(
            prediction,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.position_loss, torch.tensor(0.0))
        torch.testing.assert_close(result.velocity_loss, torch.tensor(0.0))
        torch.testing.assert_close(
            result.valid_frame_fraction,
            torch.tensor(2.0 / 3.0),
        )

    def test_single_frame_has_zero_velocity_loss(self) -> None:
        target = self._moving_target()[:, :1]
        result = subject_centroid_trajectory_loss(
            target,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.velocity_loss, torch.tensor(0.0))
        torch.testing.assert_close(
            result.valid_velocity_fraction,
            torch.tensor(0.0),
        )

    def test_sample_weights_are_applied_after_per_sample_normalization(self) -> None:
        target = self._moving_target().repeat(2, 1, 1, 1)
        prediction = target.clone()
        prediction[:, :, :, :] = 0
        prediction[:, :, 1, 4] = 1

        full = subject_centroid_trajectory_loss(
            prediction,
            target,
            sample_weight=torch.ones(2),
        )
        weighted = subject_centroid_trajectory_loss(
            prediction,
            target,
            sample_weight=torch.tensor([0.0, 0.5]),
        )

        torch.testing.assert_close(
            weighted.position_loss,
            full.per_sample_position_loss[1] * 0.25,
        )
        torch.testing.assert_close(
            weighted.velocity_loss,
            full.per_sample_velocity_loss[1] * 0.25,
        )

    def test_loss_backpropagates_to_soft_prediction(self) -> None:
        logits = torch.randn((1, 3, 3, 5), requires_grad=True)
        prediction = logits.sigmoid()
        result = subject_centroid_trajectory_loss(
            prediction,
            self._moving_target(),
            sample_weight=torch.ones(1),
        )

        (result.position_loss + result.velocity_loss).backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5])
        self.linear_timesteps_weights = torch.ones(2)


class _FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor(0.25))


class _FakePipe:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.dit = _FakeDiT()
        self.in_iteration_models = ["dit"]
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")

    def model_fn(self, *, dit, latents, timestep, **kwargs):
        del timestep, kwargs
        return latents * dit.lora_A


class SubjectTrajectoryObjectiveTests(unittest.TestCase):
    def test_objective_adds_centroid_terms_and_backpropagates_to_lora(self) -> None:
        module = importlib.import_module("scripts.wan22_subject_trajectory_train")
        pipe = _FakePipe()
        head = LatentOccupancyHead(latent_channels=1, hidden_channels=2)
        mask = torch.zeros((1, 9, 3, 5))
        for frame in range(9):
            mask[0, frame, 1, min(4, frame // 2)] = 1
        config = {
            "enable_st_iou_loss": True,
            "lambda_st": 0.03,
            "lambda_subject_flow": 0.0,
            "lambda_motion_delta": 0.0,
            "lambda_centroid_position": 0.05,
            "lambda_centroid_velocity": 0.025,
            "st_iou_eps": 1e-6,
            "subject_flow_eps": 1e-6,
            "motion_delta_eps": 1e-6,
            "trajectory_eps": 1e-6,
            "trajectory_smooth_l1_beta": 0.05,
            "st_loss_weighting": "linear_clean",
            "st_noise_threshold": 0.5,
            "aux_warmup_steps": 0,
            "latent_channels": 1,
            "hidden_channels": 2,
        }

        metrics = module.compute_subject_trajectory_objective(
            pipe=pipe,
            inputs={"input_latents": torch.randn((1, 1, 3, 3, 5))},
            subject_mask=mask,
            occupancy_head=head,
            config=config,
            optimizer_step=0,
            timestep_ids=torch.tensor([1]),
            noise=torch.zeros((1, 1, 3, 3, 5)),
        )

        self.assertGreater(
            float(metrics["centroid_position_loss"].detach()),
            0.0,
        )
        self.assertGreaterEqual(
            float(metrics["centroid_velocity_loss"].detach()),
            0.0,
        )
        expected = (
            metrics["base_loss"]
            + metrics["st_contribution"]
            + metrics["centroid_position_contribution"]
            + metrics["centroid_velocity_contribution"]
        )
        torch.testing.assert_close(metrics["total_loss"], expected)
        metrics["total_loss"].backward()
        self.assertIsNotNone(pipe.dit.lora_A.grad)
        self.assertGreater(float(pipe.dit.lora_A.grad.abs()), 0.0)


class SubjectTrajectoryRegistrationTests(unittest.TestCase):
    def test_bundle_registers_low_drift_centroid_recipe(self) -> None:
        self.assertEqual(
            BASELINE_PATH,
            discover_baseline_bundles()[BASELINE_ID],
        )
        bundle = load_baseline_bundle(BASELINE_PATH)
        trainer = bundle.value["trainer"]["config"]

        self.assertEqual(1, trainer["num_epochs"])
        self.assertEqual(234, trainer["save_steps"])
        self.assertEqual(0.03, trainer["lambda_st"])
        self.assertEqual(0.0, trainer["lambda_subject_flow"])
        self.assertEqual(0.0, trainer["lambda_motion_delta"])
        self.assertEqual(0.05, trainer["lambda_centroid_position"])
        self.assertEqual(0.025, trainer["lambda_centroid_velocity"])
        self.assertEqual(50, trainer["aux_warmup_steps"])
        self.assertEqual(
            "wan22_ti2v_5b_lora_r32_st_tube_iou_v1",
            bundle.value["model"]["initialization"]["parent_baseline_id"],
        )

    def test_split_task_compiles_to_one_balanced_epoch_and_stock_inference(self) -> None:
        bundle = load_baseline_bundle(BASELINE_PATH)
        plugin = load_baseline_plugin(bundle)
        instance = plugin.task_builder.build(
            load_dataset(LATEST_DATASET, check_assets=False),
            load_task(SPLIT_TASK),
        )
        instance.verify()
        value = instance.value

        self.assertEqual(679, len(value["canonical_plan"]["train_case_ids"]))
        self.assertEqual(76, len(value["canonical_plan"]["jobs"]))
        self.assertEqual(1, value["training"]["trainer"]["config"]["num_epochs"])
        self.assertTrue(all(
            "subject_mask" not in str(job)
            and ".st-head.safetensors" not in str(job)
            for job in value["inference"]["jobs"]
        ))

    def test_driver_fingerprints_trajectory_specific_dependencies(self) -> None:
        bundle = load_baseline_bundle(BASELINE_PATH)
        plugin = load_baseline_plugin(bundle)
        paths = plugin.driver.dependency_paths()
        expected = {
            "src/physbench/baselines/wan22_subject_trajectory.py",
            "src/physbench/baselines/wan22_subject_trajectory_model.py",
            "scripts/wan22_subject_trajectory_train.py",
            "scripts/train_wan22_subject_trajectory.sh",
        }

        self.assertTrue(expected.issubset(paths))
        self.assertTrue(all(path.is_file() for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
