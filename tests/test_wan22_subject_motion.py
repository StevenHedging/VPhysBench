from __future__ import annotations

import importlib
import unittest

import torch

from physbench.baselines.wan22_subject_motion_model import (
    masked_subject_flow_loss,
    subject_temporal_difference_loss,
)


TRAINER_MODULE_NAME = "scripts.wan22_subject_motion_train"


def _trainer_module():
    try:
        return importlib.import_module(TRAINER_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {TRAINER_MODULE_NAME} has not been implemented"
        ) from exc


class SubjectFlowLossTests(unittest.TestCase):
    def test_background_is_ignored_and_mask_area_is_normalized_per_sample(self) -> None:
        prediction = torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32)
        target = torch.zeros_like(prediction)
        mask = torch.zeros((2, 2, 2, 2), dtype=torch.float32)

        mask[0, 0, 0, 0] = 1
        prediction[0, 0, 0, 0, 0] = 2
        prediction[0, 0, 1, 1, 1] = 100

        mask[1, 0] = 1
        prediction[1, 0, 0] = 2

        result = masked_subject_flow_loss(
            prediction,
            target,
            mask,
            sample_weight=torch.ones(2),
        )

        torch.testing.assert_close(result.per_sample_loss, torch.tensor([4.0, 4.0]))
        torch.testing.assert_close(result.loss, torch.tensor(4.0))
        torch.testing.assert_close(
            result.support_fraction,
            torch.tensor([1.0 / 8.0, 4.0 / 8.0]),
        )

    def test_scheduler_weights_are_applied_before_batch_mean(self) -> None:
        prediction = torch.tensor([2.0, 3.0]).reshape(2, 1, 1, 1, 1)
        target = torch.zeros_like(prediction)
        mask = torch.ones((2, 1, 1, 1))

        result = masked_subject_flow_loss(
            prediction,
            target,
            mask,
            sample_weight=torch.tensor([0.25, 0.5]),
        )

        torch.testing.assert_close(result.per_sample_loss, torch.tensor([4.0, 9.0]))
        torch.testing.assert_close(result.loss, torch.tensor(2.75))

    def test_empty_aligned_subject_is_rejected(self) -> None:
        prediction = torch.zeros((1, 2, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "empty"):
            masked_subject_flow_loss(
                prediction,
                torch.zeros_like(prediction),
                torch.zeros((1, 2, 2, 2)),
                sample_weight=torch.ones(1),
            )


class SubjectTemporalDifferenceLossTests(unittest.TestCase):
    def test_identical_temporal_changes_have_only_charbonnier_floor(self) -> None:
        clean = torch.randn((2, 3, 3, 2, 2))
        result = subject_temporal_difference_loss(
            clean,
            clean.clone(),
            torch.ones((2, 3, 2, 2)),
            sample_weight=torch.ones(2),
        )

        torch.testing.assert_close(
            result.per_sample_loss,
            torch.full((2,), 0.001),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_error_outside_endpoint_union_support_is_ignored(self) -> None:
        estimate = torch.zeros((1, 1, 3, 1, 2))
        target = torch.zeros_like(estimate)
        estimate[0, 0, 1, 0, 1] = 100
        mask = torch.zeros((1, 3, 1, 2))
        mask[:, :, :, 0] = 1

        result = subject_temporal_difference_loss(
            estimate,
            target,
            mask,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(
            result.loss,
            torch.tensor(0.001),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_subject_temporal_change_error_is_penalized(self) -> None:
        estimate = torch.zeros((1, 1, 2, 1, 1))
        target = torch.zeros_like(estimate)
        estimate[:, :, 1] = 2

        result = subject_temporal_difference_loss(
            estimate,
            target,
            torch.ones((1, 2, 1, 1)),
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(
            result.loss,
            torch.tensor((4.0 + 1e-6) ** 0.5),
        )

    def test_noise_weights_are_applied_before_batch_mean(self) -> None:
        estimate = torch.zeros((2, 1, 2, 1, 1))
        target = torch.zeros_like(estimate)
        estimate[0, :, 1] = 1
        estimate[1, :, 1] = 2

        result = subject_temporal_difference_loss(
            estimate,
            target,
            torch.ones((2, 2, 1, 1)),
            sample_weight=torch.tensor([0.25, 0.5]),
        )

        expected = (
            0.25 * (1.0 + 1e-6) ** 0.5
            + 0.5 * (4.0 + 1e-6) ** 0.5
        ) / 2
        torch.testing.assert_close(result.loss, torch.tensor(expected))

    def test_loss_backpropagates_to_clean_estimate(self) -> None:
        estimate = torch.randn((1, 2, 3, 2, 2), requires_grad=True)
        result = subject_temporal_difference_loss(
            estimate,
            torch.zeros_like(estimate),
            torch.ones((1, 3, 2, 2)),
            sample_weight=torch.ones(1),
        )
        result.loss.backward()

        self.assertIsNotNone(estimate.grad)
        self.assertGreater(float(estimate.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(estimate.grad).all()))

    def test_single_latent_frame_is_rejected(self) -> None:
        clean = torch.zeros((1, 2, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, "two latent frames"):
            subject_temporal_difference_loss(
                clean,
                clean.clone(),
                torch.ones((1, 1, 2, 2)),
                sample_weight=torch.ones(1),
            )


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5])
        self.linear_timesteps_weights = torch.tensor([2.0, 3.0])


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
        self.last_model_latents = None

    def model_fn(self, *, dit, latents, timestep, **kwargs):
        del timestep, kwargs
        self.last_model_latents = latents.detach().clone()
        return latents * dit.lora_A


class _FailIfAuxiliaryRuns(torch.nn.Module):
    def forward(self, value):
        del value
        raise AssertionError("zero auxiliary coefficients must bypass the head")


def _objective_config(**overrides) -> dict:
    config = {
        "enable_st_iou_loss": True,
        "lambda_st": 0.2,
        "lambda_subject_flow": 0.3,
        "lambda_motion_delta": 0.4,
        "st_iou_eps": 1e-6,
        "subject_flow_eps": 1e-6,
        "motion_delta_eps": 1e-6,
        "st_loss_weighting": "none",
        "st_noise_threshold": 0.5,
        "aux_warmup_steps": 2,
        "base_loss_scale": 1.0,
    }
    config.update(overrides)
    return config


class SubjectMotionObjectiveTests(unittest.TestCase):
    @staticmethod
    def _head() -> torch.nn.Module:
        head = torch.nn.Conv3d(1, 1, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(1.0)
        return head

    def test_zero_auxiliaries_equal_stock_loss_and_use_per_sample_timesteps(self) -> None:
        module = _trainer_module()
        pipe = _FakePipe()
        clean = torch.tensor([
            [[[[1.0]], [[2.0]]]],
            [[[[2.0]], [[4.0]]]],
        ])
        noise = torch.zeros_like(clean)

        result = module.compute_subject_motion_objective(
            pipe=pipe,
            inputs={"input_latents": clean},
            subject_mask=torch.ones((2, 5, 1, 1)),
            occupancy_head=_FailIfAuxiliaryRuns(),
            config=_objective_config(
                enable_st_iou_loss=False,
                lambda_st=0.0,
                lambda_subject_flow=0.0,
                lambda_motion_delta=0.0,
            ),
            optimizer_step=0,
            timestep_ids=torch.tensor([0, 1]),
            noise=noise,
        )

        sigma = torch.tensor([1.0, 0.5]).reshape(2, 1, 1, 1, 1)
        noisy = (1.0 - sigma) * clean
        prediction = noisy * 0.25
        target = -clean
        mse = (prediction - target).square().flatten(1).mean(1)
        expected_base = (mse * torch.tensor([2.0, 3.0])).mean()
        torch.testing.assert_close(result["base_loss"], expected_base)
        torch.testing.assert_close(result["total_loss"], expected_base)
        torch.testing.assert_close(result["noise_sigma"], torch.tensor([1.0, 0.5]))
        self.assertEqual(0.0, float(result["aux_warmup_ratio"]))

    def test_combined_objective_injects_condition_and_weights_each_term(self) -> None:
        module = _trainer_module()
        pipe = _FakePipe()
        clean = torch.tensor([[[[[1.0]], [[3.0]]]]])
        first_frame = torch.tensor([[[[[7.0]]]]])
        head = self._head()

        result = module.compute_subject_motion_objective(
            pipe=pipe,
            inputs={
                "input_latents": clean,
                "first_frame_latents": first_frame,
            },
            subject_mask=torch.ones((1, 5, 1, 1)),
            occupancy_head=head,
            config=_objective_config(),
            optimizer_step=0,
            timestep_ids=torch.tensor([1]),
            noise=torch.zeros_like(clean),
        )

        self.assertIsNotNone(pipe.last_model_latents)
        torch.testing.assert_close(pipe.last_model_latents[:, :, :1], first_frame)
        self.assertAlmostEqual(0.5, float(result["aux_warmup_ratio"]))
        self.assertAlmostEqual(0.1, float(result["lambda_st_effective"]))
        self.assertAlmostEqual(0.15, float(result["lambda_subject_effective"]))
        self.assertAlmostEqual(0.2, float(result["lambda_motion_effective"]))
        expected = (
            result["base_loss"]
            + 0.1 * result["st_loss"]
            + 0.15 * result["subject_flow_loss"]
            + 0.2 * result["motion_delta_loss"]
        )
        torch.testing.assert_close(result["total_loss"], expected)
        torch.testing.assert_close(
            result["subject_flow_contribution"],
            0.15 * result["subject_flow_loss"],
        )
        torch.testing.assert_close(
            result["motion_delta_contribution"],
            0.2 * result["motion_delta_loss"],
        )

    def test_auxiliary_only_objective_reaches_lora_and_head_gradients(self) -> None:
        module = _trainer_module()
        pipe = _FakePipe()
        head = self._head()
        clean = torch.tensor([[[[[1.0]], [[3.0]]]]])

        result = module.compute_subject_motion_objective(
            pipe=pipe,
            inputs={
                "input_latents": clean,
                "first_frame_latents": clean[:, :, :1],
            },
            subject_mask=torch.ones((1, 5, 1, 1)),
            occupancy_head=head,
            config=_objective_config(base_loss_scale=0.0, aux_warmup_steps=0),
            optimizer_step=0,
            timestep_ids=torch.tensor([1]),
            noise=torch.zeros_like(clean),
        )
        result["total_loss"].backward()

        self.assertGreater(abs(float(pipe.dit.lora_A.grad)), 0.0)
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(result["subject_support_fraction"]), 0.0)
        self.assertGreater(float(result["motion_support_fraction"]), 0.0)
        self.assertTrue(bool(torch.isfinite(result["total_loss"])))


if __name__ == "__main__":
    unittest.main()
