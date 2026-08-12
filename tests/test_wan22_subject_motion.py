from __future__ import annotations

import unittest

import torch

from physbench.baselines.wan22_subject_motion_model import (
    masked_subject_flow_loss,
    subject_temporal_difference_loss,
)


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


if __name__ == "__main__":
    unittest.main()
