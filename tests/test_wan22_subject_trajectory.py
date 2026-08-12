from __future__ import annotations

import unittest

import torch

from physbench.baselines.wan22_subject_trajectory_model import (
    soft_mask_centroid_trajectory,
    subject_centroid_trajectory_loss,
)


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


if __name__ == "__main__":
    unittest.main()
