from __future__ import annotations

import importlib
import unittest

import torch


MODULE_NAME = "physbench.baselines.wan22_st_tube_iou_model"


def _model_module():
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {MODULE_NAME} has not been implemented"
        ) from exc


class SpatioTemporalTubeIoULossTests(unittest.TestCase):
    def _result(self, prediction: torch.Tensor, target: torch.Tensor):
        module = _model_module()
        return module.SpatioTemporalTubeIoULoss(eps=1e-6)(
            prediction,
            target,
        )

    def test_identical_and_both_empty_tubes_have_zero_loss(self) -> None:
        occupied = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
        )
        empty = torch.zeros_like(occupied)

        result = self._result(
            torch.cat([occupied, empty]),
            torch.cat([occupied, empty]),
        )

        torch.testing.assert_close(result.iou, torch.tensor([1.0, 1.0]))
        self.assertEqual(0.0, float(result.loss))

    def test_disjoint_and_one_empty_tubes_have_near_unit_loss(self) -> None:
        target = torch.zeros(3, 2, 2, 2)
        prediction = torch.zeros_like(target)
        target[0, 0, 0, 0] = 1.0
        prediction[0, 1, 1, 1] = 1.0
        target[1, 0, 0, 0] = 1.0
        prediction[2, 0, 0, 0] = 1.0

        result = self._result(prediction, target)

        expected = torch.tensor([
            1e-6 / (2.0 + 1e-6),
            1e-6 / (1.0 + 1e-6),
            1e-6 / (1.0 + 1e-6),
        ])
        torch.testing.assert_close(result.iou, expected)
        self.assertGreater(float(result.loss), 0.999998)

    def test_partial_and_oversized_tubes_use_one_full_volume_iou(self) -> None:
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        partial = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        oversized = torch.ones_like(target)

        result = self._result(
            torch.cat([partial, oversized]),
            torch.cat([target, target]),
        )

        expected = torch.tensor([
            (1.0 + 1e-6) / (3.0 + 1e-6),
            (2.0 + 1e-6) / (4.0 + 1e-6),
        ])
        torch.testing.assert_close(result.iou, expected)
        self.assertAlmostEqual(float(1.0 - expected.mean()), float(result.loss))

    def test_temporal_failures_are_part_of_the_same_tube_union(self) -> None:
        target = torch.zeros(4, 4, 1, 1)
        prediction = torch.zeros_like(target)
        target[:, :, 0, 0] = 1.0
        prediction[0, :2, 0, 0] = 1.0  # disappearance
        prediction[1, 2:, 0, 0] = 1.0  # delayed appearance
        prediction[2, 1:, 0, 0] = 1.0  # one-frame shift / missing first
        prediction[3, :, 0, 0] = 1.0  # exact

        result = self._result(prediction, target)

        expected = torch.tensor([
            (2.0 + 1e-6) / (4.0 + 1e-6),
            (2.0 + 1e-6) / (4.0 + 1e-6),
            (3.0 + 1e-6) / (4.0 + 1e-6),
            1.0,
        ])
        torch.testing.assert_close(result.iou, expected)

    def test_sample_weights_are_applied_before_batch_mean(self) -> None:
        target = torch.ones(2, 1, 1, 1)
        prediction = torch.tensor([[[[1.0]]], [[[0.0]]]])

        result = _model_module().SpatioTemporalTubeIoULoss(eps=1e-6)(
            prediction,
            target,
            sample_weight=torch.tensor([0.0, 0.25]),
        )

        expected_second = 1.0 - 1e-6 / (1.0 + 1e-6)
        self.assertAlmostEqual(0.25 * expected_second / 2.0, float(result.loss))


class FlowMatchCleanEstimateTests(unittest.TestCase):
    def test_exact_velocity_recovers_clean_latents(self) -> None:
        module = _model_module()
        clean = torch.tensor([1.0, -2.0]).reshape(2, 1, 1, 1, 1)
        noise = torch.tensor([5.0, 3.0]).reshape_as(clean)
        sigma = torch.tensor([0.25, 0.75])
        noisy = (1.0 - sigma.reshape(2, 1, 1, 1, 1)) * clean
        noisy = noisy + sigma.reshape(2, 1, 1, 1, 1) * noise

        estimate = module.flowmatch_clean_estimate(
            noisy,
            noise - clean,
            sigma,
        )

        torch.testing.assert_close(estimate, clean)

    def test_known_ti2v_first_latent_replaces_model_estimate(self) -> None:
        module = _model_module()
        noisy = torch.full((1, 2, 3, 1, 1), 5.0)
        velocity = torch.full_like(noisy, 2.0)
        known = torch.tensor([10.0, 20.0]).reshape(1, 2, 1, 1, 1)

        estimate = module.flowmatch_clean_estimate(
            noisy,
            velocity,
            torch.tensor([0.5]),
            first_frame_latents=known,
        )

        torch.testing.assert_close(estimate[:, :, :1], known)
        torch.testing.assert_close(
            estimate[:, :, 1:],
            torch.full((1, 2, 2, 1, 1), 4.0),
        )

    def test_alignment_matches_wan_temporal_compression_and_stays_binary(self) -> None:
        module = _model_module()
        for pixel_frames, latent_frames in ((5, 2), (9, 3), (121, 31)):
            mask = torch.zeros(1, pixel_frames, 8, 8)
            mask[:, :, 2:6, 2:6] = 1.0
            aligned = module.align_mask_tube(
                mask,
                latent_shape=(1, latent_frames, 2, 2),
            )
            self.assertEqual((1, latent_frames, 2, 2), tuple(aligned.shape))
            self.assertEqual({0.0, 1.0}, set(aligned.unique().tolist()))

        with self.assertRaisesRegex(ValueError, "temporal compression"):
            module.align_mask_tube(
                torch.zeros(1, 9, 8, 8),
                latent_shape=(1, 4, 2, 2),
            )

    def test_noise_weight_modes_follow_wan_sigma_direction(self) -> None:
        module = _model_module()
        sigma = torch.tensor([0.0, 0.25, 0.75, 1.0])
        torch.testing.assert_close(
            module.noise_weight(sigma, mode="none", threshold=0.5),
            torch.ones(4),
        )
        torch.testing.assert_close(
            module.noise_weight(sigma, mode="linear_clean", threshold=0.5),
            torch.tensor([1.0, 0.75, 0.25, 0.0]),
        )
        torch.testing.assert_close(
            module.noise_weight(sigma, mode="threshold", threshold=0.5),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )

    def test_tube_loss_backpropagates_through_head_and_clean_estimate(self) -> None:
        module = _model_module()
        torch.manual_seed(7)
        noisy = torch.randn(2, 16, 3, 4, 4, requires_grad=True)
        velocity = torch.randn_like(noisy, requires_grad=True)
        head = module.LatentOccupancyHead(latent_channels=16, hidden_channels=8)
        target = torch.zeros(2, 9, 32, 32)
        target[:, :, 8:24, 8:24] = 1.0

        clean = module.flowmatch_clean_estimate(
            noisy,
            velocity,
            torch.tensor([0.2, 0.4]),
        )
        prediction = head(clean).sigmoid().squeeze(1)
        aligned = module.align_mask_tube(
            target,
            latent_shape=tuple(prediction.shape),
        )
        result = module.SpatioTemporalTubeIoULoss()(prediction, aligned)
        result.loss.backward()

        self.assertGreater(float(velocity.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
