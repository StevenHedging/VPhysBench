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
from physbench.baselines.wan22_st_tube_iou_model import LatentOccupancyHead
from physbench.baselines.wan22_subject_anchor_trust_model import (
    anchored_centroid_displacement_loss,
    relative_lora_drift_loss,
    snapshot_lora_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "wan22_ti2v_5b_lora_r32_subject_anchor_trust_v1"
BASELINE_PATH = ROOT / "baselines/wan22_subject_anchor_trust/baseline.json"
FIXED_TUBE_BASELINE_ID = (
    "wan22_ti2v_5b_lora_r32_fixed_probe_tube_anchor_trust_v1"
)
FIXED_TUBE_BASELINE_PATH = (
    ROOT / "baselines/wan22_fixed_probe_tube_anchor_trust/baseline.json"
)
SPLIT_TASK = ROOT / "tasks/experiments/six_scene_train_five_scene_eval_v14.json"


class AnchoredCentroidDisplacementLossTests(unittest.TestCase):
    @staticmethod
    def _moving_target() -> torch.Tensor:
        target = torch.zeros((1, 4, 3, 7))
        target[0, 0, 1, 0] = 1
        target[0, 1, 1, 2] = 1
        target[0, 2, 1, 4] = 1
        target[0, 3, 1, 6] = 1
        return target

    def test_identical_motion_has_zero_loss(self) -> None:
        target = self._moving_target()

        result = anchored_centroid_displacement_loss(
            target,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.loss, torch.tensor(0.0))
        torch.testing.assert_close(result.valid_frame_fraction, torch.tensor(1.0))

    def test_stationary_prediction_is_penalized_for_missing_motion(self) -> None:
        target = self._moving_target()
        prediction = torch.zeros_like(target)
        prediction[0, :, 1, 0] = 1

        result = anchored_centroid_displacement_loss(
            prediction,
            target,
            sample_weight=torch.ones(1),
        )

        self.assertGreater(float(result.loss), 0.0)

    def test_common_spatial_offset_does_not_change_relative_motion(self) -> None:
        target = torch.zeros((1, 3, 3, 7))
        prediction = torch.zeros_like(target)
        for frame, x in enumerate((0, 1, 2)):
            target[0, frame, 1, x] = 1
            prediction[0, frame, 1, x + 3] = 1

        result = anchored_centroid_displacement_loss(
            prediction,
            target,
            sample_weight=torch.ones(1),
        )

        torch.testing.assert_close(result.loss, torch.tensor(0.0))

    def test_loss_backpropagates_to_prediction(self) -> None:
        logits = torch.randn((1, 4, 3, 7), requires_grad=True)
        result = anchored_centroid_displacement_loss(
            logits.sigmoid(),
            self._moving_target(),
            sample_weight=torch.ones(1),
        )

        result.loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


class RelativeLoraDriftLossTests(unittest.TestCase):
    class _TinyLora(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.lora_B = torch.nn.Parameter(torch.tensor([3.0]))
            self.other = torch.nn.Parameter(torch.tensor([4.0]))

    def test_snapshot_only_contains_lora_parameters(self) -> None:
        model = self._TinyLora()

        reference = snapshot_lora_parameters(model.named_parameters())

        self.assertEqual({"lora_A", "lora_B"}, set(reference))
        self.assertTrue(all(not value.requires_grad for value in reference.values()))

    def test_drift_is_zero_at_snapshot_and_penalizes_lora_only(self) -> None:
        model = self._TinyLora()
        reference = snapshot_lora_parameters(model.named_parameters())
        zero = relative_lora_drift_loss(model.named_parameters(), reference)
        model.other.data.add_(10.0)
        still_zero = relative_lora_drift_loss(model.named_parameters(), reference)
        model.lora_A.data.add_(0.5)
        drift = relative_lora_drift_loss(model.named_parameters(), reference)

        torch.testing.assert_close(zero, torch.tensor(0.0))
        torch.testing.assert_close(still_zero, torch.tensor(0.0))
        self.assertGreater(float(drift.detach()), 0.0)

    def test_drift_backpropagates_to_lora(self) -> None:
        model = self._TinyLora()
        reference = snapshot_lora_parameters(model.named_parameters())
        model.lora_B.data.add_(0.25)

        loss = relative_lora_drift_loss(model.named_parameters(), reference)
        loss.backward()

        self.assertIsNotNone(model.lora_B.grad)
        self.assertGreater(float(model.lora_B.grad.abs().sum()), 0.0)
        self.assertIsNone(model.other.grad)


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


class SubjectAnchorTrustObjectiveTests(unittest.TestCase):
    def test_objective_adds_anchor_and_trust_terms_to_lora_gradient(self) -> None:
        module = importlib.import_module(
            "scripts.wan22_subject_anchor_trust_train"
        )
        pipe = _FakePipe()
        reference = snapshot_lora_parameters(pipe.dit.named_parameters())
        pipe.dit.lora_A.data.add_(0.1)
        head = LatentOccupancyHead(latent_channels=1, hidden_channels=2)
        head.requires_grad_(False)
        mask = torch.zeros((1, 9, 3, 5))
        for frame in range(9):
            mask[0, frame, 1, min(4, frame // 2)] = 1
        config = {
            "enable_st_iou_loss": True,
            "lambda_st": 0.0,
            "lambda_subject_flow": 0.0,
            "lambda_motion_delta": 0.0,
            "lambda_anchored_displacement": 0.05,
            "lambda_lora_trust_region": 0.25,
            "st_iou_eps": 1e-6,
            "subject_flow_eps": 1e-6,
            "motion_delta_eps": 1e-6,
            "trajectory_eps": 1e-6,
            "trajectory_smooth_l1_beta": 0.05,
            "lora_trust_region_eps": 1e-12,
            "st_loss_weighting": "linear_clean",
            "st_noise_threshold": 0.5,
            "aux_warmup_steps": 0,
            "latent_channels": 1,
            "hidden_channels": 2,
        }

        metrics = module.compute_subject_anchor_trust_objective(
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

        self.assertGreater(float(metrics["anchored_displacement_loss"].detach()), 0)
        self.assertGreater(float(metrics["lora_relative_drift"].detach()), 0)
        expected = (
            metrics["base_loss"]
            + metrics["anchored_displacement_contribution"]
            + metrics["lora_trust_region_contribution"]
        )
        torch.testing.assert_close(metrics["total_loss"], expected)
        metrics["total_loss"].backward()
        self.assertIsNotNone(pipe.dit.lora_A.grad)
        self.assertGreater(float(pipe.dit.lora_A.grad.abs()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))


class SubjectAnchorTrustRegistrationTests(unittest.TestCase):
    def test_bundle_registers_fixed_probe_trust_region_recipe(self) -> None:
        self.assertEqual(
            BASELINE_PATH,
            discover_baseline_bundles()[BASELINE_ID],
        )
        bundle = load_baseline_bundle(BASELINE_PATH)
        trainer = bundle.value["trainer"]["config"]

        self.assertEqual(1, trainer["num_epochs"])
        self.assertEqual(234, trainer["save_steps"])
        self.assertEqual(0.00005, trainer["learning_rate"])
        self.assertTrue(trainer["freeze_occupancy_head"])
        self.assertEqual(0.0, trainer["lambda_st"])
        self.assertEqual(0.05, trainer["lambda_anchored_displacement"])
        self.assertEqual(0.25, trainer["lambda_lora_trust_region"])
        self.assertEqual(
            "wan22_ti2v_5b_lora_r32_st_tube_iou_v1",
            bundle.value["model"]["initialization"]["parent_baseline_id"],
        )

    def test_split_task_keeps_training_masks_out_of_inference(self) -> None:
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
        self.assertTrue(all(
            "subject_mask" not in str(job)
            and ".st-head.safetensors" not in str(job)
            for job in value["inference"]["jobs"]
        ))

    def test_driver_fingerprints_anchor_trust_dependencies(self) -> None:
        bundle = load_baseline_bundle(BASELINE_PATH)
        plugin = load_baseline_plugin(bundle)
        paths = plugin.driver.dependency_paths()
        expected = {
            "src/physbench/baselines/wan22_subject_anchor_trust.py",
            "src/physbench/baselines/wan22_subject_anchor_trust_model.py",
            "scripts/wan22_subject_anchor_trust_train.py",
            "scripts/train_wan22_subject_anchor_trust.sh",
        }

        self.assertTrue(expected.issubset(paths))
        self.assertTrue(all(path.is_file() for path in paths.values()))


class FixedProbeTubeAnchorTrustRegistrationTests(unittest.TestCase):
    def test_bundle_restores_absolute_tube_constraint(self) -> None:
        self.assertEqual(
            FIXED_TUBE_BASELINE_PATH,
            discover_baseline_bundles()[FIXED_TUBE_BASELINE_ID],
        )
        bundle = load_baseline_bundle(FIXED_TUBE_BASELINE_PATH)
        trainer = bundle.value["trainer"]["config"]

        self.assertTrue(trainer["freeze_occupancy_head"])
        self.assertEqual(0.02, trainer["lambda_st"])
        self.assertEqual(0.03, trainer["lambda_anchored_displacement"])
        self.assertEqual(0.25, trainer["lambda_lora_trust_region"])
        self.assertEqual(0.0, trainer["lambda_subject_flow"])
        self.assertEqual(0.0, trainer["lambda_motion_delta"])
        self.assertEqual(
            "wan22_ti2v_5b_lora_r32_st_tube_iou_v1",
            bundle.value["model"]["initialization"]["parent_baseline_id"],
        )

    def test_bundle_reuses_audited_fixed_probe_driver(self) -> None:
        bundle = load_baseline_bundle(FIXED_TUBE_BASELINE_PATH)
        driver_path = (
            FIXED_TUBE_BASELINE_PATH.parent
            / bundle.value["implementation"]["driver"]
        )
        driver_source = driver_path.read_text(encoding="utf-8")

        self.assertTrue(driver_path.is_file())
        self.assertIn("Wan22SubjectGeometryTrustManagedDriver", driver_source)


if __name__ == "__main__":
    unittest.main()
