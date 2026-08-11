from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from physbench.baseline_api import discover_baseline_bundles, load_baseline_bundle


MODULE_NAME = "physbench.baselines.wan22_st_tube_iou_model"
MASK_MODULE_NAME = "physbench.baselines.wan22_st_tube_iou_masks"
ADAPTER_MODULE_NAME = "physbench.baselines.wan22_st_tube_iou"
TRAINER_MODULE_NAME = "scripts.wan22_st_tube_iou_train"
ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "wan22_ti2v_5b_lora_r32_st_tube_iou_v1"
BASELINE_PATH = ROOT / "baselines" / "wan22_st_tube_iou" / "baseline.json"


def _model_module():
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {MODULE_NAME} has not been implemented"
        ) from exc


def _mask_module():
    try:
        return importlib.import_module(MASK_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {MASK_MODULE_NAME} has not been implemented"
        ) from exc


def _adapter_module():
    try:
        return importlib.import_module(ADAPTER_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {ADAPTER_MODULE_NAME} has not been implemented"
        ) from exc


def _trainer_module():
    try:
        return importlib.import_module(TRAINER_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {TRAINER_MODULE_NAME} has not been implemented"
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


class _DeterministicTubeSegmenter:
    model_id = "fixture/deterministic-segmenter"

    def segment_instances(self, frames, *, prompts, **kwargs):
        del kwargs
        height, width = frames[0].shape[:2]
        tubes = []
        for instance_index, _prompt in enumerate(prompts):
            masks = []
            for frame_index in range(len(frames)):
                mask = np.zeros((height, width), dtype=np.uint8)
                if instance_index == 0:
                    mask[1, min(width - 1, frame_index + 1)] = 255
                else:
                    mask[height - 2, max(0, width - 2 - frame_index)] = 255
                masks.append(mask)
            tubes.append(masks)
        return tubes, {
            "backend": "fixture",
            "instance_count": len(prompts),
            "frame_count": len(frames),
        }


class _FailIfCalledSegmenter:
    model_id = "fixture/deterministic-segmenter"

    def segment_instances(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("valid cached mask tube should not rerun propagation")


class MaskTubeMaterializerTests(unittest.TestCase):
    @staticmethod
    def _write_video(path: Path, *, frames: int = 5) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            24.0,
            (8, 8),
        )
        if not writer.isOpened():
            raise RuntimeError("fixture video writer failed to open")
        for index in range(frames):
            frame = np.full((8, 8, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    @staticmethod
    def _fixture(root: Path) -> tuple[Path, Path, Path]:
        canonical = root / "assets" / "scene" / "case-a" / "canonical"
        masks = canonical / "masks"
        masks.mkdir(parents=True)
        reference = canonical / "reference.mp4"
        normalized = root / "normalized.mp4"
        MaskTubeMaterializerTests._write_video(reference)
        MaskTubeMaterializerTests._write_video(normalized)
        first = np.zeros((1, 8, 8), dtype=np.uint8)
        first[0, 1:3, 1:3] = 1
        second = np.zeros((1, 8, 8), dtype=np.uint8)
        second[0, 5:7, 5:7] = 1
        np.savez_compressed(
            masks / "01.npz",
            masks=first,
            mask_ids=np.asarray(["01"]),
            object_ids=np.asarray(["object_1"]),
            frame_index=np.asarray(0),
        )
        np.savez_compressed(
            masks / "02.npz",
            masks=second,
            mask_ids=np.asarray(["02"]),
            object_ids=np.asarray(["object_2"]),
            frame_index=np.asarray(0),
        )
        manifest = masks / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.2",
            "case_id": "case-a",
            "frame_scope": "first_frame_only",
            "frame_index": 0,
            "image_shape_hw": [8, 8],
            "instances": [
                {
                    "mask_id": "01",
                    "object_id": "object_1",
                    "npz_asset": "assets/scene/case-a/canonical/masks/01.npz",
                },
                {
                    "mask_id": "02",
                    "object_id": "object_2",
                    "npz_asset": "assets/scene/case-a/canonical/masks/02.npz",
                },
            ],
        }), encoding="utf-8")
        return reference, normalized, manifest

    def test_sibling_manifest_resolution_is_contained_and_case_checked(self) -> None:
        module = _mask_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, _, manifest = self._fixture(root)

            resolved = module.resolve_training_mask_manifest(
                reference,
                dataset_root=root,
                case_id="case-a",
            )

            self.assertEqual(manifest.resolve(), resolved)
            mov_reference = reference.with_suffix(".mov")
            mov_reference.write_bytes(reference.read_bytes())
            self.assertEqual(
                manifest.resolve(),
                module.resolve_training_mask_manifest(
                    mov_reference,
                    dataset_root=root,
                    case_id="case-a",
                ),
            )
            with self.assertRaisesRegex(ValueError, "case_id"):
                module.resolve_training_mask_manifest(
                    reference,
                    dataset_root=root,
                    case_id="different-case",
                )
            outside = root.parent / "outside" / "reference.mp4"
            with self.assertRaisesRegex(ValueError, "Dataset root"):
                module.resolve_training_mask_manifest(
                    outside,
                    dataset_root=root,
                    case_id="case-a",
                )

    def test_contain_transform_uses_nearest_resize_and_zero_padding(self) -> None:
        module = _mask_module()
        source = np.zeros((4, 8), dtype=np.uint8)
        source[1:3, 2:6] = 1

        transformed = module.contain_mask(
            source,
            output_width=8,
            output_height=8,
        )

        self.assertEqual((8, 8), transformed.shape)
        self.assertEqual(0, int(transformed[:2].sum()))
        self.assertEqual(0, int(transformed[6:].sum()))
        self.assertEqual(8, int(transformed.sum()))
        self.assertEqual({0, 1}, set(np.unique(transformed).tolist()))

    def test_materializer_unions_instances_and_reuses_valid_cache(self) -> None:
        module = _mask_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, normalized, manifest = self._fixture(root)
            output = root / "tube.npz"

            audit = module.materialize_subject_mask_tube(
                normalized_video=normalized,
                mask_manifest=manifest,
                dataset_root=root,
                output=output,
                case_id="case-a",
                segmenter=_DeterministicTubeSegmenter(),
            )
            cached = module.materialize_subject_mask_tube(
                normalized_video=normalized,
                mask_manifest=manifest,
                dataset_root=root,
                output=output,
                case_id="case-a",
                segmenter=_FailIfCalledSegmenter(),
            )

            with np.load(output, allow_pickle=False) as payload:
                tube = payload["masks"]
            self.assertEqual((5, 8, 8), tube.shape)
            self.assertEqual(np.uint8, tube.dtype)
            self.assertEqual({0, 1}, set(np.unique(tube).tolist()))
            self.assertTrue(all(int(frame.sum()) == 2 for frame in tube))
            self.assertEqual("materialized", audit["status"])
            self.assertEqual("cached", cached["status"])
            self.assertEqual(2, audit["instance_count"])
            self.assertEqual(5, audit["frame_count"])
            self.assertEqual(audit["tube_sha256"], cached["tube_sha256"])

    def test_materializer_rejects_malformed_nonbinary_seed(self) -> None:
        module = _mask_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, normalized, manifest = self._fixture(root)
            with np.load(manifest.parent / "01.npz", allow_pickle=False) as payload:
                values = {key: payload[key] for key in payload.files}
            values["masks"] = values["masks"].copy()
            values["masks"][0, 0, 0] = 2
            np.savez_compressed(manifest.parent / "01.npz", **values)

            with self.assertRaisesRegex(ValueError, "binary"):
                module.materialize_subject_mask_tube(
                    normalized_video=normalized,
                    mask_manifest=manifest,
                    dataset_root=root,
                    output=root / "tube.npz",
                    case_id="case-a",
                    segmenter=_DeterministicTubeSegmenter(),
                )


class STTubeIoUBaselineRegistrationTests(unittest.TestCase):
    @staticmethod
    def _adapter_config() -> dict:
        return {
            "schema_version": "1.0",
            "baseline_id": BASELINE_ID,
            "supported_scenes": [
                "pendulum",
                "collision_1d",
                "inclined_plane_slide",
                "uniform_circular_motion",
                "parabolic_motion",
                "push_bottle",
                "vertical_spring_oscillator",
            ],
            "runtime": {
                "project_root": "/runtime",
                "python": "/runtime/python",
                "model_base": "/runtime/models",
                "cuda_visible_devices": "0,1,2,3",
                "accelerate_config": "/runtime/accelerate.yaml",
            },
            "media_adapter": {
                "width": 480,
                "height": 832,
                "fps": 24,
                "max_frames": 121,
                "min_frames": 5,
                "pad_mode": "edge",
            },
            "lora": {
                "micro_batch_size": 1,
                "global_batch_size": 8,
                "dataset_repeat": 1,
                "num_epochs": 8,
                "scene_balancing": "oversample_each_scene_to_largest_world_aligned",
                "seed": 42,
            },
            "st_tube_iou": {
                "enable_st_iou_loss": True,
                "lambda_st": 0.1,
                "st_iou_eps": 1e-6,
                "st_loss_weighting": "linear_clean",
                "st_noise_threshold": 0.5,
                "st_loss_warmup_steps": 100,
                "latent_channels": 16,
                "hidden_channels": 32,
                "mask_segmenter_model_id": "facebook/sam2.1-hiera-tiny",
            },
        }

    def test_bundle_is_discovered_with_complete_auxiliary_recipe(self) -> None:
        discovered = discover_baseline_bundles()
        self.assertIn(BASELINE_ID, discovered)
        self.assertEqual(BASELINE_PATH, discovered[BASELINE_ID])

        bundle = load_baseline_bundle(BASELINE_PATH)
        self.assertEqual("finetune_eval", bundle.value["capabilities"]["task_families"][0])
        self.assertEqual("ignored", bundle.value["input_policy"]["physics"]["usage"])
        self.assertEqual(7, len(bundle.value["supported_scenes"]))
        config = bundle.value["trainer"]["config"]
        expected = {
            "enable_st_iou_loss": True,
            "lambda_st": 0.1,
            "st_iou_eps": 1e-6,
            "st_loss_weighting": "linear_clean",
            "st_noise_threshold": 0.5,
            "st_loss_warmup_steps": 100,
        }
        self.assertEqual(expected, {
            key: config[key]
            for key in expected
        })
        self.assertEqual(1, config["micro_batch_size"])
        self.assertEqual(8, config["global_batch_size"])

    def test_four_gpu_parallelism_preserves_global_batch(self) -> None:
        adapter = _adapter_module().Wan22STTubeIoULoraAdapter(
            self._adapter_config(),
            execute=False,
        )

        contract = adapter._training_parallelism(metadata_row_count=56)

        self.assertEqual(4, contract["world_size"])
        self.assertEqual(2, contract["gradient_accumulation_steps"])
        self.assertEqual(7, contract["optimizer_steps_per_epoch"])
        self.assertEqual(56, contract["total_optimizer_steps"])

    def test_training_metadata_binds_tube_but_generation_stays_stock(self) -> None:
        adapter = _adapter_module().Wan22STTubeIoULoraAdapter(
            self._adapter_config(),
            execute=False,
        )
        row = adapter._training_metadata_row(
            case={"case_id": "case-a", "scene_id": "pendulum"},
            adaptation={
                "text_transform_id": "identity",
                "native_inputs": {"text": {"prompt": "a pendulum swings"}},
            },
            video="videos/case-a.mp4",
            subject_mask="masks/case-a.npz",
        )

        self.assertEqual("masks/case-a.npz", row["subject_mask"])
        self.assertEqual("a pendulum swings", row["prompt"])
        self.assertEqual(ROOT / "scripts" / "wan22_generate.py", adapter._generation_script())


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

    @staticmethod
    def model_fn(*, dit, latents, timestep, **kwargs):
        del timestep, kwargs
        return latents * dit.lora_A


class _FailIfAuxiliaryRuns(torch.nn.Module):
    def forward(self, value):
        del value
        raise AssertionError("disabled ST loss must bypass the occupancy head")


def _trainer_config(**overrides) -> dict:
    config = {
        "enable_st_iou_loss": True,
        "lambda_st": 1.0,
        "st_iou_eps": 1e-6,
        "st_loss_weighting": "none",
        "st_noise_threshold": 0.5,
        "st_loss_warmup_steps": 0,
        "base_loss_scale": 1.0,
    }
    config.update(overrides)
    return config


class STTubeIoUTrainerContractTests(unittest.TestCase):
    @staticmethod
    def _head() -> torch.nn.Module:
        head = torch.nn.Conv3d(1, 1, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(1.0)
        return head

    def test_effective_lambda_warms_up_by_completed_optimizer_step(self) -> None:
        module = _trainer_module()
        self.assertEqual(0.1, module.effective_st_lambda(0.5, 0, 5))
        self.assertEqual(0.3, module.effective_st_lambda(0.5, 2, 5))
        self.assertEqual(0.5, module.effective_st_lambda(0.5, 99, 5))
        self.assertEqual(0.5, module.effective_st_lambda(0.5, 0, 0))

    def test_disabled_auxiliary_is_exact_base_loss_with_per_sample_timesteps(self) -> None:
        module = _trainer_module()
        pipe = _FakePipe()
        clean = torch.tensor([
            [[[[1.0]], [[2.0]]]],
            [[[[2.0]], [[4.0]]]],
        ])
        mask = torch.ones(2, 5, 1, 1)
        noise = torch.zeros_like(clean)

        result = module.compute_flowmatch_st_objective(
            pipe=pipe,
            inputs={"input_latents": clean},
            subject_mask=mask,
            occupancy_head=_FailIfAuxiliaryRuns(),
            config=_trainer_config(enable_st_iou_loss=False),
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
        self.assertEqual(0.0, float(result["lambda_effective"]))

    def test_st_only_loss_reaches_lora_and_occupancy_head_gradients(self) -> None:
        module = _trainer_module()
        pipe = _FakePipe()
        head = self._head()
        clean = torch.tensor([[[[[1.0]], [[3.0]]]]])
        mask = torch.zeros(1, 5, 1, 1)
        mask[:, 4] = 1.0

        result = module.compute_flowmatch_st_objective(
            pipe=pipe,
            inputs={"input_latents": clean},
            subject_mask=mask,
            occupancy_head=head,
            config=_trainer_config(base_loss_scale=0.0),
            optimizer_step=0,
            timestep_ids=torch.tensor([1]),
            noise=torch.zeros_like(clean),
        )
        result["total_loss"].backward()

        self.assertGreater(abs(float(pipe.dit.lora_A.grad)), 0.0)
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)
        self.assertEqual((1,), tuple(result["st_iou_per_sample"].shape))
        self.assertTrue(bool(torch.isfinite(result["total_loss"])))

    def test_lora_export_and_head_path_cannot_pollute_stock_checkpoint(self) -> None:
        module = _trainer_module()
        state = {
            "pipe.dit.block.lora_A.default.weight": torch.ones(2, 2),
            "pipe.dit.block.lora_B.default.weight": torch.ones(2, 2),
            "occupancy_head.features.0.weight": torch.ones(1),
            "occupancy_head.output.weight": torch.ones(1),
        }

        exported = module.lora_only_state_dict(state, remove_prefix="pipe.dit.")
        paired = module.st_head_checkpoint_path(Path("step-9.safetensors"))

        self.assertEqual({
            "block.lora_A.default.weight",
            "block.lora_B.default.weight",
        }, set(exported))
        self.assertEqual(Path("step-9.st-head.safetensors"), paired)


if __name__ == "__main__":
    unittest.main()
