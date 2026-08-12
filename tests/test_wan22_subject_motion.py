from __future__ import annotations

import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.io import load_json
from physbench.tasks import load_task
from physbench.baselines.wan22_subject_motion_model import (
    align_subject_support_tube,
    masked_subject_flow_loss,
    subject_temporal_difference_loss,
)


TRAINER_MODULE_NAME = "scripts.wan22_subject_motion_train"
ADAPTER_MODULE_NAME = "physbench.baselines.wan22_subject_motion"
ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "wan22_ti2v_5b_lora_r32_subject_motion_v1"
BASELINE_ROOT = ROOT / "baselines" / "wan22_subject_motion"
BASELINE_PATH = BASELINE_ROOT / "baseline.json"
SPLIT_TASK = ROOT / "tasks/experiments/six_scene_train_five_scene_eval_v14.json"


def _trainer_module():
    try:
        return importlib.import_module(TRAINER_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {TRAINER_MODULE_NAME} has not been implemented"
        ) from exc


def _adapter_module():
    try:
        return importlib.import_module(ADAPTER_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"production module {ADAPTER_MODULE_NAME} has not been implemented"
        ) from exc


class SubjectFlowLossTests(unittest.TestCase):
    def test_area_preserving_alignment_keeps_a_subpixel_subject(self) -> None:
        mask = torch.zeros((1, 9, 32, 16), dtype=torch.float32)
        mask[0, 4, 1, 1] = 1.0

        aligned = align_subject_support_tube(
            mask,
            latent_shape=(1, 3, 2, 1),
        )

        self.assertEqual((1, 3, 2, 1), tuple(aligned.shape))
        self.assertEqual(1.0, float(aligned.max()))
        self.assertGreater(float(aligned.sum()), 0.0)
        self.assertEqual({0.0, 1.0}, set(aligned.unique().tolist()))

    def test_area_preserving_alignment_checks_wan_temporal_compression(self) -> None:
        with self.assertRaisesRegex(ValueError, "temporal compression"):
            align_subject_support_tube(
                torch.ones((1, 9, 32, 16)),
                latent_shape=(1, 4, 2, 1),
            )

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


def _adapter_config(root: Path) -> dict:
    return {
        "schema_version": "1.0",
        "baseline_id": "wan22_ti2v_5b_lora_r32_subject_motion_v1",
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
            "project_root": str(root / "runtime"),
            "python": str(root / "python"),
            "model_base": str(root / "models"),
            "cuda_visible_devices": "0,1,2,3",
            "accelerate_config": str(root / "accelerate.yaml"),
            "subject_tube_cache_dir": str(root / "cache"),
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
            "num_epochs": 2,
            "scene_balancing": "oversample_each_scene_to_largest_world_aligned",
            "seed": 42,
            "save_optimizer_state": True,
        },
        "st_tube_iou": {
            "enable_st_iou_loss": True,
            "lambda_st": 0.05,
            "lambda_subject_flow": 0.10,
            "lambda_motion_delta": 0.05,
            "st_iou_eps": 1e-6,
            "subject_flow_eps": 1e-6,
            "motion_delta_eps": 1e-6,
            "st_loss_weighting": "linear_clean",
            "st_noise_threshold": 0.5,
            "aux_warmup_steps": 100,
            "latent_channels": 48,
            "hidden_channels": 32,
            "mask_segmenter_model_id": "facebook/sam2.1-hiera-tiny",
        },
    }


def _write_cache_pair(
    root: Path,
    *,
    case_id: str = "case-a",
    stored_case_id: str | None = None,
    dtype=np.uint8,
) -> tuple[Path, Path]:
    from physbench.io import sha256_file

    root.mkdir(parents=True, exist_ok=True)
    tube_path = root / f"{case_id}.npz"
    audit_path = root / f"{case_id}.audit.json"
    masks = np.ones((5, 4, 4), dtype=dtype)
    np.savez_compressed(
        tube_path,
        masks=masks,
        layout=np.asarray("THW"),
        case_id=np.asarray(stored_case_id or case_id),
    )
    audit = {
        "schema_version": "1.0",
        "status": "materialized",
        "case_id": stored_case_id or case_id,
        "shape_thw": [5, 4, 4],
        "dtype": str(masks.dtype),
        "values": sorted(np.unique(masks).tolist()),
        "frame_count": 5,
        "foreground_fraction": float(masks.mean()),
        "source_fingerprint": {
            "case_id": stored_case_id or case_id,
            "normalized_video_sha256": "a" * 64,
        },
        "tube_sha256": sha256_file(tube_path),
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return tube_path, audit_path


class SubjectMotionAdapterTests(unittest.TestCase):
    def test_valid_cache_pair_is_seeded_and_missing_case_falls_back(self) -> None:
        module = _adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _adapter_config(root)
            source, source_audit = _write_cache_pair(root / "cache")
            adapter = module.Wan22SubjectMotionAdapter(config, execute=True)

            result = adapter._seed_subject_tube_cache(
                ["case-a", "case-missing"],
                root / "target",
            )

            target = root / "target" / source.name
            target_audit = root / "target" / source_audit.name
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(source_audit.read_bytes(), target_audit.read_bytes())
            self.assertEqual(["case-a"], result["seeded_case_ids"])
            self.assertEqual(["case-missing"], result["fallback_case_ids"])

    def test_cache_validation_rejects_case_hash_and_binary_contract_errors(self) -> None:
        module = _adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _adapter_config(root)
            adapter = module.Wan22SubjectMotionAdapter(config, execute=True)

            with self.subTest("case"):
                tube, audit = _write_cache_pair(
                    root / "case-mismatch",
                    stored_case_id="other-case",
                )
                with self.assertRaisesRegex(ValueError, "case_id"):
                    adapter._validate_cached_tube("case-a", tube, audit)

            with self.subTest("hash"):
                tube, audit = _write_cache_pair(root / "hash-mismatch")
                value = json.loads(audit.read_text())
                value["tube_sha256"] = "0" * 64
                audit.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash"):
                    adapter._validate_cached_tube("case-a", tube, audit)

            with self.subTest("dtype"):
                tube, audit = _write_cache_pair(
                    root / "dtype-mismatch",
                    dtype=np.float32,
                )
                with self.assertRaisesRegex(ValueError, "uint8"):
                    adapter._validate_cached_tube("case-a", tube, audit)

    def test_warm_start_requires_standard_lora_and_paired_head(self) -> None:
        module = _adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _adapter_config(root)
            lora = root / "step-2184.safetensors"
            config["initial_lora_checkpoint"] = str(lora)
            adapter = module.Wan22SubjectMotionAdapter(config, execute=True)

            with self.assertRaisesRegex(FileNotFoundError, "LoRA"):
                adapter._resolve_warm_start_checkpoint()
            lora.write_bytes(b"lora")
            with self.assertRaisesRegex(FileNotFoundError, "paired"):
                adapter._resolve_warm_start_checkpoint()
            head = root / "step-2184.st-head.safetensors"
            head.write_bytes(b"head")
            self.assertEqual(lora.resolve(), adapter._resolve_warm_start_checkpoint())

            bad = _adapter_config(root)
            bad["initial_lora_checkpoint"] = str(head)
            with self.assertRaisesRegex(ValueError, "ST-head"):
                module.Wan22SubjectMotionAdapter(
                    bad, execute=True
                )._resolve_warm_start_checkpoint()

    def test_launcher_is_bound_to_four_gpus_and_subject_motion_entrypoint(self) -> None:
        module = _adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _adapter_config(root)
            lora = root / "step-2184.safetensors"
            lora.write_bytes(b"lora")
            lora.with_name("step-2184.st-head.safetensors").write_bytes(b"head")
            config["initial_lora_checkpoint"] = str(lora)
            (root / "accelerate.yaml").write_text("compute_environment: LOCAL_MACHINE\n")
            adapter = module.Wan22SubjectMotionAdapter(config, execute=True)
            run_dir = root / "run"
            dataset_dir = root / "dataset"
            metadata = dataset_dir / "metadata.jsonl"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("", encoding="utf-8")

            environment = adapter._training_environment(
                run_dir,
                dataset_dir,
                metadata,
            )
            command = adapter._training_command(root / "runtime")

            self.assertEqual("0,1,2,3", environment["CUDA_VISIBLE_DEVICES"])
            self.assertEqual(str(lora.resolve()), environment["LORA_CHECKPOINT"])
            self.assertIn("SUBJECT_MOTION_CONFIG_JSON", environment)
            self.assertNotIn("ST_TUBE_IOU_CONFIG_JSON", environment)
            self.assertEqual(
                Path("scripts/train_wan22_subject_motion.sh"),
                Path(command[1]).relative_to(adapter.project_root),
            )

    def test_six_scene_balancing_aligns_to_the_global_batch(self) -> None:
        module = _adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = module.Wan22SubjectMotionAdapter(
                _adapter_config(root),
                execute=False,
            )
            rows = []
            for scene_index in range(6):
                for case_index in range(5 - (scene_index % 3)):
                    rows.append({
                        "case_id": f"case-{scene_index}-{case_index}",
                        "scene_id": f"scene-{scene_index}",
                    })

            balanced = adapter._balance_training_rows(rows, root)
            plan = load_json(root / "training_sampling_plan.json")

            self.assertEqual(48, len(balanced))
            self.assertEqual(8, plan["per_scene_target"])
            self.assertEqual(6, plan["optimizer_steps_per_epoch"])
            self.assertEqual(12, plan["total_optimizer_steps"])


class SubjectMotionRegistrationTests(unittest.TestCase):
    def test_bundle_is_discoverable_and_freezes_the_recipe(self) -> None:
        discovered = discover_baseline_bundles()
        self.assertEqual(BASELINE_PATH, discovered[BASELINE_ID])
        bundle = load_baseline_bundle(BASELINE_PATH)

        self.assertEqual("1.0.0", bundle.value["baseline_version"])
        self.assertEqual(7, len(bundle.value["supported_scenes"]))
        self.assertEqual(["finetune_eval"], bundle.value["capabilities"]["task_families"])
        self.assertEqual(["i2v"], bundle.value["capabilities"]["generation_modes"])
        self.assertEqual("ignored", bundle.value["input_policy"]["physics"]["usage"])
        trainer = bundle.value["trainer"]["config"]
        expected = {
            "lambda_st": 0.05,
            "lambda_subject_flow": 0.10,
            "lambda_motion_delta": 0.05,
            "aux_warmup_steps": 100,
            "num_epochs": 2,
            "save_steps": 234,
            "rank": 32,
            "micro_batch_size": 1,
            "global_batch_size": 8,
            "latent_channels": 48,
        }
        self.assertEqual(expected, {key: trainer[key] for key in expected})
        initialization = bundle.value["model"]["initialization"]
        self.assertEqual(
            "wan22_ti2v_5b_lora_r32_st_tube_iou_v1",
            initialization["parent_baseline_id"],
        )
        self.assertEqual(64, len(initialization["dit_lora_sha256"]))
        self.assertEqual(64, len(initialization["latent_occupancy_head_sha256"]))

    def test_local_deployment_changes_only_deployment_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bundle"
            target.mkdir()
            for name in ("baseline.json", "driver.py", "__init__.py"):
                shutil.copy2(BASELINE_ROOT / name, target / name)
            (target / "baseline.local.json").write_text(
                json.dumps({"runtime": {"python": "/first/python"}}),
                encoding="utf-8",
            )
            first = load_baseline_bundle(target / "baseline.json")
            (target / "baseline.local.json").write_text(
                json.dumps({"runtime": {"python": "/second/python"}}),
                encoding="utf-8",
            )
            second = load_baseline_bundle(target / "baseline.json")

        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.deployment_digest, second.deployment_digest)

    def test_managed_instance_uses_split_task_and_mask_free_stock_inference(self) -> None:
        bundle = load_baseline_bundle(BASELINE_PATH)
        plugin = load_baseline_plugin(bundle)
        dataset = load_dataset(LATEST_DATASET, check_assets=False)
        task = load_task(SPLIT_TASK)

        instance = plugin.task_builder.build(dataset, task)
        instance.verify()
        value = instance.value

        self.assertEqual(679, len(value["canonical_plan"]["train_case_ids"]))
        self.assertEqual(76, len(value["canonical_plan"]["jobs"]))
        self.assertEqual(
            0.10,
            value["training"]["trainer"]["config"]["lambda_subject_flow"],
        )
        self.assertTrue(all(
            "subject_mask" not in json.dumps(job)
            and "reference_video" not in json.dumps(job)
            and ".st-head.safetensors" not in json.dumps(job)
            for job in value["inference"]["jobs"]
        ))

    def test_managed_driver_fingerprints_all_subject_motion_dependencies(self) -> None:
        bundle = load_baseline_bundle(BASELINE_PATH)
        plugin = load_baseline_plugin(bundle)
        paths = plugin.driver.dependency_paths()

        expected = {
            "src/physbench/baselines/wan22_subject_motion.py",
            "src/physbench/baselines/wan22_subject_motion_model.py",
            "scripts/wan22_subject_motion_train.py",
            "scripts/train_wan22_subject_motion.sh",
        }
        self.assertTrue(expected.issubset(paths))
        self.assertTrue(all(path.is_file() for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
