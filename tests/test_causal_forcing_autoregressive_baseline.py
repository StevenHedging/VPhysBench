from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from _paths import ROOT
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
)
from physbench.baseline_runtime.adapter_loader import load_data_adapter
from physbench.baseline_runtime import build_i2v_media_contract
from physbench.baseline_runtime.compiler import ManagedTaskBuilder
from physbench.data_layout import V10_DATASET
from physbench.datasets import load_dataset
from physbench.io import load_json
from physbench.tasks import load_task


BASELINE_ROOT = ROOT / "baselines" / "causal_forcing_pp_2step_i2v"
BASELINE = BASELINE_ROOT / "baseline.json"
PHYSICS_BASELINE = BASELINE_ROOT / "physics.baseline.json"
PHYSICS_TEMPLATE = (
    ROOT
    / "src"
    / "physbench"
    / "baseline_plugins"
    / "resources"
    / "six_scene_physics_clauses_v1.json"
)
DIRECT_TASK = ROOT / "tasks" / "official" / "five_scene_direct_eval.json"
SCENES = {
    "pendulum",
    "free_fall",
    "collision_1d",
    "inclined_plane_slide",
    "parabolic_motion",
    "uniform_circular_motion",
}
CURRENT_DATASET_SCENES = SCENES - {"free_fall"}
SHAPES = {
    "pendulum": (480, 832),
    "free_fall": (480, 832),
    "collision_1d": (832, 480),
    "inclined_plane_slide": (832, 480),
    "parabolic_motion": (480, 832),
    "uniform_circular_motion": (480, 832),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CausalForcingAutoregressiveBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_baseline_bundle(BASELINE)
        cls.manifest = load_json(BASELINE)
        cls.physics_bundle = load_baseline_bundle(PHYSICS_BASELINE)
        cls.physics_manifest = load_json(PHYSICS_BASELINE)
        cls.dataset = load_dataset(V10_DATASET, check_assets=False)
        cls.task = load_task(DIRECT_TASK)
        cls.adapter = load_data_adapter(cls.bundle)
        cls.physics_adapter = load_data_adapter(cls.physics_bundle)

    def test_manifest_is_a_direct_only_generic_i2v_baseline(self) -> None:
        self.assertEqual("5.0", self.manifest["schema_version"])
        self.assertEqual(
            "causal_forcing_pp_2step_i2v_generic",
            self.manifest["baseline_id"],
        )
        self.assertEqual(
            ["direct_eval"],
            self.manifest["capabilities"]["task_families"],
        )
        self.assertEqual(
            ["i2v"], self.manifest["capabilities"]["generation_modes"]
        )
        self.assertEqual(
            "ignored", self.manifest["input_policy"]["physics"]["usage"]
        )
        self.assertEqual(SCENES, set(self.manifest["supported_scenes"]))
        self.assertEqual(
            ["config/*.yaml"],
            self.manifest["implementation"]["fingerprint_paths"],
        )

    def test_physics_manifest_is_a_structured_text_direct_i2v_variant(
        self,
    ) -> None:
        self.assertEqual("5.0", self.physics_manifest["schema_version"])
        self.assertEqual(
            "causal_forcing_pp_2step_i2v_physics",
            self.physics_manifest["baseline_id"],
        )
        self.assertEqual(
            {
                "source": "case.physics[annotated=true]",
                "usage": "required",
                "representations": ["structured_text"],
            },
            self.physics_manifest["input_policy"]["physics"],
        )
        self.assertEqual(
            {
                "type": "append_structured_text_v1",
                "template_set": "six_scene_physics_clauses_v1",
            },
            self.physics_manifest["adapter"]["physics_transform"],
        )
        self.assertEqual(
            SCENES, set(self.physics_manifest["supported_scenes"])
        )
        for field in (
            "baseline_version",
            "implementation",
            "supported_scenes",
            "capabilities",
            "model",
            "runtime",
            "runner",
        ):
            self.assertEqual(
                self.manifest[field],
                self.physics_manifest[field],
                field,
            )
        for field in (
            "kind",
            "preset",
            "first_frame_policy",
            "spatial",
            "temporal",
        ):
            self.assertEqual(
                self.manifest["adapter"][field],
                self.physics_manifest["adapter"][field],
                field,
            )

    def test_both_causal_forcing_identities_are_discoverable(self) -> None:
        discovered = discover_baseline_bundles()
        self.assertEqual(
            BASELINE.resolve(),
            discovered["causal_forcing_pp_2step_i2v_generic"],
        )
        self.assertEqual(
            PHYSICS_BASELINE.resolve(),
            discovered["causal_forcing_pp_2step_i2v_physics"],
        )

    def test_checkpoint_and_source_identities_are_frozen(self) -> None:
        model = self.manifest["model"]
        runtime = self.manifest["runtime"]
        self.assertEqual(
            "2f8eb8bb6eeb1238da9d13e5420d342a74d634a6",
            model["snapshot_revision"],
        )
        self.assertRegex(model["checkpoint_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(model["base_identity_files"])
        self.assertEqual(
            {
                "diffusion_pytorch_model.safetensors": 5676070424,
                "models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
                "Wan2.1_VAE.pth": 507609880,
            },
            model["base_required_files"],
        )
        self.assertEqual(
            "1fc7bbc19a503c1bce80ecef08158b20e702f386",
            runtime["causal_forcing_commit"],
        )

    def test_all_scenes_have_81_frame_native_i2v_adaptations(self) -> None:
        by_scene = {}
        for case in self.dataset.cases:
            if case["scene_id"] not in CURRENT_DATASET_SCENES:
                continue
            by_scene.setdefault(case["scene_id"], case)
        self.assertEqual(CURRENT_DATASET_SCENES, set(by_scene))
        for scene_id, case in by_scene.items():
            adaptation = self.adapter.adapt_case(case, role="eval")
            native = adaptation["native_inputs"]
            self.assertEqual(
                case["assets"]["first_frame"],
                native["vision"]["first_frame_asset"],
            )
            self.assertEqual(case["text"]["prompt"], native["text"]["prompt"])
            self.assertEqual({}, adaptation["used_parameters"])
            shape = native["generation_shape"]
            self.assertEqual(81, shape["num_frames"])
            self.assertEqual(16, shape["fps"])
            self.assertEqual(
                SHAPES[scene_id],
                (shape["width"], shape["height"]),
            )
            self.assertEqual(
                ["first_frame"],
                adaptation["input_contract"]["asset_access"],
            )
            self.assertEqual([], adaptation["input_contract"]["physics_channels"])

    def test_physics_adapter_fuses_audited_text_for_all_current_cases(
        self,
    ) -> None:
        seen_scenes = set()
        self.assertEqual(
            self.adapter.materialization_fingerprint,
            self.physics_adapter.materialization_fingerprint,
        )
        self.assertNotEqual(
            self.adapter.fingerprint,
            self.physics_adapter.fingerprint,
        )
        for case in self.dataset.cases:
            if case["scene_id"] not in CURRENT_DATASET_SCENES:
                continue
            seen_scenes.add(case["scene_id"])
            generic = self.adapter.adapt_case(case, role="eval")
            physics = self.physics_adapter.adapt_case(case, role="eval")
            self.assertTrue(
                physics["prompt"].startswith(
                    case["text"]["prompt"] + " Physical parameters"
                ),
                case["case_id"],
            )
            self.assertNotEqual(generic["prompt"], physics["prompt"])
            self.assertEqual(
                generic["native_inputs"]["vision"],
                physics["native_inputs"]["vision"],
            )
            self.assertEqual(
                generic["native_inputs"]["generation_shape"],
                physics["native_inputs"]["generation_shape"],
            )
            self.assertTrue(physics["used_parameters"], case["case_id"])
            for name, rendered in physics["used_parameters"].items():
                quantity = case["physics"][name]
                self.assertIs(quantity["annotated"], True)
                self.assertEqual(quantity["value"], rendered["value"])
                self.assertEqual(quantity["unit"], rendered["unit"])
            channels = physics["input_contract"]["physics_channels"]
            self.assertEqual(1, len(channels), case["case_id"])
            self.assertEqual(
                sorted(physics["used_parameters"]),
                channels[0]["used_parameters"],
            )
            self.assertEqual(
                physics["prompt"],
                physics["native_inputs"]["text"]["prompt"],
            )
        self.assertEqual(CURRENT_DATASET_SCENES, seen_scenes)

    def test_six_scene_template_preserves_multi_incident_collision_physics(
        self,
    ) -> None:
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "collision_1d"
            and item["physics"]["ball_1_initial_velocity"]["value"] != 0
            and item["physics"]["ball_2_initial_velocity"]["value"] != 0
        )
        adaptation = self.physics_adapter.adapt_case(case, role="eval")
        used = adaptation["used_parameters"]
        self.assertIn("ball_1_initial_velocity", used)
        self.assertIn("ball_2_initial_velocity", used)
        self.assertNotIn("striker_initial_velocity", used)
        self.assertIn("numbered from left to right", adaptation["prompt"])
        self.assertIn("rightward velocity is positive", adaptation["prompt"])

    def test_parabolic_physics_prompt_uses_all_four_conditionable_values(
        self,
    ) -> None:
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "parabolic_motion"
        )
        adaptation = self.physics_adapter.adapt_case(case, role="eval")
        self.assertEqual(
            {
                "initial_horizontal_velocity",
                "launch_height",
                "ball_radius",
                "ball_mass",
            },
            set(adaptation["used_parameters"]),
        )
        self.assertIn("initial horizontal speed", adaptation["prompt"])
        self.assertNotIn("photogate", adaptation["prompt"].lower())

    def test_inclined_plane_prompt_uses_all_conditionable_physics(self) -> None:
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "inclined_plane_slide"
        )
        adaptation = self.physics_adapter.adapt_case(case, role="eval")
        self.assertEqual(
            {
                "incline_angle",
                "block_length",
                "block_mass",
                "calibration_length",
                "gravity_acceleration",
                "kinetic_friction_coefficient",
                "friction_force",
                "theoretical_acceleration",
            },
            set(adaptation["used_parameters"]),
        )
        self.assertNotIn("initial_velocity", adaptation["used_parameters"])

    def test_physics_template_is_fingerprinted_and_covers_six_scenes(
        self,
    ) -> None:
        template = load_json(PHYSICS_TEMPLATE)
        self.assertEqual(
            "six_scene_physics_clauses_v1",
            template["template_set_id"],
        )
        self.assertEqual(SCENES, set(template["scenes"]))
        dependencies = self.physics_adapter.dependency_paths()
        self.assertIn(
            "src/physbench/baseline_plugins/resources/"
            "six_scene_physics_clauses_v1.json",
            dependencies,
        )

    def test_current_direct_task_compiles_exactly_658_jobs(self) -> None:
        variants = (
            (self.bundle, self.adapter),
            (self.physics_bundle, self.physics_adapter),
        )
        job_identities = []
        for bundle, adapter in variants:
            with self.subTest(baseline_id=bundle.baseline_id):
                builder = ManagedTaskBuilder(bundle, {}, adapter)
                instance = builder.build(self.dataset, self.task)
                instance.verify()
                self.assertEqual(
                    658, len(instance.value["inference"]["jobs"])
                )
                self.assertEqual(658, len(instance.value["adaptations"]))
                for job in instance.value["inference"]["jobs"]:
                    native = job["native_inputs"]
                    self.assertEqual(
                        {
                            "paradigm",
                            "first_frame_asset",
                            "first_frame_source",
                            "first_frame_policy",
                        },
                        set(native["vision"]),
                    )
                job_identities.append([
                    (job["job_id"], job["case_id"], job["seed"])
                    for job in instance.value["inference"]["jobs"]
                ])
        self.assertEqual(job_identities[0], job_identities[1])

    def test_worker_assignment_is_deterministic_and_single_gpu(self) -> None:
        module = _load_module("test_causal_forcing_driver", BASELINE_ROOT / "driver.py")
        driver = module.Driver.__new__(module.Driver)
        driver.bundle = SimpleNamespace(value={
            "runtime": {"cuda_visible_devices": "0,1,2,3,4,5,6,7"}
        })
        self.assertEqual([str(index) for index in range(8)], driver._gpu_ids())
        first = driver._worker_index("case-a__seed-42")
        self.assertEqual(first, driver._worker_index("case-a__seed-42"))
        self.assertIn(first, range(8))

    def test_prepare_job_never_exposes_reference_video(self) -> None:
        module = _load_module(
            "test_causal_forcing_driver_prepare",
            BASELINE_ROOT / "driver.py",
        )
        driver = module.Driver.__new__(module.Driver)
        driver.bundle = SimpleNamespace(
            baseline_id="causal_forcing_pp_2step_i2v_generic",
            root=BASELINE_ROOT,
            value=self.bundle.value,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (64, 64), "white").save(source / "first.png")
            run_dir = root / "run"
            adaptation = {
                "input_contract": {"asset_access": ["first_frame"]},
            }
            spec = driver.prepare_job(
                job={
                    "job_id": "job-a",
                    "case_id": "case-a",
                    "seed": 42,
                    "native_inputs": {
                        "vision": {"first_frame_asset": "first.png"},
                        "text": {"prompt": "A ball moves."},
                        "generation_shape": {
                            "width": 832,
                            "height": 480,
                            "fps": 16,
                            "num_frames": 81,
                        },
                        "media_contract": build_i2v_media_contract(
                            conditioning_asset="first.png",
                            width=832,
                            height=480,
                            temporal={
                                "fps": 16,
                                "num_frames": 81,
                                "valid_frame_rule": "4n+1",
                            },
                        ),
                    },
                },
                case={
                    "case_id": "case-a",
                    "scene_id": "collision_1d",
                    "assets": {
                        "first_frame": "first.png",
                        "reference_video": "forbidden.mp4",
                    },
                },
                adaptation=adaptation,
                source_root=source,
                run_dir=run_dir,
            )
            forbidden = {
                "reference_video",
                "physics_reference_video",
                "evaluation_reference_video",
                "visual_reference_video",
            }
            self.assertFalse(forbidden & set(spec))
            self.assertEqual(
                str((source / "first.png").resolve()),
                spec["source_first_frame"],
            )
            with Image.open(spec["first_frame"]) as conditioned:
                self.assertEqual((832, 480), conditioned.size)
            self.assertTrue(Path(spec["output_video"]).is_relative_to(run_dir))

    def test_prepare_job_forwards_fused_prompt_without_raw_physics(self) -> None:
        module = _load_module(
            "test_causal_forcing_physics_driver_prepare",
            BASELINE_ROOT / "driver.py",
        )
        driver = module.Driver.__new__(module.Driver)
        driver.bundle = SimpleNamespace(
            baseline_id="causal_forcing_pp_2step_i2v_physics",
            root=BASELINE_ROOT,
            value=self.physics_bundle.value,
        )
        case = next(
            item
            for item in self.dataset.cases
            if item["scene_id"] == "parabolic_motion"
        )
        adaptation = self.physics_adapter.adapt_case(case, role="eval")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (64, 64), "white").save(source / "first.png")
            native = adaptation["native_inputs"]
            native["vision"]["first_frame_asset"] = "first.png"
            native["media_contract"]["conditioning"]["asset"] = (
                "first.png"
            )
            spec = driver.prepare_job(
                job={
                    "job_id": "job-physics",
                    "case_id": case["case_id"],
                    "seed": 42,
                    "native_inputs": native,
                },
                case=case,
                adaptation=adaptation,
                source_root=source,
                run_dir=root / "run",
            )
        self.assertEqual(adaptation["prompt"], spec["prompt"])
        self.assertFalse(
            {
                "physics",
                "used_parameters",
                "reference_video",
                "physics_reference_video",
            }
            & set(spec)
        )

    def test_portrait_resize_preserves_orientation_and_does_not_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "conditioned.png"
            Image.new("RGB", (400, 800), (10, 20, 30)).save(source)
            contract = build_i2v_media_contract(
                conditioning_asset="source.png",
                width=480,
                height=832,
                temporal={"fps": 16, "num_frames": 81},
            )
            from physbench.baseline_runtime import (
                materialize_i2v_conditioning,
            )

            materialize_i2v_conditioning(source, output, contract)
            with Image.open(output) as conditioned:
                self.assertEqual((480, 832), conditioned.size)
                self.assertEqual((10, 20, 30), conditioned.getpixel((0, 0)))
                self.assertEqual(
                    (10, 20, 30),
                    conditioned.getpixel((479, 831)),
                )


if __name__ == "__main__":
    unittest.main()
