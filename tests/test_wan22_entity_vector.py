from __future__ import annotations

import math
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from baselines.wan22_entity_vector.adapter import (
    ENTITY_VECTOR_COMPONENTS,
    FirstEntityVectorDataAdapter,
    extract_first_entity_vector,
)
from _paths import ROOT
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.baseline_api import load_baseline_bundle
from physbench.orchestration import build_task_instance
from physbench.baselines.wan22_quantity import Wan22QuantityLoraAdapter
from physbench.baselines.wan22_quantity_model import (
    QUANTITY_CHECKPOINT_PREFIX,
    QuantityEncoder,
    expected_wan22_ti2v_5b_lora_targets,
)
from torch import nn


def _case(
    scene_id: str,
    object_1: dict,
    *,
    environment: dict | None = None,
    object_2: dict | None = None,
) -> dict:
    objects = {"object_1": object_1}
    if object_2 is not None:
        objects["object_2"] = object_2
    return {
        "case_id": f"case_{scene_id}",
        "scene_id": scene_id,
        "physics": {
            "environment": environment or {},
            "objects": objects,
        },
    }


def _quantity(value: float, unit: str) -> dict:
    return {"value": value, "unit": unit, "symbol": "x"}


class FirstEntityVectorExtractionTests(unittest.TestCase):
    def test_component_order_is_frozen(self) -> None:
        self.assertEqual(
            ENTITY_VECTOR_COMPONENTS,
            ("mass_kg", "size_m", "initial_velocity_m_per_s"),
        )

    def test_collision_uses_only_object_one(self) -> None:
        record = extract_first_entity_vector(
            _case(
                "collision_1d",
                {
                    "mass": _quantity(0.03313, "kg"),
                    "radius": _quantity(0.01, "m"),
                    "initial_velocity": _quantity(
                        0.6004202942059444,
                        "m/s",
                    ),
                },
                object_2={
                    "mass": _quantity(99.0, "kg"),
                    "radius": _quantity(88.0, "m"),
                    "initial_velocity": _quantity(77.0, "m/s"),
                },
            )
        )
        self.assertEqual(
            record["values_si"],
            [0.03313, 0.01, 0.6004202942059444],
        )
        self.assertEqual(record["source_object"], "object_1")
        self.assertEqual(
            record["source_fields"],
            ["mass", "radius", "initial_velocity"],
        )

    def test_size_field_precedence_and_rest_velocity(self) -> None:
        cases = (
            ("inclined_plane_slide", "length", 0.11),
            ("push_bottle", "height", 0.22),
            ("vertical_spring_oscillator", "radius", 0.025),
        )
        for scene_id, field, size in cases:
            with self.subTest(scene_id=scene_id):
                record = extract_first_entity_vector(
                    _case(
                        scene_id,
                        {
                            "mass": _quantity(0.5, "kg"),
                            field: _quantity(size, "m"),
                        },
                    )
                )
                self.assertEqual(record["values_si"], [0.5, size, 0.0])
                self.assertEqual(
                    record["source_fields"],
                    ["mass", field, None],
                )
                self.assertEqual(record["provenance"][2], "imputed_rest")

    def test_circular_velocity_is_radius_times_angular_speed(self) -> None:
        record = extract_first_entity_vector(
            _case(
                "uniform_circular_motion",
                {"orbit_radius": _quantity(0.02, "m")},
                environment={
                    "angular_velocity": _quantity(54.55, "deg/s")
                },
            )
        )
        self.assertAlmostEqual(record["values_si"][0], 0.0)
        self.assertAlmostEqual(record["values_si"][1], 0.02)
        self.assertAlmostEqual(
            record["values_si"][2],
            0.02 * math.radians(54.55),
        )
        self.assertEqual(
            record["source_fields"],
            [None, "orbit_radius", "orbit_radius*angular_velocity"],
        )
        self.assertEqual(
            record["provenance"],
            ["imputed_zero", "observed", "derived"],
        )

    def test_initial_horizontal_velocity_is_supported(self) -> None:
        record = extract_first_entity_vector(
            _case(
                "parabolic_motion",
                {
                    "mass": _quantity(0.06477, "kg"),
                    "radius": _quantity(0.0125, "m"),
                    "initial_horizontal_velocity": _quantity(
                        0.6956037840845855,
                        "m/s",
                    ),
                },
            )
        )
        self.assertEqual(
            record["values_si"],
            [0.06477, 0.0125, 0.6956037840845855],
        )

    def test_missing_object_one_is_rejected(self) -> None:
        case = _case("collision_1d", {})
        case["physics"]["objects"] = {
            "object_2": {"mass": _quantity(1.0, "kg")}
        }
        with self.assertRaisesRegex(ValueError, "object_1"):
            extract_first_entity_vector(case)

    def test_invalid_present_values_are_rejected(self) -> None:
        bad_objects = (
            {"mass": _quantity(1.0, "g")},
            {"radius": _quantity(float("nan"), "m")},
            {"initial_velocity": _quantity(-1.0, "m/s")},
            {"height": _quantity(True, "m")},
        )
        for object_1 in bad_objects:
            with self.subTest(object_1=object_1):
                with self.assertRaises(ValueError):
                    extract_first_entity_vector(
                        _case("push_bottle", object_1)
                    )


class FirstEntityVectorAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset(LATEST_DATASET)
        cls.case = next(
            case
            for case in dataset.cases
            if case["case_id"]
            == "collision_supp_20260729_img_1167_two_ball_opposed_incident"
        )

    def test_adapter_seals_text_vector_and_first_frame_channels(self) -> None:
        frozen = (
            ROOT
            / "run"
            / "wan22_physics_text_lora_2184_v1"
            / "frozen"
            / "baseline.json"
        )
        import json

        source = json.loads(frozen.read_text())
        source["adapter"]["kind"] = "python"
        source["adapter"]["module"] = "adapter.py"
        source["input_policy"]["physics"]["representations"] = [
            "structured_text",
            "first_entity_vector_mlp_v1",
        ]
        adapter = FirstEntityVectorDataAdapter(
            SimpleNamespace(value=source, path=Path("baseline.json"))
        )
        result = adapter.adapt_case(self.case, role="training")

        self.assertEqual(
            adapter.config["physics_transform"]["template_set"],
            "seven_scene_physics_text_v1",
        )
        prompt = result["native_inputs"]["text"]["prompt"]
        audited = result["native_inputs"]["text"]["audited_prompt"]
        self.assertEqual(prompt.count("<extra_id_0>"), 1)
        self.assertNotIn("<extra_id_0>", audited)
        self.assertIn("[m=0.03313 kg, size=0.01000 m", audited)
        vector = result["native_inputs"]["physics"]["entity_vector"]
        self.assertEqual(vector["sentinel"], "<extra_id_0>")
        self.assertEqual(
            vector["values_si"],
            [0.03313, 0.01, 0.6004202942059444],
        )
        channels = result["input_contract"]["physics_channels"]
        self.assertEqual(len(channels), 2)
        self.assertEqual(channels[1]["id"], "first_entity_vector")
        self.assertEqual(
            result["input_contract"]["media_channels"][0]["id"],
            "initial_frame",
        )


class FirstEntityVectorEncoderTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "encoder_type": "first_entity_vector_mlp_v1",
            "input_size": 3,
            "hidden_sizes": [256, 1024],
            "text_hidden_size": 4096,
        }

    def _record(self) -> dict:
        return {
            "representation": "first_entity_vector_mlp_v1",
            "components": list(ENTITY_VECTOR_COMPONENTS),
            "values_si": [0.03313, 0.01, 0.6004202942059444],
            "sentinel": "<extra_id_0>",
        }

    def test_encoder_has_exact_three_linear_layers(self) -> None:
        encoder = QuantityEncoder(self._config())
        linears = [
            module
            for module in encoder.modules()
            if isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(linears), 3)
        self.assertEqual(
            [
                (layer.in_features, layer.out_features)
                for layer in linears
            ],
            [(3, 256), (256, 1024), (1024, 4096)],
        )

    def test_encoder_outputs_one_finite_wan_context_vector(self) -> None:
        encoder = QuantityEncoder(self._config()).eval()
        output = encoder.encode_records([self._record()])
        self.assertEqual(tuple(output.shape), (1, 4096))
        self.assertTrue(output.isfinite().all().item())

    def test_encoder_rejects_wrong_record_count_or_shape(self) -> None:
        encoder = QuantityEncoder(self._config())
        record = self._record()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            encoder.encode_records([])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            encoder.encode_records([record, record])
        wrong = dict(record, values_si=[1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "three"):
            encoder.encode_records([wrong])
        nonfinite = dict(record, values_si=[1.0, float("inf"), 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            encoder.encode_records([nonfinite])

    def test_checkpoint_inventory_accepts_only_entity_topology(self) -> None:
        encoder = QuantityEncoder(self._config())
        tensors = {
            f"{QUANTITY_CHECKPOINT_PREFIX}{name}": (
                "F32",
                list(tensor.shape),
            )
            for name, tensor in encoder.state_dict().items()
        }
        for target in expected_wan22_ti2v_5b_lora_targets():
            tensors[f"{target}.lora_A.weight"] = ("F32", [32, 1])
            tensors[f"{target}.lora_B.weight"] = ("F32", [1, 32])
        offset = 0
        header = {}
        for name, (dtype, shape) in tensors.items():
            size = math.prod(shape) * 4
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size
        encoded_header = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = struct.pack("<Q", len(encoded_header)) + encoded_header
        payload += b"\0" * offset

        expected_keys = frozenset(encoder.state_dict())
        inventory = Wan22QuantityLoraAdapter._checkpoint_inventory_bytes(
            payload,
            expected_encoder_keys=expected_keys,
            expected_encoder_tensor_count=len(expected_keys),
        )
        self.assertEqual(
            inventory["quantity_encoder_tensor_count"],
            len(expected_keys),
        )
        with self.assertRaisesRegex(ValueError, "619 tensors"):
            Wan22QuantityLoraAdapter._checkpoint_inventory_bytes(payload)


class FirstEntityVectorRegistrationTests(unittest.TestCase):
    baseline_path = (
        ROOT / "baselines" / "wan22_entity_vector" / "baseline.json"
    )
    task_path = (
        ROOT
        / "tasks"
        / "experiments"
        / "seven_scene_entity_vector_finetune_eval.json"
    )

    def test_bundle_freezes_the_requested_training_recipe(self) -> None:
        bundle = load_baseline_bundle(self.baseline_path)
        value = bundle.value
        self.assertEqual(
            value["baseline_id"],
            "wan22_ti2v_5b_lora_r32_physics_text_entity_vector_v1",
        )
        self.assertEqual(
            value["input_policy"]["physics"]["representations"],
            ["structured_text", "first_entity_vector_mlp_v1"],
        )
        self.assertEqual(
            value["model"]["quantity_encoder"]["encoder_type"],
            "first_entity_vector_mlp_v1",
        )
        trainer = value["trainer"]["config"]
        self.assertEqual(trainer["dataset_repeat"], 1)
        self.assertEqual(trainer["num_epochs"], 8)
        self.assertEqual(trainer["save_steps"], 273)
        self.assertEqual(trainer["rank"], 32)
        self.assertEqual(trainer["seed"], 42)
        self.assertEqual(
            trainer["scene_balancing"],
            "oversample_each_scene_to_largest_world_aligned",
        )

    def test_task_compiles_806_train_cases_and_110_jobs(self) -> None:
        instance = build_task_instance(
            dataset_path=LATEST_DATASET,
            task_path=self.task_path,
            baseline_path=self.baseline_path,
            check_assets=False,
        )
        plan = instance.canonical_plan
        self.assertEqual(len(plan.train_case_ids), 806)
        self.assertEqual(len(plan.jobs), 110)
        self.assertEqual(
            {job["seed"] for job in plan.jobs},
            {42},
        )
        self.assertEqual(
            json.loads(self.task_path.read_text())["evaluation"]["protocol"],
            "scene_default_v14",
        )

    def test_world_aligned_balancing_reproduces_2184_steps(self) -> None:
        adapter = object.__new__(Wan22QuantityLoraAdapter)
        adapter.config = {
            "lora": {
                "scene_balancing": (
                    "oversample_each_scene_to_largest_world_aligned"
                ),
                "dataset_repeat": 1,
                "num_epochs": 8,
                "seed": 42,
            }
        }
        adapter.runtime = {
            "cuda_visible_devices": "0,1,2,3,4,5,6,7"
        }
        scene_counts = {
            "collision_1d": 310,
            "inclined_plane_slide": 80,
            "parabolic_motion": 82,
            "pendulum": 80,
            "push_bottle": 127,
            "uniform_circular_motion": 30,
            "vertical_spring_oscillator": 97,
        }
        rows = [
            {"scene_id": scene, "case_id": f"{scene}_{index:04d}"}
            for scene, count in scene_counts.items()
            for index in range(count)
        ]
        with tempfile.TemporaryDirectory() as directory:
            balanced = adapter._balance_training_rows(
                rows,
                Path(directory),
            )
            plan = json.loads(
                (Path(directory) / "training_sampling_plan.json").read_text()
            )
        self.assertEqual(len(rows), 806)
        self.assertEqual(len(balanced), 2184)
        self.assertEqual(plan["per_scene_target"], 312)
        self.assertEqual(plan["expected_optimizer_steps_per_epoch"], 273)
        self.assertEqual(plan["expected_total_optimizer_steps"], 2184)


if __name__ == "__main__":
    unittest.main()
