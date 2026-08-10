from __future__ import annotations

import math
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


if __name__ == "__main__":
    unittest.main()
