from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from _paths import ROOT
from baselines.wan22_symbol_value_cross_attention.adapter import (
    SymbolValueRegistry,
)


REGISTRY = (
    ROOT
    / "baselines"
    / "wan22_symbol_value_cross_attention"
    / "quantity_registry.json"
)


def _case() -> dict:
    return {
        "case_id": "spring_fixture",
        "scene_id": "vertical_spring_oscillator",
        "text": {
            "prompt": (
                "An oscillator of mass m and radius r is attached to a spring "
                "of stiffness k and natural length L_0 under gravity g. Its "
                "initial displacement x_0 is above equilibrium."
            )
        },
        "physics": {
            "objects": {
                "object_1": {
                    "initial_displacement": {
                        "symbol": "x_0",
                        "value": 0.1,
                        "unit": "m",
                    },
                    "mass": {
                        "symbol": "m",
                        "value": 0.5156,
                        "unit": "kg",
                    },
                    "radius": {
                        "symbol": "r",
                        "value": 0.025,
                        "unit": "m",
                    },
                }
            },
            "environment": {
                "gravity_acceleration": {
                    "symbol": "g",
                    "value": 9.80665,
                    "unit": "m/s^2",
                },
                "natural_spring_length": {
                    "symbol": "L_0",
                    "value": 0.068,
                    "unit": "m",
                },
                "spring_stiffness": {
                    "symbol": "k",
                    "value": 32.6213467096774,
                    "unit": "N/m",
                },
            },
        },
    }


class SymbolValueRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SymbolValueRegistry(REGISTRY)

    def test_render_preserves_prompt_and_emits_si_symbol_value_records(self) -> None:
        case = _case()

        prompt, records, used = self.registry.render(
            case,
            case["text"]["prompt"],
        )

        self.assertEqual(case["text"]["prompt"], prompt)
        self.assertEqual(6, len(records))
        displacement = next(
            item for item in records
            if item["name"] == "initial_displacement"
        )
        self.assertEqual("x_0", displacement["symbol"])
        self.assertEqual(0.1, displacement["si_value"])
        self.assertEqual([1, 0, 0, 0, 0, 0, 0], displacement["dimension"])
        self.assertIsInstance(displacement["unit_id"], int)
        self.assertEqual(
            {item["name"] for item in records},
            set(used),
        )
        self.assertNotIn("0.100", prompt)

    def test_si_conversion_uses_registry_scale(self) -> None:
        case = {
            "case_id": "pendulum_fixture",
            "scene_id": "pendulum",
            "text": {
                "prompt": (
                    "A bob of mass m and radius r is released at angle θ_0 "
                    "on a string of length l_s."
                )
            },
            "physics": {
                "objects": {
                    "object_1": {
                        "radius": {
                            "symbol": "r", "value": 0.01, "unit": "m"
                        },
                        "initial_angle": {
                            "symbol": "θ_0", "value": 30.0, "unit": "deg"
                        },
                        "mass": {
                            "symbol": "m", "value": 0.03, "unit": "kg"
                        },
                    }
                },
                "environment": {
                    "string_length": {
                        "symbol": "l_s", "value": 0.5, "unit": "m"
                    }
                },
            },
        }

        _, records, _ = self.registry.render(
            case,
            case["text"]["prompt"],
        )

        angle = next(item for item in records if item["name"] == "initial_angle")
        self.assertAlmostEqual(math.pi / 6.0, angle["si_value"])
        self.assertEqual("rad", angle["canonical_si_unit"])

    def test_missing_symbol_in_prompt_is_rejected(self) -> None:
        case = _case()
        case["text"]["prompt"] = case["text"]["prompt"].replace("x_0", "x")

        with self.assertRaisesRegex(ValueError, "symbol.*x_0.*absent"):
            self.registry.render(case, case["text"]["prompt"])

    def test_unexpected_unit_is_rejected(self) -> None:
        case = _case()
        case["physics"]["objects"]["object_1"]["mass"]["unit"] = "g"

        with self.assertRaisesRegex(ValueError, "unit.*g.*kg"):
            self.registry.render(case, case["text"]["prompt"])

    def test_non_finite_value_is_rejected(self) -> None:
        case = _case()
        case["physics"]["objects"]["object_1"]["mass"]["value"] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            self.registry.render(case, case["text"]["prompt"])

    def test_temporal_quantity_is_not_collapsed_to_a_scalar(self) -> None:
        case = _case()
        case["physics"]["objects"]["object_1"]["mass"] = {
            "symbol": "m",
            "samples": [{"time": 0.0, "value": 0.5}],
            "time_unit": "s",
            "unit": "kg",
        }

        with self.assertRaisesRegex(ValueError, "scalar"):
            self.registry.render(case, case["text"]["prompt"])

    def test_missing_required_quantity_is_rejected(self) -> None:
        case = _case()
        del case["physics"]["environment"]["spring_stiffness"]

        with self.assertRaisesRegex(
            ValueError,
            "required.*spring_stiffness|invalid.*environment",
        ):
            self.registry.render(case, case["text"]["prompt"])

    def test_ascii_symbol_requires_identifier_boundaries(self) -> None:
        case = _case()
        case["text"]["prompt"] = case["text"]["prompt"].replace(
            "mass m", "mass mm"
        )

        with self.assertRaisesRegex(ValueError, "symbol.*m.*absent"):
            self.registry.render(case, case["text"]["prompt"])


if __name__ == "__main__":
    unittest.main()
