from __future__ import annotations

import copy
import unittest

from _paths import FIXTURES, ROOT
from physbench.io import load_jsonl
from physbench.prompts import PromptRegistry


class PromptRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = PromptRegistry(ROOT / "configs" / "prompts")
        cls.cases = {
            case["case_id"]: case for case in load_jsonl(FIXTURES / "cases.jsonl")
        }

    def test_only_requested_profiles_are_registered(self) -> None:
        self.assertEqual(
            {"generic", "physics_natural"}, set(self.registry.profiles)
        )

    def test_generic_has_no_rendered_physical_parameters(self) -> None:
        record = self.registry.resolve(
            self.cases["pend_id_001"], "generic", role="eval"
        )
        self.assertEqual({}, record["used_parameters"])
        self.assertNotIn("0.250", record["prompt"])
        self.assertNotIn("20.0", record["prompt"])

    def test_physics_natural_is_deterministic_and_uses_case_truth(self) -> None:
        first = self.registry.resolve(
            self.cases["pend_id_001"], "physics_natural", role="eval"
        )
        second = self.registry.resolve(
            self.cases["pend_id_001"], "physics_natural", role="eval"
        )
        self.assertEqual(first, second)
        self.assertIn("string length is 0.250 meters", first["prompt"])
        self.assertIn("initial release angle is 20.0 degrees", first["prompt"])
        self.assertEqual(
            {"string_length", "bob_radius", "initial_angle"},
            set(first["used_parameters"]),
        )

    def test_missing_required_parameter_fails_loudly(self) -> None:
        case = copy.deepcopy(self.cases["pend_id_001"])
        del case["physical_parameters"]["string_length"]
        with self.assertRaisesRegex(ValueError, "string_length"):
            self.registry.resolve(case, "physics_natural", role="eval")


if __name__ == "__main__":
    unittest.main()
