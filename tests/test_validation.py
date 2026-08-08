from __future__ import annotations

import copy
import unittest

from _paths import FIXTURES, SCENES
from physbench.io import load_jsonl
from physbench.validation import errors, load_scene_configs, validate_cases


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = FIXTURES / "cases.jsonl"
        self.cases = load_jsonl(self.path)
        self.scenes = load_scene_configs(SCENES)
        # The legacy validation fixture includes one synthetic free-fall family;
        # keep its unit-level scene contract local instead of expanding the
        # release scene catalog.
        self.scenes["free_fall"] = {
            "scene_id": "free_fall",
            "id_parameters": ["ball_radius", "initial_height"],
            "ood1_factors": ["background", "ball_material"],
        }

    def test_fixture_is_valid(self) -> None:
        self.assertEqual([], errors(validate_cases(self.cases, manifest_path=self.path, scene_configs=self.scenes)))

    def test_synthetic_ood_cannot_change_physics(self) -> None:
        cases = copy.deepcopy(self.cases)
        case = next(item for item in cases if item["case_id"] == "pend_ood_bg_001")
        case["physical_parameters"]["initial_angle"]["value"] = 99
        codes = {issue.code for issue in errors(validate_cases(cases, manifest_path=self.path, scene_configs=self.scenes))}
        self.assertIn("ood1_physics_changed", codes)

    def test_visual_reference_flag_is_strict(self) -> None:
        cases = copy.deepcopy(self.cases)
        case = next(item for item in cases if item["case_id"] == "pend_ood_bg_001")
        case["has_real_reference_video"] = True
        codes = {issue.code for issue in errors(validate_cases(cases, manifest_path=self.path, scene_configs=self.scenes))}
        self.assertIn("missing_real_reference", codes)
        self.assertIn("synthetic_marked_real", codes)

    def test_prompts_are_not_required_inside_case_input_views(self) -> None:
        cases = copy.deepcopy(self.cases)
        for case in cases:
            case["text"].pop("prompt", None)
            for view in case["input_views"].values():
                view.pop("prompt", None)
        self.assertEqual(
            [],
            errors(validate_cases(cases, manifest_path=self.path, scene_configs=self.scenes)),
        )


if __name__ == "__main__":
    unittest.main()
