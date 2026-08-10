from __future__ import annotations

import copy
import hashlib
import unittest

from _paths import ROOT
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import (
    SUPPORTED_EVALUATOR_TYPES,
    SceneEvaluatorRegistry,
)
from physbench.evaluation.scenes.pendulum.v8_evaluator import (
    PendulumOpenWorldCaseEvaluatorV8,
)
from physbench.tasks.planner import load_task


V11_PROTOCOL_SHA256 = (
    "adc9133899833b80c6abc67eaaa6f5eac4f20977fbe82974c5ac49dbd68fe5d9"
)
V11_DIRECT_TASK_SHA256 = (
    "81e13b1d1469330c0662fb9a14841bc266bec495e6d60523741665af58ac02b9"
)
V11_FINETUNE_TASK_SHA256 = (
    "3440d431c0dca6ea0174089eb0681494d5668d9f12c0cfb0efb7cbac84babff7"
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationProtocolV12Tests(unittest.TestCase):
    def test_v12_changes_only_pendulum_subject_identity_contract(self) -> None:
        v11 = load_evaluation_protocol("scene_default_v11")
        v12 = load_evaluation_protocol("scene_default_v12")

        self.assertEqual("scene_default_v12", v12["protocol_id"])
        self.assertEqual(v11["robustness"], v12["robustness"])
        self.assertEqual(v11["general_metrics"], v12["general_metrics"])
        self.assertEqual(
            {
                scene_id: config
                for scene_id, config in v11["scenes"].items()
                if scene_id != "pendulum"
            },
            {
                scene_id: config
                for scene_id, config in v12["scenes"].items()
                if scene_id != "pendulum"
            },
        )

        expected = copy.deepcopy(v11["scenes"]["pendulum"])
        expected["type"] = "pendulum_state_v8"
        expected["reference_observation_policy"] = (
            "frozen_dataset_subject_identity"
        )
        expected["subject_identity"] = {
            "anchor_policy": "first_frame_subject_mask_manifest_v1",
            "failure_policy": "fail_closed_v1",
            "minimum_anchor_iou": 0.10,
            "maximum_center_distance_radii": 1.75,
            "minimum_area_ratio": 0.35,
            "maximum_area_ratio": 2.5,
            "minimum_length_ratio": 0.60,
            "maximum_length_ratio": 1.35,
            "minimum_color_similarity": 0.20,
            "minimum_score": 0.45,
            "ambiguity_margin": 0.05,
        }
        expected["open_world_observation"].update(
            {
                "v8_anchor_dilation_radius_ratio": 0.75,
                "v8_minimum_circle_anchor_containment": 0.45,
                "v8_maximum_anchor_center_distance_radii": 1.75,
                "v8_minimum_identity_source_agreement": 3,
                "v8_minimum_anchor_geometry_ratio_score": 0.45,
                "v8_condition_identity_ambiguity_margin": 0.05,
                "v8_residual_requires_pivot_string": True,
                "v8_condition_low_margin_warning": 0.08,
            }
        )
        self.assertEqual(expected, v12["scenes"]["pendulum"])

    def test_registry_resolves_only_the_new_v8_type_to_v8(self) -> None:
        self.assertIn("pendulum_state_v8", SUPPORTED_EVALUATOR_TYPES)
        protocol = load_evaluation_protocol("scene_default_v12")

        evaluator = SceneEvaluatorRegistry(protocol).resolve("pendulum")

        self.assertIsInstance(evaluator, PendulumOpenWorldCaseEvaluatorV8)
        self.assertEqual("3.0", evaluator.evaluator_version)

    def test_identity_tasks_change_only_task_id_and_protocol(self) -> None:
        pairs = (
            (
                "five_scene_direct_eval_csti.json",
                "five_scene_direct_eval_csti_identity.json",
                "five_scene_direct_eval_v13_csti_identity",
            ),
            (
                "five_scene_finetune_eval_csti.json",
                "five_scene_finetune_eval_csti_identity.json",
                "five_scene_finetune_eval_v13_csti_identity",
            ),
        )
        for old_name, new_name, expected_id in pairs:
            with self.subTest(task=new_name):
                old = load_task(ROOT / "tasks/official" / old_name)
                new = load_task(ROOT / "tasks/official" / new_name)
                expected = copy.deepcopy(old.value)
                expected["task_id"] = expected_id
                expected["evaluation"]["protocol"] = "scene_default_v12"

                self.assertEqual(expected, new.value)

    def test_v11_protocol_and_tasks_remain_byte_frozen(self) -> None:
        self.assertEqual(
            V11_PROTOCOL_SHA256,
            _sha256(
                ROOT / "configs/evaluation/protocols/scene_default_v11.json"
            ),
        )
        self.assertEqual(
            V11_DIRECT_TASK_SHA256,
            _sha256(ROOT / "tasks/official/five_scene_direct_eval_csti.json"),
        )
        self.assertEqual(
            V11_FINETUNE_TASK_SHA256,
            _sha256(ROOT / "tasks/official/five_scene_finetune_eval_csti.json"),
        )


if __name__ == "__main__":
    unittest.main()
