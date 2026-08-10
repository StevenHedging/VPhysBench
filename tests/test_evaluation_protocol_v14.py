from __future__ import annotations

import copy
import hashlib
import json
import unittest

from _paths import ROOT
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import (
    SUPPORTED_EVALUATOR_TYPES,
    SceneEvaluatorRegistry,
)
from physbench.evaluation.scenes.collision.v6_evaluator import (
    CollisionFailClosedCaseEvaluator,
)
from physbench.tasks.planner import load_task


V13_PROTOCOL_SHA256 = (
    "4b9fb5f5c992023094997b026513dbf5450705c833bd8e328d29d4c57bea7aef"
)
V13_DIRECT_TASK_SHA256 = (
    "7f41f39cfe269c7fe33051969be508acb4560059a97748098eff4b1c5f22c84a"
)
V13_FINETUNE_TASK_SHA256 = (
    "5fbe50791d755e22b4f0a3c21d08c5d05824a50edd7e9796b44e481b2a5d5f21"
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationProtocolV14Tests(unittest.TestCase):
    def test_v14_changes_only_collision_identity_contract(self) -> None:
        v13 = load_evaluation_protocol("scene_default_v13")
        v14 = load_evaluation_protocol("scene_default_v14")

        self.assertEqual("scene_default_v14", v14["protocol_id"])
        self.assertEqual(v13["robustness"], v14["robustness"])
        self.assertEqual(v13["general_metrics"], v14["general_metrics"])
        self.assertEqual(
            {
                scene_id: config
                for scene_id, config in v13["scenes"].items()
                if scene_id != "collision_1d"
            },
            {
                scene_id: config
                for scene_id, config in v14["scenes"].items()
                if scene_id != "collision_1d"
            },
        )
        expected = copy.deepcopy(v13["scenes"]["collision_1d"])
        expected["type"] = "collision_1d_state_v6"
        expected["subject_identity"] = {
            "anchor_policy": "reference_motion_validated_frame_zero_v1",
            "failure_policy": "fail_closed_v1",
            "reference_localization": {
                "minimum_motion_fraction": 0.20,
                "minimum_mean_change": 8.0,
                "maximum_entity_y_spread_px": 32.0,
                "maximum_radius_ratio": 4.5,
                "minimum_gap_radius_fraction": 0.45,
                "minimum_two_body_gap_radius_fraction": 0.75,
                "minimum_set_score_margin": 0.20,
            },
            "prediction_binding": {
                "box_expand": 1.5,
                "minimum_box_side": 24,
                "maximum_center_distance_radii": 3.0,
                "minimum_position_similarity": 0.20,
                "minimum_scale_similarity": 0.45,
                "minimum_appearance_similarity": 0.15,
                "minimum_binding_score": 0.55,
                "minimum_assignment_margin": 0.05,
                "weights": {
                    "position": 0.5,
                    "scale": 0.2,
                    "appearance": 0.3,
                },
            },
            "tracking": {
                "minimum_mask_pixels": 12,
                "maximum_mask_area_ratio": 0.02,
                "maximum_centroid_jump_px": 120,
            },
            "unexpected_participant": {
                "minimum_observed_frames": 3,
                "minimum_observed_frame_fraction": 0.25,
            },
        }
        self.assertEqual(expected, v14["scenes"]["collision_1d"])

    def test_registry_resolves_only_new_collision_type_to_v6(self) -> None:
        self.assertIn("collision_1d_state_v6", SUPPORTED_EVALUATOR_TYPES)
        evaluator = SceneEvaluatorRegistry(
            load_evaluation_protocol("scene_default_v14")
        ).resolve("collision_1d")

        self.assertIsInstance(evaluator, CollisionFailClosedCaseEvaluator)
        self.assertEqual("3.0", evaluator.evaluator_version)
        self.assertEqual("collision_1d_fail_closed_nbody", evaluator.evaluator_id)

    def test_v3_tasks_change_only_task_id_and_protocol(self) -> None:
        pairs = (
            (
                "five_scene_direct_eval_csti_identity_v2.json",
                "five_scene_direct_eval_csti_identity_v3.json",
                "five_scene_direct_eval_v14_csti_identity_v3",
            ),
            (
                "five_scene_finetune_eval_csti_identity_v2.json",
                "five_scene_finetune_eval_csti_identity_v3.json",
                "five_scene_finetune_eval_v14_csti_identity_v3",
            ),
        )
        for old_name, new_name, task_id in pairs:
            with self.subTest(task=new_name):
                old = load_task(ROOT / "tasks/official" / old_name)
                new = load_task(ROOT / "tasks/official" / new_name)
                expected = copy.deepcopy(old.value)
                expected["task_id"] = task_id
                expected["evaluation"]["protocol"] = "scene_default_v14"
                self.assertEqual(expected, new.value)

    def test_schema_declares_fail_closed_collision_v6(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v3/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        scene = schema["$defs"]["referenceScene"]
        self.assertIn(
            "collision_1d_state_v6",
            scene["properties"]["type"]["enum"],
        )
        self.assertIn("subject_identity", scene["properties"])
        collision_rule = scene["allOf"][1]
        self.assertEqual(
            {"collision_1d_state_v5", "collision_1d_state_v6"},
            set(collision_rule["if"]["properties"]["type"]["enum"]),
        )
        identity_rule = scene["allOf"][2]
        self.assertEqual(
            "collision_1d_state_v6",
            identity_rule["if"]["properties"]["type"]["const"],
        )
        self.assertIn("subject_identity", identity_rule["then"]["required"])

    def test_v13_protocol_and_tasks_remain_byte_frozen(self) -> None:
        self.assertEqual(
            V13_PROTOCOL_SHA256,
            _sha256(ROOT / "configs/evaluation/protocols/scene_default_v13.json"),
        )
        self.assertEqual(
            V13_DIRECT_TASK_SHA256,
            _sha256(
                ROOT
                / "tasks/official/five_scene_direct_eval_csti_identity_v2.json"
            ),
        )
        self.assertEqual(
            V13_FINETUNE_TASK_SHA256,
            _sha256(
                ROOT
                / "tasks/official/five_scene_finetune_eval_csti_identity_v2.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
