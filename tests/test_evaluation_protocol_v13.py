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
from physbench.evaluation.scenes.parabolic_motion.v2_evaluator import (
    ParabolicMotionCaseEvaluatorV2,
)
from physbench.tasks.planner import load_task


V12_PROTOCOL_SHA256 = (
    "12e92e7f5ef74586f95cd75378b0b1f579140fc510e0e0d2e41ab321cad67bf3"
)
V12_DIRECT_TASK_SHA256 = (
    "bb78bdd2ac565d02f8a5489ec6a3c2ba498d7370cf6c31b7ccb79a70df1b1556"
)
V12_FINETUNE_TASK_SHA256 = (
    "344e9b288d2c24769c9fb08889989b91ee08b788fa4647cd324ee82cdd991b1b"
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationProtocolV13Tests(unittest.TestCase):
    def test_v13_changes_only_parabolic_fail_closed_contract(self) -> None:
        v12 = load_evaluation_protocol("scene_default_v12")
        v13 = load_evaluation_protocol("scene_default_v13")

        self.assertEqual("scene_default_v13", v13["protocol_id"])
        self.assertEqual(v12["robustness"], v13["robustness"])
        self.assertEqual(v12["general_metrics"], v13["general_metrics"])
        self.assertEqual(
            {
                scene_id: config
                for scene_id, config in v12["scenes"].items()
                if scene_id != "parabolic_motion"
            },
            {
                scene_id: config
                for scene_id, config in v13["scenes"].items()
                if scene_id != "parabolic_motion"
            },
        )
        expected = copy.deepcopy(v12["scenes"]["parabolic_motion"])
        expected["type"] = "parabolic_motion_state_v2"
        expected["reference_observation_policy"] = (
            "frozen_dataset_subject_identity"
        )
        expected["subject_identity"] = {
            "anchor_policy": "first_frame_subject_mask_manifest_v1",
            "failure_policy": "fail_closed_v1",
        }
        expected["open_world_observation"].update(
            {
                "minimum_assignment_cost_margin": 0.15,
                "latch_identity_loss": True,
            }
        )
        expected["scoring"].update(
            {
                "minimum_binding_score": 0.55,
                "minimum_binding_appearance": 0.20,
                "composition": "strict_multiplicative_v2",
            }
        )
        self.assertEqual(expected, v13["scenes"]["parabolic_motion"])

    def test_registry_resolves_only_new_parabolic_type_to_v2(self) -> None:
        self.assertIn("parabolic_motion_state_v2", SUPPORTED_EVALUATOR_TYPES)
        evaluator = SceneEvaluatorRegistry(
            load_evaluation_protocol("scene_default_v13")
        ).resolve("parabolic_motion")

        self.assertIsInstance(evaluator, ParabolicMotionCaseEvaluatorV2)
        self.assertEqual("2.0", evaluator.evaluator_version)
        self.assertEqual("parabolic_motion_fail_closed", evaluator.evaluator_id)

    def test_v2_tasks_change_only_task_id_and_protocol(self) -> None:
        pairs = (
            (
                "five_scene_direct_eval_csti_identity.json",
                "five_scene_direct_eval_csti_identity_v2.json",
                "five_scene_direct_eval_v13_csti_identity_v2",
            ),
            (
                "five_scene_finetune_eval_csti_identity.json",
                "five_scene_finetune_eval_csti_identity_v2.json",
                "five_scene_finetune_eval_v13_csti_identity_v2",
            ),
        )
        for old_name, new_name, task_id in pairs:
            with self.subTest(task=new_name):
                old = load_task(ROOT / "tasks/official" / old_name)
                new = load_task(ROOT / "tasks/official" / new_name)
                expected = copy.deepcopy(old.value)
                expected["task_id"] = task_id
                expected["evaluation"]["protocol"] = "scene_default_v13"
                self.assertEqual(expected, new.value)

    def test_schema_declares_v8_pendulum_and_v2_parabolic(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v3/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        scene = schema["$defs"]["openWorldV2Scene"]
        self.assertTrue(
            {
                "pendulum_state_v8",
                "parabolic_motion_state_v2",
            }.issubset(set(scene["properties"]["type"]["enum"]))
        )
        self.assertIn("subject_identity", scene["properties"])
        self.assertIn(
            "frozen_dataset_subject_identity",
            scene["properties"]["reference_observation_policy"]["enum"],
        )
        self.assertEqual(
            {
                "pendulum_state_v6",
                "pendulum_state_v7",
                "pendulum_state_v8",
            },
            set(
                schema["$defs"]["openWorldV2PendulumScene"]["allOf"][1]
                ["properties"]["type"]["enum"]
            ),
        )
        self.assertEqual(
            {
                "parabolic_motion_state_v1",
                "parabolic_motion_state_v2",
            },
            set(
                schema["$defs"]["openWorldV2ParabolicScene"]["allOf"][1]
                ["properties"]["type"]["enum"]
            ),
        )

    def test_v12_protocol_and_tasks_remain_byte_frozen(self) -> None:
        self.assertEqual(
            V12_PROTOCOL_SHA256,
            _sha256(ROOT / "configs/evaluation/protocols/scene_default_v12.json"),
        )
        self.assertEqual(
            V12_DIRECT_TASK_SHA256,
            _sha256(
                ROOT / "tasks/official/five_scene_direct_eval_csti_identity.json"
            ),
        )
        self.assertEqual(
            V12_FINETUNE_TASK_SHA256,
            _sha256(
                ROOT
                / "tasks/official/five_scene_finetune_eval_csti_identity.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
