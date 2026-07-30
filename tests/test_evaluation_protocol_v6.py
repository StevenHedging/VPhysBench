from __future__ import annotations

import json
import unittest

from _paths import ROOT
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.circular_motion.v6_evaluator import (
    CircularMotionOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.collision.v5_evaluator import (
    CollisionOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.free_fall.v6_evaluator import (
    FreeFallOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.inclined_plane.v6_evaluator import (
    InclinedPlaneOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.pendulum.v6_evaluator import (
    PendulumOpenWorldCaseEvaluator,
)


V3_PROTOCOL_FINGERPRINT = (
    "14dac014a9311ce32be65052f88875e9a9432d64f044dfeb7cc5eceb9adc41c6"
)
V4_PROTOCOL_FINGERPRINT = (
    "54cefd0a75dc927e783ba7bcd8c75719b6c8b619fc07dce055eb50f94ca60532"
)
V5_PROTOCOL_FINGERPRINT = (
    "93703d6afdf8bbdea86b69d8d8a427653c68030bd7660f3801341e3b2b6209f6"
)
V6_PROTOCOL_FINGERPRINT = (
    "095f40ab72f8e42dadba104bb0c69dce7314153923d3c75d3a65ecef405288ff"
)
V5_COLLISION_FINGERPRINT = (
    "686b705e426f43d29acdc745a95b765fe249bd48628c86f5167f40f851caa156"
)
V6_EVALUATOR_FINGERPRINTS = {
    "pendulum": (
        "696605bc08595fc820d2a174a36f47768e862f90522dc12576cb463d841b8dfd"
    ),
    "free_fall": (
        "25872d31ff7ad21590ee5ee48395aa2260cf265853355be3a8c32bd941a7edc6"
    ),
    "inclined_plane_slide": (
        "31a3df4ad6d1983489920f8c0d3e61cf013b1a8e1d16b3fec6a3b6b1e4f7094c"
    ),
    "uniform_circular_motion": (
        "0fe16b1c21a51c19f13e27327f61bdeb365146827753536b7d931bc09a019666"
    ),
}


class EvaluationProtocolV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = load_evaluation_protocol("scene_default_v3")
        cls.v4 = load_evaluation_protocol("scene_default_v4")
        cls.v5 = load_evaluation_protocol("scene_default_v5")
        cls.v6 = load_evaluation_protocol("scene_default_v6")

    def test_old_protocols_and_collision_22_remain_frozen(self) -> None:
        self.assertEqual(V3_PROTOCOL_FINGERPRINT, self.v3["fingerprint"])
        self.assertEqual(V4_PROTOCOL_FINGERPRINT, self.v4["fingerprint"])
        self.assertEqual(V5_PROTOCOL_FINGERPRINT, self.v5["fingerprint"])
        self.assertEqual(
            self.v5["scenes"]["collision_1d"],
            self.v6["scenes"]["collision_1d"],
        )
        evaluator = SceneEvaluatorRegistry(self.v6).resolve("collision_1d")
        self.assertIsInstance(evaluator, CollisionOpenWorldCaseEvaluator)
        self.assertEqual("2.2", evaluator.describe()["version"])
        self.assertEqual(
            V5_COLLISION_FINGERPRINT,
            evaluator.describe()["fingerprint"],
        )

    def test_v6_routes_every_non_collision_scene_to_open_world_v2(self) -> None:
        self.assertEqual(V6_PROTOCOL_FINGERPRINT, self.v6["fingerprint"])
        expected = {
            "pendulum": (
                "pendulum_state_v6",
                PendulumOpenWorldCaseEvaluator,
            ),
            "free_fall": (
                "free_fall_state_v6",
                FreeFallOpenWorldCaseEvaluator,
            ),
            "inclined_plane_slide": (
                "inclined_plane_state_v6",
                InclinedPlaneOpenWorldCaseEvaluator,
            ),
            "uniform_circular_motion": (
                "uniform_circular_motion_state_v6",
                CircularMotionOpenWorldCaseEvaluator,
            ),
        }
        registry = SceneEvaluatorRegistry(self.v6)
        for scene_id, (evaluator_type, evaluator_class) in expected.items():
            with self.subTest(scene_id=scene_id):
                config = self.v6["scenes"][scene_id]
                self.assertEqual(evaluator_type, config["type"])
                self.assertEqual(
                    "open_world_v2", config["observer_protocol"]
                )
                self.assertIsInstance(
                    registry.resolve(scene_id), evaluator_class
                )
                self.assertEqual(
                    "scene_subject_state_similarity",
                    registry.resolve(scene_id).describe()["primary_score"],
                )
                self.assertEqual(
                    V6_EVALUATOR_FINGERPRINTS[scene_id],
                    registry.resolve(scene_id).describe()["fingerprint"],
                )

    def test_v6_configuration_has_common_audit_and_external_visualization(
        self,
    ) -> None:
        for scene_id, config in self.v6["scenes"].items():
            if scene_id == "collision_1d":
                continue
            with self.subTest(scene_id=scene_id):
                self.assertEqual(
                    0.1,
                    config["object_centric_scoring"]["assignment"][
                        "minimum_match_position_similarity"
                    ],
                )
                visualization = config["visualization"]
                self.assertTrue(visualization["enabled"])
                self.assertTrue(
                    visualization["external_root"].startswith("/mnt/nvme1/")
                )
                self.assertEqual(
                    "visualizations", visualization["repository_link"]
                )

    def test_schema_has_an_isolated_open_world_v2_scene_contract(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "v2"
                / "evaluation_protocol.schema.json"
            ).read_text(encoding="utf-8")
        )
        contract = schema["$defs"]["openWorldV2Scene"]
        self.assertEqual(
            "open_world_v2",
            contract["properties"]["observer_protocol"]["const"],
        )
        self.assertEqual(
            {
                "pendulum_state_v6",
                "free_fall_state_v6",
                "inclined_plane_state_v6",
                "uniform_circular_motion_state_v6",
            },
            set(contract["properties"]["type"]["enum"]),
        )
        self.assertIn(
            "object_centric_scoring", contract["required"]
        )
        self.assertIn("visualization", contract["required"])
        self.assertNotIn(
            "pendulum_state_v6",
            schema["$defs"]["pendulum"]["properties"]["type"]["enum"],
        )
        self.assertNotIn(
            "free_fall_state_v6",
            schema["$defs"]["referenceScene"]["properties"]["type"]["enum"],
        )
        wrappers = {
            "pendulum": "openWorldV2PendulumScene",
            "free_fall": "openWorldV2FreeFallScene",
            "inclined_plane_slide": "openWorldV2InclinedPlaneScene",
            "uniform_circular_motion": "openWorldV2CircularMotionScene",
        }
        for scene_id, definition in wrappers.items():
            references = schema["properties"]["scenes"]["properties"][
                scene_id
            ]["oneOf"]
            self.assertIn(
                {"$ref": f"#/$defs/{definition}"},
                references,
            )
        expected_types = {
            "openWorldV2PendulumScene": "pendulum_state_v6",
            "openWorldV2FreeFallScene": "free_fall_state_v6",
            "openWorldV2InclinedPlaneScene": (
                "inclined_plane_state_v6"
            ),
            "openWorldV2CircularMotionScene": (
                "uniform_circular_motion_state_v6"
            ),
        }
        for definition, evaluator_type in expected_types.items():
            self.assertEqual(
                evaluator_type,
                schema["$defs"][definition]["allOf"][1][
                    "properties"
                ]["type"]["const"],
            )


if __name__ == "__main__":
    unittest.main()
