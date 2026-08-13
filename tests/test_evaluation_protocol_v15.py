from __future__ import annotations

import copy
import json
import unittest

from _paths import ROOT
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import (
    SUPPORTED_EVALUATOR_TYPES,
    SceneEvaluatorRegistry,
)


class EvaluationProtocolV15Tests(unittest.TestCase):
    def test_v15_preserves_v14_scenes_and_adds_vertical_spring(self) -> None:
        v14 = load_evaluation_protocol("scene_default_v14")
        v15 = load_evaluation_protocol("scene_default_v15")

        self.assertEqual("scene_default_v15", v15["protocol_id"])
        self.assertEqual(v14["robustness"], v15["robustness"])
        self.assertEqual(v14["general_metrics"], v15["general_metrics"])
        self.assertEqual(v14["scenes"], {
            scene_id: copy.deepcopy(v15["scenes"][scene_id])
            for scene_id in v14["scenes"]
        })
        self.assertEqual(
            {
                "pendulum",
                "collision_1d",
                "inclined_plane_slide",
                "uniform_circular_motion",
                "parabolic_motion",
                "vertical_spring_oscillator",
            },
            set(v15["scenes"]),
        )
        self.assertEqual(
            "vertical_spring_oscillator_v1",
            v15["scenes"]["vertical_spring_oscillator"]["type"],
        )

    def test_registry_resolves_vertical_spring_v1(self) -> None:
        from physbench.evaluation.scenes.vertical_spring_oscillator.evaluator import (
            VerticalSpringOscillatorCaseEvaluator,
        )

        self.assertIn(
            "vertical_spring_oscillator_v1", SUPPORTED_EVALUATOR_TYPES
        )
        evaluator = SceneEvaluatorRegistry(
            load_evaluation_protocol("scene_default_v15")
        ).resolve("vertical_spring_oscillator")
        self.assertIsInstance(evaluator, VerticalSpringOscillatorCaseEvaluator)

    def test_schema_declares_vertical_spring_scene(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v3/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        scenes = schema["properties"]["scenes"]
        self.assertIn("vertical_spring_oscillator", scenes["properties"])
        definition_name = scenes["properties"]["vertical_spring_oscillator"][
            "$ref"
        ].rsplit("/", 1)[-1]
        self.assertEqual(
            "vertical_spring_oscillator_v1",
            schema["$defs"][definition_name]["allOf"][1]["properties"][
                "type"
            ]["const"],
        )


if __name__ == "__main__":
    unittest.main()
