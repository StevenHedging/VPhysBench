from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

from _paths import ROOT
from physbench.evaluation.common.entities import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldComparison,
    OpenWorldObservation,
    OpenWorldTrack,
    compare_open_world_tracks,
    deduplicate_frame_detections,
    detections_from_instance_masks,
    track_open_world_detections,
)
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.collision.nbody import (
    NBodyExtractionConfig,
    NBodyScoringConfig,
)
from physbench.evaluation.scenes.collision.v5_evaluator import (
    CollisionOpenWorldCaseEvaluator,
)


V3_PROTOCOL_FINGERPRINT = (
    "14dac014a9311ce32be65052f88875e9a9432d64f044dfeb7cc5eceb9adc41c6"
)
V4_PROTOCOL_FINGERPRINT = (
    "54cefd0a75dc927e783ba7bcd8c75719b6c8b619fc07dce055eb50f94ca60532"
)
V4_COLLISION_FINGERPRINT = (
    "34032a1b50b10a4ca8948ef43f8bb6900122b4863d15ec2adc4bf4187e96fc4d"
)


class EvaluationProtocolV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = load_evaluation_protocol("scene_default_v3")
        cls.v4 = load_evaluation_protocol("scene_default_v4")
        cls.v5 = load_evaluation_protocol("scene_default_v5")

    def test_v5_is_an_isolated_collision_upgrade(self) -> None:
        self.assertEqual(
            "collision_1d_state_v3",
            self.v4["scenes"]["collision_1d"]["type"],
        )
        self.assertEqual(
            "collision_1d_state_v5",
            self.v5["scenes"]["collision_1d"]["type"],
        )
        self.assertEqual(self.v4["robustness"], self.v5["robustness"])
        for scene_id in (
            "pendulum",
            "free_fall",
            "inclined_plane_slide",
            "uniform_circular_motion",
        ):
            self.assertEqual(
                self.v4["scenes"][scene_id],
                self.v5["scenes"][scene_id],
            )

    def test_v3_and_v4_identities_remain_frozen(self) -> None:
        self.assertEqual(V3_PROTOCOL_FINGERPRINT, self.v3["fingerprint"])
        self.assertEqual(V4_PROTOCOL_FINGERPRINT, self.v4["fingerprint"])
        v4_collision = SceneEvaluatorRegistry(self.v4).resolve(
            "collision_1d"
        )
        self.assertEqual(
            V4_COLLISION_FINGERPRINT,
            v4_collision.describe()["fingerprint"],
        )
        self.assertEqual("1.4", v4_collision.describe()["version"])

    def test_registry_routes_v5_to_the_open_world_evaluator(self) -> None:
        registry = SceneEvaluatorRegistry(self.v5)
        evaluator = registry.resolve("collision_1d")
        self.assertIsInstance(evaluator, CollisionOpenWorldCaseEvaluator)
        self.assertIs(evaluator, registry.resolve("collision_1d"))
        description = evaluator.describe()
        self.assertEqual("2.1", description["version"])
        self.assertEqual(
            "collision_1d_open_world_nbody",
            description["id"],
        )
        self.assertEqual(
            "case_entity_manifest_not_scene_constant",
            description["observation"]["cardinality"],
        )
        self.assertNotEqual(
            V4_COLLISION_FINGERPRINT,
            description["fingerprint"],
        )

    def test_v5_config_pins_every_new_algorithmic_component(self) -> None:
        collision = self.v5["scenes"]["collision_1d"]
        extraction = NBodyExtractionConfig(
            **collision["nbody_extraction"]
        )
        scoring = NBodyScoringConfig(**collision["nbody_scoring"])
        self.assertEqual(2, extraction.velocity_window_frames)
        self.assertAlmostEqual(
            1.0,
            sum(scoring.component_weights().values()),
        )
        content_weights = collision["object_centric_scoring"][
            "content_weights"
        ]
        self.assertEqual(
            {"nbody_physics", "shape", "appearance"},
            set(content_weights),
        )
        self.assertAlmostEqual(1.0, sum(content_weights.values()))
        observation = collision["multi_frame_observation"]
        self.assertEqual(32, observation["maximum_residual_tracks"])
        self.assertEqual(32.0, observation["maximum_entity_y_spread_px"])
        self.assertFalse(
            any(
                role in key
                for key in observation
                for role in ("striker", "target")
            )
        )
        self.assertGreater(
            observation["residual_maximum_gap_s"],
            0.0,
        )
        self.assertEqual(
            3,
            observation["residual_hough_confirmation_frames"],
        )
        self.assertEqual(
            3,
            observation["residual_motion_confirmation_frames"],
        )
        self.assertEqual(
            2.0,
            observation[
                "residual_motion_minimum_displacement_radius_ratio"
            ],
        )
        self.assertLess(
            observation["residual_minimum_radius_ratio"],
            1.0,
        )
        self.assertGreater(
            observation["residual_maximum_radius_ratio"],
            1.0,
        )
        self.assertLess(
            observation["residual_direct_duplicate_center_fraction"],
            1.0,
        )

    def test_protocol_schema_declares_and_requires_v5_contracts(
        self,
    ) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "v2"
                / "evaluation_protocol.schema.json"
            ).read_text(encoding="utf-8")
        )
        reference = schema["$defs"]["referenceScene"]
        self.assertIn(
            "collision_1d_state_v5",
            reference["properties"]["type"]["enum"],
        )
        v5_branch = next(
            branch
            for branch in reference["allOf"]
            if branch["if"]["properties"]["type"].get("const")
            == "collision_1d_state_v5"
        )
        self.assertEqual(
            {
                "sam2",
                "multi_frame_observation",
                "quality",
                "nbody_extraction",
                "nbody_scoring",
                "object_centric_scoring",
            },
            set(v5_branch["then"]["required"]),
        )
        self.assertEqual(
            {
                "nbody_physics",
                "shape",
                "appearance",
            },
            set(
                schema["$defs"]["objectCentricScoring"]["properties"][
                    "content_weights"
                ]["required"]
            ),
        )

    def test_open_world_observer_is_a_public_optional_api(self) -> None:
        expected = (
            EvidenceTier,
            ObjectDetection,
            OpenWorldComparison,
            OpenWorldObservation,
            OpenWorldTrack,
            compare_open_world_tracks,
            deduplicate_frame_detections,
            detections_from_instance_masks,
            track_open_world_detections,
        )
        self.assertTrue(all(value is not None for value in expected))
        project = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        optional = project["project"]["optional-dependencies"]
        self.assertTrue(
            any(
                value.startswith("scipy>=")
                for value in optional["scene-evaluation"]
            )
        )
        # The legacy pendulum-only alias does not expose the v5 observer.
        # ReferenceCaseEvaluator imports the timeline module directly, so the
        # alias need not grow an unrelated Hungarian-assignment dependency.
        self.assertFalse(
            any(
                value.startswith("scipy>=")
                for value in optional["pendulum-evaluation"]
            )
        )

    def test_v5_protocol_file_is_canonical_loader_input(self) -> None:
        path = Path(self.v5["path"])
        self.assertEqual("scene_default_v5.json", path.name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("path", raw)
        self.assertNotIn("fingerprint", raw)
        self.assertEqual("scene_default_v5", raw["protocol_id"])


if __name__ == "__main__":
    unittest.main()
