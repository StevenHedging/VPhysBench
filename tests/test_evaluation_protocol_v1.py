from __future__ import annotations

import json
import unittest
from copy import deepcopy

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # pragma: no cover - optional test dependency
    Draft202012Validator = None  # type: ignore[assignment,misc]

from _paths import ROOT
from physbench.evaluation import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.circular_motion.v7_evaluator import (
    CircularMotionOpenWorldCaseEvaluatorV7,
)
from physbench.evaluation.scenes.collision.v6_evaluator import (
    CollisionFailClosedCaseEvaluator,
)
from physbench.evaluation.scenes.inclined_plane.v7_evaluator import (
    InclinedPlaneOpenWorldCaseEvaluatorV7,
)
from physbench.evaluation.scenes.parabolic_motion.v2_evaluator import (
    ParabolicMotionCaseEvaluatorV2,
)
from physbench.evaluation.scenes.pendulum.v8_evaluator import (
    PendulumOpenWorldCaseEvaluatorV8,
)


class EvaluationProtocolV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_evaluation_protocol("scene_default_v1")

    def test_v1_routes_each_scene_to_the_latest_implementation(self) -> None:
        from physbench.evaluation.scenes.vertical_spring_oscillator.evaluator import (
            VerticalSpringOscillatorCaseEvaluator,
        )

        expected = {
            "pendulum": PendulumOpenWorldCaseEvaluatorV8,
            "collision_1d": CollisionFailClosedCaseEvaluator,
            "inclined_plane_slide": InclinedPlaneOpenWorldCaseEvaluatorV7,
            "uniform_circular_motion": CircularMotionOpenWorldCaseEvaluatorV7,
            "parabolic_motion": ParabolicMotionCaseEvaluatorV2,
            "vertical_spring_oscillator": VerticalSpringOscillatorCaseEvaluator,
        }
        registry = SceneEvaluatorRegistry(self.protocol)
        for scene_id, evaluator_class in expected.items():
            with self.subTest(scene_id=scene_id):
                evaluator = registry.resolve(scene_id)
                self.assertIsInstance(evaluator, evaluator_class)
                self.assertEqual("1.0", evaluator.evaluator_version)

    def test_v1_keeps_the_latest_fail_closed_scoring_contract(self) -> None:
        csti = self.protocol["general_metrics"]["csti"]
        self.assertEqual(
            "exact_full_tube_edt",
            csti["algorithm"],
        )
        self.assertEqual(1, csti["initial_frames_excluded"])
        self.assertEqual(
            "reference_tube_equivalent_diameter_v1",
            csti["spatial_tolerance_policy"],
        )
        self.assertEqual(0.5, csti["spatial_tolerance_radius_ratio"])
        self.assertNotIn("spatial_tolerance_fraction", csti)
        self.assertEqual(
            "frozen_dataset_subject_identity",
            self.protocol["scenes"]["pendulum"][
                "reference_observation_policy"
            ],
        )
        self.assertEqual(
            "fail_closed_v1",
            self.protocol["scenes"]["collision_1d"]["subject_identity"][
                "failure_policy"
            ],
        )
        self.assertEqual(
            "strict_multiplicative_v2",
            self.protocol["scenes"]["parabolic_motion"]["scoring"][
                "composition"
            ],
        )
        spring = self.protocol["scenes"]["vertical_spring_oscillator"]
        self.assertEqual(0.90, spring["quality"]["minimum_valid_frame_ratio"])
        self.assertEqual(0.30, spring["scoring"]["horizontal_trajectory_scale"])
        self.assertEqual(0.10, spring["topology"]["minimum_reference_score"])
        self.assertEqual(
            {
                "minimum_anchor_iou",
                "maximum_centroid_distance_radii",
                "minimum_anchor_area_ratio",
                "maximum_anchor_area_ratio",
            },
            set(spring["identity"]),
        )
        self.assertEqual(
            {
                "canny_low_threshold",
                "canny_high_threshold",
                "corridor_half_width_radius_ratio",
                "minimum_edge_pixels_per_row",
                "endpoint_height_radius_ratio",
                "boundary_exclusion_px",
                "connectivity_dilation_px",
                "minimum_corridor_height_radius_ratio",
                "minimum_connected_vertical_span_ratio",
                "minimum_reference_score",
            },
            set(spring["topology"]),
        )
        self.assertAlmostEqual(1.0, sum(spring["content_weights"].values()))

    @unittest.skipIf(
        Draft202012Validator is None,
        "jsonschema unavailable: install a Draft 2020-12 consumer",
    )
    def test_schema_requires_adaptive_csti_spatial_tolerance(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = deepcopy(self.protocol)
        candidate.pop("path")
        candidate.pop("fingerprint")
        csti = candidate["general_metrics"]["csti"]
        csti.pop("spatial_tolerance_fraction", None)
        csti["spatial_tolerance_policy"] = (
            "reference_tube_equivalent_diameter_v1"
        )
        csti["spatial_tolerance_radius_ratio"] = 0.5
        csti["initial_frames_excluded"] = 1
        assert Draft202012Validator is not None
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)

        self.assertEqual([], list(validator.iter_errors(candidate)))
        for key in (
            "spatial_tolerance_policy",
            "spatial_tolerance_radius_ratio",
        ):
            with self.subTest(missing=key):
                missing = deepcopy(candidate)
                missing["general_metrics"]["csti"].pop(key)
                self.assertTrue(list(validator.iter_errors(missing)))
        invalid = deepcopy(candidate)
        invalid["general_metrics"]["csti"][
            "spatial_tolerance_radius_ratio"
        ] = 0
        self.assertTrue(list(validator.iter_errors(invalid)))
        legacy = deepcopy(candidate)
        legacy["general_metrics"]["csti"][
            "spatial_tolerance_fraction"
        ] = 0.005
        self.assertTrue(list(validator.iter_errors(legacy)))

    def test_unversioned_schema_declares_only_public_v1_types(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        definitions = schema["$defs"]
        self.assertEqual(
            {
                "pendulum_v1",
                "collision_1d_v1",
                "inclined_plane_slide_v1",
                "uniform_circular_motion_v1",
                "parabolic_motion_v1",
                "vertical_spring_oscillator_v1",
            },
            {
                definitions[name]["allOf"][1]["properties"]["type"]["const"]
                for name in (
                    "pendulum",
                    "collision",
                    "inclinedPlane",
                    "circularMotion",
                    "parabolicMotion",
                    "verticalSpring",
                )
            },
        )

    def test_v1_schema_requires_the_sixth_public_scene_contract(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/evaluation_protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        scenes = schema["properties"]["scenes"]
        self.assertIn("vertical_spring_oscillator", scenes["required"])
        self.assertEqual(
            {"$ref": "#/$defs/verticalSpring"},
            scenes["properties"]["vertical_spring_oscillator"],
        )
        spring = schema["$defs"]["verticalSpring"]
        required = set(spring["allOf"][1]["required"])
        self.assertEqual(
            {
                "type",
                "evaluator_contract",
                "timeline",
                "spatial",
                "sam2",
                "quality",
                "period",
                "scoring",
                "subject_scoring",
                "identity",
                "topology",
                "content_weights",
            },
            required,
        )


if __name__ == "__main__":
    unittest.main()
