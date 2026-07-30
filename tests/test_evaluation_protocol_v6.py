from __future__ import annotations

import json
import unittest

from _paths import ROOT
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.circular_motion.v6_evaluator import (
    CircularMotionOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.circular_motion.v7_evaluator import (
    CircularMotionOpenWorldCaseEvaluatorV7,
)
from physbench.evaluation.scenes.collision.v5_evaluator import (
    CollisionOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.free_fall.v6_evaluator import (
    FreeFallOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.free_fall.v7_evaluator import (
    FreeFallOpenWorldCaseEvaluatorV7,
)
from physbench.evaluation.scenes.inclined_plane.v6_evaluator import (
    InclinedPlaneOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.inclined_plane.v7_evaluator import (
    InclinedPlaneOpenWorldCaseEvaluatorV7,
)
from physbench.evaluation.scenes.pendulum.v6_evaluator import (
    PendulumOpenWorldCaseEvaluator,
)
from physbench.evaluation.scenes.pendulum.v7_evaluator import (
    PendulumOpenWorldCaseEvaluatorV7,
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
        cls.v7 = load_evaluation_protocol("scene_default_v7")

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

    def test_v7_circular_observer_is_opt_in_and_v6_remains_frozen(
        self,
    ) -> None:
        self.assertEqual(V6_PROTOCOL_FINGERPRINT, self.v6["fingerprint"])
        self.assertEqual(
            self.v6["scenes"]["collision_1d"],
            self.v7["scenes"]["collision_1d"],
        )
        config = self.v7["scenes"]["uniform_circular_motion"]
        self.assertEqual(
            "uniform_circular_motion_state_v7", config["type"]
        )
        self.assertEqual(
            "apparatus_geometry_symmetric",
            config["reference_observation_policy"],
        )
        observation = config["color_observation"]
        self.assertEqual(
            "adaptive_dominant_hue",
            observation["disk_colour_policy"],
        )
        self.assertEqual(
            "compact_participant_v1",
            observation["open_world_component_shape_filter"],
        )
        self.assertEqual(
            "condition_support_robust_median_v1",
            observation["apparatus_palette_frame_shift_policy"],
        )
        self.assertEqual(
            45.0,
            observation["apparatus_palette_maximum_frame_shift_lab"],
        )
        evaluator = SceneEvaluatorRegistry(self.v7).resolve(
            "uniform_circular_motion"
        )
        self.assertIsInstance(
            evaluator, CircularMotionOpenWorldCaseEvaluatorV7
        )
        self.assertEqual("2.1", evaluator.describe()["version"])
        pendulum = self.v7["scenes"]["pendulum"]
        self.assertEqual("pendulum_state_v7", pendulum["type"])
        self.assertEqual(
            "condition_causal_symmetric",
            pendulum["reference_observation_policy"],
        )
        pendulum_evaluator = SceneEvaluatorRegistry(self.v7).resolve(
            "pendulum"
        )
        self.assertIsInstance(
            pendulum_evaluator, PendulumOpenWorldCaseEvaluatorV7
        )
        self.assertEqual(
            "2.1", pendulum_evaluator.describe()["version"]
        )
        rigid = {
            "free_fall": (
                "free_fall_state_v7",
                FreeFallOpenWorldCaseEvaluatorV7,
            ),
            "inclined_plane_slide": (
                "inclined_plane_state_v7",
                InclinedPlaneOpenWorldCaseEvaluatorV7,
            ),
        }
        registry = SceneEvaluatorRegistry(self.v7)
        for scene_id, (evaluator_type, evaluator_class) in rigid.items():
            with self.subTest(scene_id=scene_id):
                scene = self.v7["scenes"][scene_id]
                self.assertEqual(evaluator_type, scene["type"])
                self.assertEqual(
                    "condition_causal_symmetric",
                    scene["reference_observation_policy"],
                )
                self.assertEqual(
                    "condition_frame_apparatus_only_v1",
                    scene["rigid_body_observation"][
                        "compact_only_evidence_policy"
                    ],
                )
                self.assertEqual(
                    "condition_directed_vs_residual_v1",
                    scene["rigid_body_observation"][
                        "exclusive_tracking_partition_policy"
                    ],
                )
                self.assertNotIn(
                    "exclusive_tracking_partition_policy",
                    self.v6["scenes"][scene_id][
                        "rigid_body_observation"
                    ],
                )
                self.assertIsInstance(
                    registry.resolve(scene_id), evaluator_class
                )
                self.assertEqual(
                    "2.1",
                    registry.resolve(scene_id).describe()["version"],
                )

    def test_v7_behavioral_thresholds_are_protocol_explicit(self) -> None:
        pendulum = self.v7["scenes"]["pendulum"]
        observation = pendulum["open_world_observation"]
        expected_observation = {
            "v7_initial_angle_scale_deg": 7.5,
            "v7_minimum_line_agreement": 0.25,
            "v7_minimum_ratio_agreement": 0.55,
            "v7_minimum_edge_agreement": 0.18,
            "v7_minimum_body_agreement": 0.1,
            "v7_minimum_angle_agreement": 0.35,
            "v7_minimum_source_agreement": 3,
            "v7_minimum_condition_score": 0.3,
            "v7_condition_ambiguity_margin": 0.005,
            "v7_condition_low_margin_warning": 0.03,
            "v7_condition_minimum_visible_angle_deg": 7.0,
            "v7_condition_support_continuation_minimum_alignment": 0.9,
            "v7_condition_support_continuation_maximum_axis_distance_radius_ratio": 1.2,
            "v7_condition_support_continuation_minimum_extension_radius_ratio": 2.5,
            "v7_condition_pivot_extension_minimum_length_fraction": 0.78,
            "v7_condition_pivot_extension_maximum_scale": 4.0,
            "v7_condition_pivot_extension_canvas_margin_px": 4.0,
            "v7_condition_minimum_physics_ratio_score": 0.5,
            "v7_condition_full_string_half_width_px": 2,
            "v7_condition_full_string_edge_weight": 0.15,
            "v7_condition_pivot_extension_log_penalty": 0.12,
            "v7_minimum_residual_length_ratio": 0.6,
            "v7_maximum_residual_length_ratio": 1.35,
            "v7_minimum_residual_color_similarity": 0.15,
            "v7_minimum_residual_edge_support": 0.2,
            "v7_minimum_residual_body_contrast": 0.12,
            "v7_anchor_copy_directed_departure_radius_ratio": 2.5,
            "v7_anchor_copy_center_radius_ratio": 1.25,
            "v7_residual_minimum_frames": 3,
            "v7_residual_minimum_duration_s": 0.12,
            "v7_directed_minimum_length_ratio": 0.6,
            "v7_directed_maximum_length_ratio": 1.35,
            "v7_directed_condition_anchor_maximum_distance_radii": 2.5,
            "v7_directed_maximum_jump_radius_ratio": 5.0,
            "v7_directed_maximum_velocity_error_radius_ratio": 4.0,
            "v7_directed_maximum_recovery_gap_steps": 3.0,
            "v7_condition_preexistence_search_radius_px": 3,
            "v7_condition_preexistence_patch_radius_ratio": 1.5,
            "v7_condition_preexistence_rejection_threshold": 0.84,
            "v7_residual_minimum_directed_subject_overlap": 0.5,
            "v7_recovery_minimum_edge_support": 0.6,
            "v7_recovery_minimum_body_contrast": 0.55,
            "v7_recovery_minimum_color_similarity": 0.45,
            "v7_recovery_maximum_velocity_error_radius_ratio": 2.0,
            "v7_terminal_repair_directed_maximum_area_ratio": 0.65,
            "v7_terminal_repair_residual_minimum_area_ratio": 0.5,
            "v7_terminal_repair_residual_maximum_area_ratio": 1.8,
            "v7_terminal_repair_minimum_length_ratio": 0.78,
            "v7_terminal_repair_maximum_length_ratio": 1.2,
            "v7_terminal_repair_minimum_length_improvement": 0.025,
            "v7_terminal_repair_minimum_alignment": 0.965,
            "v7_terminal_repair_maximum_perpendicular_radii": 1.25,
            "v7_terminal_repair_maximum_separation_radii": 3.5,
            "v7_fusion_center_radius_ratio": 0.65,
            "v7_fusion_minimum_mask_iou": 0.18,
            "v7_fusion_minimum_containment": 0.55,
            "v7_near_directed_residual_suppression_radius": 2.0,
            "v7_low_matched_coverage_warning": 0.8,
            "v7_residual_fragmentation_warning_tracks": 3,
        }
        self.assertEqual(
            expected_observation,
            {
                key: observation[key]
                for key in expected_observation
            },
        )
        expected_topology = {
            "v7_topology_persistence_frames": 3,
            "v7_topology_source_disagreement_threshold": 0.45,
            "v7_broken_string_threshold": 0.55,
            "v7_branch_threshold": 0.2,
        }
        self.assertEqual(
            expected_topology,
            {
                key: pendulum["topology"][key]
                for key in expected_topology
            },
        )

        circular = self.v7["scenes"]["uniform_circular_motion"][
            "color_observation"
        ]
        circular_explicit = {
            "disk_colour_policy",
            "apparatus_coordinate_policy",
            "apparatus_appearance_policy",
            "coordinate_phase_policy",
            "physics_parent_phase_policy",
            "apparatus_palette_lab_quantization",
            "apparatus_palette_maximum_colours",
            "apparatus_palette_minimum_fraction",
            "apparatus_palette_maximum_lab_distance",
            "apparatus_condition_maximum_lab_distance",
            "apparatus_temporal_minimum_lab_distance",
            "apparatus_temporal_condition_minimum_lab_distance",
            "adaptive_disk_minimum_saturation",
            "adaptive_disk_minimum_value",
            "adaptive_disk_hue_smoothing_radius",
            "adaptive_disk_hue_tolerance",
            "adaptive_disk_maximum_hue_hypotheses",
            "adaptive_disk_close_kernel",
            "adaptive_disk_open_kernel",
            "adaptive_disk_erosion_kernel",
            "adaptive_disk_maximum_area_ratio",
            "adaptive_disk_minimum_radius_ratio",
            "adaptive_disk_maximum_radius_ratio",
            "disk_erosion_kernel",
            "minimum_disk_area_ratio",
            "minimum_component_area",
            "maximum_component_area_ratio",
            "open_world_component_shape_filter",
            "open_world_minimum_component_disk_area_ratio",
            "open_world_minimum_component_rectangularity",
            "open_world_minimum_component_solidity",
            "open_world_maximum_component_aspect_ratio",
            "open_world_minimum_component_circularity",
            "open_world_maximum_candidates_per_frame",
            "open_world_maximum_tracks",
            "open_world_maximum_gap_s",
            "open_world_maximum_assignment_cost",
            "open_world_minimum_scale_px",
            "open_world_area_weight",
            "open_world_appearance_switch_threshold_lab",
            "center_fallback_radius_ratio",
            "open_world_minimum_reference_disk_valid_ratio",
            "open_world_minimum_reference_track_ratio",
        }
        self.assertTrue(circular_explicit.issubset(circular))
        self.assertEqual(
            45.0,
            circular["open_world_appearance_switch_threshold_lab"],
        )
        self.assertEqual(
            0.08,
            circular["center_fallback_radius_ratio"],
        )

        rigid_explicit = {
            "reference_hypothesis_policy",
            "reference_hypothesis_maximum_candidates",
            "reference_candidate_maximum_area_ratio",
            "reference_candidate_minimum_compact_score",
            "reference_candidate_maximum_aspect_ratio",
            "reference_hypothesis_log_area_mad_scale",
            "reference_hypothesis_minimum_condition_shape",
            "reference_hypothesis_minimum_median_shape",
            "reference_hypothesis_minimum_continuity",
            "condition_anchor_policy",
            "exclusive_tracking_partition_policy",
            "directed_identity_swept_exclusion_policy",
            "directed_identity_swept_bridge_body_fraction",
            "directed_identity_swept_maximum_bridge_radii",
            "coupled_optical_artifact_policy",
            "coupled_artifact_minimum_overlap_frames",
            "coupled_artifact_maximum_compact_support_ratio",
            "coupled_artifact_maximum_offset_std_radii",
            "coupled_artifact_maximum_velocity_error_radii",
            "coupled_artifact_minimum_offset_radii",
            "coupled_artifact_maximum_offset_radii",
            "motion_only_participant_minimum_frames",
            "condition_apparatus_maximum_fragment_span_radii",
            "condition_apparatus_maximum_anchor_distance_radii",
            "condition_apparatus_minimum_compact_support_ratio",
            "condition_present_apparatus_policy",
            "condition_present_apparatus_maximum_span_radii",
            "directed_recovery_policy",
            "directed_recovery_maximum_gap_frames",
            "directed_recovery_maximum_distance_radii",
            "directed_recovery_minimum_area_ratio",
            "directed_recovery_maximum_area_ratio",
            "directed_recovery_minimum_color_similarity",
            "dependent_change_only_policy",
            "dependent_change_minimum_objectness",
            "compact_only_evidence_policy",
            "condition_frame_apparatus_maximum_change_fraction",
            "condition_frame_apparatus_maximum_subject_containment",
            "condition_difference_threshold",
            "temporal_difference_threshold",
            "minimum_area_ratio",
            "maximum_area_ratio",
            "direct_minimum_area_ratio",
            "direct_maximum_area_ratio",
            "minimum_anchor_color_similarity",
            "minimum_duplicate_color_similarity",
            "minimum_weak_residual_color_similarity",
            "identity_rejection_minimum_consecutive_frames",
            "identity_rejection_policy",
            "identity_continuity_maximum_prediction_error_radii",
            "identity_continuity_maximum_initial_jump_radii",
            "identity_continuity_maximum_area_ratio",
            "outside_roi_minimum_anchor_color_similarity",
            "outside_roi_minimum_area_ratio",
            "outside_roi_maximum_area_ratio",
            "roi_half_width_body_diameters",
            "roi_minimum_half_width_fraction",
            "residual_dilation_body_fraction",
            "maximum_gap_s",
            "maximum_assignment_cost",
            "maximum_tracks",
            "exit_minimum_trailing_frames",
            "exit_minimum_progress",
            "exit_minimum_forward_step_ratio",
            "exit_minimum_canvas_span_fraction",
            "exit_boundary_margin_body_radii",
            "exit_boundary_margin_fraction",
            "state_coverage_power",
        }
        rigid_scoring = {
            "free_fall": {
                "trajectory_error_scale",
                "acceleration_error_scale",
                "impact_time_error_scale",
                "horizontal_drift_scale",
                "constraint_reference_mode",
                "empirical_step_error_scale",
            },
            "inclined_plane_slide": {
                "trajectory_error_scale",
                "acceleration_error_scale",
                "descent_time_error_scale",
                "cross_track_scale",
                "orientation_std_scale_deg",
                "constraint_reference_mode",
                "empirical_step_error_scale",
            },
        }
        for scene_id, scoring_keys in rigid_scoring.items():
            with self.subTest(scene_id=scene_id):
                scene = self.v7["scenes"][scene_id]
                self.assertTrue(
                    rigid_explicit.issubset(
                        scene["rigid_body_observation"]
                    )
                )
                self.assertTrue(scoring_keys.issubset(scene["scoring"]))
        self.assertTrue(
            {"minimum_circularity", "maximum_circle_aspect_ratio"}.issubset(
                self.v7["scenes"]["free_fall"][
                    "rigid_body_observation"
                ]
            )
        )
        self.assertTrue(
            {
                "minimum_rectangularity",
                "maximum_rectangle_aspect_ratio",
            }.issubset(
                self.v7["scenes"]["inclined_plane_slide"][
                    "rigid_body_observation"
                ]
            )
        )

    def test_v7_omits_retired_single_object_selector_fields(self) -> None:
        pendulum = self.v7["scenes"]["pendulum"][
            "open_world_observation"
        ]
        self.assertTrue(
            {
                "minimum_condition_bob_angle_deg",
                "minimum_condition_structure_score",
                "direct_duplicate_overlap",
                "minimum_residual_color_similarity",
            }.isdisjoint(pendulum)
        )
        self.assertNotIn(
            "branch_penalty_weight",
            self.v7["scenes"]["pendulum"]["topology"],
        )
        circular = self.v7["scenes"]["uniform_circular_motion"][
            "color_observation"
        ]
        self.assertTrue(
            {
                "green_hsv_lower",
                "green_hsv_upper",
                "minimum_valid_frame_ratio",
                "maximum_candidates",
            }.isdisjoint(circular)
        )

    def test_v7_config_changes_do_not_mutate_frozen_protocols(self) -> None:
        expected = {
            "scene_default_v3": (
                self.v3,
                V3_PROTOCOL_FINGERPRINT,
            ),
            "scene_default_v4": (
                self.v4,
                V4_PROTOCOL_FINGERPRINT,
            ),
            "scene_default_v5": (
                self.v5,
                V5_PROTOCOL_FINGERPRINT,
            ),
            "scene_default_v6": (
                self.v6,
                V6_PROTOCOL_FINGERPRINT,
            ),
        }
        for protocol_id, (protocol, fingerprint) in expected.items():
            with self.subTest(protocol_id=protocol_id):
                self.assertEqual(fingerprint, protocol["fingerprint"])
        self.assertEqual(
            self.v5["scenes"]["collision_1d"],
            self.v6["scenes"]["collision_1d"],
        )
        self.assertEqual(
            self.v6["scenes"]["collision_1d"],
            self.v7["scenes"]["collision_1d"],
        )
        collision = SceneEvaluatorRegistry(self.v7).resolve(
            "collision_1d"
        )
        self.assertEqual("2.2", collision.describe()["version"])
        self.assertEqual(
            V5_COLLISION_FINGERPRINT,
            collision.describe()["fingerprint"],
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
                "pendulum_state_v7",
                "free_fall_state_v6",
                "free_fall_state_v7",
                "inclined_plane_state_v6",
                "inclined_plane_state_v7",
                "uniform_circular_motion_state_v6",
                "uniform_circular_motion_state_v7",
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
            "openWorldV2PendulumScene": {
                "pendulum_state_v6",
                "pendulum_state_v7",
            },
            "openWorldV2FreeFallScene": {
                "free_fall_state_v6",
                "free_fall_state_v7",
            },
            "openWorldV2InclinedPlaneScene": {
                "inclined_plane_state_v6",
                "inclined_plane_state_v7",
            },
            "openWorldV2CircularMotionScene": {
                "uniform_circular_motion_state_v6",
                "uniform_circular_motion_state_v7",
            },
        }
        for definition, evaluator_types in expected_types.items():
            self.assertEqual(
                evaluator_types,
                set(
                    schema["$defs"][definition]["allOf"][1][
                        "properties"
                    ]["type"]["enum"]
                ),
            )

    def test_every_schema_evaluator_type_is_registry_resolvable(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "v2"
                / "evaluation_protocol.schema.json"
            ).read_text(encoding="utf-8")
        )
        schema_types = set(
            schema["$defs"]["referenceScene"]["properties"]["type"]["enum"]
        )
        schema_types.update(
            schema["$defs"]["pendulum"]["properties"]["type"]["enum"]
        )
        schema_types.update(
            schema["$defs"]["openWorldV2Scene"]["properties"]["type"][
                "enum"
            ]
        )
        registry_types = SceneEvaluatorRegistry.supported_evaluator_types()
        self.assertEqual(
            schema_types,
            set(registry_types) - {"unsupported"},
        )


if __name__ == "__main__":
    unittest.main()
