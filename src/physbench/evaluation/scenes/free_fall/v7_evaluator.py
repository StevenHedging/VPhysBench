from __future__ import annotations

from .v6_evaluator import FreeFallOpenWorldCaseEvaluator


class FreeFallOpenWorldCaseEvaluatorV7(
    FreeFallOpenWorldCaseEvaluator
):
    """V7 free-fall evaluator with symmetric condition-causal observation."""

    evaluator_version = "2.1"
    sequential_evaluator_version = "2.1"
    robust_evaluator_version = "2.1"
    temporary_prefix = "physbench_free_fall_v7_"
    required_observer_version = "2.1"
    required_observer_policies = {
        "condition_anchor_policy": (
            "condition_only_ball_photometric_all_modes_v3"
        ),
        "condition_axis_policy": "condition_scene_geometry_causal_v2",
        "reference_geometry_policy": (
            "condition_frozen_same_case_coordinates_v2"
        ),
        "state_reference_geometry_policy": (
            "condition_frozen_same_case_state_v2"
        ),
        "exclusive_tracking_partition_policy": (
            "condition_directed_vs_residual_v1"
        ),
        "condition_directed_recovery_policy": (
            "anchor_or_release_geometry_bidirectional_v2"
        ),
        "directed_component_selection_policy": (
            "release_aware_ball_objectness_v3"
        ),
        "free_fall_release_site_apparatus_policy": (
            "condition_anchor_static_track_v1"
        ),
    }


__all__ = ["FreeFallOpenWorldCaseEvaluatorV7"]
