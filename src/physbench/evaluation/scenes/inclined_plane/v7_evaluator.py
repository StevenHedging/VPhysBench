from __future__ import annotations

from .v6_evaluator import InclinedPlaneOpenWorldCaseEvaluator


class InclinedPlaneOpenWorldCaseEvaluatorV7(
    InclinedPlaneOpenWorldCaseEvaluator
):
    """V7 incline evaluator with symmetric condition-causal observation."""

    evaluator_version = "2.1"
    sequential_evaluator_version = "2.1"
    robust_evaluator_version = "2.1"
    temporary_prefix = "physbench_inclined_plane_v7_"
    required_observer_version = "2.1"
    required_observer_policies = {
        "condition_anchor_policy": (
            "condition_only_compact_geometry_all_modes_v2"
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
            "anchor_validated_bidirectional_hypotheses_v1"
        ),
    }


__all__ = ["InclinedPlaneOpenWorldCaseEvaluatorV7"]
