from __future__ import annotations

from typing import Any

from .v6_evaluator import CircularMotionOpenWorldCaseEvaluator


class CircularMotionOpenWorldCaseEvaluatorV7(
    CircularMotionOpenWorldCaseEvaluator
):
    """V7 circular evaluator with hue-invariant apparatus discovery."""

    evaluator_version = "1.0"
    sequential_evaluator_version = "1.0"
    robust_evaluator_version = "1.0"

    def __init__(self, config: dict[str, Any]):
        observation = config.get("color_observation", {})
        required_observation = {
            "disk_colour_policy": "adaptive_dominant_hue",
            "open_world_component_shape_filter": (
                "compact_participant_v1"
            ),
            "apparatus_coordinate_policy": (
                "condition_frozen_shared_frame_v1"
            ),
            "apparatus_appearance_policy": (
                "condition_lab_value_temporal_residual_v1"
            ),
            "apparatus_palette_frame_shift_policy": (
                "condition_support_robust_median_v1"
            ),
            "coordinate_phase_policy": (
                "condition_frozen_absolute_phase_v1"
            ),
            "physics_parent_phase_policy": (
                "separate_condition_relative_dynamics_v1"
            ),
        }
        mismatches = {
            key: observation.get(key)
            for key, expected in required_observation.items()
            if observation.get(key) != expected
        }
        if config.get("reference_observation_policy") != (
            "apparatus_geometry_symmetric"
        ):
            mismatches["reference_observation_policy"] = config.get(
                "reference_observation_policy"
            )
        if mismatches:
            raise ValueError(
                "circular v7 observation policies require explicit opt-in: "
                f"{mismatches}"
            )
        super().__init__(config)

    def describe_observation(self) -> dict[str, Any]:
        return {
            **super().describe_observation(),
            "segmentation": (
                "multi_hue_circular_apparatus_geometry_then_all_interior_"
                "contrast_components"
            ),
            "apparatus_exclusion": (
                "condition_frozen_center_radius_with_condition_lab_value_"
                "temporal_apparatus_model_and_robust_global_illumination_"
                "alignment"
            ),
            "shape_policy": (
                "compactness_is_confidence_only_all_area_qualified_"
                "components_remain_auditable"
            ),
            "coordinate_system": (
                "condition_frozen_disk_absolute_polar_with_"
                "cartesian_center_fallback"
            ),
            "apparatus_drift": (
                "prediction_geometry_never_redefines_origin_scale_or_roi"
            ),
        }


__all__ = ["CircularMotionOpenWorldCaseEvaluatorV7"]
