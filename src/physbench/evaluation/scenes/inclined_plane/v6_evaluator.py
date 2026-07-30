from __future__ import annotations

from ..rigid_body_open_world import RigidBodyOpenWorldCaseEvaluatorBase


class InclinedPlaneOpenWorldCaseEvaluator(
    RigidBodyOpenWorldCaseEvaluatorBase
):
    """V6 incline evaluator with open-world block cardinality auditing."""

    evaluator_id = "inclined_plane_open_world"
    evaluator_version = "2.0"
    scene_id = "inclined_plane_slide"
    primary_score = "inclined_plane_open_world_similarity"
    open_world_metric = "inclined_plane_open_world_similarity"
    scene_kind = "inclined_plane"
    entity_class = "block"
    scene_name = "Open-world inclined-plane slide"
    temporary_prefix = "physbench_inclined_plane_v6_"
    minimum_span_quality_key = "minimum_motion_span_px"


__all__ = ["InclinedPlaneOpenWorldCaseEvaluator"]
