from __future__ import annotations

from ..rigid_body_open_world import RigidBodyOpenWorldCaseEvaluatorBase


class FreeFallOpenWorldCaseEvaluator(RigidBodyOpenWorldCaseEvaluatorBase):
    """V6 free-fall evaluator with open-world ball cardinality auditing."""

    evaluator_id = "free_fall_open_world"
    evaluator_version = "2.0"
    scene_id = "free_fall"
    primary_score = "free_fall_open_world_similarity"
    open_world_metric = "free_fall_open_world_similarity"
    scene_kind = "free_fall"
    entity_class = "ball"
    scene_name = "Open-world free fall"
    temporary_prefix = "physbench_free_fall_v6_"
    minimum_span_quality_key = "minimum_vertical_span_px"


__all__ = ["FreeFallOpenWorldCaseEvaluator"]
