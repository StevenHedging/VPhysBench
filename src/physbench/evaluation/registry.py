from __future__ import annotations

from typing import Any

from .contracts import SceneCaseEvaluator
from .scenes.unsupported import UnsupportedSceneEvaluator


class SceneEvaluatorRegistry:
    """Resolve and reuse one evaluator instance per scene."""

    def __init__(self, protocol: dict[str, Any]):
        self.protocol = protocol
        self._instances: dict[str, SceneCaseEvaluator] = {}

    def resolve(self, scene_id: str) -> SceneCaseEvaluator:
        if scene_id in self._instances:
            return self._instances[scene_id]
        config = self.protocol["scenes"].get(
            scene_id, {"type": "unsupported"}
        )
        evaluator_type = config.get("type", "unsupported")
        if evaluator_type in {
            "pendulum_state_v1",
            "pendulum_state_v2",
            "pendulum_state_v3",
        }:
            from .scenes.pendulum.evaluator import PendulumCaseEvaluator

            evaluator: SceneCaseEvaluator = PendulumCaseEvaluator(config)
        elif evaluator_type in {"free_fall_state_v1", "free_fall_state_v2"}:
            from .scenes.free_fall.evaluator import FreeFallCaseEvaluator

            evaluator = FreeFallCaseEvaluator(config)
        elif evaluator_type in {
            "inclined_plane_state_v1",
            "inclined_plane_state_v2",
        }:
            from .scenes.inclined_plane.evaluator import (
                InclinedPlaneCaseEvaluator,
            )

            evaluator = InclinedPlaneCaseEvaluator(config)
        elif evaluator_type in {
            "uniform_circular_motion_state_v1",
            "uniform_circular_motion_state_v2",
        }:
            from .scenes.circular_motion.evaluator import (
                CircularMotionCaseEvaluator,
            )

            evaluator = CircularMotionCaseEvaluator(config)
        elif evaluator_type in {
            "collision_1d_state_v1",
            "collision_1d_state_v2",
        }:
            from .scenes.collision.evaluator import CollisionCaseEvaluator

            evaluator = CollisionCaseEvaluator(config)
        elif evaluator_type == "unsupported":
            evaluator = UnsupportedSceneEvaluator(scene_id, config)
        else:
            raise ValueError(
                f"unknown evaluator type {evaluator_type!r} for scene {scene_id}"
            )
        self._instances[scene_id] = evaluator
        return evaluator

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {}
        for scene_id, config in sorted(self.protocol["scenes"].items()):
            instance = self._instances.get(scene_id)
            description[scene_id] = (
                {**instance.describe(), "loaded": True}
                if instance is not None
                else {
                    "scene_id": scene_id,
                    "type": config.get("type", "unsupported"),
                    "loaded": False,
                }
            )
        return description
