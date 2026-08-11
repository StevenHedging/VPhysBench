from __future__ import annotations

from typing import Any

from .contracts import SceneCaseEvaluator
from .scenes.unsupported import UnsupportedSceneEvaluator


SUPPORTED_EVALUATOR_TYPES = frozenset(
    {
        "pendulum_v1",
        "collision_1d_v1",
        "inclined_plane_slide_v1",
        "uniform_circular_motion_v1",
        "parabolic_motion_v1",
        "unsupported",
    }
)


class SceneEvaluatorRegistry:
    """Resolve and reuse one evaluator instance per scene."""

    def __init__(self, protocol: dict[str, Any]):
        self.protocol = protocol
        self._instances: dict[str, SceneCaseEvaluator] = {}

    @staticmethod
    def supported_evaluator_types() -> frozenset[str]:
        """Return the evaluator types that can be resolved by this build."""

        return SUPPORTED_EVALUATOR_TYPES

    def resolve(self, scene_id: str) -> SceneCaseEvaluator:
        if scene_id in self._instances:
            return self._instances[scene_id]
        raw_config = self.protocol["scenes"].get(
            scene_id, {"type": "unsupported"}
        )
        config = dict(raw_config)
        if "general_metrics" in self.protocol:
            config["general_metrics"] = self.protocol["general_metrics"]
        evaluator_type = config.get("type", "unsupported")
        if evaluator_type == "pendulum_v1":
            from .scenes.pendulum.v8_evaluator import (
                PendulumOpenWorldCaseEvaluatorV8,
            )

            evaluator: SceneCaseEvaluator = PendulumOpenWorldCaseEvaluatorV8(
                config
            )
        elif evaluator_type == "inclined_plane_slide_v1":
            from .scenes.inclined_plane.v7_evaluator import (
                InclinedPlaneOpenWorldCaseEvaluatorV7,
            )

            evaluator = InclinedPlaneOpenWorldCaseEvaluatorV7(config)
        elif evaluator_type == "uniform_circular_motion_v1":
            from .scenes.circular_motion.v7_evaluator import (
                CircularMotionOpenWorldCaseEvaluatorV7,
            )

            evaluator = CircularMotionOpenWorldCaseEvaluatorV7(config)
        elif evaluator_type == "collision_1d_v1":
            from .scenes.collision.v6_evaluator import (
                CollisionFailClosedCaseEvaluator,
            )

            evaluator = CollisionFailClosedCaseEvaluator(config)
        elif evaluator_type == "parabolic_motion_v1":
            from .scenes.parabolic_motion.v2_evaluator import (
                ParabolicMotionCaseEvaluatorV2,
            )

            evaluator = ParabolicMotionCaseEvaluatorV2(config)
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
