from __future__ import annotations

from typing import Any

from physbench.baseline_plugins.wan22_subject_motion import (
    Wan22SubjectMotionExecutionEngine,
)
from physbench.baselines.wan22_subject_trajectory import (
    Wan22SubjectTrajectoryAdapter,
)


_TRAJECTORY_CONFIG_KEYS = (
    "lambda_centroid_position",
    "lambda_centroid_velocity",
    "trajectory_eps",
    "trajectory_smooth_l1_beta",
)


class Wan22SubjectTrajectoryExecutionEngine(Wan22SubjectMotionExecutionEngine):
    """Managed WAN engine for centroid-trajectory fine-tuning."""

    def _adapter_class(self):
        return Wan22SubjectTrajectoryAdapter

    def _legacy_config(self, instance_value: dict[str, Any]) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        trainer = config["lora"]
        config["adapter"] = "wan22_subject_trajectory"
        config["st_tube_iou"].update({
            key: trainer[key] for key in _TRAJECTORY_CONFIG_KEYS
        })
        return config


__all__ = ["Wan22SubjectTrajectoryExecutionEngine"]
