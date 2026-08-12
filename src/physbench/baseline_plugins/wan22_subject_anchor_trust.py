from __future__ import annotations

from typing import Any

from physbench.baseline_plugins.wan22_subject_motion import (
    Wan22SubjectMotionExecutionEngine,
)
from physbench.baselines.wan22_subject_anchor_trust import (
    Wan22SubjectAnchorTrustAdapter,
)


_ANCHOR_TRUST_CONFIG_KEYS = (
    "freeze_occupancy_head",
    "lambda_anchored_displacement",
    "lambda_lora_trust_region",
    "trajectory_eps",
    "trajectory_smooth_l1_beta",
    "lora_trust_region_eps",
)


class Wan22SubjectAnchorTrustExecutionEngine(Wan22SubjectMotionExecutionEngine):
    """Managed WAN engine for fixed-probe anchored-motion training."""

    def _adapter_class(self):
        return Wan22SubjectAnchorTrustAdapter

    def _legacy_config(self, instance_value: dict[str, Any]) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        trainer = config["lora"]
        config["adapter"] = "wan22_subject_anchor_trust"
        config["st_tube_iou"].update({
            key: trainer[key] for key in _ANCHOR_TRUST_CONFIG_KEYS
        })
        return config


__all__ = ["Wan22SubjectAnchorTrustExecutionEngine"]
