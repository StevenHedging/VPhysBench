from __future__ import annotations

from typing import Any

from physbench.baseline_plugins.wan22_subject_anchor_trust import (
    Wan22SubjectAnchorTrustExecutionEngine,
)
from physbench.baselines.wan22_subject_geometry_trust import (
    Wan22SubjectGeometryTrustAdapter,
)


_GEOMETRY_CONFIG_KEYS = (
    "lambda_subject_mass",
    "lambda_subject_covariance",
    "lambda_framewise_iou",
    "geometry_eps",
    "geometry_smooth_l1_beta",
)


class Wan22SubjectGeometryTrustExecutionEngine(
    Wan22SubjectAnchorTrustExecutionEngine
):
    """Managed WAN engine for fixed-probe subject-geometry training."""

    def _adapter_class(self):
        return Wan22SubjectGeometryTrustAdapter

    def _legacy_config(self, instance_value: dict[str, Any]) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        trainer = config["lora"]
        config["adapter"] = "wan22_subject_geometry_trust"
        config["st_tube_iou"].update({
            key: trainer[key] for key in _GEOMETRY_CONFIG_KEYS
        })
        return config


__all__ = ["Wan22SubjectGeometryTrustExecutionEngine"]
