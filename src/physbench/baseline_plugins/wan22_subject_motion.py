from __future__ import annotations

from typing import Any

from physbench.baseline_plugins.wan22 import Wan22ExecutionEngine
from physbench.baselines.wan22_subject_motion import Wan22SubjectMotionAdapter


_SUBJECT_MOTION_CONFIG_KEYS = (
    "enable_st_iou_loss",
    "lambda_st",
    "lambda_subject_flow",
    "lambda_motion_delta",
    "st_iou_eps",
    "subject_flow_eps",
    "motion_delta_eps",
    "st_loss_weighting",
    "st_noise_threshold",
    "aux_warmup_steps",
    "latent_channels",
    "hidden_channels",
    "mask_segmenter_model_id",
    "mask_supervision",
)


class Wan22SubjectMotionExecutionEngine(Wan22ExecutionEngine):
    """Managed WAN engine for warm-start subject-motion fine-tuning."""

    def _adapter_class(self):
        return Wan22SubjectMotionAdapter

    def _legacy_config(self, instance_value: dict[str, Any]) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        trainer = config["lora"]
        runtime = config["runtime"]
        warm_start = runtime.get("warm_start_checkpoint")
        if not isinstance(warm_start, str) or not warm_start.strip():
            raise ValueError(
                "runtime.warm_start_checkpoint is required for subject-motion training"
            )
        config.update({
            "adapter": "wan22_subject_motion",
            "initial_lora_checkpoint": warm_start,
            "st_tube_iou": {
                key: trainer[key] for key in _SUBJECT_MOTION_CONFIG_KEYS
            },
        })
        return config


__all__ = ["Wan22SubjectMotionExecutionEngine"]
