from __future__ import annotations

from typing import Any

from physbench.baseline_plugins.wan22 import Wan22ExecutionEngine
from physbench.baselines.wan22_st_tube_iou import Wan22STTubeIoULoraAdapter


_ST_CONFIG_KEYS = (
    "enable_st_iou_loss",
    "lambda_st",
    "st_iou_eps",
    "st_loss_weighting",
    "st_noise_threshold",
    "st_loss_warmup_steps",
    "latent_channels",
    "hidden_channels",
    "mask_segmenter_model_id",
    "mask_supervision",
)


class Wan22STTubeIoUExecutionEngine(Wan22ExecutionEngine):
    """Managed WAN engine with a training-only tube-IoU objective."""

    def _adapter_class(self):
        return Wan22STTubeIoULoraAdapter

    def _legacy_config(self, instance_value: dict[str, Any]) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        trainer = config["lora"]
        config.update({
            "adapter": "wan22_st_tube_iou",
            "st_tube_iou": {key: trainer[key] for key in _ST_CONFIG_KEYS},
        })
        return config


__all__ = ["Wan22STTubeIoUExecutionEngine"]
