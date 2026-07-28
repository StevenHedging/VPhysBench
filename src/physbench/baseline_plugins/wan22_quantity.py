from __future__ import annotations

from pathlib import Path
from typing import Any

from physbench.baseline_plugins.wan22 import Wan22ExecutionEngine
from physbench.baselines.wan22_quantity import Wan22QuantityLoraAdapter


class Wan22QuantityExecutionEngine(Wan22ExecutionEngine):
    """Managed WAN engine that preserves the structured quantity channel."""

    def _adapter_class(self):
        return Wan22QuantityLoraAdapter

    def _worker_script(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "wan22_quantity_generate_batch.py"
        )

    def _legacy_config(
        self,
        instance_value: dict[str, Any],
    ) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        config.update({
            "adapter": "wan22_quantity_embedding",
            "quantity_encoder": self.bundle.value["model"][
                "quantity_encoder"
            ],
        })
        return config

    def _finalize_prepared_job(
        self,
        prepared: dict[str, Any],
        raw_job: dict[str, Any],
    ) -> None:
        expected = raw_job["native_inputs"]["physics"]
        actual = prepared["model_input"].get("quantity_payload")
        if actual != expected:
            raise AssertionError(
                "quantity adapter changed the sealed model-side payload"
            )
        if "physical_parameters" in prepared["model_input"]:
            raise AssertionError(
                "quantity model input contains an undeclared physics channel"
            )


__all__ = ["Wan22QuantityExecutionEngine"]
