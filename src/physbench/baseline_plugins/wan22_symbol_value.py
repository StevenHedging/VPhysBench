from __future__ import annotations

from pathlib import Path
from typing import Any

from physbench.baseline_plugins.wan22 import Wan22ExecutionEngine
from physbench.baselines.wan22_symbol_value import Wan22SymbolValueLoraAdapter


class Wan22SymbolValueExecutionEngine(Wan22ExecutionEngine):
    """Managed WAN engine that preserves the sealed symbol/value channel."""

    def _adapter_class(self):
        return Wan22SymbolValueLoraAdapter

    def _worker_script(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "wan22_symbol_value_generate_batch.py"
        )

    def _legacy_config(self, instance_value: dict[str, Any]) -> dict[str, Any]:
        config = super()._legacy_config(instance_value)
        config.update({
            "adapter": "wan22_symbol_value_cross_attention",
            "symbol_value_conditioner": self.bundle.value["model"][
                "symbol_value_conditioner"
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
            raise AssertionError("symbol-value adapter changed the sealed payload")
        if "physical_parameters" in prepared["model_input"]:
            raise AssertionError("model input contains an undeclared physics channel")


__all__ = ["Wan22SymbolValueExecutionEngine"]

