from __future__ import annotations

from pathlib import Path
from typing import Any

from ...baseline_plugins.wan22_symbol_value import (
    Wan22SymbolValueExecutionEngine,
)
from .wan22 import Wan22ManagedDriver


class Wan22SymbolValueManagedDriver(Wan22ManagedDriver):
    def _execution_engine_class(self):
        return Wan22SymbolValueExecutionEngine

    def _adapter_recipe(self) -> dict[str, Any]:
        return self.bundle.value["adapter"]["config"]

    def dependency_paths(self) -> dict[str, Path]:
        root = Path(__file__).resolve().parents[4]
        paths = super().dependency_paths()
        relative = (
            "src/physbench/baseline_runtime/drivers/wan22_symbol_value.py",
            "src/physbench/baseline_plugins/wan22_symbol_value.py",
            "src/physbench/baselines/wan22_symbol_value.py",
            "src/physbench/baselines/wan22_symbol_value_model.py",
            "scripts/train_wan22_symbol_value.sh",
            "scripts/wan22_symbol_value_train.py",
            "scripts/wan22_symbol_value_generate.py",
            "scripts/wan22_symbol_value_generate_batch.py",
        )
        paths.update({name: root / name for name in relative})
        missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"WAN symbol-value runtime dependencies missing: {missing}"
            )
        return paths


__all__ = ["Wan22SymbolValueManagedDriver"]
