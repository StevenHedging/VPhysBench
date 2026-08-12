from __future__ import annotations

from pathlib import Path

from ...baseline_plugins.wan22_subject_motion import (
    Wan22SubjectMotionExecutionEngine,
)
from .wan22 import Wan22ManagedDriver


class Wan22SubjectMotionManagedDriver(Wan22ManagedDriver):
    """Managed runtime for the WAN subject-motion auxiliary baseline."""

    def _execution_engine_class(self):
        return Wan22SubjectMotionExecutionEngine

    def dependency_paths(self) -> dict[str, Path]:
        root = Path(__file__).resolve().parents[4]
        paths = super().dependency_paths()
        relative = (
            "src/physbench/baseline_runtime/drivers/wan22_subject_motion.py",
            "src/physbench/baseline_plugins/wan22_subject_motion.py",
            "src/physbench/baselines/wan22_subject_motion.py",
            "src/physbench/baselines/wan22_subject_motion_model.py",
            "src/physbench/baselines/wan22_st_tube_iou.py",
            "src/physbench/baselines/wan22_st_tube_iou_model.py",
            "src/physbench/baselines/wan22_st_tube_iou_masks.py",
            "scripts/train_wan22_subject_motion.sh",
            "scripts/wan22_subject_motion_train.py",
            "scripts/wan22_st_tube_iou_train.py",
        )
        paths.update({name: root / name for name in relative})
        missing = [
            f"{name}={path}"
            for name, path in paths.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"WAN subject-motion dependencies missing: {missing}"
            )
        return paths


__all__ = ["Wan22SubjectMotionManagedDriver"]
