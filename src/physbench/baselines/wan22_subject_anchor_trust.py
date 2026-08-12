from __future__ import annotations

from pathlib import Path
from typing import Any

from .wan22_subject_motion import Wan22SubjectMotionAdapter


class Wan22SubjectAnchorTrustAdapter(Wan22SubjectMotionAdapter):
    """WAN fine-tuning with fixed-probe motion and LoRA trust region."""

    def _training_command(self, runtime_root: Path) -> list[str]:
        del runtime_root
        return [
            "bash",
            str(self.project_root / "scripts/train_wan22_subject_anchor_trust.sh"),
        ]

    def _training_environment(
        self,
        run_dir: Path,
        dataset_dir: Path,
        metadata: Path,
    ) -> dict[str, str]:
        environment = super()._training_environment(run_dir, dataset_dir, metadata)
        environment["SUBJECT_MOTION_METRICS_PATH"] = str(
            run_dir
            / "artifacts"
            / "wan22"
            / "training_subject_anchor_trust_metrics.jsonl"
        )
        return environment

    def train(
        self,
        prepared_training: dict[str, Any],
        job_path: Path,
    ) -> dict[str, Any]:
        result = super().train(prepared_training, job_path)
        return {**result, "adapter": "wan22_subject_anchor_trust"}


__all__ = ["Wan22SubjectAnchorTrustAdapter"]
