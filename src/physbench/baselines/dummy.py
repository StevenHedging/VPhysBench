from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaselineAdapter


class DummyAdapter(BaselineAdapter):
    def prepare_training(self, train_case_ids: list[str], run_dir: Path) -> dict[str, Any]:
        value = super().prepare_training(train_case_ids, run_dir)
        if train_case_ids:
            value.update(status="placeholder", note="No model is installed; training contract only.")
        return value

    def generate(self, prepared_job: dict[str, Any], job_path: Path) -> dict[str, Any]:
        return {
            "job_id": prepared_job["job_id"],
            "case_id": prepared_job["case_id"],
            "baseline_id": self.baseline_id,
            "prompt_profile_id": prepared_job["prompt_profile_id"],
            "evaluation_partition": prepared_job["evaluation_partition"],
            "status": "placeholder",
            "video_path": None,
            "manual_scores": {},
            "note": "Dummy adapter validates orchestration but does not generate video.",
            "job_spec": str(job_path),
        }
