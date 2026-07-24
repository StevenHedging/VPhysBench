from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .base import BaselineAdapter


class CommandAdapter(BaselineAdapter):
    def _command(self, stage: str, job_path: Path) -> list[str]:
        template = self.config.get("commands", {}).get(stage)
        if not isinstance(template, list) or not all(isinstance(token, str) for token in template):
            raise ValueError(f"missing command token list for stage={stage}")
        return [token.replace("{job_json}", str(job_path)) for token in template]

    def train(self, prepared_training: dict[str, Any], job_path: Path) -> dict[str, Any]:
        if prepared_training["status"] == "not_requested":
            return prepared_training
        capabilities = self.config.get("capabilities", {})
        stage = "finetune" if capabilities.get("finetune") else "train"
        command = self._command(stage, job_path)
        if not self.execute:
            return {
                **prepared_training,
                "status": "planned",
                "command": command,
                "note": "Training command execution disabled; review the frozen job and rerun with --execute.",
            }
        completed = subprocess.run(command, check=False)
        return {
            **prepared_training,
            "status": "complete" if completed.returncode == 0 else "failed",
            "command": command,
            "return_code": completed.returncode,
        }

    def generate(self, prepared_job: dict[str, Any], job_path: Path) -> dict[str, Any]:
        if not self.execute:
            return {
                "job_id": prepared_job["job_id"],
                "case_id": prepared_job["case_id"],
                "baseline_id": self.baseline_id,
                "prompt_profile_id": prepared_job["prompt_profile_id"],
                "evaluation_partition": prepared_job["evaluation_partition"],
                "status": "planned",
                "video_path": None,
                "manual_scores": {},
                "command": self._command("generate", job_path),
                "note": "Command execution disabled; rerun with --execute after reviewing the frozen job.",
            }
        command = self._command("generate", job_path)
        completed = subprocess.run(command, check=False)
        output = Path(prepared_job["output_video"])
        status = "complete" if completed.returncode == 0 and output.is_file() else "failed"
        return {
            "job_id": prepared_job["job_id"],
            "case_id": prepared_job["case_id"],
            "baseline_id": self.baseline_id,
            "prompt_profile_id": prepared_job["prompt_profile_id"],
            "evaluation_partition": prepared_job["evaluation_partition"],
            "status": status,
            "video_path": str(output) if output.is_file() else None,
            "manual_scores": {},
            "return_code": completed.returncode,
        }
