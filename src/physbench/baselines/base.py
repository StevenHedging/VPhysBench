from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaselineAdapter(ABC):
    def __init__(self, config: dict[str, Any], execute: bool = False):
        self.config = config
        self.execute = execute

    @property
    def baseline_id(self) -> str:
        return self.config["baseline_id"]

    @property
    def input_view(self) -> str:
        return self.config["input_view"]

    def prepare_job(self, job: dict[str, Any], case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        if self.input_view not in case["input_views"]:
            raise ValueError(f"case {case['case_id']} does not provide input view {self.input_view}")
        profile_id = job.get("prompt_profile_id", "unprofiled")
        output = run_dir / "predictions" / profile_id / f"{job['job_id']}.mp4"
        model_input = dict(case["input_views"][self.input_view])
        if job.get("resolved_prompt"):
            model_input["prompt"] = job["resolved_prompt"]["prompt"]
        return {
            **job,
            "baseline_id": self.baseline_id,
            "input_view": self.input_view,
            "model_input": model_input,
            "physical_parameters": case["physical_parameters"],
            "training_artifact_dir": str(run_dir / "artifacts"),
            "data_context": str(run_dir / "data_context.json"),
            "output_video": str(output),
        }

    def prepare_training(self, train_case_ids: list[str], run_dir: Path) -> dict[str, Any]:
        capabilities = self.config.get("capabilities", {})
        if train_case_ids and not (capabilities.get("train") or capabilities.get("finetune")):
            raise ValueError(f"baseline {self.baseline_id} cannot train or finetune")
        return {
            "status": "not_requested" if not train_case_ids else "planned",
            "case_ids": train_case_ids,
            "cases_manifest": str(run_dir / "frozen_cases.jsonl"),
            "task_plan": str(run_dir / "plan.json"),
            "resolved_prompts": str(run_dir / "resolved_prompts.jsonl"),
            "frozen_prompt_profiles": str(run_dir / "frozen_prompt_profiles.json"),
            "data_context": str(run_dir / "data_context.json"),
            "artifact_dir": str(run_dir / "artifacts"),
            "checkpoint_manifest": str(run_dir / "artifacts" / "checkpoint.json"),
        }

    def train(self, prepared_training: dict[str, Any], job_path: Path) -> dict[str, Any]:
        """Execute or stage training. Adapters override this when they own a trainer."""
        return prepared_training

    @abstractmethod
    def generate(self, prepared_job: dict[str, Any], job_path: Path) -> dict[str, Any]:
        raise NotImplementedError
