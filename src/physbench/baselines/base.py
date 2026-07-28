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

    def prepare_training(self, train_case_ids: list[str], run_dir: Path) -> dict[str, Any]:
        capabilities = self.config.get("capabilities", {})
        if train_case_ids and not (capabilities.get("train") or capabilities.get("finetune")):
            raise ValueError(f"baseline {self.baseline_id} cannot train or finetune")
        return {
            "status": "not_requested" if not train_case_ids else "planned",
            "case_ids": train_case_ids,
            "cases_manifest": str(run_dir / "frozen_cases.jsonl"),
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
