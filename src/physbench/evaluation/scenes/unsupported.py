from __future__ import annotations

from typing import Any

from ..contracts import CaseEvaluationRequest, CaseEvaluationResult


class UnsupportedSceneEvaluator:
    evaluator_id = "unsupported_scene"
    evaluator_version = "1.0"

    def __init__(self, scene_id: str, config: dict[str, Any] | None = None):
        self.scene_id = scene_id
        self.config = config or {}

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.evaluator_id,
            "version": self.evaluator_version,
            "scene_id": self.scene_id,
            "implemented": False,
        }

    def evaluate(self, request: CaseEvaluationRequest) -> CaseEvaluationResult:
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=self.describe(),
            status="unsupported",
            score=None,
            reason_code="scene_evaluator_not_implemented",
            reason=(
                f"case evaluator for scene {request.case['scene_id']} "
                "has not been implemented"
            ),
        )
