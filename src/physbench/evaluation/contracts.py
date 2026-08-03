from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


CASE_STATUSES = {"evaluated", "unavailable", "unsupported", "error"}


@dataclass(frozen=True)
class CaseEvaluationRequest:
    job: dict[str, Any]
    case: dict[str, Any]
    case_catalog: dict[str, dict[str, Any]]
    prediction: dict[str, Any] | None
    asset_root: Path
    artifact_dir: Path
    evaluator_config: dict[str, Any]
    run_id: str | None = None
    save_visualizations: bool = False
    visualization_root: Path | None = None


@dataclass
class CaseEvaluationResult:
    job_id: str
    case_id: str
    scene_id: str
    evaluator: dict[str, Any]
    status: str
    score: float | None
    reason_code: str | None = None
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in CASE_STATUSES:
            raise ValueError(f"invalid case evaluation status: {self.status}")
        if self.status == "evaluated" and self.score is None:
            raise ValueError("evaluated case requires a score")
        if self.status != "evaluated" and self.score is not None:
            raise ValueError(f"{self.status} case cannot carry a score")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(f"case score outside [0,1]: {self.score}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CaseEvaluationResult":
        """Validate and reconstruct the canonical serialized result contract."""
        if not isinstance(value, dict):
            raise TypeError("case evaluation result must be an object")
        required = {
            "job_id",
            "case_id",
            "scene_id",
            "evaluator",
            "status",
            "score",
            "reason_code",
            "reason",
            "metrics",
            "quality",
            "artifacts",
            "provenance",
        }
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing or extra:
            raise ValueError(
                "case evaluation result fields differ from the canonical "
                f"contract: missing={missing}, extra={extra}"
            )
        for name in ("job_id", "case_id", "scene_id"):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError(
                    f"case evaluation result {name} must be a non-empty string"
                )
        for name in (
            "evaluator",
            "metrics",
            "quality",
            "artifacts",
            "provenance",
        ):
            if not isinstance(value[name], dict):
                raise TypeError(
                    f"case evaluation result {name} must be an object"
                )
        evaluator = value["evaluator"]
        for name in ("id", "version", "scene_id"):
            if not isinstance(evaluator.get(name), str) or not evaluator[name]:
                raise ValueError(
                    f"case evaluator {name} must be a non-empty string"
                )
        if evaluator["scene_id"] != value["scene_id"]:
            raise ValueError(
                "case evaluator scene_id differs from result scene_id"
            )
        for name in ("reason_code", "reason"):
            if value[name] is not None and not isinstance(value[name], str):
                raise TypeError(
                    f"case evaluation result {name} must be null or a string"
                )
        score = value["score"]
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("case evaluation score must be a finite number")
        return cls(**value)


class SceneCaseEvaluator(Protocol):
    evaluator_id: str
    evaluator_version: str
    scene_id: str

    def evaluate(self, request: CaseEvaluationRequest) -> CaseEvaluationResult:
        ...

    def describe(self) -> dict[str, Any]:
        ...
