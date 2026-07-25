from __future__ import annotations

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


class SceneCaseEvaluator(Protocol):
    evaluator_id: str
    evaluator_version: str
    scene_id: str

    def evaluate(self, request: CaseEvaluationRequest) -> CaseEvaluationResult:
        ...

    def describe(self) -> dict[str, Any]:
        ...
