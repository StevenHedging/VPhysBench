from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_STATUSES = {"evaluated", "unavailable", "not_applicable", "error"}


@dataclass
class MetricResult:
    name: str
    status: str
    score: float | None
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid metric status {self.status}")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(f"metric score outside [0,1]: {self.score}")
        if self.status == "evaluated" and self.score is None:
            raise ValueError("evaluated metric requires a score")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manual_score(prediction: dict[str, Any], key: str) -> float | None:
    value = prediction.get("manual_scores", {}).get(key)
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"manual score {key} outside [0,1]")
    return value

