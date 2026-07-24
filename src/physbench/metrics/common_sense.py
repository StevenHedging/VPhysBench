from __future__ import annotations

from typing import Any

from .base import MetricResult, manual_score


def evaluate(case: dict[str, Any], prediction: dict[str, Any], scene: dict[str, Any], plugin: dict[str, Any]) -> MetricResult:
    score = manual_score(prediction, "common_sense")
    checks = scene.get("metric_spec", {}).get("common_sense_checks", [])
    if score is not None:
        return MetricResult("common_sense", "evaluated", score, "externally supplied judge score", {"checks": checks})
    return MetricResult(
        "common_sense", "unavailable", None,
        "VLM/common-sense judge is not configured; placeholder did not fabricate a score",
        {"planned_checks": checks, "plugin": plugin},
    )

